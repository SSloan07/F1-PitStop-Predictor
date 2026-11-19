"""Gradient-boosting diagnostic: which inputs carry pit-entry signal?

Trains LightGBM on single-row summaries of the same causal samples used by the
LSTM (train 2022-2023, validation 2024; 2025 is never loaded) and compares
four input sets:

* ``equivalente_lstm``: last visible value and window mean of every LSTM
  numeric input, the last dynamic categories and the static IDs.
* ``telemetria``: the telemetry-only projection (no static context, no
  race-constant weather).
* ``telemetria_estrategia``: telemetry plus new strategy features.
* ``todo``: everything.

Strategy features are rebuilt from the lap matrix using only rows/events whose
timestamp is not later than ``decision_time_seconds`` (start of the predicted
lap). The number of boosting rounds is chosen with grouped CV by race on train;
validation is used only for the final report and the F2 threshold, mirroring
the LSTM protocol. This is a diagnostic, not a replacement model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import load_split, sha256, verify_preprocessed, write_json
from f1_pitstop.strategy_features import SEASONS, median_stint_length_train, scheduled_laps, strategy_features

PACKAGE = ROOT / "EDA/dataset/muestras_preprocesadas_v1_3"
PACKAGE_MANIFEST_SHA256 = "08afa3c1f406f6bbb33c27a5e688b0d0fd46658ec8398a095f8981d74abdbf66"
LAPS = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl"
SAMPLE_INDEX = ROOT / "EDA/dataset/muestras_causales_v1_3/sample_index.csv.gz"
CACHE = ROOT / "EDA/dataset/cache_fastf1_carreras"
REMOVED_WEATHER = ("Weather_AirTemp", "Weather_Humidity", "Weather_Pressure", "Weather_TrackTemp",
                   "Weather_WindDirection_sin", "Weather_WindDirection_cos", "Weather_WindSpeed")
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "verbose": -1, "seed": 42, "deterministic": True, "force_row_wise": True, "num_threads": 4}


# --------------------------------------------------------------------------- base features
def base_features(split, state):
    """Per-sample summary of exactly what the LSTM sees (already causal and masked)."""
    schema = state["schema"]
    numeric_names = schema["numeric_feature_names"]
    arrays, positions = split.arrays, split.positions
    values = arrays["X_numeric"][positions].astype(np.float64)
    reasons = arrays["mask_reason_codes"][positions][..., :len(numeric_names)]
    present = reasons == 0
    masked = np.where(present, values, np.nan)
    frame = {}
    width = values.shape[1]
    for index, name in enumerate(numeric_names):
        column = masked[:, :, index]
        has = present[:, :, index]
        last_slot = np.where(has.any(1), width - 1 - np.argmax(has[:, ::-1], axis=1), -1)
        last = np.where(last_slot >= 0, column[np.arange(len(column)), np.clip(last_slot, 0, None)], np.nan)
        frame[f"{name}__last"] = last
        with np.errstate(all="ignore"):
            frame[f"{name}__mean_w8"] = np.nanmean(column, axis=1) if has.any() else np.nan
    dynamic = arrays["X_dynamic_categorical_ids"][positions]
    visible = ~arrays["padding_mask"][positions] & arrays["slot_exists_mask"][positions]
    for index, name in enumerate(schema["categorical_feature_names"]):
        ids = np.where(visible, dynamic[:, :, index], 0)
        last_slot = width - 1 - np.argmax((ids > 0)[:, ::-1], axis=1)
        frame[f"{name}__last"] = ids[np.arange(len(ids)), last_slot]
    static = arrays["X_static_categorical_ids"][positions]
    for index, name in enumerate(schema["static_feature_names"]):
        frame[f"{name}__id"] = static[:, index]
    out = pd.DataFrame(frame)
    categorical = [c for c in out if c.endswith("__id") or c in
                   {f"{n}__last" for n in schema["categorical_feature_names"]}]
    return out, categorical


# --------------------------------------------------------------------------- modelling
def _cv_rounds(lgb, X, y, groups, categorical):
    from sklearn.model_selection import GroupKFold
    best = []
    for fit_idx, hold_idx in GroupKFold(n_splits=5).split(X, y, groups):
        fit = lgb.Dataset(X.iloc[fit_idx], y[fit_idx], categorical_feature=categorical, free_raw_data=False)
        hold = lgb.Dataset(X.iloc[hold_idx], y[hold_idx], categorical_feature=categorical, reference=fit)
        booster = lgb.train({**LGB_PARAMS, "metric": "average_precision"}, fit, num_boost_round=2000,
                            valid_sets=[hold], callbacks=[lgb.early_stopping(100, verbose=False)])
        best.append(booster.best_iteration)
    return int(np.median(best)), best


def _permutation_importance(booster, X, y, columns, repeats=3, seed=42):
    rng = np.random.default_rng(seed)
    base = binary_metrics(y, booster.predict(X), 0.5)["average_precision"]
    result = {}
    for column in columns:
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            drops.append(base - binary_metrics(y, booster.predict(shuffled), 0.5)["average_precision"])
        result[column] = float(np.mean(drops))
    return base, result


def _group_permutation(booster, X, y, groups, repeats=3, seed=42):
    rng = np.random.default_rng(seed)
    base = binary_metrics(y, booster.predict(X), 0.5)["average_precision"]
    result = {}
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            order = rng.permutation(len(X))
            shuffled[columns] = X[columns].to_numpy()[order]
            drops.append(base - binary_metrics(y, booster.predict(shuffled), 0.5)["average_precision"])
        result[name] = float(np.mean(drops))
    return result


def _feature_group(column):
    if column.startswith(("Driver", "Team", "GranPremio")):
        return "contexto_estatico"
    if column.startswith("Weather_"):
        return "clima"
    if column.startswith("Telemetry_"):
        return "telemetria_coche"
    if column.startswith(("LapTime", "Sector")):
        return "ritmo"
    if column.startswith(("TyreLife", "Compound", "tyre_life", "stops_done", "laps_since", "dry_comp",
                          "two_dry", "used_wet", "pace_")):
        return "neumatico_y_stint"
    if column.startswith(("LapNumber", "race_fraction", "laps_remaining")):
        return "progreso_carrera"
    if column.startswith(("track_phase",)):
        return "estado_pista"
    return "posicion_y_rivales"


def run(output, root=ROOT):
    import lightgbm as lgb
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    _, state = verify_preprocessed(PACKAGE, PACKAGE_MANIFEST_SHA256)
    splits = {name: load_split(PACKAGE, name, state) for name in ("train", "validation")}
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(SEASONS)].copy()
    index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
    total_laps = scheduled_laps(CACHE)
    stint_median = median_stint_length_train(laps)
    frames, targets, meta = {}, {}, {}
    for name, split in splits.items():
        base, categorical = base_features(split, state)
        samples = split.metadata.iloc[split.positions][["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]]
        samples = samples.astype({"DriverNumber": str}).merge(
            index[["sample_id", "decision_time_seconds"]], on="sample_id", how="left", validate="one_to_one")
        if samples.decision_time_seconds.isna().any():
            raise ValueError("Missing decision times")
        strategy = strategy_features(samples, laps, total_laps, stint_median)
        if not np.array_equal(strategy.sample_id.to_numpy(), samples.sample_id.to_numpy()):
            strategy = samples[["sample_id"]].merge(strategy, on="sample_id", how="left", validate="one_to_one")
        frames[name] = pd.concat([base.reset_index(drop=True), strategy.drop(columns="sample_id").reset_index(drop=True)], axis=1)
        targets[name] = split.targets.astype(int)
        meta[name] = samples.reset_index(drop=True)
    strategy_columns = [c for c in frames["train"] if "__" not in c]
    base_columns = [c for c in frames["train"] if "__" in c]
    static_columns = [c for c in base_columns if c.endswith("__id")]
    weather_removed = [c for c in base_columns if c.split("__")[0] in REMOVED_WEATHER]
    telemetry_columns = [c for c in base_columns if c not in static_columns and c not in weather_removed]
    variants = {
        "equivalente_lstm": base_columns,
        "telemetria": telemetry_columns,
        "telemetria_estrategia": telemetry_columns + strategy_columns,
        "todo": base_columns + strategy_columns,
    }
    groups = (meta["train"].Temporada.astype(str) + "-" + meta["train"].Ronda.astype(str)).to_numpy()
    results, models = {}, {}
    for variant, columns in variants.items():
        cats = [c for c in categorical if c in columns]
        X_train, X_val = frames["train"][columns], frames["validation"][columns]
        rounds, fold_rounds = _cv_rounds(lgb, X_train, targets["train"], groups, cats)
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, targets["train"], categorical_feature=cats),
                            num_boost_round=rounds)
        probabilities = booster.predict(X_val)
        selected = select_metric_threshold(targets["validation"], probabilities, THRESHOLD_GRID, metric="f2")
        results[variant] = {"n_features": len(columns), "boosting_rounds": rounds, "cv_fold_rounds": fold_rounds,
                            "validation_selected_threshold": selected,
                            "validation_average_precision": selected["metrics"]["average_precision"],
                            "train_average_precision": binary_metrics(targets["train"], booster.predict(X_train), 0.5)["average_precision"]}
        models[variant] = (booster, columns, probabilities)
        print(variant, json.dumps({k: results[variant][k] for k in ("n_features", "boosting_rounds", "validation_average_precision")}),
              "F2", round(selected["metrics"]["f2"], 5), flush=True)
    output.mkdir(parents=True)
    importance_rows = []
    for variant in ("telemetria_estrategia", "todo"):
        booster, columns, probabilities = models[variant]
        X_val = frames["validation"][columns]
        gain = dict(zip(booster.feature_name(), booster.feature_importance("gain")))
        contributions = booster.predict(X_val, pred_contrib=True)[:, :-1]
        shap = dict(zip(columns, np.abs(contributions).mean(0)))
        _, permutation = _permutation_importance(booster, X_val, targets["validation"], columns)
        total_gain = sum(gain.values())
        for column in columns:
            importance_rows.append({"variant": variant, "feature": column, "group": _feature_group(column),
                                    "gain_share": gain[column] / total_gain, "mean_abs_shap": float(shap[column]),
                                    "permutation_ap_drop": permutation[column]})
        grouped = {}
        for column in columns:
            grouped.setdefault(_feature_group(column), []).append(column)
        results[variant]["group_permutation_ap_drop"] = _group_permutation(booster, X_val, targets["validation"], grouped)
    importance = pd.DataFrame(importance_rows).sort_values(["variant", "permutation_ap_drop"], ascending=[True, False])
    importance.to_csv(output / "importancia_variables.csv", index=False)
    # Track-phase breakdown for the strategy model.
    booster, columns, probabilities = models["telemetria_estrategia"]
    threshold = results["telemetria_estrategia"]["validation_selected_threshold"]["threshold"]
    phase_ids = {entry["id"]: entry["category"] for entry in state["vocabularies"]["dynamic"]["track_phase_asof"]}
    phase = frames["validation"]["track_phase_asof__last"].map(lambda v: phase_ids.get(int(v), "OTRO")).to_numpy()
    breakdown = {}
    for label, mask in (("GREEN", phase == "GREEN"), ("NO_GREEN", phase != "GREEN")):
        breakdown[label] = {"samples": int(mask.sum()), "positives": int(targets["validation"][mask].sum()),
                            **binary_metrics(targets["validation"][mask], probabilities[mask], threshold)}
    results["telemetria_estrategia"]["by_track_phase_at_decision"] = breakdown
    # Univariate signal of strategy features (validation positive rate by bins).
    univariate = {}
    val = frames["validation"].assign(target=targets["validation"])
    for column in ("stops_done", "two_dry_compounds_used", "field_pit_entries_last2laps", "ahead_pitted_last2laps"):
        univariate[column] = val.groupby(val[column].clip(upper=3)).target.agg(["count", "mean"]).reset_index().to_dict("records")
    results["telemetria_estrategia"]["validation_positive_rate_by_value"] = univariate
    feature_table = pd.concat([meta["validation"], frames["validation"][strategy_columns]], axis=1)
    feature_table.to_csv(output / "variables_estrategia_validacion.csv.gz", index=False, compression="gzip")
    write_json(output / "metrics.json", {
        "protocol": {"train": [2022, 2023], "validation": [2024], "test_2025_loaded": False,
                     "rounds_selection": "median best iteration of 5-fold GroupKFold by race on train (AP early stopping)",
                     "threshold_selection": "F2 on validation grid 0.05..0.95, same rule as LSTM",
                     "lightgbm_params": LGB_PARAMS, "lightgbm_version": lgb.__version__},
        "train_eligible": int(len(targets["train"])), "validation_eligible": int(len(targets["validation"])),
        "validation_positive_rate": float(targets["validation"].mean()),
        "stint_median_train": stint_median, "variants": results,
        "strategy_features": strategy_columns,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "gbm_diagnostic_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "script": sha256(Path(__file__))},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evidencia_gbm_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
