"""Second-round strategy features: train-only grouped CV ablation, then one 2024 confirmation.

Selection uses only train races (2022-2023): 5-fold GroupKFold by race, with
early stopping on held-out races inside each training fold. All variants share
the same folds, so differences are paired by fold. The selection rule is fixed
before looking at results:

    keep a group if removing it from "base + all groups" lowers out-of-fold AP
    on average by more than 0.002 and in at least 4 of 5 folds.

Validation 2024 is used once, to confirm base vs base+selected vs base+all.
The base is the round-1 GBM input set (telemetry + round-1 strategy). 2025 is
never loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diagnostico_gbm import (CACHE, LAPS, LGB_PARAMS, PACKAGE, PACKAGE_MANIFEST_SHA256, REMOVED_WEATHER,
                             SAMPLE_INDEX, THRESHOLD_GRID, _cv_rounds, _permutation_importance, base_features)
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import load_split, sha256, verify_preprocessed, write_json
from f1_pitstop.strategy_features import SEASONS, median_stint_length_train, scheduled_laps, strategy_features
from f1_pitstop.strategy_features_r2 import FEATURE_GROUPS, pit_loss_table, strategy_features_round2

MIN_MEAN_DROP = 0.002
MIN_FOLDS = 4


def build_frames():
    _, state = verify_preprocessed(PACKAGE, PACKAGE_MANIFEST_SHA256)
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(SEASONS)].copy()
    index = pd.read_csv(SAMPLE_INDEX, dtype={"DriverNumber": str})
    total_laps = scheduled_laps(CACHE)
    stint_median = median_stint_length_train(laps)
    losses = pit_loss_table(laps)
    frames = {}
    for name in ("train", "validation"):
        split = load_split(PACKAGE, name, state)
        base, categorical = base_features(split, state)
        samples = split.metadata.iloc[split.positions][["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]]
        samples = samples.astype({"DriverNumber": str}).merge(
            index[["sample_id", "decision_time_seconds"]], on="sample_id", how="left", validate="one_to_one")
        round1 = strategy_features(samples, laps, total_laps, stint_median)
        round2 = strategy_features_round2(samples, laps, total_laps, stint_median, losses)
        for table in (round1, round2):
            if not np.array_equal(table.sample_id.to_numpy(), samples.sample_id.to_numpy()):
                raise ValueError("Strategy feature order differs from samples")
        if set(round1.columns) & set(round2.columns) - {"sample_id"}:
            raise ValueError("Round-2 feature names collide with round 1")
        frame = pd.concat([base.reset_index(drop=True), round1.drop(columns="sample_id"),
                           round2.drop(columns="sample_id")], axis=1)
        frames[name] = (frame, split.targets.astype(int), samples.reset_index(drop=True))
    base_columns = [c for c in frames["train"][0] if "__" in c and not c.endswith("__id")
                    and c.split("__")[0] not in REMOVED_WEATHER]
    round1_columns = [c for c in frames["train"][0] if "__" not in c and c not in sum(FEATURE_GROUPS.values(), [])]
    return frames, [c for c in categorical if c in base_columns], base_columns + round1_columns


def cv_scores(lgb, X, y, races, categorical, folds):
    """Per-fold out-of-fold AP with early stopping on held-out races inside the training fold."""
    from sklearn.model_selection import GroupShuffleSplit
    oof = np.zeros(len(y))
    fold_ap = []
    for fit_idx, hold_idx in folds:
        inner_fit, inner_stop = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                                     .split(fit_idx, groups=races[fit_idx]))
        fit_rows, stop_rows = fit_idx[inner_fit], fit_idx[inner_stop]
        fit = lgb.Dataset(X.iloc[fit_rows], y[fit_rows], categorical_feature=categorical, free_raw_data=False)
        stop = lgb.Dataset(X.iloc[stop_rows], y[stop_rows], categorical_feature=categorical, reference=fit)
        booster = lgb.train({**LGB_PARAMS, "metric": "average_precision"}, fit, num_boost_round=2000,
                            valid_sets=[stop], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[hold_idx] = booster.predict(X.iloc[hold_idx], num_iteration=booster.best_iteration)
        fold_ap.append(binary_metrics(y[hold_idx], oof[hold_idx], 0.5)["average_precision"])
    selected = select_metric_threshold(y, oof, THRESHOLD_GRID, metric="f2")
    return {"fold_ap": fold_ap, "oof_ap": binary_metrics(y, oof, 0.5)["average_precision"],
            "oof_f2": selected["metrics"]["f2"], "oof_threshold": selected["threshold"]}


def run(output):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frames, categorical, base = build_frames()
    X_train, y_train, meta_train = frames["train"]
    races = (meta_train.Temporada.astype(str) + "-" + meta_train.Ronda.astype(str)).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X_train, y_train, races))
    all_new = sum(FEATURE_GROUPS.values(), [])
    variants = {"base": base, "base+todas": base + all_new}
    for group, columns in FEATURE_GROUPS.items():
        variants[f"base+{group}"] = base + columns
        variants[f"base+todas-{group}"] = [c for c in base + all_new if c not in columns]
    cv = {}
    for name, columns in variants.items():
        cats = [c for c in categorical if c in columns]
        cv[name] = cv_scores(lgb, X_train[columns], y_train, races, cats, folds)
        print(f"{name:32s} OOF AP {cv[name]['oof_ap']:.4f}  F2 {cv[name]['oof_f2']:.4f}", flush=True)

    def paired(first, second):
        diff = np.array(cv[second]["fold_ap"]) - np.array(cv[first]["fold_ap"])
        return {"mean_delta_ap": float(diff.mean()), "folds_positive": int((diff > 0).sum()), "per_fold": diff.tolist()}

    groups = {}
    for group in FEATURE_GROUPS:
        added = paired("base", f"base+{group}")
        removed = paired(f"base+todas-{group}", "base+todas")  # positive = group helps on top of the others
        groups[group] = {"features": FEATURE_GROUPS[group], "add_to_base": added, "drop_from_all": removed,
                         "selected": removed["mean_delta_ap"] > MIN_MEAN_DROP and removed["folds_positive"] >= MIN_FOLDS}
    selected = [g for g, v in groups.items() if v["selected"]]
    final = base + sum((FEATURE_GROUPS[g] for g in selected), [])
    print("Grupos seleccionados:", selected, flush=True)

    # One confirmation on validation 2024.
    X_val, y_val, _ = frames["validation"]
    confirmation, models = {}, {}
    for name, columns in (("base", base), ("base+seleccion", final), ("base+todas", base + all_new)):
        if name == "base+seleccion" and columns == base:
            continue
        cats = [c for c in categorical if c in columns]
        rounds, _ = _cv_rounds(lgb, X_train[columns], y_train, races, cats)
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_train[columns], y_train, categorical_feature=cats),
                            num_boost_round=rounds)
        probabilities = booster.predict(X_val[columns])
        chosen = select_metric_threshold(y_val, probabilities, THRESHOLD_GRID, metric="f2")
        confirmation[name] = {"n_features": len(columns), "boosting_rounds": rounds, "threshold": chosen["threshold"],
                              **{k: chosen["metrics"][k] for k in ("average_precision", "f2", "f1", "precision",
                                                                   "recall", "tp", "fp", "fn")}}
        models[name] = (booster, columns)
        print(f"validación {name:16s} AP {confirmation[name]['average_precision']:.4f} F2 {confirmation[name]['f2']:.4f}", flush=True)
    importance_name = "base+seleccion" if "base+seleccion" in models else "base+todas"
    booster, columns = models[importance_name]
    _, permutation = _permutation_importance(booster, X_val[columns], y_val, columns)
    new_columns = [c for c in columns if c in all_new]
    importance = pd.DataFrame({"feature": columns, "permutation_ap_drop": [permutation[c] for c in columns],
                               "round2": [c in new_columns for c in columns]}).sort_values("permutation_ap_drop", ascending=False)
    output.mkdir(parents=True)
    importance.to_csv(output / "importancia_validacion.csv", index=False)
    write_json(output / "metrics.json", {
        "protocol": {"selection_split": "train 2022-2023 only, 5-fold GroupKFold by race, inner 20% race early stopping",
                     "selection_rule": f"drop-from-all mean ΔAP > {MIN_MEAN_DROP} and positive in >= {MIN_FOLDS}/5 folds",
                     "confirmation_split": "validation 2024, used once", "test_2025_loaded": False,
                     "base": "round-1 GBM telemetry + strategy set", "lightgbm_params": LGB_PARAMS,
                     "lightgbm_version": lgb.__version__},
        "cv": cv, "groups": groups, "selected_groups": selected, "confirmation_2024": confirmation,
        "importance_model": importance_name,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "gbm_round2_diagnostic_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "script": sha256(Path(__file__)),
                          "round2_module": sha256(ROOT / "src/f1_pitstop/strategy_features_r2.py")},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evidencia_gbm_ronda2_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
