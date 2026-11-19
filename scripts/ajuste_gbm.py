"""LightGBM hyperparameter search with train-only grouped CV, then one 2024 confirmation.

Input set: telemetry + strategy rounds 1-3 (66 features, the current best).
Search: the current default parameters plus N random configurations drawn with
a fixed seed from a declared grid. Each is scored by out-of-fold AP on the
same five race-grouped train folds (inner 20 % race early stopping, as in the
feature rounds). Pre-declared rule:

    adopt the best configuration if it beats the default by more than 0.002
    mean fold AP and in at least 4 of 5 folds; otherwise keep the default.

Validation 2024 is used once to compare default vs adopted (both as the mean
of seeds 42-44, number of trees chosen by train CV). 2025 is never loaded.
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
from diagnostico_gbm import CACHE, LAPS, LGB_PARAMS, PACKAGE, SAMPLE_INDEX, THRESHOLD_GRID
from diagnostico_gbm_ronda2 import build_frames
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import sha256, write_json
from f1_pitstop.strategy_features import SEASONS, scheduled_laps
from f1_pitstop.strategy_features_r2 import FEATURE_GROUPS
from f1_pitstop.strategy_features_r3 import FEATURES as ROUND3, stop_priors, strategy_features_round3

GRID = {
    "num_leaves": [7, 15, 31, 63, 127],
    "min_child_samples": [20, 50, 100, 200, 400],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "feature_fraction": [0.5, 0.7, 0.8, 1.0],
    "bagging_fraction": [0.6, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 5.0, 20.0],
    "min_gain_to_split": [0.0, 0.1, 0.5],
    "max_bin": [63, 255],
}
N_RANDOM = 30
SEARCH_SEED = 2026
MIN_MEAN_GAIN = 0.002
MIN_FOLDS = 4
SEEDS = (42, 43, 44)


def frames_round3():
    frames, categorical, base_r1 = build_frames()
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(SEASONS)].copy()
    total_laps = scheduled_laps(CACHE)
    race_stops, long_stint = stop_priors(laps)
    for name, (frame, y, meta) in frames.items():
        round3 = strategy_features_round3(meta, laps, total_laps, race_stops, long_stint)
        if not np.array_equal(round3.sample_id.to_numpy(), meta.sample_id.to_numpy()):
            raise ValueError("Round-3 feature order differs from samples")
        frames[name] = (pd.concat([frame, round3.drop(columns="sample_id")], axis=1), y, meta)
    columns = base_r1 + sum(FEATURE_GROUPS.values(), []) + ROUND3
    return frames, [c for c in categorical if c in columns], columns


def configurations():
    rng = np.random.default_rng(SEARCH_SEED)
    configs = [{"name": "default", **{k: LGB_PARAMS[k] for k in ("num_leaves", "min_child_samples", "learning_rate",
                                                                   "feature_fraction", "bagging_fraction", "lambda_l2")}}]
    seen = set()
    while len(configs) < N_RANDOM + 1:
        config = {key: values[rng.integers(len(values))] for key, values in GRID.items()}
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append({"name": f"random_{len(configs):02d}", **config})
    return configs


def params_for(config, seed=42):
    params = {**LGB_PARAMS, "seed": seed}
    params.update({k: v for k, v in config.items() if k != "name"})
    if params.get("bagging_fraction", 1.0) >= 1.0:
        params["bagging_freq"] = 0
    return params


def cv_fold_ap(lgb, params, X, y, races, categorical, folds):
    from sklearn.model_selection import GroupShuffleSplit
    fold_ap, best_iterations, oof = [], [], np.zeros(len(y))
    for fit_idx, hold_idx in folds:
        inner_fit, inner_stop = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                                     .split(fit_idx, groups=races[fit_idx]))
        fit_rows, stop_rows = fit_idx[inner_fit], fit_idx[inner_stop]
        fit = lgb.Dataset(X.iloc[fit_rows], y[fit_rows], categorical_feature=categorical, free_raw_data=False)
        stop = lgb.Dataset(X.iloc[stop_rows], y[stop_rows], categorical_feature=categorical, reference=fit)
        booster = lgb.train({**params, "metric": "average_precision"}, fit, num_boost_round=5000,
                            valid_sets=[stop], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[hold_idx] = booster.predict(X.iloc[hold_idx], num_iteration=booster.best_iteration)
        fold_ap.append(binary_metrics(y[hold_idx], oof[hold_idx], 0.5)["average_precision"])
        best_iterations.append(booster.best_iteration)
    return fold_ap, best_iterations, binary_metrics(y, oof, 0.5)["average_precision"]


def full_rounds(lgb, params, X, y, races, categorical):
    """Trees for the final fit: median best iteration over GroupKFold on all train (as before)."""
    from sklearn.model_selection import GroupKFold
    best = []
    for fit_idx, hold_idx in GroupKFold(n_splits=5).split(X, y, races):
        fit = lgb.Dataset(X.iloc[fit_idx], y[fit_idx], categorical_feature=categorical, free_raw_data=False)
        hold = lgb.Dataset(X.iloc[hold_idx], y[hold_idx], categorical_feature=categorical, reference=fit)
        booster = lgb.train({**params, "metric": "average_precision"}, fit, num_boost_round=5000,
                            valid_sets=[hold], callbacks=[lgb.early_stopping(150, verbose=False)])
        best.append(booster.best_iteration)
    return int(np.median(best))


def run(output):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frames, categorical, columns = frames_round3()
    X_train, y_train, meta_train = frames["train"]
    X_train = X_train[columns]
    races = (meta_train.Temporada.astype(str) + "-" + meta_train.Ronda.astype(str)).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X_train, y_train, races))
    search = []
    for config in configurations():
        fold_ap, iterations, oof_ap = cv_fold_ap(lgb, params_for(config), X_train, y_train, races, categorical, folds)
        search.append({**config, "mean_fold_ap": float(np.mean(fold_ap)), "oof_ap": oof_ap,
                       "fold_ap": fold_ap, "best_iterations": iterations})
        print(f"{config['name']:10s} AP medio {np.mean(fold_ap):.4f} OOF {oof_ap:.4f} "
              f"{ {k: v for k, v in config.items() if k != 'name'} }", flush=True)
    table = pd.DataFrame(search).sort_values("mean_fold_ap", ascending=False)
    default = next(row for row in search if row["name"] == "default")
    best = max(search, key=lambda row: row["mean_fold_ap"])
    diff = np.array(best["fold_ap"]) - np.array(default["fold_ap"])
    adopted = bool(best["name"] != "default" and diff.mean() > MIN_MEAN_GAIN and (diff > 0).sum() >= MIN_FOLDS)
    decision = {"best": best["name"], "mean_delta_ap_vs_default": float(diff.mean()),
                "folds_positive": int((diff > 0).sum()), "per_fold": diff.tolist(), "adopted": adopted}
    print("Decisión:", decision, flush=True)
    chosen_config = best if adopted else default

    X_val, y_val, _ = frames["validation"]
    X_val = X_val[columns]
    confirmation = {}
    for label, config in (("default", default), ("ajustado", chosen_config)):
        if label == "ajustado" and not adopted:
            continue
        probabilities, rounds = [], []
        for seed in SEEDS:
            params = params_for(config, seed)
            n = full_rounds(lgb, params, X_train, y_train, races, categorical)
            booster = lgb.train(params, lgb.Dataset(X_train, y_train, categorical_feature=categorical),
                                num_boost_round=n)
            probabilities.append(booster.predict(X_val))
            rounds.append(n)
        mean = np.mean(probabilities, axis=0)
        selected = select_metric_threshold(y_val, mean, THRESHOLD_GRID, metric="f2")
        confirmation[label] = {"config": {k: v for k, v in config.items() if k not in ("fold_ap", "best_iterations")},
                               "boosting_rounds_by_seed": rounds, "threshold": selected["threshold"],
                               **{k: selected["metrics"][k] for k in ("average_precision", "f2", "f1", "precision",
                                                                      "recall", "tp", "fp", "fn")},
                               "single_seed_ap": [binary_metrics(y_val, p, 0.5)["average_precision"] for p in probabilities]}
        print(f"validación {label:9s} AP {confirmation[label]['average_precision']:.4f} "
              f"F2 {confirmation[label]['f2']:.4f}", flush=True)
    output.mkdir(parents=True)
    table.to_csv(output / "busqueda.csv", index=False)
    write_json(output / "metrics.json", {
        "protocol": {"features": "telemetry + strategy rounds 1-3 (66)", "grid": GRID, "random_configs": N_RANDOM,
                     "search_seed": SEARCH_SEED,
                     "selection": "mean out-of-fold AP over 5 race-grouped train folds (inner 20% race early stopping)",
                     "adoption_rule": f"best beats default by > {MIN_MEAN_GAIN} mean fold AP and in >= {MIN_FOLDS}/5 folds",
                     "confirmation": "validation 2024, mean of seeds 42-44, used once", "test_2025_loaded": False,
                     "lightgbm_version": lgb.__version__},
        "default_params": LGB_PARAMS, "decision": decision, "confirmation_2024": confirmation,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "gbm_tuning_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "script": sha256(Path(__file__))},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evidencia_ajuste_gbm_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
