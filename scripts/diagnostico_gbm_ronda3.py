"""Stop-count features (round 3): train-only grouped CV, one 2024 confirmation, false-alarm check.

Base: the current best LightGBM input set (telemetry + round-1 + all round-2
strategy features, 61 columns). Candidate: base + the five round-3 stop-plan
features. Pre-declared rule, decided before looking at results:

    adopt round 3 if it raises out-of-fold AP on average by more than 0.002
    and in at least 4 of the 5 race-grouped train folds.

Validation 2024 is then used once to confirm (AP/F2 at the F2 threshold) and
to check whether "pure" false alarms (next real entry 11+ laps away or no
further stop) drop, which is what motivated these features. 2025 is never loaded.
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
from analisis_errores import EVENTS, bucket, next_entry_distance
from diagnostico_gbm import CACHE, LAPS, LGB_PARAMS, PACKAGE, SAMPLE_INDEX, THRESHOLD_GRID, _cv_rounds, _permutation_importance
from diagnostico_gbm_ronda2 import build_frames, cv_scores
from f1_pitstop.experiment_metrics import select_metric_threshold
from f1_pitstop.rnn_data import sha256, write_json
from f1_pitstop.strategy_features import SEASONS, scheduled_laps
from f1_pitstop.strategy_features_r2 import FEATURE_GROUPS
from f1_pitstop.strategy_features_r3 import FEATURES, stop_priors, strategy_features_round3

MIN_MEAN_GAIN = 0.002
MIN_FOLDS = 4


def run(output):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frames, categorical, base_r1 = build_frames()
    laps = pd.read_pickle(LAPS)
    laps = laps.loc[laps.Temporada.isin(SEASONS)].copy()
    total_laps = scheduled_laps(CACHE)
    race_stops, long_stint = stop_priors(laps)
    for name, (frame, y, meta) in frames.items():
        round3 = strategy_features_round3(meta, laps, total_laps, race_stops, long_stint)
        if not np.array_equal(round3.sample_id.to_numpy(), meta.sample_id.to_numpy()):
            raise ValueError("Round-3 feature order differs from samples")
        if set(round3.columns) & set(frame.columns):
            raise ValueError("Round-3 feature names collide with existing features")
        frames[name] = (pd.concat([frame, round3.drop(columns="sample_id")], axis=1), y, meta)
    base = base_r1 + sum(FEATURE_GROUPS.values(), [])
    candidate = base + FEATURES
    X_train, y_train, meta_train = frames["train"]
    races = (meta_train.Temporada.astype(str) + "-" + meta_train.Ronda.astype(str)).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X_train, y_train, races))
    cv = {}
    for name, columns in (("base_r1_r2", base), ("base_r1_r2+paradas", candidate)):
        cats = [c for c in categorical if c in columns]
        cv[name] = cv_scores(lgb, X_train[columns], y_train, races, cats, folds)
        print(f"{name:22s} OOF AP {cv[name]['oof_ap']:.4f}  F2 {cv[name]['oof_f2']:.4f}", flush=True)
    diff = np.array(cv["base_r1_r2+paradas"]["fold_ap"]) - np.array(cv["base_r1_r2"]["fold_ap"])
    decision = {"mean_delta_ap": float(diff.mean()), "folds_positive": int((diff > 0).sum()),
                "per_fold": diff.tolist(),
                "adopted": bool(diff.mean() > MIN_MEAN_GAIN and (diff > 0).sum() >= MIN_FOLDS)}
    print("Decisión:", decision, flush=True)

    X_val, y_val, meta_val = frames["validation"]
    meta_val = meta_val.astype({"DriverNumber": str})
    meta_val["DriverNumber"] = meta_val.DriverNumber.str.removesuffix(".0")
    events = pd.read_csv(EVENTS, dtype={"DriverNumber": str}, keep_default_na=False)
    events = events.loc[events.Temporada == 2024].copy()
    events["DriverNumber"] = events.DriverNumber.astype(str).str.removesuffix(".0")
    distance, _ = next_entry_distance(meta_val, events)
    buckets = bucket(distance)
    confirmation, models = {}, {}
    for name, columns in (("base_r1_r2", base), ("base_r1_r2+paradas", candidate)):
        cats = [c for c in categorical if c in columns]
        rounds, _ = _cv_rounds(lgb, X_train[columns], y_train, races, cats)
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_train[columns], y_train, categorical_feature=cats),
                            num_boost_round=rounds)
        probabilities = booster.predict(X_val[columns])
        chosen = select_metric_threshold(y_val, probabilities, THRESHOLD_GRID, metric="f2")
        alert = probabilities >= chosen["threshold"]
        fp = alert & (y_val == 0)
        fp_buckets = pd.Series(buckets[fp]).value_counts().to_dict()
        pure = int(sum(fp_buckets.get(k, 0) for k in ("+11 o más", "sin_parada_posterior")))
        confirmation[name] = {"n_features": len(columns), "boosting_rounds": rounds, "threshold": chosen["threshold"],
                              **{k: chosen["metrics"][k] for k in ("average_precision", "f2", "f1", "precision",
                                                                   "recall", "tp", "fp", "fn")},
                              "false_positives_by_laps_to_next_entry": fp_buckets, "pure_false_alarms": pure}
        models[name] = (booster, columns)
        print(f"validación {name:22s} AP {confirmation[name]['average_precision']:.4f} "
              f"F2 {confirmation[name]['f2']:.4f} FP {confirmation[name]['fp']} alarmas puras {pure}", flush=True)
    booster, columns = models["base_r1_r2+paradas"]
    _, permutation = _permutation_importance(booster, X_val[columns], y_val, columns)
    importance = pd.DataFrame({"feature": columns, "permutation_ap_drop": [permutation[c] for c in columns],
                               "round3": [c in FEATURES for c in columns]}).sort_values("permutation_ap_drop", ascending=False)
    output.mkdir(parents=True)
    importance.to_csv(output / "importancia_validacion.csv", index=False)
    write_json(output / "metrics.json", {
        "protocol": {"selection_split": "train 2022-2023 only, 5-fold GroupKFold by race, inner 20% race early stopping",
                     "selection_rule": f"mean fold ΔAP > {MIN_MEAN_GAIN} and positive in >= {MIN_FOLDS}/5 folds",
                     "confirmation_split": "validation 2024, used once for this round", "test_2025_loaded": False,
                     "base": "telemetry + round-1 + round-2 strategy (61 features)",
                     "lightgbm_params": LGB_PARAMS, "lightgbm_version": lgb.__version__},
        "train_priors": {"long_stint_p90": long_stint,
                         "races_with_stop_counts": len(race_stops)},
        "cv": cv, "decision": decision, "confirmation_2024": confirmation,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "gbm_round3_diagnostic_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "events": sha256(EVENTS),
                          "script": sha256(Path(__file__)),
                          "round3_module": sha256(ROOT / "src/f1_pitstop/strategy_features_r3.py")},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Resultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/aclaraciones/evidencia_gbm_ronda3_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
