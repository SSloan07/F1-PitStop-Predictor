"""Ensemble of LightGBM (round-1 + round-2 strategy) and the round-1 strategy LSTM32 on validation 2024.

Pre-declared combination, no weights fitted on validation:

* GBM: mean probability of three LightGBM models (seeds 42, 43, 44) on the
  "base + all round-2 groups" input set; rounds chosen by GroupKFold on train.
* LSTM: mean probability of the three round-1 strategy LSTM32 runs
  (seeds 42, 43, 44; their exported validation predictions).
* Ensemble (primary): 0.5 * GBM + 0.5 * LSTM.

Each candidate then gets its F2 threshold on the same validation grid as every
other experiment, so all numbers are development metrics. Event coverage uses
the shared evaluator logic. 2025 is never loaded.
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
from diagnostico_gbm import LAPS, LGB_PARAMS, PACKAGE, SAMPLE_INDEX, THRESHOLD_GRID
from diagnostico_gbm_ronda2 import build_frames
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import read_json, sha256, write_json
from f1_pitstop.rnn_experiment import attach_decision_times, event_coverage
from f1_pitstop.strategy_features_r2 import FEATURE_GROUPS

SEEDS = (42, 43, 44)
EVIDENCE = ROOT / "docs/aclaraciones/evidencia_ensamble_v1"
LSTM_INPUTS = [EVIDENCE / f"entradas_lstm/lstm32_estrategia_r1_seed{seed}_validation_predictions.csv.gz" for seed in SEEDS]
EVENTS_CONFIG = ROOT / "configs/experimento_lstm32_estrategia_candidate_cpu_seed42_v1.json"
KEYS = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]


def _rounds(lgb, params, X, y, groups, categorical):
    from sklearn.model_selection import GroupKFold
    best = []
    for fit_idx, hold_idx in GroupKFold(n_splits=5).split(X, y, groups):
        fit = lgb.Dataset(X.iloc[fit_idx], y[fit_idx], categorical_feature=categorical, free_raw_data=False)
        hold = lgb.Dataset(X.iloc[hold_idx], y[hold_idx], categorical_feature=categorical, reference=fit)
        booster = lgb.train({**params, "metric": "average_precision"}, fit, num_boost_round=2000,
                            valid_sets=[hold], callbacks=[lgb.early_stopping(100, verbose=False)])
        best.append(booster.best_iteration)
    return int(np.median(best))


def gbm_probabilities(lgb):
    frames, categorical, base = build_frames()
    X_train, y_train, meta_train = frames["train"]
    X_val, y_val, meta_val = frames["validation"]
    columns = base + sum(FEATURE_GROUPS.values(), [])
    cats = [c for c in categorical if c in columns]
    races = (meta_train.Temporada.astype(str) + "-" + meta_train.Ronda.astype(str)).to_numpy()
    probabilities = {}
    for seed in SEEDS:
        params = {**LGB_PARAMS, "seed": seed}
        rounds = _rounds(lgb, params, X_train[columns], y_train, races, cats)
        booster = lgb.train(params, lgb.Dataset(X_train[columns], y_train, categorical_feature=cats),
                            num_boost_round=rounds)
        probabilities[seed] = booster.predict(X_val[columns])
        print(f"GBM semilla {seed}: {rounds} árboles, AP "
              f"{binary_metrics(y_val, probabilities[seed], 0.5)['average_precision']:.4f}", flush=True)
    return probabilities, y_val, meta_val


def lstm_probabilities(meta_val, y_val):
    probabilities = {}
    for seed, path in zip(SEEDS, LSTM_INPUTS):
        table = pd.read_csv(path, dtype={"DriverNumber": str})
        if not np.array_equal(table.sample_id.to_numpy(), meta_val.sample_id.to_numpy()):
            raise ValueError(f"LSTM predictions are not aligned with validation samples: {path.name}")
        if not np.array_equal(table.target.to_numpy(), y_val):
            raise ValueError(f"LSTM targets differ from validation labels: {path.name}")
        probabilities[seed] = table.probability.to_numpy(dtype=float)
    return probabilities


def evaluate(name, probabilities, y, meta, events, sample_index):
    chosen = select_metric_threshold(y, probabilities, THRESHOLD_GRID, metric="f2")
    threshold = chosen["threshold"]
    table = meta[KEYS].copy()
    table["target"] = y
    table["probability"] = probabilities
    table["alert"] = probabilities >= threshold
    table = attach_decision_times(table, sample_index)
    _, event_metrics = event_coverage(events, table)
    result = {"threshold": threshold, **{k: chosen["metrics"][k] for k in (
        "average_precision", "f2", "f1", "precision", "recall", "tp", "fp", "fn")},
        "covered_events": event_metrics["covered_events"],
        "evaluable_events": event_metrics["events_with_evaluable_opportunity"]}
    print(f"{name:26s} AP {result['average_precision']:.4f} F2 {result['f2']:.4f} "
          f"P {result['precision']:.3f} R {result['recall']:.3f} eventos {result['covered_events']}", flush=True)
    return result


def run(output):
    import lightgbm as lgb
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    gbm, y_val, meta_val = gbm_probabilities(lgb)
    meta_val = meta_val.astype({"DriverNumber": str})
    lstm = lstm_probabilities(meta_val, y_val)
    evaluation = read_json(EVENTS_CONFIG)["evaluation"]
    events_path, index_path = ROOT / evaluation["events_source"], ROOT / evaluation["decision_index_source"]
    if sha256(events_path) != evaluation["events_sha256"] or sha256(index_path) != evaluation["decision_index_sha256"]:
        raise ValueError("Event source or decision index differs from the pinned publication")
    events = pd.read_csv(events_path, dtype={"DriverNumber": str}, keep_default_na=False)
    events = events.loc[events.Temporada == 2024]
    sample_index = pd.read_csv(index_path, dtype={"DriverNumber": str})
    gbm_mean = np.mean([gbm[s] for s in SEEDS], axis=0)
    lstm_mean = np.mean([lstm[s] for s in SEEDS], axis=0)
    candidates = {
        **{f"gbm_seed{s}": gbm[s] for s in SEEDS},
        **{f"lstm_seed{s}": lstm[s] for s in SEEDS},
        "gbm_media_3": gbm_mean,
        "lstm_media_3": lstm_mean,
        "ensamble_50_50": 0.5 * gbm_mean + 0.5 * lstm_mean,
    }
    results = {name: evaluate(name, p, y_val, meta_val, events, sample_index) for name, p in candidates.items()}
    correlation = float(np.corrcoef(gbm_mean, lstm_mean)[0, 1])
    output.mkdir(parents=True)
    pd.DataFrame({**{k: meta_val[k] for k in KEYS}, "target": y_val, "gbm_media_3": gbm_mean,
                  "lstm_media_3": lstm_mean, "ensamble_50_50": candidates["ensamble_50_50"]}).to_csv(
        output / "predicciones_validacion.csv.gz", index=False, compression="gzip")
    write_json(output / "metrics.json", {
        "protocol": {"combination": "0.5 * mean(GBM seeds 42-44) + 0.5 * mean(LSTM seeds 42-44); no fitted weights",
                     "threshold": "F2 on validation grid 0.05..0.95, same rule as all experiments",
                     "gbm_inputs": "round-1 base + all round-2 strategy groups (61 features)",
                     "lstm_inputs": "muestras_preprocesadas_v1_3_telemetria_estrategia_v1 (local CPU runs, preliminary)",
                     "test_2025_loaded": False, "lightgbm_params": LGB_PARAMS, "lightgbm_version": lgb.__version__},
        "gbm_lstm_probability_correlation": correlation, "results": results,
    })
    write_json(output / "manifest.json", {
        "artifact_version": "gbm_lstm_ensemble_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "script": sha256(Path(__file__)),
                          **{p.name: sha256(p) for p in LSTM_INPUTS}},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    })
    print(f"Correlación GBM-LSTM: {correlation:.3f}\nResultados: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "resultados")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
