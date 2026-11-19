"""Freeze the final LightGBM model before any look at the 2025 test season.

Trains the adopted configuration exactly as in the tuning confirmation
(docs/aclaraciones/34_ajuste_hiperparametros_gbm.md): 66 features (telemetry +
strategy rounds 1-3), parameters ``random_06``, three seeds (42, 43, 44) fitted
on train 2022-2023 with the number of trees chosen by race-grouped CV on train.
The prediction is the mean of the three probabilities; the alert threshold is
the F2-selected 0.14 from validation 2024.

The script refuses to publish unless the frozen models reproduce the
documented validation metrics. It writes the boosters as LightGBM text files,
the exact feature order, parameters, threshold, validation predictions and a
manifest of hashes. 2025 is never loaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ajuste_gbm import SEEDS, frames_round3, full_rounds, params_for
from diagnostico_gbm import LAPS, PACKAGE, SAMPLE_INDEX, THRESHOLD_GRID
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import read_json, sha256, write_json

TUNING = ROOT / "docs/aclaraciones/evidencia_ajuste_gbm_v1/metrics.json"
SOURCES = ["src/f1_pitstop/strategy_features.py", "src/f1_pitstop/strategy_features_r2.py",
           "src/f1_pitstop/strategy_features_r3.py", "scripts/diagnostico_gbm.py",
           "scripts/diagnostico_gbm_ronda2.py", "scripts/ajuste_gbm.py", "scripts/congelar_modelo_final.py"]
TOLERANCE = 1e-9


def run(output):
    import lightgbm as lgb
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tuning = read_json(TUNING)
    if not tuning["decision"]["adopted"]:
        raise ValueError("Tuning did not adopt a configuration")
    documented = tuning["confirmation_2024"]["ajustado"]
    config = {k: v for k, v in documented["config"].items() if k not in ("mean_fold_ap", "oof_ap")}
    frames, categorical, columns = frames_round3()
    X_train, y_train, meta_train = frames["train"]
    X_val, y_val, meta_val = frames["validation"]
    X_train, X_val = X_train[columns], X_val[columns]
    races = (meta_train.Temporada.astype(str) + "-" + meta_train.Ronda.astype(str)).to_numpy()
    boosters, probabilities, rounds = {}, [], []
    for seed in SEEDS:
        params = params_for(config, seed)
        n = full_rounds(lgb, params, X_train, y_train, races, categorical)
        booster = lgb.train(params, lgb.Dataset(X_train, y_train, categorical_feature=categorical), num_boost_round=n)
        boosters[seed], rounds = booster, rounds + [n]
        probabilities.append(booster.predict(X_val))
    if rounds != documented["boosting_rounds_by_seed"]:
        raise ValueError(f"Tree counts {rounds} differ from the documented {documented['boosting_rounds_by_seed']}")
    mean = np.mean(probabilities, axis=0)
    selected = select_metric_threshold(y_val, mean, THRESHOLD_GRID, metric="f2")
    threshold = documented["threshold"]
    if selected["threshold"] != threshold:
        raise ValueError("Validation threshold differs from the documented one")
    metrics = binary_metrics(y_val, mean, threshold)
    for key in ("average_precision", "f2", "precision", "recall"):
        if abs(metrics[key] - documented[key]) > TOLERANCE:
            raise ValueError(f"Frozen model does not reproduce documented {key}: {metrics[key]} vs {documented[key]}")
    output.mkdir(parents=True)
    for seed, booster in boosters.items():
        booster.save_model(str(output / f"lightgbm_seed{seed}.txt"))
    # Reload and check the saved files give identical predictions.
    reloaded = np.mean([lgb.Booster(model_file=str(output / f"lightgbm_seed{s}.txt")).predict(X_val) for s in SEEDS], axis=0)
    if not np.allclose(reloaded, mean, rtol=0, atol=1e-12):
        shutil.rmtree(output)
        raise ValueError("Saved boosters do not reproduce in-memory predictions")
    write_json(output / "features.json", {"order": columns, "categorical": categorical, "count": len(columns)})
    write_json(output / "model.json", {
        "model": "lightgbm_mean_of_seeds", "seeds": list(SEEDS), "boosting_rounds_by_seed": rounds,
        "params_by_seed": {str(s): params_for(config, s) for s in SEEDS}, "config_name": config["name"],
        "combination": "mean of seed probabilities", "threshold": threshold,
        "threshold_source": "maximize F2 on validation 2024, grid 0.05..0.95, tie: precision then threshold",
        "train_seasons": [2022, 2023], "validation_season": 2024, "test_season_reserved": 2025,
        "lightgbm_version": lgb.__version__,
        "validation_2024": {k: metrics[k] for k in ("average_precision", "f2", "f1", "precision", "recall",
                                                    "tp", "fp", "fn", "n_samples", "positives")},
    })
    table = meta_val[["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]].copy()
    table["target"], table["probability"] = y_val, mean
    table["alert"] = mean >= threshold
    table.to_csv(output / "validation_predictions.csv.gz", index=False, compression="gzip")
    write_json(output / "manifest.json", {
        "artifact_version": "frozen_lightgbm_final_v1",
        "inputs_sha256": {"preprocessed_manifest": sha256(PACKAGE / "manifest.json"), "laps": sha256(LAPS),
                          "sample_index": sha256(SAMPLE_INDEX), "tuning_metrics": sha256(TUNING)},
        "sources_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "files_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
        "test_2025_loaded": False,
    })
    print(f"Modelo congelado en {output}")
    print({k: round(metrics[k], 4) for k in ("average_precision", "f2", "precision", "recall")}, "umbral", threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "modelos/lightgbm_final_v1")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
