"""Freeze the three-seed strategy LSTM32 (round 1) before evaluating it on 2025.

Takes three completed training bundles (configs
experimento_lstm32_estrategia_candidate_cpu_seed{42,43,44}_v1.json), checks
their hashes, status, configs and that no test data was used, copies them into
``modelos/lstm32_estrategia_v1/seed{seed}/`` and fixes the decision rule for the
ensemble **using validation 2024 only**: mean of the three probabilities and
the threshold that maximizes F2 on the usual grid (same rule as every model).

This is declared as a post-hoc comparison: the frozen LightGBM was already
evaluated on 2025, but nothing here looks at 2025.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.experiment_metrics import binary_metrics, select_metric_threshold
from f1_pitstop.rnn_data import read_json, sha256, write_json

SEEDS = (42, 43, 44)
THRESHOLD_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]
PACKAGE = "EDA/dataset/muestras_preprocesadas_v1_3_telemetria_estrategia_v1"


def verified_bundle(directory, seed):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("artifact_version") != "rnn_reference_bundle_v1" or manifest.get("test_used") is not False
            or read_json(directory / "status.json").get("status") != "complete"):
        raise ValueError(f"Bundle seed {seed} is incomplete or used test data")
    for name, digest in manifest["files_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Bundle seed {seed} hash mismatch: {name}")
    config = read_json(directory / "config.json")
    declared = read_json(ROOT / f"configs/experimento_lstm32_estrategia_candidate_cpu_seed{seed}_v1.json")
    if (config["training"]["seed"] != seed or config["preprocessed_package"] != PACKAGE
            or config["preprocessed_manifest_sha256"] != declared["preprocessed_manifest_sha256"]):
        raise ValueError(f"Bundle seed {seed} does not match its declared config")
    if sha256(ROOT / "src/f1_pitstop/lstm_model.py") != manifest["model_source_sha256"]:
        raise ValueError("Current LSTM source differs from the bundle's frozen source")
    return manifest


def run(runs, output):
    runs, output = Path(runs), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tables, manifests = [], {}
    for seed in SEEDS:
        manifests[seed] = verified_bundle(runs / f"seed{seed}", seed)
        tables.append(pd.read_csv(runs / f"seed{seed}" / "validation_predictions.csv.gz", dtype={"DriverNumber": str}))
    for table in tables[1:]:
        if not (np.array_equal(table.sample_id, tables[0].sample_id) and np.array_equal(table.target, tables[0].target)):
            raise ValueError("Seed predictions are not aligned")
    y = tables[0].target.to_numpy()
    mean = np.mean([t.probability.to_numpy() for t in tables], axis=0)
    selected = select_metric_threshold(y, mean, THRESHOLD_GRID, metric="f2")
    threshold = selected["threshold"]
    metrics = binary_metrics(y, mean, threshold)
    output.mkdir(parents=True)
    for seed in SEEDS:
        shutil.copytree(runs / f"seed{seed}", output / f"seed{seed}")
    keys = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap", "target"]
    predictions = tables[0][keys].copy()
    predictions["probability"], predictions["alert"] = mean, mean >= threshold
    predictions.to_csv(output / "validation_predictions.csv.gz", index=False, compression="gzip")
    per_seed = {str(s): {"threshold": read_json(output / f"seed{s}" / "threshold.json")["threshold"],
                         "best_epoch": manifests[s]["best_epoch"],
                         "validation_2024": {k: v for k, v in read_json(output / f"seed{s}" / "metrics.json")
                                             ["selected_threshold"]["metrics"].items()
                                             if k in ("average_precision", "f2", "f1", "precision", "recall")}}
                for s in SEEDS}
    write_json(output / "model.json", {
        "model": "lstm32_strategy_r1_mean_of_seeds", "seeds": list(SEEDS),
        "combination": "mean of seed probabilities", "threshold": threshold,
        "threshold_source": "maximize F2 on validation 2024 for the seed mean, grid 0.05..0.95",
        "inputs": PACKAGE, "train_seasons": [2022, 2023], "validation_season": 2024,
        "post_hoc_note": "Frozen after the LightGBM 2025 evaluation; selected and thresholded on 2024 only.",
        "parameter_count": manifests[42]["parameter_count"], "per_seed": per_seed,
        "validation_2024": {k: metrics[k] for k in ("average_precision", "f2", "f1", "precision", "recall",
                                                    "tp", "fp", "fn", "n_samples", "positives")},
    })
    files = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {
        "artifact_version": "frozen_lstm_strategy_v1", "files_sha256": files, "test_2025_loaded": False,
        "sources_sha256": {name: sha256(ROOT / name) for name in (
            "src/f1_pitstop/lstm_model.py", "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/strategy_features.py",
            "scripts/preparar_entradas_telemetria.py", "scripts/preparar_entradas_estrategia.py",
            "scripts/congelar_lstm_estrategia.py")},
    })
    print(f"LSTM congelada en {output}; umbral {threshold}; "
          + ", ".join(f"{k} {metrics[k]:.4f}" for k in ("average_precision", "f2", "precision", "recall")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Directory with seed42/ seed43/ seed44/ bundles")
    parser.add_argument("--output", type=Path, default=ROOT / "modelos/lstm32_estrategia_v1")
    args = parser.parse_args()
    run(args.runs, args.output)


if __name__ == "__main__":
    main()
