"""Describe a validated capacity pair without fitting weights or thresholds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/f1-rnn-matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pair", "evaluation", "historical-reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new comparison directory")
    pair_manifest = read(args.pair / "pair_manifest.json")
    if not pair_manifest["comparison_authorized"] or pair_manifest["test_used"]:
        raise ValueError("The pair did not pass its comparison gates")
    for name, expected in pair_manifest["files_sha256"].items():
        if digest(args.pair / name) != expected:
            raise ValueError(f"Pair artifact hash mismatch: {name}")
    names = ["rnn32_reference", "rnn16_candidate"]
    local, training, histories, curves, predictions = [], [], [], [], []
    input_hashes = {"pair_manifest": digest(args.pair / "pair_manifest.json")}
    rows = []
    for name in names:
        evaluation = args.evaluation / name
        manifest = read(evaluation / "manifest.json")
        for file, expected in manifest["files_sha256"].items():
            if digest(evaluation / file) != expected:
                raise ValueError(f"Evaluation artifact hash mismatch: {name}/{file}")
        result = read(evaluation / "metrics.json")
        if not result["parity_passed"] or result["threshold_decision_differences"] != 0:
            raise ValueError("Comparison requires parity with no changed decisions")
        local.append(result)
        training.append(read(args.pair / name / "metrics.json"))
        histories.append(pd.read_csv(args.pair / name / "history.csv"))
        curves.append(pd.read_csv(args.pair / name / "threshold_curve.csv"))
        predictions.append(pd.read_csv(evaluation / "predictions.csv.gz", dtype={"DriverNumber": str}))
        child = read(args.pair / name / "manifest.json")
        rows.append({"model": name, "parameters": child["parameter_count"],
                     "selected_epoch": child["best_epoch"], "epochs": len(histories[-1]),
                     "threshold": result["threshold_frozen"], **result["metrics"],
                     **result["event_metrics"],
                     "train_f2": training[-1]["train_selected_checkpoint"]["validation_selected_threshold"]["f2"],
                     "train_average_precision": training[-1]["train_selected_checkpoint"]["validation_selected_threshold"]["average_precision"],
                     "train_bce": training[-1]["train_selected_checkpoint"]["bce"], "validation_bce": result["bce"]})
        input_hashes[name + "_evaluation_manifest"] = digest(evaluation / "manifest.json")
    keys = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap", "target"]
    if not predictions[0][keys].equals(predictions[1][keys]) or not predictions[0].Temporada.eq(2024).all():
        raise ValueError("Evaluated populations differ or include a season outside 2024")
    historical = pd.read_csv(args.historical_reference / "validation_predictions.csv.gz")
    repeated = pd.read_csv(args.pair / names[0] / "validation_predictions.csv.gz")
    historical_check = {
        "same_weights_bytes": digest(args.historical_reference / "weights.pt") == digest(args.pair / names[0] / "weights.pt"),
        "same_sample_ids": np.array_equal(historical.sample_id, repeated.sample_id),
        "same_exported_probabilities": np.array_equal(historical.probability, repeated.probability),
    }
    input_hashes["historical_manifest"] = digest(args.historical_reference / "manifest.json")
    actual = predictions[0].target.eq(1).to_numpy()
    first, second = [p.alert.to_numpy(dtype=bool) for p in predictions]
    transitions = {
        "positive_rescued": int(np.sum(actual & ~first & second)),
        "positive_lost": int(np.sum(actual & first & ~second)),
        "positive_missed_by_both": int(np.sum(actual & ~first & ~second)),
        "old_false_positives_avoided": int(np.sum(~actual & first & ~second)),
        "new_false_positives": int(np.sum(~actual & ~first & second)),
    }
    summary = {"scope": "Paired development validation 2024; no model or threshold changes",
               "models": rows, "historical_reference_reproduction": historical_check,
               "decision_transitions": transitions, "test_used": False}
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, (label, color) in enumerate([("RNN32", "#1f77b4"), ("RNN16", "#d97706")]):
        h = histories[i]
        axes[0, 0].plot(h.epoch, h.train_evaluation_loss, color=color, label=label + " train")
        axes[0, 0].plot(h.epoch, h.validation_loss, color=color, linestyle="--", label=label + " validación")
        axes[0, 1].plot(h.epoch, h.train_f2_at_0_5, color=color, label=label + " train")
        axes[0, 1].plot(h.epoch, h.validation_f2_at_0_5, color=color, linestyle="--", label=label + " validación")
        axes[0, 1].axvline(rows[i]["selected_epoch"], color=color, linestyle=":", alpha=0.6)
        axes[1, 0].plot(curves[i].threshold, curves[i].f2, color=color, label=label)
        axes[1, 0].scatter([rows[i]["threshold"]], [rows[i]["f2"]], color=color)
    axes[0, 0].set(title="Pérdida al cierre de cada época", xlabel="Época", ylabel="BCE")
    axes[0, 1].set(title="F2 con umbral fijo 0,50", xlabel="Época", ylabel="F2", ylim=(0, 0.5))
    axes[1, 0].set(title="F2 con checkpoints congelados", xlabel="Umbral", ylabel="F2", xlim=(0.05, 0.5), ylim=(0, 0.5))
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.legend(fontsize=9)
    fields = ["f2", "average_precision", "precision", "recall", "recall_per_evaluable_event"]
    positions = np.arange(len(fields))
    for i, (label, color) in enumerate([("RNN32", "#1f77b4"), ("RNN16", "#d97706")]):
        axes[1, 1].bar(positions + (i - 0.5) * 0.36, [rows[i][f] for f in fields], width=0.36, color=color, label=label)
    axes[1, 1].set_xticks(positions, ["F2", "AP", "Precisión", "Recall", "Cobertura\neventos"])
    axes[1, 1].set(title="Validación con cada umbral seleccionado", ylabel="Métrica", ylim=(0, 1))
    axes[1, 1].legend()
    environment = read(args.pair / "pair_environment.json")["children"][names[0]]
    hardware = environment["gpu"] or environment["device"].upper()
    fig.suptitle(f"Capacidad RNN32 → RNN16 · {hardware} · validación 2024", fontsize=14)
    fig.savefig(args.output / "capacity_comparison.png", dpi=160)
    plt.close(fig)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    manifest = {"artifact_version": "rnn_capacity_local_comparison_v1", "test_used": False,
                "inputs_sha256": input_hashes, "matplotlib_version": matplotlib.__version__,
                "files_sha256": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"models": rows, "historical_reference_reproduction": historical_check,
                      "decision_transitions": transitions}, indent=2))


if __name__ == "__main__":
    main()
