"""Reproduce descriptive plots of the frozen RNN run; never fits a model or threshold."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "evaluation", "events", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new analysis directory")
    config = json.loads((args.bundle / "config.json").read_text())
    selection_metric = config["selection"]["primary_metric"].removeprefix("positive_class_")
    if selection_metric not in {"f1", "f2"}:
        raise ValueError("Unsupported selection metric")
    metric_label = selection_metric.upper()
    if digest(args.events) != config["evaluation"]["events_sha256"]:
        raise ValueError("Events do not match the frozen run")
    history = pd.read_csv(args.bundle / "history.csv")
    curve = pd.read_csv(args.bundle / "threshold_curve.csv")
    model_metrics = json.loads((args.bundle / "metrics.json").read_text())
    local = json.loads((args.evaluation / "metrics.json").read_text())
    if local["split"] != "validation" or not local["parity_passed"]:
        raise ValueError("Analysis requires successful local validation parity")
    frozen = local["threshold_frozen"]
    selected_epoch = model_metrics["selection_epoch"]
    metrics = local["metrics"]
    events = pd.read_csv(args.events)
    names = events.loc[events.Temporada.eq(2024), ["Temporada", "Ronda", "GranPremio"]].drop_duplicates()
    races = pd.read_csv(args.evaluation / "metrics_by_race.csv").merge(names, on=["Temporada", "Ronda"], validate="one_to_one")
    predictions = pd.read_csv(args.evaluation / "predictions.csv.gz")
    if not predictions.Temporada.eq(2024).all():
        raise ValueError("Only validation 2024 is allowed")
    # Descriptive diagnostic only, not an alternative trained baseline.
    positives, n = metrics["positives"], metrics["n_samples"]
    always_alert_f2 = 5 * positives / (4 * positives + n)
    always_alert_f1 = 2 * positives / (positives + n)
    summary = {
        "scope": "Descriptive validation analysis; no model/threshold changes",
        "selection_metric": selection_metric,
        "selected_epoch": selected_epoch, "epochs_completed": len(history),
        "threshold_frozen": frozen, "prevalence": positives / n,
        "always_alert_f2_sanity_control": always_alert_f2,
        "always_alert_f1_sanity_control": always_alert_f1,
        "alert_count": metrics["tp"] + metrics["fp"],
        "false_fraction_of_alerts": metrics["fp"] / (metrics["tp"] + metrics["fp"]),
        "missed_events": local["event_metrics"]["events_with_evaluable_opportunity"] - local["event_metrics"]["covered_events"],
        "train_validation_f2_gap": model_metrics["train_selected_checkpoint"]["validation_selected_threshold"]["f2"] - metrics["f2"],
        "train_validation_f1_gap": model_metrics["train_selected_checkpoint"]["validation_selected_threshold"]["f1"] - metrics["f1"],
        "test_used": False,
    }
    args.output.mkdir(parents=True)
    races.to_csv(args.output / "metrics_by_named_race.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(history.epoch, history.train_evaluation_loss, label="Train", marker=".")
    ax.plot(history.epoch, history.validation_loss, label="Validación", marker=".")
    ax.axvline(selected_epoch, color="gray", linestyle="--", label=f"Checkpoint: época {selected_epoch}")
    ax.set(title="Pérdida con pesos al cierre de cada época", xlabel="Época", ylabel="BCE")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(history.epoch, history[f"train_{selection_metric}_at_0_5"], label="Train", marker=".")
    ax.plot(history.epoch, history[f"validation_{selection_metric}_at_0_5"], label="Validación", marker=".")
    ax.axvline(selected_epoch, color="gray", linestyle="--")
    ax.set(title=f"{metric_label} durante entrenamiento: umbral fijo 0,50", xlabel="Época", ylabel=metric_label, ylim=(0, 1))
    ax.legend()
    ax = axes[1, 0]
    for field, label in [("f1", "F1"), ("f2", "F2"), ("precision", "Precisión"), ("recall", "Recall")]:
        ax.plot(curve.threshold, curve[field], label=label)
    ax.axvline(frozen, color="gray", linestyle="--", label=f"Umbral congelado: {frozen:.2f}")
    control_score = always_alert_f1 if selection_metric == "f1" else always_alert_f2
    ax.axhline(control_score, color="gray", linestyle=":", label=f"{metric_label} de alertar siempre")
    ax.set(title="Regla de alerta con checkpoint congelado", xlabel="Umbral", ylabel="Métrica", ylim=(0, 1))
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black", fontsize=16)
    ax.set_xticks([0, 1], ["Sin alerta", "Alerta"])
    ax.set_yticks([0, 1], ["Negativa", "Positiva"])
    ax.set(title=f"Matriz de confusión: umbral {frozen:.2f}", xlabel="Decisión del modelo", ylabel="Etiqueta real")
    fig.suptitle(f"RNN v1 · Selección {metric_label} · Validación 2024 (con selección sobre validación)", fontsize=14)
    fig.savefig(args.output / "diagnostics.png", dpi=160)
    plt.close(fig)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    manifest = {
        "artifact_version": "rnn_validation_descriptive_analysis_v1", "test_used": False,
        "input_hashes": {"bundle_manifest": digest(args.bundle / "manifest.json"),
                         "local_evaluation_manifest": digest(args.evaluation / "manifest.json"),
                         "events": digest(args.events)},
        "matplotlib_version": matplotlib.__version__,
        "files_sha256": {p.name: digest(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
