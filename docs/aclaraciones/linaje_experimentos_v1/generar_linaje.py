"""Render the measured experiment family without fitting or evaluating models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(relative, sources):
    path = ROOT / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def fmt(value, digits=4):
    return f"{value:.{digits}f}".replace(".", ",")


def build_registry():
    sources = {}
    f2 = load("eval/rnn_v1_validation/metrics.json", sources)
    f1 = load("eval/rnn_f1_v1_validation/metrics.json", sources)
    control = load("docs/aclaraciones/evidencia_control_f1_v1/control.json", sources)
    base = load("runs/rnn_v1_seed42/manifest.json", sources)
    trial = load("runs/rnn_f1_v1_seed42/manifest.json", sources)
    base_environment = load("runs/rnn_v1_seed42/environment.json", sources)
    pair_root = "runs/rnn16_vs_rnn32_f2_v1_seed42"
    eval_root = "eval/rnn16_vs_rnn32_f2_v1_seed42"
    pair = load(f"{pair_root}/pair_manifest.json", sources)
    pair_environment = load(f"{pair_root}/pair_environment.json", sources)
    reference = load(f"{pair_root}/rnn32_reference/manifest.json", sources)
    candidate = load(f"{pair_root}/rnn16_candidate/manifest.json", sources)
    reference_model = load(f"{pair_root}/rnn32_reference/model_spec.json", sources)
    candidate_model = load(f"{pair_root}/rnn16_candidate/model_spec.json", sources)
    reference_eval = load(f"{eval_root}/rnn32_reference/metrics.json", sources)
    candidate_eval = load(f"{eval_root}/rnn16_candidate/metrics.json", sources)
    dropout_root = "runs/rnn32_dropout02_f2_v2_seed42"
    dropout_eval_root = "eval/rnn32_dropout02_f2_v2_seed42"
    dropout_pair = load(f"{dropout_root}/pair_manifest.json", sources)
    dropout_environment = load(f"{dropout_root}/pair_environment.json", sources)
    dropout_summary = load(f"{dropout_root}/pair_summary.json", sources)
    dropout_reference = load(f"{dropout_root}/rnn32_reference/manifest.json", sources)
    dropout_candidate = load(f"{dropout_root}/rnn32_dropout/manifest.json", sources)
    dropout_reference_model = load(f"{dropout_root}/rnn32_reference/model_spec.json", sources)
    dropout_candidate_model = load(f"{dropout_root}/rnn32_dropout/model_spec.json", sources)
    dropout_reference_eval = load(f"{dropout_eval_root}/rnn32_reference/metrics.json", sources)
    dropout_candidate_eval = load(f"{dropout_eval_root}/rnn32_dropout/metrics.json", sources)
    lstm_root = "runs/rnn32_lstm_vs_rnn_f2_v1_seed42"
    lstm_eval_root = "eval/rnn32_lstm_vs_rnn_f2_v1_seed42"
    lstm_pair = load(f"{lstm_root}/pair_manifest.json", sources)
    lstm_environment = load(f"{lstm_root}/pair_environment.json", sources)
    lstm_summary = load(f"{lstm_root}/pair_summary.json", sources)
    lstm_reference = load(f"{lstm_root}/rnn32_reference/manifest.json", sources)
    lstm_candidate = load(f"{lstm_root}/lstm32_candidate/manifest.json", sources)
    lstm_reference_model = load(f"{lstm_root}/rnn32_reference/model_spec.json", sources)
    lstm_candidate_model = load(f"{lstm_root}/lstm32_candidate/model_spec.json", sources)
    lstm_reference_eval = load(f"{lstm_eval_root}/rnn32_reference/metrics.json", sources)
    lstm_candidate_eval = load(f"{lstm_eval_root}/lstm32_candidate/metrics.json", sources)
    lstm_dropout_root = "runs/lstm32_dropout02_f2_v1_seed42"
    lstm_dropout_eval_root = "eval/lstm32_dropout02_f2_v1_seed42"
    lstm_dropout_pair = load(f"{lstm_dropout_root}/pair_manifest.json", sources)
    lstm_dropout_environment = load(f"{lstm_dropout_root}/pair_environment.json", sources)
    lstm_dropout_summary = load(f"{lstm_dropout_root}/pair_summary.json", sources)
    lstm_dropout_reference = load(f"{lstm_dropout_root}/lstm32_reference/manifest.json", sources)
    lstm_dropout_candidate = load(f"{lstm_dropout_root}/lstm32_dropout/manifest.json", sources)
    lstm_dropout_reference_model = load(f"{lstm_dropout_root}/lstm32_reference/model_spec.json", sources)
    lstm_dropout_candidate_model = load(f"{lstm_dropout_root}/lstm32_dropout/model_spec.json", sources)
    lstm_dropout_reference_eval = load(f"{lstm_dropout_eval_root}/lstm32_reference/metrics.json", sources)
    lstm_dropout_candidate_eval = load(f"{lstm_dropout_eval_root}/lstm32_dropout/metrics.json", sources)
    lstm_dropout_comparison = load(f"{lstm_dropout_eval_root}/comparison/comparison.json", sources)
    scheduler_root = "runs/lstm32_scheduler_f2_v1_seed42"
    scheduler_eval_root = "eval/lstm32_scheduler_f2_v1_seed42"
    scheduler_pair = load(f"{scheduler_root}/pair_manifest.json", sources)
    scheduler_environment = load(f"{scheduler_root}/pair_environment.json", sources)
    scheduler_summary = load(f"{scheduler_root}/pair_summary.json", sources)
    scheduler_reference = load(f"{scheduler_root}/lstm32_reference/manifest.json", sources)
    scheduler_candidate = load(f"{scheduler_root}/lstm32_scheduler/manifest.json", sources)
    scheduler_reference_model = load(f"{scheduler_root}/lstm32_reference/model_spec.json", sources)
    scheduler_candidate_model = load(f"{scheduler_root}/lstm32_scheduler/model_spec.json", sources)
    scheduler_reference_eval = load(f"{scheduler_eval_root}/lstm32_reference/metrics.json", sources)
    scheduler_candidate_eval = load(f"{scheduler_eval_root}/lstm32_scheduler/metrics.json", sources)
    scheduler_comparison = load(f"{scheduler_eval_root}/comparison/comparison.json", sources)
    assert f2["parity_passed"] and f1["parity_passed"]
    assert reference_eval["parity_passed"] and candidate_eval["parity_passed"]
    assert all(item["test_used"] is False for item in [base, trial, pair, reference, candidate])
    assert base["preprocessed_manifest_sha256"] == trial["preprocessed_manifest_sha256"]
    assert base["model_source_sha256"] == trial["model_source_sha256"]
    assert pair["artifact_version"] == "rnn_capacity_pair_v1"
    assert pair["comparison_authorized"] and pair_environment["comparison_authorized"]
    assert pair["selection_metric"] == "f2" and pair["selection_beta"] == 2
    assert pair["changed_factor"] == {"name": "model.hidden_size", "reference": 32, "candidate": 16}
    assert pair["seed"] == 42
    assert pair["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert reference["preprocessed_manifest_sha256"] == candidate["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert reference["model_source_sha256"] == candidate["model_source_sha256"] == base["model_source_sha256"]
    assert reference_model["hidden_size"] == 32 and candidate_model["hidden_size"] == 16
    assert {key: value for key, value in reference_model.items() if key != "hidden_size"} == {key: value for key, value in candidate_model.items() if key != "hidden_size"}
    for role in ["rnn32_reference", "rnn16_candidate"]:
        for name in ["manifest.json", "model_spec.json"]:
            assert sources[f"{pair_root}/{role}/{name}"] == pair["files_sha256"][f"{role}/{name}"]
    assert sources[f"{pair_root}/pair_environment.json"] == pair["files_sha256"]["pair_environment.json"]
    ref_environment = pair_environment["children"]["rnn32_reference"]
    candidate_environment = pair_environment["children"]["rnn16_candidate"]
    assert ref_environment == candidate_environment
    assert ref_environment["device"] == "cuda" and "T4" in ref_environment["gpu"]
    same_historical_environment = ref_environment == base_environment
    same_historical_weights = reference["files_sha256"]["weights.pt"] == base["files_sha256"]["weights.pt"]
    assert all(item["metrics"]["n_samples"] == 25635 for item in [f2, f1, reference_eval, candidate_eval])
    assert control["new_training"] is False
    assert dropout_pair["artifact_version"] == "rnn_dropout_pair_v1"
    assert dropout_pair["comparison_authorized"] and dropout_environment["comparison_authorized"]
    assert dropout_pair["test_used"] is False and dropout_pair["seed"] == 42
    assert dropout_pair["changed_factor"] == {"name": "model.dropout_rate", "reference": 0.0, "candidate": 0.2, "location": "fusion_head_after_relu"}
    assert dropout_pair["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert dropout_reference["model_source_sha256"] == dropout_candidate["model_source_sha256"]
    assert dropout_reference_model["dropout_rate"] == 0.0 and dropout_candidate_model["dropout_rate"] == 0.2
    assert {key: value for key, value in dropout_reference_model.items() if key != "dropout_rate"} == {key: value for key, value in dropout_candidate_model.items() if key != "dropout_rate"}
    assert dropout_environment["children"]["rnn32_reference"] == dropout_environment["children"]["rnn32_dropout"]
    assert dropout_environment["children"]["rnn32_reference"]["device"] == "cuda" and "T4" in dropout_environment["children"]["rnn32_reference"]["gpu"]
    assert dropout_reference["files_sha256"]["weights.pt"] == base["files_sha256"]["weights.pt"]
    assert dropout_summary["reference"]["best_epoch"] == dropout_summary["candidate"]["best_epoch"] == 10
    for result in [dropout_reference_eval, dropout_candidate_eval]:
        assert result["parity_passed"] and result["threshold_decision_differences"] == 0
        assert result["metrics"]["n_samples"] == 25635 and result["split"] == "validation"
    for role in ["rnn32_reference", "rnn32_dropout"]:
        for name in ["manifest.json", "model_spec.json"]:
            assert sources[f"{dropout_root}/{role}/{name}"] == dropout_pair["files_sha256"][f"{role}/{name}"]
    assert sources[f"{dropout_root}/pair_environment.json"] == dropout_pair["files_sha256"]["pair_environment.json"]
    assert lstm_pair["artifact_version"] == "rnn_lstm_pair_v1"
    assert lstm_pair["comparison_authorized"] and lstm_environment["comparison_authorized"]
    assert lstm_pair["test_used"] is False and lstm_pair["seed"] == 42
    assert lstm_pair["changed_factor"] == {"name": "model.recurrent_cell", "reference": "rnn", "candidate": "lstm", "location": "causal_recurrent_branch"}
    assert lstm_pair["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert lstm_environment["children"]["rnn32_reference"] == lstm_environment["children"]["lstm32_candidate"]
    assert lstm_environment["children"]["rnn32_reference"]["device"] == "cuda" and "T4" in lstm_environment["children"]["rnn32_reference"]["gpu"]
    assert lstm_reference["files_sha256"]["weights.pt"] == base["files_sha256"]["weights.pt"]
    assert lstm_reference.get("model_source_path", "rnn_model.py") == "rnn_model.py"
    assert lstm_candidate["model_source_path"] == "lstm_model.py"
    assert lstm_candidate_model["recurrent_cell"] == "lstm"
    assert {key: value for key, value in lstm_reference_model.items() if key != "recurrent_cell"} == {key: value for key, value in lstm_candidate_model.items() if key != "recurrent_cell"}
    assert lstm_reference["preprocessed_manifest_sha256"] == lstm_candidate["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    for result in [lstm_reference_eval, lstm_candidate_eval]:
        assert result["parity_passed"] and result["threshold_decision_differences"] == 0
        assert result["metrics"]["n_samples"] == 25635 and result["split"] == "validation"
        assert result["event_metrics"]["events_with_evaluable_opportunity"] == 824
    for role in ["rnn32_reference", "lstm32_candidate"]:
        for name in ["manifest.json", "model_spec.json"]:
            assert sources[f"{lstm_root}/{role}/{name}"] == lstm_pair["files_sha256"][f"{role}/{name}"]
    assert sources[f"{lstm_root}/pair_environment.json"] == lstm_pair["files_sha256"]["pair_environment.json"]
    assert lstm_dropout_pair["artifact_version"] == "lstm_dropout_pair_v1"
    assert lstm_dropout_pair["comparison_authorized"] and lstm_dropout_environment["comparison_authorized"]
    assert lstm_dropout_pair["test_used"] is False and lstm_dropout_pair["seed"] == 42
    assert lstm_dropout_pair["changed_factor"] == {"name": "model.dropout_rate", "reference": 0.0, "candidate": 0.2, "location": "fusion_head_after_relu"}
    assert lstm_dropout_pair["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert lstm_dropout_environment["children"]["lstm32_reference"] == lstm_dropout_environment["children"]["lstm32_dropout"]
    assert lstm_dropout_environment["children"]["lstm32_reference"]["device"] == "cuda" and "T4" in lstm_dropout_environment["children"]["lstm32_reference"]["gpu"]
    assert lstm_dropout_reference["model_source_path"] == lstm_dropout_candidate["model_source_path"] == "lstm_model.py"
    assert lstm_dropout_reference["model_source_sha256"] == lstm_dropout_candidate["model_source_sha256"] == lstm_candidate["model_source_sha256"]
    assert lstm_dropout_reference_model["recurrent_cell"] == lstm_dropout_candidate_model["recurrent_cell"] == "lstm"
    assert lstm_dropout_reference_model["dropout_rate"] == 0.0 and lstm_dropout_candidate_model["dropout_rate"] == 0.2
    assert {key: value for key, value in lstm_dropout_reference_model.items() if key != "dropout_rate"} == {key: value for key, value in lstm_dropout_candidate_model.items() if key != "dropout_rate"}
    assert lstm_dropout_reference["files_sha256"]["weights.pt"] == lstm_candidate["files_sha256"]["weights.pt"]
    assert lstm_dropout_comparison["reference_weights_match_prior_lstm"] is True
    assert lstm_dropout_summary["reference"]["best_epoch"] == lstm_dropout_summary["candidate"]["best_epoch"] == 27
    for result in [lstm_dropout_reference_eval, lstm_dropout_candidate_eval]:
        assert result["parity_passed"] and result["threshold_decision_differences"] == 0
        assert result["metrics"]["n_samples"] == 25635 and result["split"] == "validation"
        assert result["event_metrics"]["events_with_evaluable_opportunity"] == 824
    assert lstm_dropout_candidate_eval["threshold_frozen"] == 0.05
    assert lstm_dropout_comparison["threshold_grid_boundary_warning"] is True
    for role in ["lstm32_reference", "lstm32_dropout"]:
        for name in ["manifest.json", "model_spec.json"]:
            assert sources[f"{lstm_dropout_root}/{role}/{name}"] == lstm_dropout_pair["files_sha256"][f"{role}/{name}"]
    assert sources[f"{lstm_dropout_root}/pair_environment.json"] == lstm_dropout_pair["files_sha256"]["pair_environment.json"]
    assert scheduler_pair["artifact_version"] == "lstm_scheduler_pair_v1"
    assert scheduler_pair["comparison_authorized"] and scheduler_environment["comparison_authorized"]
    assert scheduler_pair["test_used"] is False and scheduler_pair["seed"] == 42
    assert scheduler_pair["changed_factor"]["name"] == "training.lr_scheduler"
    assert scheduler_pair["changed_factor"]["reference"] == {"name": "constant"}
    assert scheduler_pair["changed_factor"]["candidate"]["name"] == "reduce_on_plateau"
    assert scheduler_pair["preprocessed_manifest_sha256"] == base["preprocessed_manifest_sha256"]
    assert scheduler_environment["children"]["lstm32_reference"] == scheduler_environment["children"]["lstm32_scheduler"]
    assert scheduler_environment["children"]["lstm32_reference"]["device"] == "cuda" and "T4" in scheduler_environment["children"]["lstm32_reference"]["gpu"]
    assert scheduler_reference["model_source_path"] == scheduler_candidate["model_source_path"] == "lstm_model.py"
    assert scheduler_reference_model == scheduler_candidate_model
    assert scheduler_reference_model["recurrent_cell"] == "lstm" and scheduler_reference_model["hidden_size"] == 32
    assert scheduler_reference_model["dropout_rate"] == scheduler_candidate_model["dropout_rate"] == 0.2
    assert scheduler_summary["reference"]["reduction_epochs"] == scheduler_summary["candidate"]["reduction_epochs"] == []
    assert scheduler_comparison["learning_rate_was_reduced"] is False
    assert scheduler_comparison["candidate_weights_equal_reference"] is True
    assert scheduler_reference["files_sha256"]["weights.pt"] == scheduler_candidate["files_sha256"]["weights.pt"]
    assert scheduler_reference["files_sha256"]["weights.pt"] == lstm_dropout_candidate["files_sha256"]["weights.pt"]
    assert scheduler_comparison["reference_weights_match_prior_lstm_dropout02"] is True
    assert scheduler_candidate_eval["threshold_frozen"] == 0.05
    assert scheduler_comparison["probability_comparison"]["identical_after_cpu_replay"] is True
    assert scheduler_comparison["probability_comparison"]["threshold_decision_differences"] == 0
    for result in [scheduler_reference_eval, scheduler_candidate_eval]:
        assert result["parity_passed"] and result["threshold_decision_differences"] == 0
        assert result["metrics"]["n_samples"] == 25635 and result["split"] == "validation"
        assert result["event_metrics"]["events_with_evaluable_opportunity"] == 824
    for role in ["lstm32_reference", "lstm32_scheduler"]:
        for name in ["manifest.json", "model_spec.json"]:
            assert sources[f"{scheduler_root}/{role}/{name}"] == scheduler_pair["files_sha256"][f"{role}/{name}"]
    assert sources[f"{scheduler_root}/pair_environment.json"] == scheduler_pair["files_sha256"]["pair_environment.json"]
    for key in ["tp", "fp", "fn", "tn", "f1", "f2"]:
        assert control["selection"]["metrics"][key] == f1["metrics"][key]
    pair_report = "docs/aclaraciones/12_resultados_rnn16_pareada.md"
    nodes = [
        {"id": "E0", "title": "Referencia F2", "kind": "Corrida", "status": "Completada",
         "color": "#dff3ff", "device": "T4 / CUDA", "epoch": base["best_epoch"],
         "threshold": f2["threshold_frozen"], "metrics": f2["metrics"],
         "event_metrics": f2["event_metrics"], "hidden_size": 32,
         "note": "BCE, semilla 42. Referencia de comparación; no cambia el objetivo al dibujar el árbol.",
         "report": "docs/aclaraciones/06_resultados_rnn_v1_validacion_2024.md"},
        {"id": "C1", "title": "Control de umbral F1", "kind": "Control sin entrenamiento", "status": "Completado",
         "color": "#def4e8", "device": "Pesos de E0", "epoch": control["frozen_checkpoint_epoch"],
         "threshold": control["selection"]["threshold"], "metrics": control["selection"]["metrics"],
         "event_metrics": control["event_coverage"], "hidden_size": 32,
         "note": "Reutiliza los pesos de E0. Solo cambia la selección de umbral; es el control más limpio del efecto de la decisión.",
         "report": "docs/aclaraciones/07_control_f1_sobre_pesos_f2.md"},
        {"id": "E1", "title": "Corrida con selección F1", "kind": "Corrida nueva desde cero", "status": "Completada; entorno distinto",
         "color": "#fff0d5", "device": "CPU", "epoch": trial["best_epoch"],
         "threshold": f1["threshold_frozen"], "metrics": f1["metrics"],
         "event_metrics": f1["event_metrics"], "hidden_size": 32,
         "note": "Selección por F1 y entorno efectivo CPU/Python distinto. No heredó los pesos de E0; el parentesco indica configuración de referencia.",
         "report": "docs/aclaraciones/08_resultados_rnn_f1_y_comparacion.md"},
        {"id": "E2", "title": "Referencia RNN32 pareada", "kind": "Repetición desde cero", "status": "Completada; pareja controlada",
         "color": "#dff3ff", "device": "T4 / CUDA", "epoch": reference["best_epoch"],
         "threshold": reference_eval["threshold_frozen"], "metrics": reference_eval["metrics"],
         "event_metrics": reference_eval["event_metrics"], "hidden_size": 32,
         "note": "Repite la especificación de E0 en la pareja de capacidad. Se entrenó desde cero; no heredó pesos históricos. " +
                 ("El entorno y las versiones coinciden con E0. " if same_historical_environment else "El entorno difiere del histórico. ") +
                 ("Los pesos resultantes tienen el mismo SHA-256 que E0." if same_historical_weights else "Los pesos resultantes difieren de E0."),
         "report": pair_report},
        {"id": "P1", "title": "Candidata RNN16 pareada", "kind": "Corrida nueva desde cero", "status": "Completada; pareja controlada",
         "color": "#ede9fe", "device": "T4 / CUDA", "epoch": candidate["best_epoch"],
         "threshold": candidate_eval["threshold_frozen"], "metrics": candidate_eval["metrics"],
         "event_metrics": candidate_eval["event_metrics"], "hidden_size": 16,
         "note": "La propuesta P1 ya fue ejecutada. Se compara con E2 cambiando solo hidden_size 32→16; mantiene F2, BCE, datos, semilla y entorno pareado. Se entrenó desde cero, sin heredar pesos de E2.",
         "report": pair_report},
        {"id": "D1", "title": "RNN32 dropout 0,2", "kind": "Regularización pareada", "status": "Completada; evaluación CPU con paridad",
         "color": "#fce7f3", "device": "T4 / CUDA", "epoch": dropout_candidate["best_epoch"],
         "threshold": dropout_candidate_eval["threshold_frozen"], "metrics": dropout_candidate_eval["metrics"],
         "event_metrics": dropout_candidate_eval["event_metrics"], "hidden_size": 32, "dropout_rate": 0.2,
         "note": "Cambia solo dropout 0→0,2 después de ReLU. AP y BCE mejoran, pero F2 de validación baja 0,00122 y se cubren 17 eventos menos; una semilla, sin evidencia de mejora robusta.",
         "report": "docs/aclaraciones/14_resultados_dropout_rnn32.md"},
        {"id": "L1", "title": "LSTM32 causal pareada", "kind": "Comparación de arquitectura", "status": "Completada; evaluación CPU con paridad",
         "color": "#dbeafe", "device": "T4 / CUDA", "epoch": lstm_candidate["best_epoch"],
         "threshold": lstm_candidate_eval["threshold_frozen"], "metrics": lstm_candidate_eval["metrics"],
         "event_metrics": lstm_candidate_eval["event_metrics"], "hidden_size": 32,
         "recurrent_cell": "lstm", "parameter_count": lstm_candidate["parameter_count"],
         "note": "Cambia solo la celda RNN→LSTM a ancho 32. F2 baja 0,00727, AP sube 0,00731, BCE empeora 0,07102 y se cubren 10 eventos menos. Usa 12.377 parámetros; evidencia de una semilla.",
         "report": "docs/aclaraciones/16_resultados_lstm_rnn32.md"},
        {"id": "L2", "title": "LSTM32 dropout 0,2", "kind": "Regularización pareada", "status": "Completada; evaluación CPU con paridad",
         "color": "#fce7f3", "device": "T4 / CUDA", "epoch": lstm_dropout_candidate["best_epoch"],
         "threshold": lstm_dropout_candidate_eval["threshold_frozen"], "metrics": lstm_dropout_candidate_eval["metrics"],
         "event_metrics": lstm_dropout_candidate_eval["event_metrics"], "hidden_size": 32, "dropout_rate": 0.2,
         "recurrent_cell": "lstm", "parameter_count": lstm_dropout_candidate["parameter_count"],
         "note": "Cambia solo dropout 0→0,2 dentro de LSTM32. F2 sube 0,00650 y cubre 40 eventos más, con 1.107 FP adicionales y AP menor. El umbral 0,05 toca el límite de la cuadrícula; una semilla.",
         "report": "docs/aclaraciones/18_resultados_lstm_dropout.md"},
        {"id": "L3", "title": "LSTM32 ReduceLROnPlateau", "kind": "Scheduler pareado", "status": "Completada; scheduler no activado",
         "color": "#ccfbf1", "device": "T4 / CUDA", "epoch": scheduler_candidate["best_epoch"],
         "threshold": scheduler_candidate_eval["threshold_frozen"], "metrics": scheduler_candidate_eval["metrics"],
         "event_metrics": scheduler_candidate_eval["event_metrics"], "hidden_size": 32, "dropout_rate": 0.2,
         "recurrent_cell": "lstm", "parameter_count": scheduler_candidate["parameter_count"],
         "note": "ReduceLROnPlateau monitorizó BCE train, pero nunca redujo LR: ambas corridas usaron 0,001 las 30 épocas y comparten pesos exactos. ΔF2=0; el scheduler no tuvo efecto observado.",
         "report": "docs/aclaraciones/20_resultados_lstm_scheduler.md"},
    ]
    by_id = {node["id"]: node for node in nodes}
    definitions = [
        {"parent": "E0", "child": "C1", "relation": "derived_decision",
         "changes": ["Criterio de umbral F2 → F1", "Umbral 0,12 → 0,22", "Pesos congelados"], "isolated_comparison": True},
        {"parent": "E0", "child": "E1", "relation": "reference_comparison",
         "changes": ["Selección de checkpoint y umbral F2 → F1", "T4/CUDA → CPU", "Python 3.13.16 → 3.13.18"], "isolated_comparison": False,
         "caveat": "Esta comparación cambió el entorno efectivo además del criterio de selección; los deltas no pueden atribuirse a un único factor."},
        {"parent": "E0", "child": "E2", "relation": "reference_repeat",
         "changes": ["Misma especificación RNN32 y selección F2", "Repetición desde cero, sin herencia de pesos", "Mismo entorno T4 y versiones" if same_historical_environment else "Entorno efectivo distinto del histórico"],
         "isolated_comparison": same_historical_environment,
         "same_historical_weights_sha256": same_historical_weights,
         "caveat": None if same_historical_environment else "La repetición de referencia usa un entorno distinto; revisar la comparación con el histórico."},
        {"parent": "E2", "child": "P1", "relation": "paired_capacity_comparison",
         "changes": ["hidden_size 32 → 16", "Misma pareja T4: datos, versiones, F2, BCE y semilla fijos"], "isolated_comparison": True,
         "pair_id": pair["pair_id"]},
        {"parent": "E2", "child": "D1", "relation": "paired_dropout_comparison",
         "changes": ["Dropout de fusión 0,0 → 0,2 después de ReLU", "Misma pareja T4: datos, versiones, F2, BCE y semilla fijos"],
         "isolated_comparison": True, "pair_id": dropout_pair["pair_id"],
         "cpu_probability_parity_max_abs": {"reference": dropout_reference_eval["max_probability_difference"], "candidate": dropout_candidate_eval["max_probability_difference"]}},
        {"parent": "E2", "child": "L1", "relation": "paired_architecture_comparison",
         "changes": ["Celda recurrente RNN tanh → LSTM causal; ancho 32", "Misma pareja T4: datos, FFNN, máscaras, F2, BCE y semilla fijos"],
         "isolated_comparison": True, "pair_id": lstm_pair["pair_id"],
         "parameter_count_delta": lstm_candidate["parameter_count"] - lstm_reference["parameter_count"],
         "reference_weights_match_historical": lstm_reference["files_sha256"]["weights.pt"] == base["files_sha256"]["weights.pt"],
         "cpu_probability_parity_max_abs": {"reference": lstm_reference_eval["max_probability_difference"], "candidate": lstm_candidate_eval["max_probability_difference"]}},
        {"parent": "L1", "child": "L2", "relation": "paired_dropout_comparison",
         "changes": ["Dropout en cabeza de fusión 0,0 → 0,2 después de ReLU", "Misma pareja T4: LSTM32, datos, versiones, F2, BCE y semilla fijos"],
         "isolated_comparison": True, "pair_id": lstm_dropout_pair["pair_id"],
         "reference_weights_match_parent": lstm_dropout_reference["files_sha256"]["weights.pt"] == lstm_candidate["files_sha256"]["weights.pt"],
         "threshold_grid_lower_boundary": lstm_dropout_comparison["threshold_grid_boundary_warning"],
         "cpu_probability_parity_max_abs": {"reference": lstm_dropout_reference_eval["max_probability_difference"], "candidate": lstm_dropout_candidate_eval["max_probability_difference"]}},
        {"parent": "L2", "child": "L3", "relation": "paired_scheduler_comparison",
         "changes": ["Tasa constante 0,001 → ReduceLROnPlateau sobre BCE de entrenamiento", "LSTM32, dropout 0,2, Adam, datos, semilla y selección F2 fijos"],
         "isolated_comparison": True, "pair_id": scheduler_pair["pair_id"],
         "learning_rate_reduction_epochs": scheduler_summary["candidate"]["reduction_epochs"],
         "weights_identical_to_parent": scheduler_candidate["files_sha256"]["weights.pt"] == lstm_dropout_candidate["files_sha256"]["weights.pt"],
         "cpu_probability_identical_between_children": scheduler_comparison["probability_comparison"]["identical_after_cpu_replay"],
         "cpu_probability_parity_max_abs": {"reference": scheduler_reference_eval["max_probability_difference"], "candidate": scheduler_candidate_eval["max_probability_difference"]}},
    ]
    edges = []
    for definition in definitions:
        parent, child = by_id[definition["parent"]], by_id[definition["child"]]
        delta = {key: child["metrics"][key] - parent["metrics"][key]
                 for key in ["f1", "f2", "precision", "recall", "tp", "fp", "fn", "tn"]}
        delta["event_recall"] = child["event_metrics"]["recall_per_evaluable_event"] - parent["event_metrics"]["recall_per_evaluable_event"]
        edges.append({**definition, "observed_delta": delta})
    return {"schema_version": "experiment_family_v1", "sources_sha256": sources,
            "context": {"train": "2022–2023", "validation": "2024", "validation_samples": 25635,
                        "window_length": 8, "seed": 42, "loss": "BCE sin ponderación",
                        "final_test": "2025 reservada", "preprocessed_manifest_sha256": base["preprocessed_manifest_sha256"]},
            "nodes": nodes, "edges": edges,
            "result_comparison": {"nodes": ["C1", "E1"], "relation": "same_observed_decisions_at_0_22",
                                  "note": "Coinciden los conteos; pesos y probabilidades difieren numéricamente. No es herencia ni identidad del modelo."}}


def node_label(node):
    lines = [f"{node['id']} · {node['title']}", node["kind"], node["status"],
             f"RNN {node['hidden_size']} · {node['device']}"]
    if node.get("recurrent_cell") == "lstm":
        lines.append(f"LSTM causal · {node['parameter_count']:,} parámetros")
    if "dropout_rate" in node:
        lines.append(f"Dropout {fmt(node['dropout_rate'], 1)}")
    if node["metrics"]:
        m = node["metrics"]
        lines += [f"Época {node['epoch']} · umbral {fmt(node['threshold'], 2)}",
                  f"F1 {fmt(m['f1'])} | F2 {fmt(m['f2'])}",
                  f"P {fmt(100*m['precision'], 2)}% | R {fmt(100*m['recall'], 2)}%",
                  f"Eventos {node['event_metrics']['covered_events']}/{node['event_metrics']['events_with_evaluable_opportunity']}"]
    else:
        lines += ["Métrica de selección F2", "Métricas pendientes"]
    return json.dumps("\n".join(lines), ensure_ascii=False)


def render_dot(registry):
    lines = ['digraph Experimentos {',
             'graph [rankdir=TB, bgcolor="white", nodesep=0.45, ranksep=0.8, pad=0.25, fontname="DejaVu Sans"];',
             'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=12, margin="0.18,0.14", color="#334155"];',
             'edge [fontname="DejaVu Sans", fontsize=10, color="#475569", arrowsize=0.7];',
             'S0 [label="CONTEXTO FIJO\\nTrain 2022–2023 · Validación 2024 · W=8 · semilla 42\\n25.635 muestras · preprocesador congelado · prueba 2025 reservada", fillcolor="#f8fafc"];']
    for node in registry["nodes"]:
        style = "rounded,filled,dashed" if node["kind"] == "Propuesta" else "rounded,filled"
        lines.append(f'{node["id"]} [id="node-{node["id"]}", label={node_label(node)}, fillcolor="{node["color"]}", style="{style}"];')
    lines.append('S0 -> E0 [label="Entrenamiento de referencia"];')
    for edge in registry["edges"]:
        relation = edge["relation"]
        if relation == "derived_decision":
            style = 'color="#15803d", penwidth=2'
            delta = edge["observed_delta"]
            label = ["SOLO DECISIÓN · pesos congelados", *edge["changes"][:2],
                     f"ΔF1 {delta['f1']:+.4f} · ΔF2 {delta['f2']:+.4f}".replace(".", ","),
                     f"FP {delta['fp']:+,} · FN {delta['fn']:+,}".replace(",", ".")]
        elif relation == "reference_comparison":
            style = 'style=dashed, color="#b45309", penwidth=2'
            label = ["NUEVO ENTRENAMIENTO", *edge["changes"], "Comparación con factores adicionales"]
        elif relation == "reference_repeat":
            style = 'color="#475569", penwidth=1.5'
            label = ["REPETICIÓN DE REFERENCIA", *edge["changes"], "Parentesco de configuración; no pesos heredados"]
        elif relation == "paired_capacity_comparison":
            style = 'color="#7c3aed", penwidth=2'
            delta = edge["observed_delta"]
            label = ["CAPACIDAD · PAREJA CONTROLADA", *edge["changes"],
                     f"ΔF1 {delta['f1']:+.4f} · ΔF2 {delta['f2']:+.4f}".replace(".", ","),
                     f"FP {delta['fp']:+,} · FN {delta['fn']:+,}".replace(",", ".")]
        elif relation == "paired_dropout_comparison":
            style = 'color="#be185d", penwidth=2'
            delta = edge["observed_delta"]
            label = ["REGULARIZACIÓN · PAREJA CONTROLADA", *edge["changes"],
                     f"ΔF1 {delta['f1']:+.4f} · ΔF2 {delta['f2']:+.4f}".replace(".", ","),
                     f"FP {delta['fp']:+,} · FN {delta['fn']:+,}".replace(",", ".")]
        elif relation == "paired_architecture_comparison":
            style = 'color="#0369a1", penwidth=2'
            delta = edge["observed_delta"]
            label = ["ARQUITECTURA · PAREJA CONTROLADA", *edge["changes"],
                     f"ΔF1 {delta['f1']:+.4f} · ΔF2 {delta['f2']:+.4f}".replace(".", ","),
                     f"Parámetros {edge['parameter_count_delta']:+,} · FP {delta['fp']:+,} · FN {delta['fn']:+,}".replace(",", ".")]
        elif relation == "paired_scheduler_comparison":
            style = 'color="#0f766e", penwidth=2'
            delta = edge["observed_delta"]
            label = ["OPTIMIZACIÓN · PAREJA CONTROLADA", *edge["changes"],
                     "El scheduler no redujo la tasa", f"ΔF2 {delta['f2']:+.4f} · pesos idénticos".replace(".", ",")]
        elif relation == "proposed_comparison":
            style = 'style=dotted, color="#64748b"'
            label = ["PROPUESTO", *edge["changes"], "Sin resultados"]
        else:
            raise ValueError(f"Unknown edge relation: {relation}")
        label = json.dumps("\n".join(label), ensure_ascii=False)
        lines.append(f'{edge["parent"]} -> {edge["child"]} [{style}, label={label}];')
    left, right = registry["result_comparison"]["nodes"]
    lines += [f'{left} -> {right} [dir=none, constraint=false, style=dashed, color="#2563eb", label="Comparación de resultados\\nMismos conteos a umbral 0,22\\nNo implica pesos idénticos"];', '}']
    return "\n".join(lines) + "\n"


def render_html(svg, registry, output):
    # JSON is embedded as inert data, and the graph works offline without CDNs.
    data = json.dumps(registry, ensure_ascii=False).replace("<", "\\u003c")
    root_prefix = json.dumps(Path(os.path.relpath(ROOT, output.resolve())).as_posix() + "/")
    return '''<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Linaje de experimentos · F1 PitStop Predictor</title>
<style>
body{font:16px system-ui,sans-serif;color:#172033;background:#f5f7fb;margin:0;padding:24px}
h1{font-size:26px;margin:0 0 12px}p{line-height:1.5}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(290px,1fr);gap:20px}
.card{background:white;border:1px solid #dce2eb;border-radius:12px;padding:20px}.graph{overflow:auto}.graph svg{width:100%;height:auto;min-width:680px}
.node{cursor:pointer}.node:focus{outline:3px solid #2563eb}.selected polygon,.selected path{stroke:#1d4ed8!important;stroke-width:3px!important}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:7px;border-bottom:1px solid #e5e7eb}.note{background:#fff5da;padding:12px;border-radius:8px}
.small{font-size:13px;color:#475569}a{color:#1d4ed8}code{word-break:break-all}@media(max-width:1050px){.layout{grid-template-columns:1fr}}
</style>
<h1>Árbol de experimentos</h1>
<p>Cada nodo es una corrida o un control. Selecciona un nodo para ver resultados y cambios respecto a su referencia de comparación.</p>
<div class="layout"><div class="card graph">''' + svg + '''</div><section class="card" id="detail" aria-live="polite"></section></div>
<p class="small">Verde: control con pesos congelados. Naranja: comparación con diferencias adicionales de entorno. Gris: repetición de referencia. Morado: capacidad pareada. Rosa: dropout pareado. Azul: arquitectura LSTM pareada. Verde azulado: scheduler pareado. Azul oscuro sin dirección: comparación de resultados, sin relación de herencia.</p>
<p class="small">Las flechas indican procedencia o referencia de comparación; no demuestran causalidad. Los modelos nuevos se entrenan desde cero. Validación fue usada para selección y no es prueba independiente. 2025 permanece reservada.</p>
<script id="registry" type="application/json">''' + data + '''</script>
<script>
const registry=JSON.parse(document.getElementById('registry').textContent);
const rootPrefix=''' + root_prefix + ''';
const percent=x=>(100*x).toFixed(2)+' %';
const score=x=>x.toFixed(4);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function select(id){
 const n=registry.nodes.find(n=>n.id===id), edge=registry.edges.find(e=>e.child===id);
 for(const g of document.querySelectorAll('g.node'))g.classList.toggle('selected',g.id==='node-'+id);
 let content='<h2>'+escape(n.id+' · '+n.title)+'</h2><p>'+escape(n.kind+' · '+n.status)+'</p><p>'+escape(n.note)+'</p>';
 if(n.metrics){const m=n.metrics,e=n.event_metrics;
  const rows=[['Umbral',n.threshold.toFixed(2)],['F1',score(m.f1)],['F2',score(m.f2)],['Precisión',percent(m.precision)],['Recall por muestra',percent(m.recall)],['FP',m.fp],['FN',m.fn],['Eventos cubiertos',e.covered_events+'/'+e.events_with_evaluable_opportunity+' ('+percent(e.recall_per_evaluable_event)+')']];
  content+='<table><tbody>'+rows.map(r=>'<tr><th>'+escape(r[0])+'</th><td>'+escape(r[1])+'</td></tr>').join('')+'</tbody></table>';
 }else content+='<p><strong>Sin métricas:</strong> todavía no se ha ejecutado.</p>';
 if(edge){const parent=registry.nodes.find(p=>p.id===edge.parent);
  content+='<h3>Comparación frente a '+escape(parent.id+' · '+parent.title)+'</h3><ul>'+edge.changes.map(c=>'<li>'+escape(c)+'</li>').join('')+'</ul>';
  if(edge.isolated_comparison===false)content+='<p class="note">'+escape(edge.caveat||'Esta comparación incluye factores adicionales; los deltas observados no pueden atribuirse a un único cambio.')+'</p>';
  if(edge.observed_delta){const d=edge.observed_delta,sign=x=>x>0?'+':'';
   content+='<p>ΔF1: '+sign(d.f1)+score(d.f1)+' · ΔF2: '+sign(d.f2)+score(d.f2)+'<br>ΔFP: '+sign(d.fp)+d.fp+' · ΔFN: '+sign(d.fn)+d.fn+'</p>';
  }
 }
 content+='<p><a href="'+rootPrefix+encodeURI(n.report)+'">Abrir informe</a></p>';
 document.getElementById('detail').innerHTML=content;
}
for(const n of registry.nodes){const g=document.getElementById('node-'+n.id);g.setAttribute('tabindex','0');g.setAttribute('role','button');g.setAttribute('aria-label',n.title);g.addEventListener('click',()=>select(n.id));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(n.id)}})}
select('E0');
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    registry = build_registry()
    args.output.mkdir(parents=True)
    (args.output / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    (args.output / "arbol.dot").write_text(render_dot(registry))
    subprocess.run(["dot", "-Tsvg", str(args.output / "arbol.dot"), "-o", str(args.output / "arbol.svg")], check=True)
    svg = (args.output / "arbol.svg").read_text()
    svg = svg[svg.index("<svg"):]
    (args.output / "arbol.html").write_text(render_html(svg, registry, args.output))
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in args.output.iterdir() if p.is_file()}
    (args.output / "manifest.json").write_text(json.dumps({
        "artifact_version": "experiment_family_view_v1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources_sha256": registry["sources_sha256"], "files_sha256": artifacts,
        "test_used": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
