"""Small RNN reference experiment and frozen local validation evaluation."""
import copy
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .experiment_metrics import binary_metrics, select_f2_threshold, threshold_curve
from .rnn_data import KEY, MODEL_ARRAYS, load_split, read_json, sha256, verify_preprocessed, write_json
from .rnn_model import SmallPitStopRNN, model_spec_from_preprocessor


def choose_device(requested):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def _loader(data, batch_size, shuffle=False, seed=42):
    tensors = []
    for name in MODEL_ARRAYS:
        selected = data.arrays[name][data.positions]
        if name == "mask_reason_codes":
            # Published reasons cover numeric and categorical features. The
            # recurrent numeric availability bits consume only the numeric prefix;
            # categorical absence is already encoded by its reserved IDs.
            selected = selected[..., :data.arrays["X_numeric"].shape[-1]]
        value = torch.from_numpy(np.array(selected, copy=True))
        if name == "X_numeric":
            value = value.float()
        elif name in {"X_dynamic_categorical_ids", "X_static_categorical_ids", "mask_reason_codes"}:
            value = value.long()
        else:
            value = value.bool()
        tensors.append(value)
    dataset = TensorDataset(*tensors, torch.from_numpy(data.targets).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def _forward(model, batch, device):
    return model(*(value.to(device) for value in batch[:-1]))


def predict(model, loader, device):
    model.eval()
    probabilities, loss_sum, count = [], 0.0, 0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            logits = _forward(model, batch, device)
            targets = batch[-1].to(device)
            loss_sum += float(criterion(logits, targets).item())
            count += len(targets)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64), loss_sum / count


def prediction_table(data, probabilities, threshold):
    table = data.metadata.iloc[data.positions].reset_index(drop=True).copy()
    table["target"] = data.targets.astype(np.int8)
    table["probability"] = probabilities
    table["alert"] = probabilities >= threshold
    table["threshold"] = float(threshold)
    table["error_type"] = np.select(
        [(table.target == 1) & ~table.alert, (table.target == 0) & table.alert],
        ["false_negative", "false_positive"], default="correct")
    return table


def _validate_config(config):
    if config.get("experiment_id") != "rnn_reference_v1":
        raise ValueError("This implementation requires the rnn_reference_v1 protocol")
    protocol = config["training"]
    for name in ("batch_size", "max_epochs", "patience"):
        if type(protocol[name]) is not int or protocol[name] < 1:
            raise ValueError(f"Invalid {name}")
    if protocol["learning_rate"] <= 0 or protocol["gradient_clip_norm"] <= 0:
        raise ValueError("Learning rate and gradient clip must be positive")
    if protocol.get("loss") != "unweighted_bce_with_logits" or protocol.get("optimizer") != "Adam":
        raise ValueError("Loss/optimizer differs from the declared reference")
    if config["selection"]["checkpoint_threshold"] != 0.5:
        raise ValueError("Reference checkpoint threshold must remain 0.5")
    grid = np.asarray(config["selection"]["threshold_grid"], dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any((grid < 0) | (grid > 1)):
        raise ValueError("Invalid declared threshold grid")


def train_experiment(config_path, output, root, device_override=None):
    config_path, output, root = Path(config_path), Path(output), Path(root)
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new run directory and ZIP filename")
    config = read_json(config_path)
    _validate_config(config)
    package = root / config["preprocessed_package"]
    input_manifest, state = verify_preprocessed(package, config["preprocessed_manifest_sha256"])
    train, validation = load_split(package, "train", state), load_split(package, "validation", state)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = choose_device(device_override or config["training"]["device"])
    torch.set_num_threads(int(config["training"].get("cpu_num_threads", 2)))
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    protocol = config["training"]
    model_spec = model_spec_from_preprocessor(state, config["model"])
    model = SmallPitStopRNN(model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    training_loader = _loader(train, protocol["batch_size"], shuffle=True, seed=seed)
    training_evaluation_loader = _loader(train, protocol["batch_size"])
    validation_loader = _loader(validation, protocol["batch_size"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running", "purpose": config.get("run_purpose", "reference_training")})
    effective_config = copy.deepcopy(config)
    effective_config["training"]["device_effective"] = str(device)
    effective_config["training"]["device_override"] = device_override
    write_json(output / "config.json", effective_config)
    best_f2, best_loss, best_epoch, stale, history = -1.0, float("inf"), 0, 0, []
    start = time.perf_counter()
    try:
        for epoch in range(1, protocol["max_epochs"] + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum, count = 0.0, 0
            for batch in training_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, batch, device)
                targets = batch[-1].to(device)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip_norm"])
                optimizer.step()
                loss_sum += float(loss.item()) * len(targets)
                count += len(targets)
            train_probabilities, train_eval_loss = predict(model, training_evaluation_loader, device)
            train_metrics = binary_metrics(train.targets, train_probabilities, 0.5)
            probabilities, val_loss = predict(model, validation_loader, device)
            metrics = binary_metrics(validation.targets, probabilities, 0.5)
            improved = metrics["f2"] > best_f2 or (metrics["f2"] == best_f2 and val_loss < best_loss)
            if improved:
                best_f2, best_loss, best_epoch, stale = metrics["f2"], val_loss, epoch, 0
                torch.save({k: value.detach().cpu().clone() for k, value in model.state_dict().items()},
                           output / "weights.pt")
            else:
                stale += 1
            row = {"epoch": epoch, "train_loss": loss_sum / count,
                   "train_evaluation_loss": train_eval_loss,
                   "train_f2_at_0_5": train_metrics["f2"],
                   "train_precision_at_0_5": train_metrics["precision"],
                   "train_recall_at_0_5": train_metrics["recall"], "validation_loss": val_loss,
                   "validation_f2_at_0_5": metrics["f2"], "validation_precision_at_0_5": metrics["precision"],
                   "validation_recall_at_0_5": metrics["recall"], "checkpoint_selected": improved,
                   "epoch_seconds": time.perf_counter() - epoch_start}
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(json.dumps(row), flush=True)
            if stale >= protocol["patience"]:
                break
        model.load_state_dict(torch.load(output / "weights.pt", map_location=device, weights_only=True))
        probabilities, val_loss = predict(model, validation_loader, device)
        train_probabilities, train_eval_loss = predict(model, training_evaluation_loader, device)
        selected = select_f2_threshold(validation.targets, probabilities, config["selection"]["threshold_grid"])
        threshold = selected["threshold"]
        write_json(output / "threshold.json", selected)
        write_json(output / "model_spec.json", model_spec)
        shutil.copyfile(package / "preprocessor.json", output / "preprocessor.json")
        write_json(output / "schema.json", state["schema"])
        write_json(output / "metrics.json", {
            "split": "validation", "selection_epoch": best_epoch,
            "checkpoint_selection": "maximum validation F2 at fixed threshold 0.5; tie lower validation BCE",
            "validation_loss_selected_checkpoint": val_loss,
            "fixed_threshold_0_5": binary_metrics(validation.targets, probabilities, 0.5),
            "selected_threshold": selected,
            "train_selected_checkpoint": {
                "bce": train_eval_loss,
                "fixed_threshold_0_5": binary_metrics(train.targets, train_probabilities, 0.5),
                "validation_selected_threshold": binary_metrics(train.targets, train_probabilities, threshold),
            },
            "warning": "Validation was used for checkpoint and threshold selection; final test remains reserved",
        })
        pd.DataFrame(threshold_curve(validation.targets, probabilities,
                     config["selection"]["threshold_grid"])).to_csv(output / "threshold_curve.csv", index=False)
        prediction_table(validation, probabilities, threshold).to_csv(
            output / "validation_predictions.csv.gz", index=False, compression="gzip")
        source_names = ["rnn_model.py", "rnn_data.py", "rnn_experiment.py", "experiment_metrics.py"]
        sources = output / "sources"
        sources.mkdir()
        for name in source_names:
            shutil.copyfile(Path(__file__).parent / name, sources / name)
        write_json(output / "environment.json", {
            "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "seed": seed, "deterministic_algorithms": True,
        })
        write_json(output / "status.json", {"status": "complete"})
        files = {str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file()}
        write_json(output / "manifest.json", {
            "artifact_version": "rnn_reference_bundle_v1", "experiment_id": config["experiment_id"],
            "files_sha256": files, "preprocessed_manifest_sha256": sha256(package / "manifest.json"),
            "preprocessor_sha256": input_manifest["output_files_sha256"]["preprocessor.json"],
            "model_source_sha256": sha256(Path(__file__).parent / "rnn_model.py"),
            "fit_split": "train", "selection_split": "validation", "test_used": False,
            "best_epoch": best_epoch, "train_eligible_samples": len(train.positions),
            "validation_eligible_samples": len(validation.positions),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": time.perf_counter() - start,
            "parity_atol": config["evaluation"]["parity_atol"],
            "parity_rtol": config["evaluation"]["parity_rtol"],
            "reproducibility_note": "No bitwise equality across GPU/CPU or framework versions is promised",
        })
        with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
            for file in output.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(output))
        return output
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def event_coverage(events, predictions):
    """Count confirmed entries covered by at least one eligible prior alert."""
    records = []
    for row in events.loc[events.event_status.eq("pit_stop")].itertuples(index=False):
        driver = str(row.DriverNumber).removesuffix(".0")
        entry_time = pd.Timedelta(row.entry_time).total_seconds()
        same = predictions.loc[(predictions.Temporada == int(row.Temporada))
            & (predictions.Ronda == int(row.Ronda))
            & (predictions.DriverNumber.astype(str).str.removesuffix(".0") == driver)]
        opportunities = same.loc[(same.prediction_lap <= row.entry_lap)
            & (same.prediction_lap + 2 >= row.entry_lap)
            & (same.decision_time_seconds < entry_time)]
        if len(opportunities) and not opportunities.target.eq(1).all():
            raise ValueError(f"Event/target contradiction for {row.event_id}")
        alerts = opportunities.loc[opportunities.alert]
        records.append({"event_id": row.event_id, "Temporada": int(row.Temporada),
                        "Ronda": int(row.Ronda), "DriverNumber": driver,
                        "entry_lap": int(row.entry_lap), "evaluable_opportunities": len(opportunities),
                        "covered": bool(len(alerts)), "alert_opportunities": len(alerts),
                        "maximum_lead_laps": None if alerts.empty else int(row.entry_lap - alerts.prediction_lap.min())})
    table = pd.DataFrame(records, columns=["event_id", "Temporada", "Ronda", "DriverNumber", "entry_lap",
                         "evaluable_opportunities", "covered", "alert_opportunities", "maximum_lead_laps"])
    evaluable = table.evaluable_opportunities.gt(0)
    covered = int(table.loc[evaluable, "covered"].sum())
    count = int(evaluable.sum())
    return table, {"confirmed_events": len(table), "events_with_evaluable_opportunity": count,
                   "events_without_evaluable_opportunity": int((~evaluable).sum()),
                   "covered_events": covered, "recall_per_evaluable_event": covered / count if count else 0.0}


def attach_decision_times(predictions, sample_index):
    """Join the original causal decision anchor without changing prediction order."""
    if sample_index.sample_id.isna().any() or sample_index.sample_id.duplicated().any():
        raise ValueError("Causal sample index must have unique complete sample_id")
    anchors = sample_index.loc[sample_index.Temporada.eq(2024), KEY + ["decision_time_seconds"]]
    if "decision_time_seconds" in predictions:
        predictions = predictions.drop(columns="decision_time_seconds")
    merged = predictions.merge(anchors, on=KEY, how="left", validate="one_to_one", sort=False)
    if not np.array_equal(merged.sample_id, predictions.sample_id):
        raise ValueError("Decision anchor join changed sample order")
    if not np.isfinite(merged.decision_time_seconds.to_numpy(dtype=float)).all():
        raise ValueError("Missing or invalid causal decision anchor for an eligible prediction")
    return merged


def evaluate_bundle(bundle, data, output, device_name="cpu", root=None):
    bundle, data, output = Path(bundle), Path(data), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation directory")
    manifest = read_json(bundle / "manifest.json")
    if manifest.get("artifact_version") != "rnn_reference_bundle_v1" or manifest.get("test_used") is not False:
        raise ValueError("Unsupported experiment bundle")
    for name, expected in manifest["files_sha256"].items():
        file = (bundle / name).resolve()
        if not file.is_relative_to(bundle.resolve()) or sha256(file) != expected:
            raise ValueError(f"Exported bundle hash mismatch: {name}")
    if read_json(bundle / "status.json").get("status") != "complete":
        raise ValueError("Run is not complete")
    if sha256(Path(__file__).parent / "rnn_model.py") != manifest["model_source_sha256"]:
        raise ValueError("Model implementation differs from the exported run; use its frozen source version")
    input_manifest, state = verify_preprocessed(data, manifest["preprocessed_manifest_sha256"])
    if sha256(data / "preprocessor.json") != sha256(bundle / "preprocessor.json"):
        raise ValueError("Evaluator must use the exported frozen preprocessor")
    validation = load_split(data, "validation", state)
    model = SmallPitStopRNN(read_json(bundle / "model_spec.json"))
    device = choose_device(device_name)
    model.load_state_dict(torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device)
    config = read_json(bundle / "config.json")
    probabilities, loss = predict(model, _loader(validation, config["training"]["batch_size"]), device)
    threshold = read_json(bundle / "threshold.json")["threshold"]
    reference = pd.read_csv(bundle / "validation_predictions.csv.gz", dtype={"DriverNumber": str})
    expected_ids = validation.metadata.iloc[validation.positions].sample_id.to_numpy()
    if not np.array_equal(reference.sample_id, expected_ids) or not np.array_equal(reference.target, validation.targets):
        raise ValueError("Exported validation predictions/targets do not align with current data")
    delta = np.abs(reference.probability.to_numpy() - probabilities)
    parity = bool(np.allclose(reference.probability, probabilities, atol=manifest["parity_atol"],
                              rtol=manifest["parity_rtol"]))
    if not parity:
        raise ValueError(f"Local/Colab probability parity failed; maximum difference {delta.max()}")
    table = prediction_table(validation, probabilities, threshold)
    metrics = binary_metrics(validation.targets, probabilities, threshold)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    events_path = root / config["evaluation"]["events_source"]
    if sha256(events_path) != config["evaluation"]["events_sha256"]:
        raise ValueError("Event source differs from the pinned label publication")
    index_path = root / config["evaluation"]["decision_index_source"]
    index_hash = sha256(index_path)
    if (index_hash != config["evaluation"]["decision_index_sha256"]
            or index_hash != input_manifest["source_hashes"]["sample_index"]):
        raise ValueError("Decision index differs from the causal preprocessing source")
    sample_index = pd.read_csv(index_path, dtype={"DriverNumber": str})
    table = attach_decision_times(table, sample_index)
    events = pd.read_csv(events_path, dtype={"DriverNumber": str}, keep_default_na=False)
    # Only validation events participate in local development evaluation.
    event_table, event_metrics = event_coverage(events.loc[events.Temporada == 2024], table)
    by_race = []
    for (year, rnd), group in table.groupby(["Temporada", "Ronda"]):
        by_race.append({"Temporada": int(year), "Ronda": int(rnd),
                        **binary_metrics(group.target, group.probability, threshold)})
    by_driver = []
    for driver, group in table.groupby("DriverNumber"):
        by_driver.append({"DriverNumber": str(driver),
                          **binary_metrics(group.target, group.probability, threshold)})
    output.mkdir(parents=True, exist_ok=False)
    table.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    table.loc[table.error_type != "correct"].to_csv(output / "errors.csv.gz", index=False, compression="gzip")
    pd.DataFrame(by_race).to_csv(output / "metrics_by_race.csv", index=False)
    pd.DataFrame(by_driver).to_csv(output / "metrics_by_driver.csv", index=False)
    event_table.to_csv(output / "event_coverage.csv", index=False)
    write_json(output / "metrics.json", {
        "split": "validation", "metrics": metrics, "bce": loss, "threshold_frozen": threshold,
        "parity_passed": parity, "max_probability_difference": float(delta.max()),
        "threshold_decision_differences": int(np.sum(reference.alert.to_numpy() != table.alert.to_numpy())),
        "excluded_samples": len(validation.metadata) - len(validation.positions),
        "event_metrics": event_metrics,
        "note": "Diagnostic validation results, not an independent final test evaluation",
    })
    write_json(output / "manifest.json", {
        "artifact_version": "rnn_local_validation_v1", "bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "data_manifest_sha256": sha256(data / "manifest.json"), "device": str(device),
        "events_source_sha256": sha256(events_path),
        "decision_index_sha256": index_hash,
        "files_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    })
    return output
