"""Train a controlled causal RNN32/LSTM32 F2 pair in one effective GPU environment."""

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_data import read_json, sha256, write_json
from f1_pitstop.rnn_experiment import (
    _selection_protocol, _validate_config, choose_device, train_experiment,
)

_METADATA = {
    "experiment_id", "question", "run_purpose", "orthogonalization", "pairing",
    "companion_configs",
}
_ENVIRONMENT_KEYS = (
    "python", "platform", "torch", "numpy", "pandas", "device", "gpu", "cuda",
)
_METRICS = ("f1", "f2", "average_precision", "precision", "recall")
_CHILDREN = ("rnn32_reference", "lstm32_candidate")


def _same_json(first, second):
    return json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second, sort_keys=True, allow_nan=False
    )


def _material_controls(config):
    return copy.deepcopy({key: value for key, value in config.items() if key not in _METADATA})


def _check_config_hashes(paths, expected):
    if any(sha256(path) != digest for path, digest in zip(paths, expected)):
        raise ValueError("Declared pair config changed during the run")


def _validate_pair_configs(reference, candidate):
    expected_cells = ("rnn", "lstm")
    for config, expected_cell in zip((reference, candidate), expected_cells):
        _validate_config(config)
        if config.get("experiment_id") != "rnn_reference_v1":
            raise ValueError("LSTM pair requires the established F2 selection protocol")
        if type(config["model"].get("hidden_size")) is not int or config["model"]["hidden_size"] != 32:
            raise ValueError("Both pair models must use hidden_size 32")
        if config["model"].get("recurrent_cell", "rnn") != expected_cell:
            raise ValueError(f"Expected recurrent_cell={expected_cell}")
        if float(config["model"].get("dropout_rate", 0.0)) != 0.0:
            raise ValueError("Dropout must remain zero in the architecture comparison")
        if _selection_protocol(config) != ("f2", 2):
            raise ValueError("LSTM comparison requires F2 selection")
        if type(config["training"].get("seed")) is not int or config["training"]["seed"] != 42:
            raise ValueError("LSTM pair requires seed 42")
        if config["training"].get("device") != "cuda":
            raise ValueError("Training requires explicit CUDA; CPU fallback is prohibited")
    reference_controls = _material_controls(reference)
    candidate_controls = _material_controls(candidate)
    reference_controls["model"].pop("recurrent_cell", None)
    candidate_controls["model"].pop("recurrent_cell", None)
    if not _same_json(reference_controls, candidate_controls):
        raise ValueError("Material controls differ beyond model.recurrent_cell")

def _capture_environment(protocol):
    device = choose_device(protocol["device"])
    if device.type != "cuda":
        raise ValueError("CUDA GPU is required for this experiment")
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "pandas": pd.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda,
        "cuda_device_index": torch.cuda.current_device() if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_num_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _check_environment(baseline, snapshot, child, protocol):
    # train_experiment intentionally sets threads/determinism. Their initial
    # process values may differ; compare hardware and versions across all three
    # snapshots, then verify the effective training flags after each child.
    for key in (*_ENVIRONMENT_KEYS, "cuda_device_index"):
        if snapshot[key] != baseline[key]:
            raise ValueError(f"Effective environment changed: {key}")
    for key in _ENVIRONMENT_KEYS:
        if child.get(key) != baseline[key]:
            raise ValueError(f"Child environment differs from the pair: {key}")
    if child.get("seed") != 42 or child.get("deterministic_algorithms") is not True:
        raise ValueError("Child seed/determinism differs from the pair protocol")
    if (snapshot["deterministic_algorithms"] is not True
            or snapshot["cpu_num_threads"] != protocol.get("cpu_num_threads", 2)
            or snapshot["cudnn_benchmark"] is not False):
        raise ValueError("Effective training runtime differs from the pair protocol")


def _verified_child_manifest(directory):
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("artifact_version") != "rnn_reference_bundle_v1"
            or manifest.get("test_used") is not False
            or read_json(directory / "status.json").get("status") != "complete"):
        raise ValueError("A child run is incomplete or used the final test")
    for name, digest in manifest["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError(f"Child artifact hash mismatch: {name}")
    if manifest.get("selection_metric") != "f2" or manifest.get("selection_beta") != 2:
        raise ValueError("A child run did not select by F2")
    return manifest


def _check_populations(reference_dir, candidate_dir, reference, candidate):
    for key in (
        "preprocessed_manifest_sha256", "preprocessor_sha256",
        "train_eligible_samples", "validation_eligible_samples", "fit_split",
        "selection_split", "test_used", "parity_atol", "parity_rtol",
    ):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Child populations or frozen controls differ: {key}")
    for name in ("rnn_model.py", "rnn_data.py", "rnn_experiment.py", "experiment_metrics.py"):
        key = f"sources/{name}"
        if reference["files_sha256"][key] != candidate["files_sha256"][key]:
            raise ValueError(f"Child source implementations differ: {name}")
    expected_sources = (("rnn_model.py", reference), ("lstm_model.py", candidate))
    for source_name, manifest in expected_sources:
        if manifest.get("model_source_path", "rnn_model.py") != source_name:
            raise ValueError(f"Unexpected model source for {source_name}")
        if manifest["files_sha256"].get(f"sources/{source_name}") != manifest["model_source_sha256"]:
            raise ValueError(f"Model source hash missing or inconsistent: {source_name}")
    first = pd.read_csv(reference_dir / "validation_predictions.csv.gz", dtype={"DriverNumber": str})
    second = pd.read_csv(candidate_dir / "validation_predictions.csv.gz", dtype={"DriverNumber": str})
    keys = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap", "target"]
    if not first[keys].equals(second[keys]):
        raise ValueError("Child validation populations/targets differ")


def _child_summary(directory, manifest):
    metrics = read_json(directory / "metrics.json")
    return {
        "directory": directory.name,
        "hidden_size": read_json(directory / "model_spec.json")["hidden_size"],
        "recurrent_cell": read_json(directory / "model_spec.json").get("recurrent_cell", "rnn"),
        "dropout_rate": read_json(directory / "model_spec.json")["dropout_rate"],
        "parameter_count": manifest["parameter_count"], "best_epoch": manifest["best_epoch"],
        "elapsed_seconds": manifest["elapsed_seconds"],
        "threshold": metrics["selected_threshold"]["threshold"],
        "validation": {
            "bce": metrics["validation_loss_selected_checkpoint"],
            "fixed_threshold_0_5": metrics["fixed_threshold_0_5"],
            "selected_threshold": metrics["selected_threshold"]["metrics"],
        },
        "train": {
            "bce": metrics["train_selected_checkpoint"]["bce"],
            "fixed_threshold_0_5": metrics["train_selected_checkpoint"]["fixed_threshold_0_5"],
            "selected_threshold": metrics["train_selected_checkpoint"]["validation_selected_threshold"],
        },
    }


def train_lstm_pair(reference_config, candidate_config, output, root=ROOT):
    """Run both model conditions and publish after all comparison gates pass.

    Config mismatches are rejected before creating output. Runtime failures keep
    their partial child runs and a failed parent status; they do not publish a
    valid parent ZIP. An I/O failure may leave an incomplete archive.
    The parent ZIP contains every file listed in its manifest plus the manifest;
    the ZIP files created by individual training runs are deliberately excluded.
    """
    root, output = Path(root).resolve(), Path(output)
    reference_config, candidate_config = Path(reference_config), Path(candidate_config)
    if not reference_config.is_absolute():
        reference_config = root / reference_config
    if not candidate_config.is_absolute():
        candidate_config = root / candidate_config
    if not output.is_absolute():
        output = root / output
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("Choose a new pair directory and ZIP filename")
    config_paths = [reference_config, candidate_config]
    config_hashes = [sha256(path) for path in config_paths]
    configs = [read_json(path) for path in config_paths]
    _validate_pair_configs(*configs)
    _check_config_hashes(config_paths, config_hashes)
    protocol = configs[0]["training"]
    baseline = _capture_environment(protocol)
    output.mkdir(parents=True, exist_ok=False)
    environment = {"snapshots": {"before": baseline}, "children": {}, "comparison_authorized": False}
    write_json(output / "pair_environment.json", environment)
    write_json(output / "status.json", {"status": "running", "comparison_authorized": False})
    stage = "reference_training"
    try:
        sources = output / "pair_sources"
        sources.mkdir()
        runner_source = sources / Path(__file__).name
        shutil.copyfile(__file__, runner_source)
        config_sources = output / "pair_configs"
        config_sources.mkdir()
        for path, name, digest in zip(config_paths, ("reference.json", "candidate.json"), config_hashes):
            shutil.copyfile(path, config_sources / name)
            if sha256(config_sources / name) != digest:
                raise ValueError("Declared pair config changed before its snapshot")
        for name, config_path, snapshot_name in zip(
            _CHILDREN, (reference_config, candidate_config), ("between", "after")
        ):
            stage = f"{name}_training"
            _check_config_hashes(config_paths, config_hashes)
            train_experiment(config_path, output / name, root)
            stage = f"{name}_environment_gate"
            child_environment = read_json(output / name / "environment.json")
            environment["children"][name] = child_environment
            snapshot = _capture_environment(protocol)
            environment["snapshots"][snapshot_name] = snapshot
            write_json(output / "pair_environment.json", environment)
            _check_environment(baseline, snapshot, child_environment, protocol)
        stage = "population_and_hash_gate"
        _check_config_hashes(config_paths, config_hashes)
        directories = [output / name for name in _CHILDREN]
        manifests = [_verified_child_manifest(directory) for directory in directories]
        effective_configs = [read_json(directory / "config.json") for directory in directories]
        _validate_pair_configs(*effective_configs)
        for declared, effective in zip(configs, effective_configs):
            expected = copy.deepcopy(declared)
            expected["selection"]["beta"] = 2
            expected["training"].update({"device_effective": baseline["device"], "device_override": None})
            if not _same_json(_material_controls(expected), _material_controls(effective)):
                raise ValueError("Child effective config differs from the declared pair controls")
        specs = [read_json(directory / "model_spec.json") for directory in directories]
        if [spec["hidden_size"] for spec in specs] != [32, 32]:
            raise ValueError("Both child models must use hidden_size 32")
        if [spec.get("recurrent_cell", "rnn") for spec in specs] != ["rnn", "lstm"]:
            raise ValueError("Child recurrent cells differ from the declared comparison")
        if not _same_json(
            {key: value for key, value in specs[0].items() if key != "recurrent_cell"},
            {key: value for key, value in specs[1].items() if key != "recurrent_cell"},
        ):
            raise ValueError("Child model specs differ beyond recurrent_cell")
        _check_populations(*directories, *manifests)
        summaries = [_child_summary(directory, manifest) for directory, manifest in zip(directories, manifests)]
        summary = {
            "pair_id": configs[0].get("pairing", {}).get("pair_id", output.name),
            "comparison_authorized": True, "selection_metric": "f2", "selection_beta": 2,
            "seed": 42, "test_used": False,
            "changed_factor": {"name": "model.recurrent_cell", "reference": "rnn", "candidate": "lstm",
                               "location": "causal_recurrent_branch"},
            "reference": summaries[0], "candidate": summaries[1],
            "delta_candidate_minus_reference": {
                f"{population}_{threshold}": {
                    metric: summaries[1][population][threshold][metric] - summaries[0][population][threshold][metric]
                    for metric in _METRICS
                }
                for population in ("train", "validation")
                for threshold in ("fixed_threshold_0_5", "selected_threshold")
            },
            "note": "Validation selects checkpoints and thresholds; final test 2025 remains reserved. Historical runs are context only.",
        }
        write_json(output / "pair_summary.json", summary)
        environment["comparison_authorized"] = True
        write_json(output / "pair_environment.json", environment)
        write_json(output / "status.json", {"status": "complete", "comparison_authorized": True})
        files = {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.suffix.lower() != ".zip"
            and path != output / "pair_manifest.json"
        }
        write_json(output / "pair_manifest.json", {
            "artifact_version": "rnn_lstm_pair_v1", "pair_id": summary["pair_id"],
            "files_sha256": files, "comparison_authorized": True,
            "selection_metric": "f2", "selection_beta": 2, "seed": 42, "test_used": False,
            "children": list(_CHILDREN), "changed_factor": summary["changed_factor"],
            "config_sources_sha256": dict(zip(("reference", "candidate"), config_hashes)),
            "runner_source": runner_source.relative_to(output).as_posix(),
            "runner_source_sha256": sha256(runner_source),
            "preprocessed_manifest_sha256": manifests[0]["preprocessed_manifest_sha256"],
            "preprocessor_sha256": manifests[0]["preprocessor_sha256"],
            "train_eligible_samples": manifests[0]["train_eligible_samples"],
            "validation_eligible_samples": manifests[0]["validation_eligible_samples"],
            "nested_zips_included": False,
        })
        stage = "parent_zip"
        with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
            for name in [*files, "pair_manifest.json"]:
                archive.write(output / name, name)
        return output
    except Exception as exc:
        environment["comparison_authorized"] = False
        write_json(output / "pair_environment.json", environment)
        write_json(output / "status.json", {
            "status": "failed", "comparison_authorized": False, "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config", type=Path, default=Path("configs/experimento_lstm_rnn_ref_v1.json"))
    parser.add_argument("--candidate-config", type=Path, default=Path("configs/experimento_lstm32_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/rnn32_lstm_vs_rnn_f2_v1_seed42"))
    args = parser.parse_args()
    print(train_lstm_pair(args.reference_config, args.candidate_config, args.output, ROOT))


if __name__ == "__main__":
    main()
