"""Train a controlled LSTM32 input-package pair in one CPU session.

Both children are the best LSTM32 recipe (dropout 0.2, BCE pos_weight=2, F2
checkpoint/threshold selection). Only the preprocessed input package changes,
and the candidate package must declare itself derived from the reference one:

* ``lstm32_telemetria_f2_v1``: full v1_3 inputs -> telemetry-only projection.
* ``lstm32_estrategia_f2_v1``: telemetry-only -> telemetry plus strategy state.
"""

import argparse
import copy
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f1_pitstop.rnn_data import read_json, sha256, verify_preprocessed, write_json
from f1_pitstop.rnn_experiment import _selection_protocol, _validate_config, train_experiment
from entrenar_comparacion_pos_weight import (
    _METRICS, _capture_environment, _check_config_hashes, _check_environment,
    _child_summary, _material_controls, _same_json, _verified_child_manifest,
)

PAIRS = {
    "lstm32_telemetria_f2_v1": {
        "reference_package": "EDA/dataset/muestras_preprocesadas_v1_3",
        "children": ("lstm32_inputs_completos", "lstm32_solo_telemetria"),
    },
    "lstm32_estrategia_f2_v1": {
        "reference_package": "EDA/dataset/muestras_preprocesadas_v1_3_telemetria_v1",
        "children": ("lstm32_solo_telemetria", "lstm32_telemetria_estrategia"),
    },
}
_INPUT_KEYS = ("preprocessed_package", "preprocessed_manifest_sha256")


def _validate_pair_configs(reference, candidate, root=ROOT):
    experiment_id = reference.get("experiment_id")
    if experiment_id not in PAIRS or candidate.get("experiment_id") != experiment_id:
        raise ValueError("Both runs must declare the same supported input-pair experiment_id")
    for config in (reference, candidate):
        _validate_config(config)
        if _selection_protocol(config) != ("f2", 2):
            raise ValueError("Both runs must use the LSTM32 F2 protocol")
        model, training = config["model"], config["training"]
        if model.get("recurrent_cell") != "lstm" or model.get("hidden_size") != 32:
            raise ValueError("Both models must be causal LSTM32")
        if model.get("num_recurrent_layers", 1) != 1 or float(model.get("dropout_rate", 0.0)) != 0.2:
            raise ValueError("Both models must keep one LSTM layer and dropout 0.2")
        if training.get("loss") != "weighted_bce_with_logits" or training.get("pos_weight") != 2.0:
            raise ValueError("Both models must keep BCE pos_weight=2.0")
        if training.get("lr_scheduler") != {"name": "constant"}:
            raise ValueError("Both models must use a constant learning rate")
        if training.get("learning_rate") != 0.001 or training.get("optimizer") != "Adam":
            raise ValueError("Adam and learning rate 0.001 must remain fixed")
        if training.get("device") != "cpu":
            raise ValueError("This Colab package is CPU-only")
    if reference["preprocessed_package"] != PAIRS[experiment_id]["reference_package"]:
        raise ValueError(f"Reference must use {PAIRS[experiment_id]['reference_package']}")
    if candidate["preprocessed_package"] == reference["preprocessed_package"]:
        raise ValueError("Candidate must use a different input package")
    left, right = _material_controls(reference), _material_controls(candidate)
    for key in _INPUT_KEYS:
        left.pop(key), right.pop(key)
    if not _same_json(left, right):
        raise ValueError("Material controls differ beyond the input package")
    candidate_manifest = read_json(Path(root) / candidate["preprocessed_package"] / "manifest.json")
    derived = candidate_manifest.get("derived_from", {})
    if (derived.get("package") != reference["preprocessed_package"]
            or derived.get("manifest_sha256") != reference["preprocessed_manifest_sha256"]):
        raise ValueError("Candidate inputs must be a declared projection of the reference package")


def _check_populations(reference_dir, candidate_dir, reference, candidate):
    for key in ("train_eligible_samples", "validation_eligible_samples", "fit_split",
                "selection_split", "test_used", "parity_atol", "parity_rtol"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Child populations or frozen controls differ: {key}")
    if reference["preprocessed_manifest_sha256"] == candidate["preprocessed_manifest_sha256"]:
        raise ValueError("Children unexpectedly used the same input package")
    for name in ("rnn_model.py", "rnn_data.py", "rnn_experiment.py", "experiment_metrics.py", "lstm_model.py"):
        key = f"sources/{name}"
        if reference["files_sha256"].get(key) != candidate["files_sha256"].get(key):
            raise ValueError(f"Child source implementations differ: {name}")
    for manifest in (reference, candidate):
        if (manifest.get("model_source_path") != "lstm_model.py"
                or manifest["files_sha256"].get("sources/lstm_model.py") != manifest["model_source_sha256"]):
            raise ValueError("Both children must use the frozen LSTM source")
    first = pd.read_csv(reference_dir / "validation_predictions.csv.gz", dtype={"DriverNumber": str})
    second = pd.read_csv(candidate_dir / "validation_predictions.csv.gz", dtype={"DriverNumber": str})
    keys = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap", "target"]
    if not first[keys].equals(second[keys]):
        raise ValueError("Child validation populations/targets differ")


def _input_description(root, config):
    package = Path(root) / config["preprocessed_package"]
    _, state = verify_preprocessed(package, config["preprocessed_manifest_sha256"])
    schema = state["schema"]
    return {"package": config["preprocessed_package"],
            "numeric_features": schema["numeric_feature_names"],
            "dynamic_categorical_features": schema["categorical_feature_names"],
            "static_features": schema["static_feature_names"]}


def train_telemetry_pair(reference_config, candidate_config, output, root=ROOT):
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
    _validate_pair_configs(*configs, root=root)
    _check_config_hashes(config_paths, config_hashes)
    protocol = configs[0]["training"]
    children = PAIRS[configs[0]["experiment_id"]]["children"]
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
        shutil.copyfile(Path(__file__).with_name("entrenar_comparacion_pos_weight.py"),
                        sources / "entrenar_comparacion_pos_weight.py")
        config_sources = output / "pair_configs"
        config_sources.mkdir()
        for path, name, digest in zip(config_paths, ("reference.json", "candidate.json"), config_hashes):
            shutil.copyfile(path, config_sources / name)
            if sha256(config_sources / name) != digest:
                raise ValueError("Declared pair config changed before its snapshot")
        for name, config_path, snapshot_name in zip(children, config_paths, ("between", "after")):
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
        directories = [output / name for name in children]
        manifests = [_verified_child_manifest(directory, "f2", 2) for directory in directories]
        effective_configs = [read_json(directory / "config.json") for directory in directories]
        _validate_pair_configs(*effective_configs, root=root)
        for declared, effective in zip(configs, effective_configs):
            expected = copy.deepcopy(declared)
            expected["selection"]["beta"] = _selection_protocol(declared)[1]
            expected["training"].update({"device_effective": baseline["device"], "device_override": None})
            if not _same_json(_material_controls(expected), _material_controls(effective)):
                raise ValueError("Child effective config differs from the declared pair controls")
        specs = [read_json(directory / "model_spec.json") for directory in directories]
        architecture = ("hidden_size", "embedding_dim", "static_hidden_size", "head_hidden_size",
                        "window_length", "num_recurrent_layers", "dropout_rate", "recurrent_cell")
        if any(specs[0].get(key) != specs[1].get(key) for key in architecture):
            raise ValueError("Child architectures differ beyond input dimensions")
        _check_populations(*directories, *manifests)
        summaries = [_child_summary(directory, manifest) for directory, manifest in zip(directories, manifests)]
        inputs = [_input_description(root, config) for config in configs]
        summary = {
            "pair_id": configs[0]["pairing"]["pair_id"],
            "comparison_authorized": True, "selection_metric": "f2", "selection_beta": 2,
            "seed": protocol["seed"], "test_used": False,
            "changed_factor": {"name": "preprocessed_package", "reference": inputs[0], "candidate": inputs[1],
                               "removed_numeric": sorted(set(inputs[0]["numeric_features"]) - set(inputs[1]["numeric_features"])),
                               "added_numeric": [n for n in inputs[1]["numeric_features"] if n not in inputs[0]["numeric_features"]],
                               "removed_static": [n for n in inputs[0]["static_features"] if n not in inputs[1]["static_features"]]},
            "model_specs": {"reference": specs[0], "candidate": specs[1]},
            "reference": summaries[0], "candidate": summaries[1],
            "delta_candidate_minus_reference": {
                f"{population}_{threshold}": {
                    metric: summaries[1][population][threshold][metric] - summaries[0][population][threshold][metric]
                    for metric in _METRICS
                }
                for population in ("train", "validation")
                for threshold in ("fixed_threshold_0_5", "selected_threshold")
            },
            "note": "Validation selects checkpoints and thresholds; final test 2025 remains reserved.",
        }
        write_json(output / "pair_summary.json", summary)
        environment["comparison_authorized"] = True
        write_json(output / "pair_environment.json", environment)
        write_json(output / "status.json", {"status": "complete", "comparison_authorized": True})
        files = {
            path.relative_to(output).as_posix(): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.suffix.lower() != ".zip" and path != output / "pair_manifest.json"
        }
        write_json(output / "pair_manifest.json", {
            "artifact_version": "lstm_telemetry_pair_v1", "pair_id": summary["pair_id"],
            "files_sha256": files, "comparison_authorized": True, "selection_metric": "f2",
            "seed": protocol["seed"], "test_used": False, "children": list(children),
            "changed_factor": "preprocessed_package",
            "preprocessed_manifest_sha256": {
                "reference": manifests[0]["preprocessed_manifest_sha256"],
                "candidate": manifests[1]["preprocessed_manifest_sha256"]},
            "config_sources_sha256": dict(zip(("reference", "candidate"), config_hashes)),
            "runner_source": runner_source.relative_to(output).as_posix(),
            "runner_source_sha256": sha256(runner_source),
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
    parser.add_argument("--reference-config", type=Path,
                        default=Path("configs/experimento_lstm32_telemetria_ref_cpu_v1.json"))
    parser.add_argument("--candidate-config", type=Path,
                        default=Path("configs/experimento_lstm32_telemetria_candidate_cpu_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/lstm32_telemetria_f2_cpu_v1_seed42"))
    args = parser.parse_args()
    print(train_telemetry_pair(args.reference_config, args.candidate_config, args.output, ROOT))


if __name__ == "__main__":
    main()
