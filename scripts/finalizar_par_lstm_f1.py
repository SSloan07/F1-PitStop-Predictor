"""Finalize a completed F2/F1 pair after a post-training gate interrupted publication."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f1_pitstop.rnn_data import read_json, sha256, write_json
from entrenar_comparacion_lstm_f1 import (
    _METRICS, _check_populations, _material_controls, _same_json,
    _selection_protocol, _validate_pair_configs, _verified_child_manifest,
    _child_summary,
)


def finalize_pair(output, reference_config, candidate_config, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    if not output.is_absolute():
        output = root / output
    if not output.is_dir():
        raise FileNotFoundError(f"Partial pair directory not found: {output}")
    if output.with_suffix(".zip").exists() or (output / "pair_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a published pair")
    status_path = output / "status.json"
    parent_status = read_json(status_path)
    if parent_status.get("status") != "failed" or parent_status.get("stage") != "population_and_hash_gate":
        raise ValueError("Recovery only supports a failed final population/hash gate")
    config_paths = [Path(reference_config), Path(candidate_config)]
    config_paths = [p if p.is_absolute() else root / p for p in config_paths]
    config_hashes = [sha256(p) for p in config_paths]
    configs = [read_json(p) for p in config_paths]
    _validate_pair_configs(*configs)
    for path, name, digest in zip(config_paths, ("reference.json", "candidate.json"), config_hashes):
        snap = output / "pair_configs" / name
        if not snap.is_file() or sha256(snap) != digest:
            raise ValueError(f"Run config snapshot does not match current config: {name}")
    children = ("lstm32_reference", "lstm32_f1")
    directories = [output / name for name in children]
    manifests = [
        _verified_child_manifest(directories[0], "f2", 2),
        _verified_child_manifest(directories[1], "f1", 1),
    ]
    effective_configs = [read_json(d / "config.json") for d in directories]
    _validate_pair_configs(*effective_configs)
    pair_environment_path = output / "pair_environment.json"
    environment = read_json(pair_environment_path)
    child_keys = ("python", "platform", "torch", "numpy", "pandas", "device", "gpu", "cuda")
    snapshot_keys = (*child_keys, "cuda_device_index")
    child_envs = [environment.get("children", {}).get(name, {}) for name in children]
    if not child_envs[0] or any(child_envs[0].get(k) != child_envs[1].get(k) for k in child_keys):
        raise ValueError("The completed child runs do not share the same environment")
    if child_envs[0].get("device") != "cuda" or "T4" not in child_envs[0].get("gpu", ""):
        raise ValueError("The pair did not use the required Colab T4 GPU")
    snapshots = environment.get("snapshots", {})
    baseline = snapshots.get("before", {})
    for name in ("before", "between", "after"):
        snap=snapshots.get(name)
        if not snap or any(snap.get(k) != baseline.get(k) for k in snapshot_keys):
            raise ValueError(f"Runtime environment changed at snapshot {name}")
    if any(baseline.get(k) != child_envs[0].get(k) for k in child_keys):
        raise ValueError("Child environment differs from the training snapshots")
    for declared, effective in zip(configs, effective_configs):
        expected = copy.deepcopy(declared)
        expected["selection"]["beta"] = _selection_protocol(declared)[1]
        expected["training"].update({"device_effective": child_envs[0]["device"], "device_override": None})
        if not _same_json(_material_controls(expected), _material_controls(effective)):
            raise ValueError("Completed child effective config differs from its declaration")
    specs = [read_json(d / "model_spec.json") for d in directories]
    if specs[0] != specs[1] or specs[0].get("recurrent_cell") != "lstm" or specs[0].get("hidden_size") != 32 or specs[0].get("dropout_rate") != 0.2:
        raise ValueError("Completed child architectures are not the declared common LSTM32/dropout 0.2")
    _check_populations(*directories, *manifests)
    summaries = [_child_summary(d, m) for d, m in zip(directories, manifests)]
    summary = {
        "pair_id": configs[0].get("pairing", {}).get("pair_id", output.name),
        "comparison_authorized": True, "selection_metric": "f1", "selection_beta": 1,
        "selection_metrics": {"reference": "f2", "candidate": "f1"},
        "seed": 42, "test_used": False,
        "changed_factor": {"name": "selection.primary_metric", "reference": "f2", "candidate": "f1",
                            "scope": "checkpoint_and_frozen_weight_threshold"},
        "reference": summaries[0], "candidate": summaries[1],
        "delta_candidate_minus_reference": {
            f"{population}_{threshold}": {
                metric: summaries[1][population][threshold][metric] - summaries[0][population][threshold][metric]
                for metric in _METRICS
            }
            for population in ("train", "validation")
            for threshold in ("fixed_threshold_0_5", "selected_threshold")
        },
        "recovery": {"recovered_completed_children": True, "failed_stage": "population_and_hash_gate",
                     "reason": "Resolved selection beta differs by objective; comparison gate now normalizes this derived field."},
        "note": "Validation selects checkpoints and thresholds; final test 2025 remains reserved.",
    }
    recovery_dir=output/"pair_recovery"
    recovery_dir.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), recovery_dir/"finalizar_par_lstm_f1.py")
    shutil.copyfile(ROOT/"scripts/entrenar_comparacion_lstm_f1.py", recovery_dir/"entrenar_comparacion_lstm_f1.py")
    write_json(output/"pair_summary.json",summary)
    environment["comparison_authorized"]=True
    environment["recovered_after_gate_failure"]=True
    write_json(pair_environment_path,environment)
    write_json(status_path,{"status":"complete","comparison_authorized":True,"recovered_after_gate_failure":True})
    files={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.suffix.lower()!=".zip" and p!=output/"pair_manifest.json"}
    run_source=output/"pair_sources"/"entrenar_comparacion_lstm_f1.py"
    if not run_source.is_file(): raise FileNotFoundError("Original pair runner snapshot missing")
    recovery_source=recovery_dir/"finalizar_par_lstm_f1.py"
    write_json(output/"pair_manifest.json",{
        "artifact_version":"lstm_f1_objective_pair_v1","pair_id":summary["pair_id"],
        "files_sha256":files,"comparison_authorized":True,
        "selection_metrics":{"reference":"f2","candidate":"f1"},"seed":42,"test_used":False,
        "children":list(children),"changed_factor":summary["changed_factor"],
        "config_sources_sha256":dict(zip(("reference","candidate"),config_hashes)),
        "runner_source":"pair_sources/entrenar_comparacion_lstm_f1.py","runner_source_sha256":sha256(run_source),
        "recovery_source":"pair_recovery/finalizar_par_lstm_f1.py","recovery_source_sha256":sha256(recovery_source),
        "preprocessed_manifest_sha256":manifests[0]["preprocessed_manifest_sha256"],
        "preprocessor_sha256":manifests[0]["preprocessor_sha256"],
        "train_eligible_samples":manifests[0]["train_eligible_samples"],
        "validation_eligible_samples":manifests[0]["validation_eligible_samples"],
        "nested_zips_included":False,"recovered_after_gate_failure":True,
    })
    with zipfile.ZipFile(output.with_suffix(".zip"),"x",zipfile.ZIP_DEFLATED) as archive:
        for name in [*files,"pair_manifest.json"]: archive.write(output/name,name)
    return output.with_suffix(".zip")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference-config",type=Path,default=Path("configs/experimento_lstm32_f1_ref_v1.json"))
    parser.add_argument("--candidate-config",type=Path,default=Path("configs/experimento_lstm32_f1_candidate_v1.json"))
    args=parser.parse_args()
    print(finalize_pair(args.output,args.reference_config,args.candidate_config,ROOT))

if __name__=="__main__": main()
