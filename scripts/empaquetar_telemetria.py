"""Create a verified Colab CPU input ZIP for the LSTM32 full-inputs versus telemetry-only pair."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f1_pitstop.rnn_data import read_json, sha256
from entrenar_comparacion_telemetria import _validate_pair_configs

FILES = [
    "requirements-colab-experiments.txt",
    "scripts/entrenar_rnn.py", "scripts/evaluar_rnn.py",
    "scripts/entrenar_comparacion_pos_weight.py", "scripts/entrenar_comparacion_telemetria.py",
    "scripts/preparar_entradas_telemetria.py", "configs/entradas_telemetria_v1.json",
    "src/f1_pitstop/__init__.py", "src/f1_pitstop/rnn_data.py",
    "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/lstm_model.py", "src/f1_pitstop/rnn_experiment.py",
    "src/f1_pitstop/experiment_metrics.py",
]
DEFAULT_REFERENCE_CONFIG = "configs/experimento_lstm32_telemetria_ref_cpu_v1.json"
DEFAULT_CANDIDATE_CONFIG = "configs/experimento_lstm32_telemetria_candidate_cpu_v1.json"
DATA_FILES = ("manifest.json", "preprocessor.json", "config.json",
              *[f"{split}_{suffix}" for split in ("train", "validation")
                for suffix in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")])


def package(output, reference_config=DEFAULT_REFERENCE_CONFIG, candidate_config=DEFAULT_CANDIDATE_CONFIG):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config_names = [Path(reference_config).as_posix(), Path(candidate_config).as_posix()]
    config_paths = [(ROOT / name).resolve() for name in config_names]
    if any(not path.is_relative_to(ROOT.resolve()) for path in config_paths):
        raise ValueError("Pair configurations must be inside the repository")
    configs = [read_json(path) for path in config_paths]
    _validate_pair_configs(*configs, root=ROOT)
    names = FILES + config_names
    for config in configs:
        data = config["preprocessed_package"]
        if sha256(ROOT / data / "manifest.json") != config["preprocessed_manifest_sha256"]:
            raise ValueError(f"Preprocessed data manifest hash mismatch: {data}")
        names += [f"{data}/{name}" for name in DATA_FILES]
    paths = [(name, ROOT / name) for name in names]
    missing = [name for name, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")
    manifest = {
        "artifact_version": "lstm_colab_telemetry_cpu_inputs_v1",
        "files_sha256": {name: sha256(path) for name, path in paths},
        "experiment_config": config_names[1],
        "companion_configs": [config_names[0]],
        "preprocessed_manifest_sha256": {
            "reference": configs[0]["preprocessed_manifest_sha256"],
            "candidate": configs[1]["preprocessed_manifest_sha256"],
        },
        "test_arrays_included": False,
        "training_device_policy": "cpu_required_no_cuda",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths:
            archive.write(path, name)
        archive.writestr("colab_package_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Package: {output}")
    print(f"SHA-256: {sha256(output)}")
    print(f"Files: {len(paths)}; test arrays included: false")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/lstm_colab_telemetria_cpu_v1.zip"))
    parser.add_argument("--reference-config", type=Path, default=Path(DEFAULT_REFERENCE_CONFIG))
    parser.add_argument("--candidate-config", type=Path, default=Path(DEFAULT_CANDIDATE_CONFIG))
    args = parser.parse_args()
    package(args.output, args.reference_config, args.candidate_config)


if __name__ == "__main__":
    main()
