"""Create a verified Colab input ZIP for a paired LSTM32 positive-class weighting experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_data import read_json
from entrenar_comparacion_pos_weight import _validate_pair_configs
FILES = [
    "requirements-colab-experiments.txt",
    "scripts/entrenar_rnn.py", "scripts/evaluar_rnn.py",
    "scripts/entrenar_comparacion_pos_weight.py",
    "src/f1_pitstop/__init__.py", "src/f1_pitstop/rnn_data.py",
    "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/lstm_model.py", "src/f1_pitstop/rnn_experiment.py",
    "src/f1_pitstop/experiment_metrics.py",
]
DEFAULT_REFERENCE_CONFIG = "configs/experimento_lstm32_f2_ref_pos_weight_cpu_v1.json"
DEFAULT_CANDIDATE_CONFIG = "configs/experimento_lstm32_f2_pos_weight2_cpu_v1.json"
DATA = "EDA/dataset/muestras_preprocesadas_v1_3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output, reference_config=DEFAULT_REFERENCE_CONFIG, candidate_config=DEFAULT_CANDIDATE_CONFIG):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config_names = [str(reference_config), str(candidate_config)]
    config_paths = [(ROOT / name).resolve() for name in config_names]
    if any(not path.is_relative_to(ROOT.resolve()) for path in config_paths):
        raise ValueError("Pair configurations must be inside the repository")
    reference, candidate = [read_json(path) for path in config_paths]
    _validate_pair_configs(reference, candidate)
    device = reference["training"]["device"]
    if device not in {"cpu", "cuda"} or candidate["training"]["device"] != device:
        raise ValueError("Both configs must require the same explicit CPU or CUDA device")
    names = FILES + config_names
    data_manifest = ROOT / DATA / "manifest.json"
    if sha256(data_manifest) != reference["preprocessed_manifest_sha256"]:
        raise ValueError("Preprocessed data manifest hash mismatch")
    names += [f"{DATA}/{name}" for name in (
        "manifest.json", "preprocessor.json", "config.json",
        *[f"{split}_{suffix}" for split in ("train", "validation")
          for suffix in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")],
    )]
    paths = [(name, ROOT / name) for name in names]
    if any(not path.is_file() for _, path in paths):
        missing = [name for name, path in paths if not path.is_file()]
        raise FileNotFoundError(f"Missing package inputs: {missing}")
    hashes = {name: sha256(path) for name, path in paths}
    manifest = {
        "artifact_version": "lstm_colab_pos_weight_cpu_inputs_v1" if device == "cuda" else "lstm_colab_pos_weight_cpu_inputs_v1",
        "files_sha256": hashes,
        "experiment_config": config_names[1],
        "companion_configs": [config_names[0]],
        "preprocessed_manifest_sha256": sha256(data_manifest),
        "test_arrays_included": False,
        "training_device_policy": "cuda_required_no_cpu_fallback" if device == "cuda" else "cpu_required_no_cuda",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in paths:
            archive.write(path, name)
        archive.writestr("colab_package_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Package: {output}")
    print(f"SHA-256: {sha256(output)}")
    print(f"Files: {len(hashes)}; test arrays included: false")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/lstm_colab_pos_weight2_cpu_v1.zip"))
    parser.add_argument("--reference-config", type=Path, default=Path(DEFAULT_REFERENCE_CONFIG))
    parser.add_argument("--candidate-config", type=Path, default=Path(DEFAULT_CANDIDATE_CONFIG))
    args = parser.parse_args()
    package(args.output, args.reference_config, args.candidate_config)


if __name__ == "__main__":
    main()
