"""Create the verified Colab input ZIP for the paired causal LSTM experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_data import read_json
from entrenar_comparacion_lstm import _validate_pair_configs
FILES = [
    "requirements-colab-experiments.txt",
    "scripts/entrenar_rnn.py", "scripts/evaluar_rnn.py",
    "scripts/entrenar_comparacion_lstm.py",
    "src/f1_pitstop/__init__.py", "src/f1_pitstop/rnn_data.py",
    "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/lstm_model.py", "src/f1_pitstop/rnn_experiment.py",
    "src/f1_pitstop/experiment_metrics.py",
    "configs/experimento_lstm_rnn_ref_v1.json",
    "configs/experimento_lstm32_v1.json",
]
DATA = "EDA/dataset/muestras_preprocesadas_v1_3"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(output):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference = read_json(ROOT / FILES[-2])
    candidate = read_json(ROOT / FILES[-1])
    _validate_pair_configs(reference, candidate)
    if reference["training"]["device"] != "cuda" or candidate["training"]["device"] != "cuda":
        raise ValueError("Both configs must require CUDA explicitly")
    data_manifest = ROOT / DATA / "manifest.json"
    if sha256(data_manifest) != reference["preprocessed_manifest_sha256"]:
        raise ValueError("Preprocessed data manifest hash mismatch")
    names = FILES + [f"{DATA}/{name}" for name in (
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
        "artifact_version": "rnn_colab_lstm_inputs_v3",
        "files_sha256": hashes,
        "experiment_config": "configs/experimento_lstm32_v1.json",
        "companion_configs": ["configs/experimento_lstm_rnn_ref_v1.json"],
        "preprocessed_manifest_sha256": sha256(data_manifest),
        "test_arrays_included": False,
        "training_device_policy": "cuda_required_no_cpu_fallback",
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
    parser.add_argument("--output", type=Path, default=Path("runs/rnn_colab_lstm_v3.zip"))
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()
