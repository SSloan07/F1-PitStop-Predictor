"""Package only shared experiment code and frozen train/validation inputs for Colab."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_data import read_json, sha256, verify_preprocessed

CODE = ["src/f1_pitstop/__init__.py", "src/f1_pitstop/rnn_data.py",
        "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/rnn_experiment.py",
        "src/f1_pitstop/experiment_metrics.py", "scripts/entrenar_rnn.py",
        "scripts/evaluar_rnn.py", "requirements-experiments.txt"]


def package_experiment(config_path, output, root=ROOT):
    root, output, config_path = Path(root).resolve(), Path(output), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing Colab ZIP")
    config = read_json(config_path)
    data_dir = root / config["preprocessed_package"]
    manifest, _ = verify_preprocessed(data_dir, config["preprocessed_manifest_sha256"])
    relative_config = config_path.relative_to(root)
    files = [root / name for name in CODE] + [config_path, data_dir / "manifest.json"]
    files += [data_dir / name for name in ["preprocessor.json", "config.json"] + [
        f"{split}_{suffix}" for split in ("train", "validation")
        for suffix in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")]]
    entries = {}
    for path in files:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("An input file points outside the project")
        entries[path.relative_to(root).as_posix()] = sha256(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
        archive.writestr("colab_package_manifest.json", json.dumps({
            "artifact_version": "rnn_colab_inputs_v1", "files_sha256": entries,
            "experiment_config": relative_config.as_posix(),
            "preprocessed_manifest_sha256": sha256(data_dir / "manifest.json"),
            "test_arrays_included": False,
            "note": "The upstream manifest has hashes/metadata of other artifacts; only the listed files are packaged",
        }, indent=2) + "\n")
    return {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size,
            "file_count": len(files), "test_arrays_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experimento_rnn_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(package_experiment(config, args.output), indent=2))


if __name__ == "__main__":
    main()
