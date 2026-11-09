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
        "scripts/evaluar_rnn.py", "scripts/entrenar_comparacion_capacidad.py",
        "requirements-experiments.txt"]


def _companion_configs(config, root, main_config):
    """Validate the explicit config allowlist before creating any ZIP output."""
    declared = config.get("companion_configs", [])
    if not isinstance(declared, list):
        raise ValueError("companion_configs must be a list of relative configs/*.json paths")
    configs_directory = root / "configs"
    resolved_configs_directory = configs_directory.resolve()
    if declared and not resolved_configs_directory.is_relative_to(root):
        raise ValueError("Companion configs directory points outside the project")
    companions, seen = [], {main_config}
    for name in declared:
        if not isinstance(name, str) or not name:
            raise ValueError("Every companion config must be a relative configs/*.json path")
        relative = Path(name)
        if (relative.is_absolute() or len(relative.parts) != 2
                or relative.parts[0] != "configs" or relative.suffix != ".json"
                or any(part in {".", ".."} for part in name.split("/"))):
            raise ValueError("Every companion config must be a relative configs/*.json path")
        path = root / relative
        resolved = path.resolve()
        if (not resolved.is_relative_to(resolved_configs_directory)
                or not resolved.is_relative_to(root) or resolved.suffix != ".json"):
            raise ValueError("Companion config symlink leaves configs/*.json")
        if resolved in seen:
            raise ValueError("Companion configs must be unique and different from the main config")
        if not path.is_file():
            raise ValueError(f"Companion config must be an existing file: {name}")
        if not isinstance(read_json(path), dict):
            raise ValueError(f"Companion config must contain a JSON object: {name}")
        seen.add(resolved)
        companions.append(path)
    return companions


def package_experiment(config_path, output, root=ROOT):
    root, output, config_path = Path(root).resolve(), Path(output), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing Colab ZIP")
    config = read_json(config_path)
    companions = _companion_configs(config, root, config_path)
    data_dir = root / config["preprocessed_package"]
    manifest, _ = verify_preprocessed(data_dir, config["preprocessed_manifest_sha256"])
    relative_config = config_path.relative_to(root)
    files = [root / name for name in CODE] + [config_path, *companions, data_dir / "manifest.json"]
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
            "companion_configs": [path.relative_to(root).as_posix() for path in companions],
            "companion_configs_sha256": {
                path.relative_to(root).as_posix(): entries[path.relative_to(root).as_posix()]
                for path in companions},
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
