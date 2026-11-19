"""Create a verified Colab CPU input ZIP for a three-seed LSTM32 strategy-input experiment (round 1 or round 2)."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f1_pitstop.rnn_data import read_json, sha256
from empaquetar_telemetria import DATA_FILES, FILES as TELEMETRY_FILES
from entrenar_comparacion_telemetria import _validate_pair_configs

SEEDS = (42, 43, 44)
FILES = [
    *TELEMETRY_FILES,
    "scripts/preparar_entradas_estrategia.py", "configs/entradas_estrategia_v1.json",
    "src/f1_pitstop/strategy_features.py",
]
EXPERIMENTS = {
    "r1": {"prefix": "experimento_lstm32_estrategia", "artifact": "lstm_colab_strategy_cpu_inputs_v1",
           "extra_files": [], "output": "runs/lstm_colab_estrategia_cpu_v1.zip"},
    "r2": {"prefix": "experimento_lstm32_estrategia_r2", "artifact": "lstm_colab_strategy_r2_cpu_inputs_v1",
           "extra_files": ["src/f1_pitstop/strategy_features_r2.py", "configs/entradas_estrategia_r2_v1.json"],
           "output": "runs/lstm_colab_estrategia_r2_cpu_v1.zip"},
}


def pair_configs(seed, experiment="r1"):
    prefix = EXPERIMENTS[experiment]["prefix"]
    return (f"configs/{prefix}_ref_cpu_seed{seed}_v1.json", f"configs/{prefix}_candidate_cpu_seed{seed}_v1.json")


def package(output, seeds=SEEDS, experiment="r1"):
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    names, packages = list(dict.fromkeys(FILES + EXPERIMENTS[experiment]["extra_files"])), {}
    for seed in seeds:
        configs = [read_json(ROOT / name) for name in pair_configs(seed, experiment)]
        _validate_pair_configs(*configs, root=ROOT)
        if any(config["training"]["seed"] != seed for config in configs):
            raise ValueError(f"Config seed differs from {seed}")
        names += pair_configs(seed, experiment)
        for config in configs:
            packages[config["preprocessed_package"]] = config["preprocessed_manifest_sha256"]
    for data, digest in packages.items():
        if sha256(ROOT / data / "manifest.json") != digest:
            raise ValueError(f"Preprocessed data manifest hash mismatch: {data}")
        names += [f"{data}/{name}" for name in (*DATA_FILES, "report.json")]
    paths = [(name, ROOT / name) for name in names]
    missing = [name for name, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing package inputs: {missing}")
    manifest = {
        "artifact_version": EXPERIMENTS[experiment]["artifact"],
        "files_sha256": {name: sha256(path) for name, path in paths},
        "seeds": list(seeds),
        "pair_configs": {str(seed): list(pair_configs(seed, experiment)) for seed in seeds},
        "preprocessed_manifest_sha256": packages,
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
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="r1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package(args.output or Path(EXPERIMENTS[args.experiment]["output"]), experiment=args.experiment)


if __name__ == "__main__":
    main()
