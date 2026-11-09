"""Build the verified Colab CPU package for pos_weight replications 43 and 44."""
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
    "scripts/entrenar_comparacion_pos_weight.py",
    "src/f1_pitstop/__init__.py", "src/f1_pitstop/rnn_data.py",
    "src/f1_pitstop/rnn_model.py", "src/f1_pitstop/lstm_model.py",
    "src/f1_pitstop/rnn_experiment.py", "src/f1_pitstop/experiment_metrics.py",
]
PAIRS = [
    ("configs/experimento_lstm32_f2_ref_pos_weight_cpu_seed43_v1.json", "configs/experimento_lstm32_f2_pos_weight2_cpu_seed43_v1.json"),
    ("configs/experimento_lstm32_f2_ref_pos_weight_cpu_seed44_v1.json", "configs/experimento_lstm32_f2_pos_weight2_cpu_seed44_v1.json"),
]
DATA = "EDA/dataset/muestras_preprocesadas_v1_3"

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def package(output):
    output=Path(output)
    if not output.is_absolute(): output=ROOT/output
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    all_configs=[]
    pairs_manifest=[]
    for reference_name,candidate_name in PAIRS:
        reference,candidate=[read_json(ROOT/name) for name in (reference_name,candidate_name)]
        _validate_pair_configs(reference,candidate)
        if reference["training"]["device"] != "cpu" or candidate["training"]["device"] != "cpu":
            raise ValueError("This package only supports CPU")
        if reference["training"]["seed"] != candidate["training"]["seed"]:
            raise ValueError("Paired configs must use the same seed")
        all_configs += [reference_name,candidate_name]
        pairs_manifest.append({"seed":reference["training"]["seed"],"pair_id":reference["pairing"]["pair_id"],"configs":[reference_name,candidate_name]})
    if [p["seed"] for p in pairs_manifest] != [43,44]: raise ValueError("Expected replication seeds 43 and 44")
    data_manifest=ROOT/DATA/"manifest.json"
    first=read_json(ROOT/PAIRS[0][0])
    if sha256(data_manifest) != first["preprocessed_manifest_sha256"]: raise ValueError("Preprocessed data manifest hash mismatch")
    names=FILES+all_configs+[f"{DATA}/{name}" for name in (
        "manifest.json","preprocessor.json","config.json",
        *[f"{split}_{suffix}" for split in ("train","validation") for suffix in ("inputs.npz","metadata.csv.gz","labels.csv.gz")],
    )]
    paths=[(name,ROOT/name) for name in names]
    missing=[name for name,path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing package inputs: {missing}")
    hashes={name:sha256(path) for name,path in paths}
    manifest={"artifact_version":"lstm_colab_pos_weight_replicates_cpu_inputs_v1","files_sha256":hashes,"pairs":pairs_manifest,"preprocessed_manifest_sha256":sha256(data_manifest),"test_arrays_included":False,"training_device_policy":"cpu_required_no_cuda_fallback"}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",zipfile.ZIP_DEFLATED) as z:
        for name,path in paths:z.write(path,name)
        z.writestr("colab_package_manifest.json",json.dumps(manifest,indent=2)+"\n")
    print(f"Package: {output}")
    print(f"SHA-256: {sha256(output)}")
    print(f"Files: {len(hashes)}; seed pairs: 43, 44; test arrays included: false")
    return output

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("runs/lstm_colab_pos_weight2_cpu_replicates_43_44_v1.zip"))
    args=parser.parse_args()
    package(args.output)
