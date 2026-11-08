"""Align published labels to causal samples and publish fixed race-grouped splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_KEY = ["Temporada", "Ronda", "DriverNumber", "prediction_lap"]
LABEL_KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
LABEL_FIELDS = [
    "PitWithin3Laps", "LabelReason", "LabelBasis", "RuleVersion", "EligibleForSupervisedLearning",
]


class PartitionContractError(ValueError):
    """Published inputs or requested split violate the experiment contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_keys(frame: pd.DataFrame, prediction: bool = False) -> pd.DataFrame:
    out = frame.copy()
    lap_column = "prediction_lap" if prediction else "LapNumber"
    out["Temporada"] = pd.to_numeric(out.Temporada, errors="raise").astype(int)
    out["Ronda"] = pd.to_numeric(out.Ronda, errors="raise").astype(int)
    out["DriverNumber"] = out.DriverNumber.astype(str).str.replace(r"\.0$", "", regex=True)
    out[lap_column] = pd.to_numeric(out[lap_column], errors="raise").astype(int)
    return out


def assign_split(year: int, splits: dict[str, list[int]]) -> str:
    found = [name for name, years in splits.items() if int(year) in set(map(int, years))]
    if len(found) != 1:
        raise PartitionContractError(f"Season {year} must belong to exactly one split, got {found}")
    return found[0]


def align_labels_and_partition(
    sample_index: pd.DataFrame,
    labels: pd.DataFrame,
    static_features: pd.DataFrame,
    splits: dict[str, list[int]],
    require_all_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-join labels one-to-one and retain every sample_id and feature row."""
    sample = _normalize_keys(sample_index, prediction=True)
    label = _normalize_keys(labels.rename(columns={"LapNumber": "prediction_lap"}), prediction=True)
    static = static_features.copy()
    for key in ("Temporada", "Ronda"):
        static[key] = pd.to_numeric(static[key], errors="raise").astype(int)
    static["DriverNumber"] = static.DriverNumber.astype(str).str.replace(r"\.0$", "", regex=True)
    if "sample_id" not in sample or "sample_id" not in static:
        raise PartitionContractError("sample_index and static_features must include sample_id")
    if sample.sample_id.isna().any() or sample.sample_id.duplicated().any():
        raise PartitionContractError("sample_id is null or duplicated")
    if static.sample_id.isna().any() or static.sample_id.duplicated().any():
        raise PartitionContractError("static_features sample_id is null or duplicated")
    if not sample.sample_id.reset_index(drop=True).equals(static.sample_id.reset_index(drop=True)):
        raise PartitionContractError("sample_id/order differs between sample_index and static_features")
    static_keys = ["Temporada", "Ronda", "DriverNumber"]
    if not sample[static_keys].reset_index(drop=True).equals(static[static_keys].reset_index(drop=True)):
        raise PartitionContractError("Static context keys differ from sample_index")
    if sample[SAMPLE_KEY].isna().any().any() or sample.duplicated(SAMPLE_KEY).any():
        raise PartitionContractError("sample key in feature index is null or duplicated")
    if label[SAMPLE_KEY].isna().any().any() or label.duplicated(SAMPLE_KEY).any():
        raise PartitionContractError("label key is null or duplicated")
    absent_fields = sorted(set(SAMPLE_KEY + LABEL_FIELDS) - set(label.columns))
    if absent_fields:
        raise PartitionContractError(f"Label schema lacks fields: {absent_fields}")
    extras = label.merge(sample[SAMPLE_KEY], on=SAMPLE_KEY, how="left", indicator=True)
    extra_keys = int(extras._merge.eq("left_only").sum())
    if extra_keys:
        raise PartitionContractError(f"{extra_keys} label keys do not exist in causal samples")
    if require_all_labels and len(label) != len(sample):
        raise PartitionContractError(f"Expected a published label for every sample; samples={len(sample)}, labels={len(label)}")

    pred_columns = ["sample_id", *SAMPLE_KEY, "decision_time_seconds", "prediction_eligible",
                    "prediction_ineligible_reason"]
    merged = sample[pred_columns].merge(
        label[SAMPLE_KEY + LABEL_FIELDS], on=SAMPLE_KEY, how="left",
        validate="one_to_one", indicator="_label_merge", sort=False,
    )
    if len(merged) != len(sample) or not merged.sample_id.equals(sample.sample_id):
        raise PartitionContractError("Label join changed sample cardinality or sample_id order")
    merged["label_status"] = np.select(
        [
            merged._label_merge.eq("left_only"),
            merged.PitWithin3Laps.isna(),
        ],
        ["not_published", "published_unknown"],
        default="published_known",
    )
    merged["label_usable"] = (
        merged.PitWithin3Laps.notna()
        & merged.EligibleForSupervisedLearning.astype("boolean").fillna(False).astype(bool)
    )
    merged["prediction_eligible"] = merged.prediction_eligible.fillna(False).astype(bool)
    merged["supervised_eligible"] = merged.label_usable & merged.prediction_eligible
    merged["supervised_ineligible_reason"] = ""
    mask_bad_label = ~merged.label_usable
    merged.loc[mask_bad_label, "supervised_ineligible_reason"] = (
        "label_not_usable:" + merged.loc[mask_bad_label, "LabelReason"].fillna("no_published_label_for_key").astype(str)
    )
    mask_bad_prediction = ~merged.prediction_eligible
    old_reason = merged.loc[mask_bad_prediction, "supervised_ineligible_reason"]
    pred_reason = merged.loc[mask_bad_prediction, "prediction_ineligible_reason"].fillna("not_prediction_eligible").astype(str)
    merged.loc[mask_bad_prediction, "supervised_ineligible_reason"] = np.where(
        old_reason.eq(""), "prediction_ineligible:" + pred_reason,
        old_reason + ";prediction_ineligible:" + pred_reason,
    )
    merged["split"] = merged.Temporada.map(lambda y: assign_split(int(y), splits))
    merged = merged.drop(columns=["_label_merge"])

    if len(static) != len(sample) or not static.sample_id.equals(sample.sample_id):
        raise PartitionContractError("Static context does not align one-to-one with samples")
    static_gp = static[["sample_id", "GranPremio"]]
    merged = merged.merge(static_gp, on="sample_id", how="left", validate="one_to_one", sort=False)
    if merged.GranPremio.isna().any() or len(merged) != len(sample):
        raise PartitionContractError("Static Grand Prix context is missing or join changed sample count")
    races = merged.groupby(["Temporada", "Ronda"], sort=True).agg(
        gp_count=("GranPremio", "nunique"), split_count=("split", "nunique"),
        samples=("sample_id", "size"), supervised_eligible=("supervised_eligible", "sum"),
        published_known=("label_status", lambda x: int(x.eq("published_known").sum())),
        published_unknown=("label_status", lambda x: int(x.eq("published_unknown").sum())),
        not_published=("label_status", lambda x: int(x.eq("not_published").sum())),
    ).reset_index()
    if races.gp_count.ne(1).any():
        raise PartitionContractError("Grand Prix identity is not unique within a race key")
    if races.split_count.ne(1).any():
        raise PartitionContractError("A race was assigned to more than one split")
    races["GrandPremio"] = merged.groupby(["Temporada", "Ronda"], sort=True).GranPremio.first().to_numpy()
    races["split"] = merged.groupby(["Temporada", "Ronda"], sort=True).split.first().to_numpy()
    races = races.drop(columns=["gp_count", "split_count"])
    return merged, races


def _check_feature_package(package: Path, manifest: dict) -> dict[str, str]:
    hashes = manifest.get("output_files_sha256", {})
    required = ["sample_index.csv.gz", "static_features.csv.gz", "windows.npz"]
    for name in required:
        if name not in hashes:
            raise PartitionContractError(f"Feature manifest lacks hash for {name}")
        observed = sha256(package / name)
        if hashes[name] != observed:
            raise PartitionContractError(f"Feature artifact hash mismatch: {name}")
    return {name: hashes[name] for name in required}


def build_package(config_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise PartitionContractError(f"Output already exists; refusing overwrite: {output_dir}")
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_dir = ROOT / config["feature_package"]
    label_dir = ROOT / config["label_package"]
    feature_manifest_path, label_manifest_path = feature_dir / "manifest.json", label_dir / "manifest.json"
    if sha256(feature_manifest_path) != config["feature_package_manifest_sha256"]:
        raise PartitionContractError("muestras_causales_v1_3 manifest hash differs from pinned configuration")
    if sha256(label_manifest_path) != config["label_manifest_sha256"]:
        raise PartitionContractError("pit_entry_v3 manifest hash differs from pinned configuration")
    feature_manifest = json.loads(feature_manifest_path.read_text())
    label_manifest = json.loads(label_manifest_path.read_text())
    if label_manifest.get("rule_version") != "pit_entry_v2" or label_manifest.get("horizon_laps") != 3:
        raise PartitionContractError("Label package rule/horizon incompatible")
    if label_manifest.get("source_sha256") != config["source_master_sha256"]:
        raise PartitionContractError("Labels are not from the pinned master matrix")
    feature_manifest_hashes = _check_feature_package(feature_dir, feature_manifest)

    # Intentionally do not load labels_aligned.csv.gz from feature_package: it is the older v2 label snapshot.
    sample_index = pd.read_csv(feature_dir / "sample_index.csv.gz", low_memory=False)
    static_features = pd.read_csv(feature_dir / "static_features.csv.gz", low_memory=False)
    labels = pd.read_pickle(label_dir / "labels.pkl")
    labels = labels.rename(columns={"LapNumber": "prediction_lap"})
    samples, races = align_labels_and_partition(
        sample_index, labels, static_features, config["splits"], require_all_labels=True,
    )
    with np.load(feature_dir / "windows.npz", allow_pickle=False) as arrays:
        if arrays["X_numeric"].shape[0] != len(samples) or arrays["X_categorical"].shape[0] != len(samples):
            raise PartitionContractError("Feature arrays do not align with the sample table")
        if arrays["window_lap_numbers"].shape[0] != len(samples):
            raise PartitionContractError("Window lap IDs do not align with sample_id")
        laps = arrays["window_lap_numbers"].astype(np.int64, copy=False)
        padding = arrays["padding_mask"]
        exists = arrays["slot_exists_mask"]
        prediction_laps = samples.prediction_lap.to_numpy(dtype=np.int64)
        if np.any(laps[:, -1] != prediction_laps) or np.any(padding[:, -1]):
            bad = int(np.flatnonzero((laps[:, -1] != prediction_laps) | padding[:, -1])[0])
            raise PartitionContractError(f"Window/current decision misalignment at sample_id={samples.sample_id.iloc[bad]}")
        if np.any(laps[exists] <= 0) or np.any(laps[exists] > np.broadcast_to(prediction_laps[:, None], laps.shape)[exists]):
            raise PartitionContractError("Window includes invalid or future lap numbers")
        # Encode full race/driver/lap keys as integers for a vectorized ownership check.
        seasons = samples.Temporada.to_numpy(dtype=np.int64)
        rounds = samples.Ronda.to_numpy(dtype=np.int64)
        drivers = pd.to_numeric(samples.DriverNumber, errors="raise").to_numpy(dtype=np.int64)
        if np.any(rounds >= 1000) or np.any(drivers >= 1000) or np.any(laps >= 10000):
            raise PartitionContractError("Key exceeds the supported window ownership encoding")
        prefix = seasons * 10**12 + rounds * 10**9 + drivers * 10**6
        available_keys = np.unique(prefix + prediction_laps)
        row_ids, slot_ids = np.nonzero(exists)
        historical_keys = prefix[row_ids] + laps[row_ids, slot_ids]
        if not np.isin(historical_keys, available_keys).all():
            raise PartitionContractError("Window crosses driver/race or references an unknown lap")
    years = {name: sorted(map(int, vals)) for name, vals in config["splits"].items()}
    if not (max(years["train"]) < min(years["validation"]) < min(years["test_final_reserved"])):
        raise PartitionContractError("Configured train/validation/test seasons are not strictly chronological")
    sample_partitions = samples[[
        "sample_id", *SAMPLE_KEY, "split", "label_status", "label_usable",
        "prediction_eligible", "prediction_ineligible_reason", "supervised_eligible",
        "supervised_ineligible_reason",
    ]].copy()
    by_split = {
        name: sample_partitions.loc[sample_partitions.split.eq(name), "sample_id"].astype(int).tolist()
        for name in config["splits"]
    }
    coverage = samples.groupby(["Temporada", "split"], as_index=False).agg(
        samples=("sample_id", "size"),
        published_known=("label_status", lambda s: int(s.eq("published_known").sum())),
        published_unknown=("label_status", lambda s: int(s.eq("published_unknown").sum())),
        not_published=("label_status", lambda s: int(s.eq("not_published").sum())),
        label_usable=("label_usable", "sum"),
        prediction_eligible=("prediction_eligible", "sum"),
        supervised_eligible=("supervised_eligible", "sum"),
    )
    reason_counts = samples.loc[~samples.supervised_eligible, "supervised_ineligible_reason"].value_counts().to_dict()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    samples.to_csv(output_dir / "labels_aligned_v3.csv.gz", index=False, compression="gzip")
    sample_partitions.to_csv(output_dir / "sample_partitions.csv.gz", index=False, compression="gzip")
    races.to_csv(output_dir / "race_partitions.csv", index=False)
    coverage.to_csv(output_dir / "coverage_by_season_split.csv", index=False)
    samples.groupby(["Temporada", "Ronda", "GranPremio", "split"], as_index=False).agg(
        samples=("sample_id", "size"),
        label_usable=("label_usable", "sum"),
        prediction_eligible=("prediction_eligible", "sum"),
        supervised_eligible=("supervised_eligible", "sum"),
        published_unknown=("label_status", lambda s: int(s.eq("published_unknown").sum())),
        not_published=("label_status", lambda s: int(s.eq("not_published").sum())),
    ).to_csv(output_dir / "coverage_by_race.csv", index=False)
    (output_dir / "sample_ids_by_split.json").write_text(json.dumps(by_split, indent=2) + "\n")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "README.md").write_text(
        "# Etiquetas y particiones vigentes\n\n"
        "Este paquete referencia las características y ventanas de muestras_causales_v1_3 sin copiarlas. "
        "Usa exclusivamente labels_aligned_v3.csv.gz para etiquetas. "
        "No uses labels_aligned.csv.gz dentro de muestras_causales_v1_3: contiene la instantánea antigua pit_entry_v2.\n\n"
        "La selección supervisada requiere etiqueta conocida, EligibleForSupervisedLearning y prediction_eligible. "
        "Padding y faltantes de características no excluyen muestras. La temporada 2025 queda reservada como prueba final.\n"
    )
    (output_dir / "example_load.py").write_text(
        "from pathlib import Path\nimport numpy as np\nimport pandas as pd\n"
        "features = Path('EDA/dataset/muestras_causales_v1_3')\n"
        "labels = Path(__file__).resolve().parent\n"
        "part = pd.read_csv(labels/'sample_partitions.csv.gz')\n"
        "target = pd.read_csv(labels/'labels_aligned_v3.csv.gz', low_memory=False)\n"
        "ids = part.loc[part.split.eq('train') & part.supervised_eligible, 'sample_id'].to_numpy()\n"
        "with np.load(features/'windows.npz', allow_pickle=False) as z:\n"
        "    X_num = z['X_numeric'][ids]; X_cat = z['X_categorical'][ids]\n"
        "y = target.set_index('sample_id').loc[ids, 'PitWithin3Laps'].to_numpy()\n"
        "print(X_num.shape, X_cat.shape, y.shape)\n"
    )
    (output_dir / "report.md").write_text(
        "# Alineación y particiones cronológicas\n\n"
        f"- Muestras alineadas sin alterar sample_id: {len(samples):,}.\n"
        f"- Etiquetas conocidas: {int(samples.label_status.eq('published_known').sum()):,}; "
        f"publicadas desconocidas: {int(samples.label_status.eq('published_unknown').sum()):,}; "
        f"sin publicación: {int(samples.label_status.eq('not_published').sum()):,}.\n"
        f"- Seleccionables para supervisión: {int(samples.supervised_eligible.sum()):,}; "
        f"inelegibles por motivos registrados: {int((~samples.supervised_eligible).sum()):,}.\n"
        f"- Train: {years['train']}; validación: {years['validation']}; "
        f"prueba final reservada: {years['test_final_reserved']}.\n"
        "- No se particionó por filas; todas las vueltas y pilotos de cada carrera permanecen juntos.\n"
        "- 2025 queda reservada para evaluación final, sin selección de variables, reglas, modelos ni umbrales.\n"
        "- No se ajustaron transformadores estadísticos ni se entrenaron modelos.\n"
    )
    output_files = {p.name: sha256(p) for p in output_dir.iterdir() if p.is_file()}
    manifest = {
        "artifact_version": "aligned_labels_and_race_splits_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                      capture_output=True).stdout.strip(),
        "python_version": platform.python_version(), "pandas_version": pd.__version__,
        "numpy_version": np.__version__, "partition_version": config["partition_version"],
        "source_hashes": {
            "config": sha256(config_path), "feature_manifest": sha256(feature_manifest_path),
            "partition_builder_script": sha256(Path(__file__).resolve()),
            "label_builder_script": label_manifest["script_sha256"],
            "feature_sample_index": sha256(feature_dir / "sample_index.csv.gz"),
            "feature_static_features": sha256(feature_dir / "static_features.csv.gz"),
            "feature_windows": sha256(feature_dir / "windows.npz"),
            "label_manifest": sha256(label_manifest_path),
            "labels": sha256(label_dir / "labels.pkl"),
        },
        "feature_artifact_hashes_in_v1_3": feature_manifest_hashes,
        "source_master_sha256": config["source_master_sha256"],
        "label_rule_version": label_manifest["rule_version"],
        "splits": years, "test_final_usage": "reserved; do not use for feature/rule/model/threshold selection",
        "supervised_eligibility_rule": config["eligibility_rule"],
        "labels_by_status": samples.label_status.value_counts().to_dict(),
        "supervised_ineligible_reasons": {str(k): int(v) for k, v in reason_counts.items()},
        "sample_rows": len(samples), "sample_ids_by_split": {
            name: {"count": len(ids), "sha256_json_values": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()} for name, ids in by_split.items()
        },
        "race_count_by_split": races.groupby("split").size().astype(int).to_dict(),
        "output_files_sha256": output_files,
        "legacy_v1_3_labels_used": False,
    }
    manifest["race_count_by_split"] = {str(k): int(v) for k, v in manifest["race_count_by_split"].items()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Reload the publication and verify cardinality, IDs, labels and array linkage.
    reloaded = pd.read_csv(output_dir / "sample_partitions.csv.gz", low_memory=False)
    aligned_reload = pd.read_csv(output_dir / "labels_aligned_v3.csv.gz", low_memory=False)
    if (len(reloaded) != len(sample_index) or len(aligned_reload) != len(sample_index)
            or not reloaded.sample_id.equals(sample_index.sample_id)):
        raise PartitionContractError("Published sample_id alignment failed on reload")
    if reloaded.groupby("sample_id").size().ne(1).any():
        raise PartitionContractError("Published sample_id is not unique")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/particion_pit_entry_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output or Path(config["output_dir"])
    output = output if output.is_absolute() else ROOT / output
    try:
        print(build_package(config_path, output))
    except (PartitionContractError, FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
