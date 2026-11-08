"""Fit train-only preprocessing and materialize train/validation inputs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.f1_pitstop.preprocessing import (  # noqa: E402
    InputPreprocessor,
    PreprocessingContractError,
    RESERVED_CATEGORY_IDS,
    category_value,
    validate_input_arrays,
)


SAMPLE_KEY = ["Temporada", "Ronda", "DriverNumber", "prediction_lap"]
MASK_NAMES = [
    "padding_mask", "slot_exists_mask", "slot_row_missing_mask",
    "temporal_unavailable_mask", "feature_missing_mask", "mask_reason_codes",
]
LABEL_OUTPUT_COLUMNS = [
    "sample_id", "PitWithin3Laps", "label_status", "LabelReason", "LabelBasis", "RuleVersion",
    "EligibleForSupervisedLearning", "label_usable", "prediction_eligible",
    "prediction_ineligible_reason", "supervised_eligible", "supervised_ineligible_reason",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_sample_ids(sample_ids: np.ndarray) -> str:
    little_endian = np.asarray(sample_ids, dtype="<i8")
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest_files(directory: Path, manifest: dict[str, Any], required: list[str]) -> dict[str, str]:
    files = manifest.get("output_files_sha256", {})
    hashes = {}
    for name in required:
        if name not in files:
            raise PreprocessingContractError(f"Manifest lacks expected artifact hash: {name}")
        observed = sha256(directory / name)
        if observed != files[name]:
            raise PreprocessingContractError(f"Input artifact hash mismatch: {directory / name}")
        hashes[name] = observed
    return hashes


def normalize_key_types(frame: pd.DataFrame, lap_column: str = "prediction_lap") -> pd.DataFrame:
    out = frame.copy()
    for name in ("Temporada", "Ronda", lap_column):
        out[name] = pd.to_numeric(out[name], errors="raise").astype("int64")
    out["DriverNumber"] = pd.to_numeric(out.DriverNumber, errors="raise").astype("int64")
    return out


def validate_split_metadata(partitions: pd.DataFrame, split_config: dict[str, list[int]]) -> dict[str, int]:
    expected = {name: set(map(int, years)) for name, years in split_config.items()}
    if set(expected) != {"train", "validation", "test_final_reserved"}:
        raise PreprocessingContractError("Split configuration must define train, validation, test_final_reserved")
    seen: dict[tuple[int, int], str] = {}
    for split, years in expected.items():
        overlap = set().union(*(x for name, x in expected.items() if name != split)) & years
        if overlap:
            raise PreprocessingContractError(f"Configured seasons overlap across splits: {sorted(overlap)}")
    observed = partitions.groupby("split").Temporada.apply(lambda x: set(map(int, x.unique()))).to_dict()
    if observed != expected:
        raise PreprocessingContractError(f"Partition seasons incompatible with configuration: observed={observed}, expected={expected}")
    races = partitions.groupby(["Temporada", "Ronda"], sort=False).split.nunique()
    if races.ne(1).any():
        raise PreprocessingContractError("A race is assigned to multiple splits")
    for row in partitions[["Temporada", "Ronda", "split"]].drop_duplicates().itertuples(index=False):
        race_key = (int(row.Temporada), int(row.Ronda))
        prior = seen.setdefault(race_key, row.split)
        if prior != row.split:
            raise PreprocessingContractError(f"Race {race_key} crosses split boundary")
    counts = partitions.groupby("split").sample_id.size().astype(int).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def load_train_validation_labels(path: Path, partitions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Read target values only for train/validation; inspect test rows for key/split metadata only."""
    partition_by_id = partitions.set_index("sample_id", verify_integrity=True)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen: set[int] = set()
    total = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", *SAMPLE_KEY, "split", "label_status", "PitWithin3Laps"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PreprocessingContractError("Aligned labels schema is incompatible")
        for record in reader:
            total += 1
            sid = int(record["sample_id"])
            if sid in seen or sid not in partition_by_id.index:
                raise PreprocessingContractError(f"Duplicate or unexpected label sample_id={sid}")
            seen.add(sid)
            expected = partition_by_id.loc[sid]
            if record["split"] != expected.split:
                raise PreprocessingContractError(f"Label split mismatch for sample_id={sid}")
            key = (int(record["Temporada"]), int(record["Ronda"]), int(record["DriverNumber"]), int(record["prediction_lap"]))
            expected_key = tuple(int(expected[name]) for name in SAMPLE_KEY)
            if key != expected_key:
                raise PreprocessingContractError(f"Label key mismatch for sample_id={sid}")
            split = record["split"]
            if split == "test_final_reserved":
                # Deliberately do not parse target, reason, or eligibility values from final test rows.
                continue
            if split not in rows:
                raise PreprocessingContractError(f"Unsupported split for sample_id={sid}: {split}")
            target_text = record["PitWithin3Laps"]
            target = True if target_text.casefold() == "true" else False if target_text.casefold() == "false" else pd.NA
            rows[split].append({
                "sample_id": sid, "PitWithin3Laps": target,
                "label_status": record["label_status"], "LabelReason": record.get("LabelReason", ""),
                "LabelBasis": record.get("LabelBasis", ""), "RuleVersion": record.get("RuleVersion", ""),
                "EligibleForSupervisedLearning": record.get("EligibleForSupervisedLearning", "False").casefold() == "true",
                "label_usable": record.get("label_usable", "False").casefold() == "true",
                "prediction_eligible": record.get("prediction_eligible", "False").casefold() == "true",
                "prediction_ineligible_reason": record.get("prediction_ineligible_reason", ""),
                "supervised_eligible": record.get("supervised_eligible", "False").casefold() == "true",
                "supervised_ineligible_reason": record.get("supervised_ineligible_reason", ""),
            })
    if total != len(partitions) or seen != set(map(int, partitions.sample_id)):
        raise PreprocessingContractError("Aligned labels do not cover partition sample_ids one-to-one")
    return {name: pd.DataFrame(values, columns=LABEL_OUTPUT_COLUMNS) for name, values in rows.items()}


def _select_masks(arrays: Any, positions: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(arrays[name][positions]) for name in MASK_NAMES}


def build_preprocessing(config_path: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise PreprocessingContractError(f"Output exists; refusing to overwrite: {output_dir}")
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("preprocessing_version") != "train_only_preprocessor_v1":
        raise PreprocessingContractError("Unsupported preprocessing configuration version")

    feature_dir = ROOT / config["feature_package"]
    split_dir = ROOT / config["partition_package"]
    label_dir = ROOT / config["label_package"]
    feature_manifest_path = feature_dir / "manifest.json"
    split_manifest_path = split_dir / "manifest.json"
    label_manifest_path = label_dir / "manifest.json"
    for path, expected, description in [
        (feature_manifest_path, config["feature_manifest_sha256"], "feature package manifest"),
        (split_manifest_path, config["partition_manifest_sha256"], "partition package manifest"),
        (label_manifest_path, config["label_manifest_sha256"], "label package manifest"),
    ]:
        if sha256(path) != expected:
            raise PreprocessingContractError(f"Pinned {description} hash mismatch")
    feature_manifest, split_manifest, label_manifest = map(read_json, [feature_manifest_path, split_manifest_path, label_manifest_path])
    if split_manifest.get("legacy_v1_3_labels_used") is not False:
        raise PreprocessingContractError("Partition package does not certify v3 labels separately from old v1_3 labels")
    if split_manifest.get("source_hashes", {}).get("label_manifest") != sha256(label_manifest_path):
        raise PreprocessingContractError("Partition manifest points to another labels publication")
    if label_manifest.get("rule_version") != "pit_entry_v2" or label_manifest.get("horizon_laps") != 3:
        raise PreprocessingContractError("Current label package has incompatible rule or horizon")
    label_files = label_manifest.get("output_files_sha256", {})
    if not label_files.get("labels.pkl") or sha256(label_dir / "labels.pkl") != label_files["labels.pkl"]:
        raise PreprocessingContractError("Current labels artifact hash differs from its manifest")
    if split_manifest.get("source_hashes", {}).get("labels") != label_files["labels.pkl"]:
        raise PreprocessingContractError("Partition package was aligned against another labels artifact")
    if label_manifest.get("source_sha256") != split_manifest.get("source_master_sha256"):
        raise PreprocessingContractError("Label package and split package refer to different master matrix")
    if split_manifest.get("splits") != config.get("splits"):
        raise PreprocessingContractError("Partition manifest split definition differs from preprocessing config")

    feature_files = verify_manifest_files(feature_dir, feature_manifest, [
        "sample_index.csv.gz", "static_features.csv.gz", "windows.npz", "feature_registry.json", "mask_codes.json",
    ])
    split_files = verify_manifest_files(split_dir, split_manifest, [
        "sample_partitions.csv.gz", "labels_aligned_v3.csv.gz",
    ])
    if feature_files["windows.npz"] != split_manifest["feature_artifact_hashes_in_v1_3"].get("windows.npz"):
        raise PreprocessingContractError("Partition package references a different v1_3 window array")

    sample_index = normalize_key_types(pd.read_csv(feature_dir / "sample_index.csv.gz", low_memory=False))
    static_frame = pd.read_csv(feature_dir / "static_features.csv.gz", low_memory=False)
    partitions = normalize_key_types(pd.read_csv(split_dir / "sample_partitions.csv.gz", low_memory=False))
    aligned_metadata = pd.read_csv(split_dir / "labels_aligned_v3.csv.gz", usecols=[
        "sample_id", *SAMPLE_KEY, "split", "label_status", "label_usable", "prediction_eligible", "supervised_eligible",
    ], low_memory=False)
    static_names = read_json(feature_dir / "feature_registry.json")["static"]
    numeric_names = read_json(feature_dir / "feature_registry.json")["numeric_sequence"]
    categorical_names = read_json(feature_dir / "feature_registry.json")["categorical_sequence"]
    mask_codes_raw = read_json(feature_dir / "mask_codes.json")
    mask_codes = {int(code): reason for code, reason in mask_codes_raw.items()}

    if sample_index.sample_id.isna().any() or sample_index.sample_id.duplicated().any():
        raise PreprocessingContractError("Feature sample_id is null or duplicated")
    sample_ids = sample_index.sample_id.to_numpy(dtype=np.int64)
    if not np.array_equal(sample_ids, np.arange(len(sample_ids), dtype=np.int64)):
        raise PreprocessingContractError("sample_id order is not the NPZ row order 0..N-1")
    if not static_frame.sample_id.equals(sample_index.sample_id):
        raise PreprocessingContractError("Static context sample_id/order differs from sample_index")
    static_frame = normalize_key_types(static_frame.assign(prediction_lap=sample_index.prediction_lap))
    if not static_frame[["sample_id", *SAMPLE_KEY]].reset_index(drop=True).equals(
        sample_index[["sample_id", *SAMPLE_KEY]].reset_index(drop=True)
    ):
        raise PreprocessingContractError("Static feature keys do not align with sample_index")
    if partitions.sample_id.isna().any() or partitions.sample_id.duplicated().any():
        raise PreprocessingContractError("Partition sample_id is null or duplicated")
    if not partitions.sample_id.equals(sample_index.sample_id):
        raise PreprocessingContractError("Partition sample_id/order differs from v1_3")
    if aligned_metadata.sample_id.duplicated().any() or not aligned_metadata.sample_id.equals(sample_index.sample_id):
        raise PreprocessingContractError("Aligned labels sample_id/order differs from v1_3")
    for name in SAMPLE_KEY:
        if not partitions[name].equals(sample_index[name]) or not aligned_metadata[name].equals(sample_index[name]):
            raise PreprocessingContractError(f"Key mismatch across feature/partition/label files: {name}")
    for name in ("split", "label_status", "label_usable", "prediction_eligible", "supervised_eligible"):
        if not partitions[name].equals(aligned_metadata[name]):
            raise PreprocessingContractError(f"Partition/label metadata mismatch: {name}")
    if not np.array_equal(
        partitions.supervised_eligible.to_numpy(dtype=bool),
        partitions.label_usable.to_numpy(dtype=bool) & partitions.prediction_eligible.to_numpy(dtype=bool),
    ):
        raise PreprocessingContractError("supervised_eligible does not match its documented definition")

    split_counts = validate_split_metadata(partitions, config["splits"])
    train_ids = partitions.loc[partitions.split.eq("train"), "sample_id"].to_numpy(dtype=np.int64)
    validation_ids = partitions.loc[partitions.split.eq("validation"), "sample_id"].to_numpy(dtype=np.int64)
    test_metadata = partitions.loc[partitions.split.eq("test_final_reserved"), ["sample_id", *SAMPLE_KEY, "split"]]
    if test_metadata.empty or not test_metadata.Temporada.eq(2025).all():
        raise PreprocessingContractError("Final test metadata is not exclusively 2025")
    fit_ids = partitions.loc[partitions.split.eq("train") & partitions.supervised_eligible, "sample_id"].to_numpy(dtype=np.int64)
    if fit_ids.size == 0:
        raise PreprocessingContractError("No supervised-eligible training samples")
    fit_local_rows = partitions.loc[partitions.split.eq("train"), "supervised_eligible"].to_numpy(dtype=bool)
    fit_ids_from_train = train_ids[fit_local_rows]
    if not np.array_equal(fit_ids, fit_ids_from_train):
        raise PreprocessingContractError("Fit sample population is not aligned to train order")

    labels = load_train_validation_labels(split_dir / "labels_aligned_v3.csv.gz", partitions)
    for split, split_ids in (("train", train_ids), ("validation", validation_ids)):
        if not np.array_equal(labels[split].sample_id.to_numpy(dtype=np.int64), split_ids):
            raise PreprocessingContractError(f"Target table sample_id/order mismatch in {split}")
        if not np.array_equal(labels[split].supervised_eligible.to_numpy(dtype=bool),
                              partitions.loc[partitions.split.eq(split), "supervised_eligible"].to_numpy(dtype=bool)):
            raise PreprocessingContractError(f"Target eligibility mismatch in {split}")

    with np.load(feature_dir / "windows.npz", allow_pickle=False) as archive:
        required_arrays = {"X_numeric", "X_categorical", "window_lap_numbers", "lineage_ids", *MASK_NAMES,
                           "numeric_feature_names", "categorical_feature_names"}
        if not required_arrays.issubset(archive.files):
            raise PreprocessingContractError(f"Feature arrays missing: {sorted(required_arrays - set(archive.files))}")
        if archive["numeric_feature_names"].tolist() != numeric_names:
            raise PreprocessingContractError("Numeric feature order differs from feature registry")
        if archive["categorical_feature_names"].tolist() != categorical_names:
            raise PreprocessingContractError("Dynamic categorical order differs from feature registry")
        if archive["X_numeric"].shape[0] != len(sample_index) or archive["X_categorical"].shape[0] != len(sample_index):
            raise PreprocessingContractError("Feature arrays do not align with sample_index")
        if archive["X_numeric"].shape[1] != 8 or config.get("window_length") != 8:
            raise PreprocessingContractError("W=8 is required by the current published samples")
        schema = {
            "window_length": int(config["window_length"]),
            "numeric_feature_names": numeric_names,
            "categorical_feature_names": categorical_names,
            "static_feature_names": static_names,
            "mask_codes": {str(code): name for code, name in mask_codes.items()},
            "mask_feature_order": [*numeric_names, *categorical_names],
            "feature_contract_version": feature_manifest["contract_version"],
        }
        numeric_train = archive["X_numeric"][train_ids]
        categorical_train = archive["X_categorical"][train_ids]
        numeric_validation = archive["X_numeric"][validation_ids]
        categorical_validation = archive["X_categorical"][validation_ids]
        train_masks = _select_masks(archive, train_ids)
        validation_masks = _select_masks(archive, validation_ids)
        train_window_laps = np.array(archive["window_lap_numbers"][train_ids], copy=True)
        validation_window_laps = np.array(archive["window_lap_numbers"][validation_ids], copy=True)
        train_lineage = np.array(archive["lineage_ids"][train_ids], copy=True)
        validation_lineage = np.array(archive["lineage_ids"][validation_ids], copy=True)
        train_static = static_frame.loc[static_frame.sample_id.isin(train_ids), static_names].to_numpy(dtype=str)
        validation_static = static_frame.loc[static_frame.sample_id.isin(validation_ids), static_names].to_numpy(dtype=str)
        if not np.array_equal(static_frame.loc[static_frame.sample_id.isin(train_ids), "sample_id"].to_numpy(), train_ids):
            raise PreprocessingContractError("Train static context order mismatch")
        if not np.array_equal(static_frame.loc[static_frame.sample_id.isin(validation_ids), "sample_id"].to_numpy(), validation_ids):
            raise PreprocessingContractError("Validation static context order mismatch")
        validate_input_arrays(numeric_train, categorical_train, train_static, train_masks, schema)
        validate_input_arrays(numeric_validation, categorical_validation, validation_static, validation_masks, schema)
        fit_local_sha = hash_sample_ids(fit_ids_from_train)
        preprocessor = InputPreprocessor().fit(
            numeric_train, categorical_train, train_static, train_masks, schema,
            fit_local_rows, fit_local_sha,
        )
        frozen_state_sha = hashlib.sha256(json.dumps(preprocessor.to_dict(), sort_keys=True).encode()).hexdigest()
        train_result = preprocessor.transform(numeric_train, categorical_train, train_static, train_masks, schema)
        validation_result = preprocessor.transform(numeric_validation, categorical_validation, validation_static,
                                                  validation_masks, schema)

    fit_state_sha_after = hashlib.sha256(json.dumps(preprocessor.to_dict(), sort_keys=True).encode()).hexdigest()
    if frozen_state_sha != fit_state_sha_after:
        raise PreprocessingContractError("Preprocessor state changed during transformation")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    preprocessor_path = output_dir / "preprocessor.json"
    preprocessor.save(preprocessor_path)
    feature_names = {
        "numeric": np.asarray(numeric_names, dtype="U64"),
        "dynamic_categorical": np.asarray(categorical_names, dtype="U64"),
        "static_categorical": np.asarray(static_names, dtype="U64"),
    }
    for split, ids, result, window_laps, lineage in (
        ("train", train_ids, train_result, train_window_laps, train_lineage),
        ("validation", validation_ids, validation_result, validation_window_laps, validation_lineage),
    ):
        values = {
            "sample_id": ids.astype(np.int64),
            "X_numeric": result["X_numeric"],
            "X_dynamic_categorical_ids": result["X_dynamic_categorical_ids"],
            "X_static_categorical_ids": result["X_static_categorical_ids"],
            "window_lap_numbers": window_laps,
            "lineage_ids": lineage,
            **feature_names,
        }
        values.update({name: result[name] for name in MASK_NAMES})
        np.savez_compressed(output_dir / f"{split}_inputs.npz", **values)
        metadata = partitions.loc[partitions.split.eq(split)].copy()
        metadata.to_csv(output_dir / f"{split}_metadata.csv.gz", index=False, compression="gzip")
        labels[split].to_csv(output_dir / f"{split}_labels.csv.gz", index=False, compression="gzip")

    train_cov: dict[str, Any] = {}
    validation_cov: dict[str, Any] = {}
    fit_masks = {name: value[fit_local_rows] for name, value in train_masks.items()}
    for idx, feature in enumerate(numeric_names):
        fit_count = int(preprocessor.to_dict()["numeric"]["observed_cell_count"][feature])
        train_cov[feature] = {
            "fit_observed_cells": fit_count,
            "fit_window_positions": int(len(fit_ids) * config["window_length"]),
            "fit_padding_positions": int(fit_masks["padding_mask"].sum()),
            "fit_row_missing_positions": int(fit_masks["slot_row_missing_mask"].sum()),
            "fit_temporally_unavailable_cells": int(fit_masks["temporal_unavailable_mask"][:, :, idx].sum()),
            "fit_feature_missing_cells": int(fit_masks["feature_missing_mask"][:, :, idx].sum()),
            "physical_zero_fit_cells": int(preprocessor.to_dict()["numeric"]["observed_physical_zero_count"][feature]),
            "constant": feature in preprocessor.to_dict()["numeric"]["constant_features_scale_one"],
        }
        validation_cov[feature] = {
            "observed_cells": int((validation_masks["mask_reason_codes"][:, :, idx] == 0).sum()),
            "temporally_unavailable_cells": int(validation_masks["temporal_unavailable_mask"][:, :, idx].sum()),
            "missing_cells": int(validation_masks["feature_missing_mask"][:, :, idx].sum()),
        }
    cat_idx = len(numeric_names)
    val_oov_dynamic = dict(zip(categorical_names, map(int, validation_result["_oov_dynamic_counts"])))
    val_unknown_dynamic = dict(zip(categorical_names, map(int, validation_result["_unknown_dynamic_counts"])))
    val_oov_static = dict(zip(static_names, map(int, validation_result["_oov_static_counts"])))
    for i, feature in enumerate(categorical_names):
        code_col = cat_idx + i
        validation_cov[feature] = {
            "out_of_vocabulary_cells": val_oov_dynamic[feature],
            "source_unknown_cells": val_unknown_dynamic[feature],
            "absent_cells": int((validation_result["X_dynamic_categorical_ids"][:, :, i]
                                 == RESERVED_CATEGORY_IDS["absent"]).sum()),
            "observed_cells": int((validation_masks["mask_reason_codes"][:, :, code_col] == 0).sum()),
        }
    for feature in static_names:
        validation_cov[f"static:{feature}"] = {"out_of_vocabulary_samples": val_oov_static[feature]}

    state = preprocessor.to_dict()
    report = {
        "fit_population": "train AND supervised_eligible only",
        "fit_sample_count": state["fit_sample_count"],
        "train_samples_transformed": len(train_ids),
        "validation_samples_transformed": len(validation_ids),
        "test_samples_transformed": 0,
        "test_metadata_rows_verified_only": split_counts["test_final_reserved"],
        "test_split": "reserved; no feature values transformed or summarized",
        "dynamic_numeric_coverage_and_fit": train_cov,
        "validation_numeric_coverage": {k: v for k, v in validation_cov.items() if k in numeric_names},
        "validation_categorical_coverage": {k: v for k, v in validation_cov.items() if k not in numeric_names},
        "vocabulary_cardinalities": {
            branch: {name: len(entries) for name, entries in vocab.items()}
            for branch, vocab in state["vocabularies"].items()
        },
        "reserved_category_ids": RESERVED_CATEGORY_IDS,
        "overlapping_window_weighting": (
            "Each observed dynamic cell occurrence in an eligible training window contributes once. "
            "A source lap/value repeated in multiple overlapping windows is therefore weighted by its number of occurrences."
        ),
        "fit_state_sha256_before_transform": frozen_state_sha,
        "fit_state_sha256_after_transform": fit_state_sha_after,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "example_load.py").write_text(
        "\"\"\"Load fitted inputs and reproduce the validation transform from raw v1_3 arrays.\"\"\"\n"
        "from pathlib import Path\nimport sys\nimport numpy as np\nimport pandas as pd\n"
        "root = Path(__file__).resolve().parent\nrepo = root.parents[2]\nsys.path.insert(0, str(repo))\n"
        "from src.f1_pitstop.preprocessing import InputPreprocessor\n"
        "config = __import__('json').loads((root/'config.json').read_text(encoding='utf-8'))\n"
        "feature_dir = repo/config['feature_package']\n"
        "preprocessor = InputPreprocessor.load(root/'preprocessor.json')\n"
        "validation = np.load(root/'validation_inputs.npz', allow_pickle=False)\n"
        "metadata = pd.read_csv(root/'validation_metadata.csv.gz')\n"
        "assert np.array_equal(validation['sample_id'], metadata.sample_id.to_numpy())\n"
        "sample_ids = validation['sample_id'].astype(np.int64, copy=False)\n"
        "registry = __import__('json').loads((feature_dir/'feature_registry.json').read_text(encoding='utf-8'))\n"
        "static_frame = pd.read_csv(feature_dir/'static_features.csv.gz')\n"
        "static_names = registry['static']\n"
        "static = static_frame.loc[static_frame.sample_id.isin(sample_ids), static_names].to_numpy(dtype=str)\n"
        "assert np.array_equal(static_frame.loc[static_frame.sample_id.isin(sample_ids), 'sample_id'].to_numpy(), sample_ids)\n"
        "mask_names = ['padding_mask','slot_exists_mask','slot_row_missing_mask','temporal_unavailable_mask','feature_missing_mask','mask_reason_codes']\n"
        "with np.load(feature_dir/'windows.npz', allow_pickle=False) as raw:\n"
        "    masks = {name: raw[name][sample_ids] for name in mask_names}\n"
        "    transformed = preprocessor.transform(raw['X_numeric'][sample_ids], raw['X_categorical'][sample_ids], static, masks, preprocessor.to_dict()['schema'])\n"
        "for name in ['X_numeric','X_dynamic_categorical_ids','X_static_categorical_ids', *mask_names]:\n"
        "    assert np.array_equal(transformed[name], validation[name]), name\n"
        "print('validación reproducida:', transformed['X_numeric'].shape, transformed['X_dynamic_categorical_ids'].shape, transformed['X_static_categorical_ids'].shape)\n"
    , encoding="utf-8")

    # Reload the fitted state and both materialized splits; verify array/mask/metadata alignment.
    reloaded = InputPreprocessor.load(preprocessor_path)
    for split, expected_ids in (("train", train_ids), ("validation", validation_ids)):
        with np.load(output_dir / f"{split}_inputs.npz", allow_pickle=False) as transformed:
            if not np.array_equal(transformed["sample_id"], expected_ids):
                raise PreprocessingContractError(f"Reloaded {split} sample_id alignment failed")
            if transformed["X_numeric"].shape != (len(expected_ids), 8, len(numeric_names)):
                raise PreprocessingContractError(f"Reloaded {split} numeric dimensions failed")
            if transformed["X_dynamic_categorical_ids"].shape != (len(expected_ids), 8, len(categorical_names)):
                raise PreprocessingContractError(f"Reloaded {split} categorical dimensions failed")
            if transformed["X_static_categorical_ids"].shape != (len(expected_ids), len(static_names)):
                raise PreprocessingContractError(f"Reloaded {split} static dimensions failed")
            for name in MASK_NAMES:
                if not np.array_equal(transformed[name], (train_masks if split == "train" else validation_masks)[name]):
                    raise PreprocessingContractError(f"Reloaded {split} mask changed: {name}")
    if reloaded.to_dict() != preprocessor.to_dict():
        raise PreprocessingContractError("Saved/loaded preprocessor state differs")

    output_files = {file.name: sha256(file) for file in output_dir.iterdir() if file.is_file()}
    manifest = {
        "artifact_version": "preprocessed_causal_inputs_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       capture_output=True).stdout.strip(),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "fit_state_sha256": frozen_state_sha,
        "fit_split": "train", "fit_eligibility": "supervised_eligible",
        "split_membership": split_counts,
        "source_hashes": {
            "config": sha256(config_path), "preprocessing_script": sha256(Path(__file__).resolve()),
            "preprocessing_module": sha256(Path(__file__).resolve().parents[1] / "src/f1_pitstop/preprocessing.py"),
            "contract_document": sha256(ROOT / "docs/specs/10_contrato_preprocesamiento.md"),
            "feature_manifest": sha256(feature_manifest_path), "partition_manifest": sha256(split_manifest_path),
            "labels_manifest": sha256(label_manifest_path), "feature_arrays": sha256(feature_dir / "windows.npz"),
            "sample_index": sha256(feature_dir / "sample_index.csv.gz"),
            "static_features": sha256(feature_dir / "static_features.csv.gz"),
            "partition_table": sha256(split_dir / "sample_partitions.csv.gz"),
            "aligned_labels": sha256(split_dir / "labels_aligned_v3.csv.gz"),
        },
        "fit_sample_ids_sha256_little_endian_int64": state["fit_sample_ids_sha256"],
        "feature_order": schema,
        "test_features_transformed": False,
        "test_label_values_used": False,
        "output_files_sha256": output_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocesador_entradas_v1.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = args.output or Path(config["output_dir"])
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        print(build_preprocessing(config_path, output_dir))
    except (PreprocessingContractError, FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
