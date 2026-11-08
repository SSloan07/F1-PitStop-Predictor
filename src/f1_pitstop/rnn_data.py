"""Verified train/validation readers; final test is deliberately out of scope."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MASKS = ["padding_mask", "slot_exists_mask", "slot_row_missing_mask",
         "temporal_unavailable_mask", "feature_missing_mask", "mask_reason_codes"]
MODEL_ARRAYS = ["X_numeric", "X_dynamic_categorical_ids", "X_static_categorical_ids",
                "mask_reason_codes", "padding_mask", "slot_exists_mask", "slot_row_missing_mask"]
KEY = ["sample_id", "Temporada", "Ronda", "DriverNumber", "prediction_lap"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def verify_preprocessed(directory, expected_manifest=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if expected_manifest and sha256(manifest_path) != expected_manifest:
        raise ValueError("Preprocessed manifest differs from the pinned experiment input")
    manifest = read_json(manifest_path)
    if manifest.get("artifact_version") != "preprocessed_causal_inputs_v1":
        raise ValueError("Unsupported preprocessed artifact version")
    required = ["preprocessor.json"] + [f"{s}_{kind}" for s in ("train", "validation")
               for kind in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")]
    for name in required:
        expected = manifest.get("output_files_sha256", {}).get(name)
        if not expected or sha256(directory / name) != expected:
            raise ValueError(f"Preprocessed artifact hash mismatch: {name}")
    state = read_json(directory / "preprocessor.json")
    # Match the serialization used by preparar_preprocesador.py for the
    # published fit_state_sha256 (including its default spaces/ASCII escaping).
    state_hash = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    if manifest.get("fit_state_sha256") != state_hash or state.get("fit_split") != "train":
        raise ValueError("Preprocessor state is not the verified frozen train state")
    return manifest, state


@dataclass
class SplitData:
    arrays: dict
    metadata: pd.DataFrame
    labels: pd.DataFrame
    positions: np.ndarray
    targets: np.ndarray


def load_split(directory, split, state):
    if split not in {"train", "validation"}:
        raise ValueError("Only train/validation are enabled; final test remains reserved")
    directory = Path(directory)
    with np.load(directory / f"{split}_inputs.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = pd.read_csv(directory / f"{split}_metadata.csv.gz", dtype={"DriverNumber": str})
    labels = pd.read_csv(directory / f"{split}_labels.csv.gz", dtype={"DriverNumber": str},
                         keep_default_na=False)
    n, width, numeric_count = arrays["X_numeric"].shape
    schema = state["schema"]
    categorical_count = len(schema["categorical_feature_names"])
    if (len(metadata) != n or len(labels) != n or width != schema["window_length"]
            or numeric_count != len(schema["numeric_feature_names"])):
        raise ValueError("Split dimensions differ from the frozen preprocessing schema")
    ids = arrays.get("sample_id")
    if ids is None or not np.array_equal(ids, metadata.sample_id) or not np.array_equal(ids, labels.sample_id):
        raise ValueError("sample_id order mismatch between inputs, metadata and labels")
    if metadata.sample_id.duplicated().any() or metadata.sample_id.isna().any():
        raise ValueError("sample_id must be unique and complete")
    for key in KEY:
        if key in labels and not np.array_equal(metadata[key].astype(str), labels[key].astype(str)):
            raise ValueError(f"Metadata/label key mismatch: {key}")
    expected_years = {2022, 2023} if split == "train" else {2024}
    if not set(metadata.Temporada.astype(int)).issubset(expected_years):
        raise ValueError("Unexpected season in split; final test cannot enter this experiment")
    if "split" in metadata and not metadata.split.eq(split).all():
        raise ValueError("Split membership mismatch")
    expected_shapes = {"X_dynamic_categorical_ids": (n, width, categorical_count),
                       "X_static_categorical_ids": (n, len(schema["static_feature_names"]))}
    expected_shapes.update({k: (n, width) for k in MASKS[:3]})
    expected_shapes.update({k: (n, width, numeric_count + categorical_count) for k in MASKS[3:]})
    for name, shape in expected_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f"Input shape mismatch: {name}")
    reason = arrays["mask_reason_codes"]
    if np.any(reason > 5):
        raise ValueError("Unknown mask reason")
    for name in MASKS[:-1]:
        if arrays[name].dtype != np.bool_:
            raise ValueError(f"Mask must be boolean: {name}")
    if not np.array_equal(reason == 3, arrays["temporal_unavailable_mask"]):
        raise ValueError("Temporal mask differs from reason codes")
    if not np.array_equal(np.isin(reason, [2, 4, 5]), arrays["feature_missing_mask"]):
        raise ValueError("Missing mask differs from reason codes")
    if not np.array_equal(reason == 1, np.broadcast_to(arrays["padding_mask"][..., None], reason.shape)):
        raise ValueError("Padding mask differs from reason codes")
    if not np.array_equal(reason == 2, np.broadcast_to(arrays["slot_row_missing_mask"][..., None], reason.shape)):
        raise ValueError("Missing-row mask differs from reason codes")
    if not np.isfinite(arrays["X_numeric"]).all():
        raise ValueError("Non-finite preprocessed numeric input")
    if np.any(arrays["X_numeric"][reason[..., :numeric_count] != 0] != 0):
        raise ValueError("Masked numeric inputs must retain technical zero")
    eligible_column = labels["supervised_eligible"]
    if not eligible_column.isin([True, False]).all():
        raise ValueError("Supervised eligibility is not boolean")
    eligible = eligible_column.to_numpy(dtype=bool)
    if "supervised_eligible" in metadata and not np.array_equal(eligible, metadata.supervised_eligible):
        raise ValueError("Eligibility differs between labels and metadata")
    positions = np.flatnonzero(eligible)
    if not positions.size:
        raise ValueError("No supervised-eligible samples in split")
    targets = labels.iloc[positions].PitWithin3Laps.astype(str).str.lower()
    if not targets.isin(["true", "false"]).all():
        raise ValueError("An eligible sample has no valid binary target")
    y = targets.eq("true").to_numpy(dtype=np.float32)
    return SplitData(arrays, metadata, labels, positions, y)
