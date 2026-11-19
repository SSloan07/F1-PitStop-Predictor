"""Derive a telemetry-focused input package from the frozen v1_3 preprocessed inputs.

The derivation only selects columns. It never refits statistics, never reads
the 2025 test split and keeps sample order, labels, metadata and masks. Because
standardization and vocabularies are fitted independently per feature, the
selected statistics are exactly those a train-only refit on the subset would
produce. Static context (Driver, Team, GranPremio) is replaced by one constant
placeholder column whose every ID is the reserved "absent" ID 1, so the frozen
LSTM architecture keeps its static branch but receives no static information.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.rnn_data import load_split, read_json, sha256, verify_preprocessed, write_json

SPLITS = ("train", "validation")
FEATURE_AXIS_MASKS = ("temporal_unavailable_mask", "feature_missing_mask", "mask_reason_codes")
ABSENT_ID = 1
STATIC_PLACEHOLDER = "SinContextoEstatico"


def _state_hash(state):
    # Same serialization that rnn_data.verify_preprocessed checks.
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _validate_selection(config, schema):
    numeric, dynamic = schema["numeric_feature_names"], schema["categorical_feature_names"]
    keep_numeric, keep_dynamic = config["keep_numeric"], config["keep_dynamic_categorical"]
    for name, kept, available in (("keep_numeric", keep_numeric, numeric),
                                  ("keep_dynamic_categorical", keep_dynamic, dynamic)):
        if not kept or len(set(kept)) != len(kept):
            raise ValueError(f"{name} must be a nonempty list of unique names")
        unknown = [feature for feature in kept if feature not in available]
        if unknown:
            raise ValueError(f"{name} has features absent from the source schema: {unknown}")
        # Preserve the frozen source order so the derivation is a pure projection.
        if kept != [feature for feature in available if feature in kept]:
            raise ValueError(f"{name} must follow the source feature order")
    declared_removed = set(config["removed_numeric"])
    if declared_removed != set(numeric) - set(keep_numeric):
        raise ValueError("removed_numeric must list exactly the numeric features that are dropped")
    if set(config["removed_static"]) != set(schema["static_feature_names"]):
        raise ValueError("removed_static must list every source static feature")
    if config.get("static_policy") != "constant_absent_placeholder":
        raise ValueError("Only static_policy=constant_absent_placeholder is supported")
    placeholder = config.get("static_placeholder_name")
    if placeholder != STATIC_PLACEHOLDER:
        raise ValueError(f"static_placeholder_name must be {STATIC_PLACEHOLDER}")
    return ([numeric.index(feature) for feature in keep_numeric],
            [dynamic.index(feature) for feature in keep_dynamic])


def derive_state(state, config):
    numeric_index, dynamic_index = _validate_selection(config, state["schema"])
    schema = state["schema"]
    keep_numeric, keep_dynamic = config["keep_numeric"], config["keep_dynamic_categorical"]
    placeholder = config["static_placeholder_name"]
    derived = copy.deepcopy(state)
    derived["schema"]["numeric_feature_names"] = list(keep_numeric)
    derived["schema"]["categorical_feature_names"] = list(keep_dynamic)
    derived["schema"]["static_feature_names"] = [placeholder]
    derived["schema"]["mask_feature_order"] = [*keep_numeric, *keep_dynamic]
    for key, value in state["numeric"].items():
        if isinstance(value, dict):
            derived["numeric"][key] = {feature: value[feature] for feature in keep_numeric}
    derived["numeric"]["constant_features_scale_one"] = [
        feature for feature in state["numeric"]["constant_features_scale_one"] if feature in keep_numeric
    ]
    derived["vocabularies"]["dynamic"] = {
        feature: state["vocabularies"]["dynamic"][feature] for feature in keep_dynamic
    }
    # Empty vocabulary: cardinality 4 (reserved IDs only); every sample uses ID 1.
    derived["vocabularies"]["static"] = {placeholder: []}
    if schema["mask_feature_order"] != [*schema["numeric_feature_names"], *schema["categorical_feature_names"]]:
        raise ValueError("Source mask order is not numeric followed by categorical features")
    mask_index = numeric_index + [len(schema["numeric_feature_names"]) + index for index in dynamic_index]
    return derived, numeric_index, dynamic_index, mask_index


def derive_arrays(arrays, numeric_index, dynamic_index, mask_index):
    out = {}
    for name, value in arrays.items():
        if name == "X_numeric":
            out[name] = np.ascontiguousarray(value[..., numeric_index])
        elif name == "X_dynamic_categorical_ids":
            out[name] = np.ascontiguousarray(value[..., dynamic_index])
        elif name == "X_static_categorical_ids":
            out[name] = np.full((value.shape[0], 1), ABSENT_ID, dtype=value.dtype)
        elif name in FEATURE_AXIS_MASKS:
            out[name] = np.ascontiguousarray(value[..., mask_index])
        elif name == "numeric":
            out[name] = value[numeric_index]
        elif name == "dynamic_categorical":
            out[name] = value[dynamic_index]
        elif name == "static_categorical":
            out[name] = np.array([STATIC_PLACEHOLDER], dtype=value.dtype)
        else:
            out[name] = value
    return out


def within_race_variance_share(split_data, names):
    """Share of each feature's train variance that lies within races (and driver-races).

    Uses present, visible cells of supervised-eligible windows, with the same
    overlapping-window weighting as the frozen preprocessor fit. Values are
    already standardized, which does not change the variance ratio.
    """
    arrays, positions = split_data.arrays, split_data.positions
    metadata = split_data.metadata.iloc[positions].reset_index(drop=True)
    width = arrays["X_numeric"].shape[1]
    race = np.repeat((metadata.Temporada.astype(str) + "-" + metadata.Ronda.astype(str)).to_numpy(), width)
    driver_race = np.repeat(metadata.DriverNumber.astype(str).to_numpy(), width)
    driver_race = race + "-" + driver_race
    values = arrays["X_numeric"][positions].reshape(-1, len(names))
    reasons = arrays["mask_reason_codes"][positions][..., :len(names)].reshape(-1, len(names))
    result = {}
    for index, name in enumerate(names):
        present = reasons[:, index] == 0
        value = values[present, index].astype(np.float64)
        total = value.var()
        shares = {}
        for label, groups in (("within_race", race[present]), ("within_driver_race", driver_race[present])):
            if total == 0:
                shares[label] = 0.0
                continue
            _, inverse = np.unique(groups, return_inverse=True)
            means = np.bincount(inverse, weights=value) / np.bincount(inverse)
            shares[label] = float(np.mean((value - means[inverse]) ** 2) / total)
        result[name] = {**shares, "present_cells": int(present.sum())}
    return result


def prepare(config_path, root=ROOT):
    root = Path(root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_json(config_path)
    source = root / config["source_package"]
    output = root / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_manifest, state = verify_preprocessed(source, config["source_manifest_sha256"])
    derived_state, numeric_index, dynamic_index, mask_index = derive_state(state, config)
    output.mkdir(parents=True)
    try:
        write_json(output / "preprocessor.json", derived_state)
        report = {"splits": {}}
        for split in SPLITS:
            with np.load(source / f"{split}_inputs.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            derived = derive_arrays(arrays, numeric_index, dynamic_index, mask_index)
            for key, schema_key in (("numeric", "numeric_feature_names"),
                                    ("dynamic_categorical", "categorical_feature_names"),
                                    ("static_categorical", "static_feature_names")):
                if key in derived and list(derived[key]) != derived_state["schema"][schema_key]:
                    raise ValueError(f"Embedded feature names differ from schema: {key}")
            np.savez_compressed(output / f"{split}_inputs.npz", **derived)
            for suffix in ("metadata.csv.gz", "labels.csv.gz"):
                shutil.copyfile(source / f"{split}_{suffix}", output / f"{split}_{suffix}")
            # Re-read through the training loader: shapes, masks and alignment must pass.
            source_split = load_split(source, split, state)
            derived_split = load_split(output, split, derived_state)
            if split == "train":
                report["source_within_race_variance_share_train"] = within_race_variance_share(
                    source_split, state["schema"]["numeric_feature_names"])
            if (not np.array_equal(source_split.positions, derived_split.positions)
                    or not np.array_equal(source_split.targets, derived_split.targets)):
                raise ValueError(f"Eligible population or targets changed in {split}")
            if not np.array_equal(derived_split.arrays["X_numeric"],
                                  source_split.arrays["X_numeric"][..., numeric_index]):
                raise ValueError(f"Selected numeric values changed in {split}")
            report["splits"][split] = {
                "samples": int(len(derived_split.metadata)),
                "supervised_eligible": int(len(derived_split.positions)),
                "positives": int(derived_split.targets.sum()),
                "X_numeric_shape": list(derived["X_numeric"].shape),
                "X_dynamic_categorical_ids_shape": list(derived["X_dynamic_categorical_ids"].shape),
                "X_static_categorical_ids_shape": list(derived["X_static_categorical_ids"].shape),
                "static_ids_unique": sorted(int(v) for v in np.unique(derived["X_static_categorical_ids"])),
            }
        report.update({
            "kept_numeric": config["keep_numeric"],
            "kept_dynamic_categorical": config["keep_dynamic_categorical"],
            "removed_numeric": config["removed_numeric"],
            "removed_static": config["removed_static"],
            "static_policy": config["static_policy"],
            "statistics_refit": False,
            "test_features_transformed": False,
        })
        write_json(output / "report.json", report)
        shutil.copyfile(config_path, output / "config.json")
        output_files = ["preprocessor.json", "report.json", "config.json",
                        *[f"{split}_{suffix}" for split in SPLITS
                          for suffix in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")]]
        schema = derived_state["schema"]
        write_json(output / "manifest.json", {
            "artifact_version": "preprocessed_causal_inputs_v1",
            "derivation_version": "feature_projection_v1",
            "derived_from": {
                "package": config["source_package"],
                "manifest_sha256": config["source_manifest_sha256"],
                "fit_state_sha256": source_manifest["fit_state_sha256"],
            },
            "fit_state_sha256": _state_hash(derived_state),
            "fit_split": "train",
            "fit_eligibility": source_manifest.get("fit_eligibility", "supervised_eligible"),
            "split_membership": source_manifest.get("split_membership"),
            "source_hashes": {
                **source_manifest["source_hashes"],
                "derivation_config": sha256(config_path),
                "derivation_script": sha256(Path(__file__)),
            },
            "feature_order": {
                "window_length": schema["window_length"],
                "numeric_feature_names": schema["numeric_feature_names"],
                "categorical_feature_names": schema["categorical_feature_names"],
                "static_feature_names": schema["static_feature_names"],
                "mask_codes": schema["mask_codes"],
                "mask_feature_order": schema["mask_feature_order"],
                "feature_contract_version": schema["feature_contract_version"],
            },
            "test_features_transformed": False,
            "test_label_values_used": False,
            "output_files_sha256": {name: sha256(output / name) for name in output_files},
        })
        verify_preprocessed(output)
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"Paquete derivado: {output}")
    print(f"SHA-256 del manifiesto: {sha256(output / 'manifest.json')}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/entradas_telemetria_v1.json"))
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
