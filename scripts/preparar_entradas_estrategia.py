"""Add causal race-strategy features, per window slot, to the telemetry-only inputs.

For a sample predicting lap N, window slot k shows lap L_k <= N. Each strategy
feature in that slot is the value known at the decision time of lap L_k (the
start of lap L_k), which is never later than the sample's own decision time.
The last slot therefore carries exactly the strategy state at the decision.

Values are standardized with mean/std fitted only on present cells of
supervised-eligible train windows (same rule as the frozen preprocessor),
typical stint lengths are fitted on train seasons, and 2025 is never loaded.
Existing inputs, masks, labels and sample order are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from f1_pitstop.rnn_data import load_split, read_json, sha256, verify_preprocessed, write_json
from f1_pitstop.strategy_features import SEASONS, median_stint_length_train, scheduled_laps, strategy_features
from f1_pitstop.strategy_features_r2 import pit_loss_table, strategy_features_round2

STRATEGY_MODULES = {1: "src/f1_pitstop/strategy_features.py", 2: "src/f1_pitstop/strategy_features_r2.py"}
from preparar_entradas_telemetria import SPLITS, _state_hash

KEY = ["Temporada", "Ronda", "DriverNumber"]
PRESENT, PADDING, ROW_MISSING, TEMPORAL, FEATURE_MISSING = 0, 1, 2, 3, 4


def decision_point_features(sample_index, laps, cache, names, strategy_round=1):
    """Strategy features for every 2022-2024 decision point (all laps, not only eligible)."""
    index = sample_index.loc[sample_index.Temporada.isin(SEASONS)].copy()
    if index.Temporada.eq(2025).any():
        raise ValueError("Final test rows must not enter the derivation")
    index["DriverNumber"] = index.DriverNumber.astype(str).str.removesuffix(".0")
    laps = laps.loc[laps.Temporada.isin(SEASONS)].copy()
    total_laps = scheduled_laps(cache)
    stint_median = median_stint_length_train(laps)
    samples = index[["sample_id", *KEY, "prediction_lap", "decision_time_seconds"]]
    if strategy_round == 1:
        table = strategy_features(samples, laps, total_laps, stint_median)
    elif strategy_round == 2:
        table = strategy_features_round2(samples, laps, total_laps, stint_median, pit_loss_table(laps))
    else:
        raise ValueError("strategy_round must be 1 or 2")
    table = samples.merge(table, on="sample_id", how="left", validate="one_to_one")
    missing = [name for name in names if name not in table]
    if missing:
        raise ValueError(f"Unknown strategy features: {missing}")
    table = table[[*KEY, "prediction_lap", "decision_time_seconds", *names]].rename(
        columns={"prediction_lap": "slot_lap", "decision_time_seconds": "slot_decision_time"})
    return table, stint_median, total_laps


def slot_values(metadata, arrays, table, sample_index, names):
    """Raw values and reason codes, shape (samples, W, features)."""
    n, width = arrays["window_lap_numbers"].shape
    meta = metadata[["sample_id", *KEY]].copy()
    meta["DriverNumber"] = meta.DriverNumber.astype(str).str.removesuffix(".0")
    decision = sample_index[["sample_id", "decision_time_seconds"]]
    meta = meta.merge(decision, on="sample_id", how="left", validate="one_to_one")
    slots = pd.DataFrame({
        "row": np.repeat(np.arange(n), width),
        "slot_lap": arrays["window_lap_numbers"].reshape(-1).astype(int),
    })
    for column in [*KEY, "decision_time_seconds"]:
        slots[column] = np.repeat(meta[column].to_numpy(), width)
    merged = slots.merge(table, on=[*KEY, "slot_lap"], how="left", validate="many_to_one", sort=False)
    if len(merged) != len(slots) or not np.array_equal(merged.row.to_numpy(), slots.row.to_numpy()):
        raise ValueError("Slot lookup changed order")
    values = merged[names].to_numpy(dtype=np.float64).reshape(n, width, len(names))
    reasons = np.full((n, width, len(names)), PRESENT, dtype=np.uint8)
    reasons[~np.isfinite(values)] = FEATURE_MISSING
    late = (merged.slot_decision_time.to_numpy() > merged.decision_time_seconds.to_numpy()).reshape(n, width)
    reasons[late] = TEMPORAL
    reasons[~arrays["slot_exists_mask"]] = FEATURE_MISSING
    reasons[arrays["slot_row_missing_mask"]] = ROW_MISSING
    reasons[arrays["padding_mask"]] = PADDING
    return values, reasons


def prepare(config_path, root=ROOT):
    root = Path(root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = read_json(config_path)
    source, output = root / config["source_package"], root / config["output_dir"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    names = list(config["strategy_features"])
    if not names or len(set(names)) != len(names):
        raise ValueError("strategy_features must be a nonempty list of unique names")
    source_manifest, state = verify_preprocessed(source, config["source_manifest_sha256"])
    if set(names) & set(state["schema"]["numeric_feature_names"]):
        raise ValueError("Strategy features already exist in the source schema")
    for key in ("laps", "sample_index"):
        if sha256(root / config[key]) != config[f"{key}_sha256"]:
            raise ValueError(f"Pinned input changed: {config[key]}")
    sample_index = pd.read_csv(root / config["sample_index"], dtype={"DriverNumber": str})
    sample_index = sample_index.loc[sample_index.Temporada.isin(SEASONS)]
    laps = pd.read_pickle(root / config["laps"])
    strategy_round = int(config.get("strategy_round", 1))
    table, stint_median, total_laps = decision_point_features(
        sample_index, laps, root / config["cache"], names, strategy_round)

    splits, raw = {}, {}
    for split in SPLITS:
        data = load_split(source, split, state)
        values, reasons = slot_values(data.metadata, data.arrays, table, sample_index, names)
        splits[split], raw[split] = data, (values, reasons)
    # Train-only fit: present cells of supervised-eligible train windows (overlaps counted).
    train_values, train_reasons = raw["train"]
    positions = splits["train"].positions
    fit = {"mean": {}, "scale": {}, "observed_cell_count": {}, "observed_physical_zero_count": {}}
    constant = []
    for index, name in enumerate(names):
        cells = train_values[positions, :, index][train_reasons[positions, :, index] == PRESENT]
        if not len(cells):
            raise ValueError(f"No observed train values for {name}")
        mean, std = float(cells.mean()), float(cells.std())
        if std == 0.0:
            std = 1.0
            constant.append(name)
        fit["mean"][name], fit["scale"][name] = mean, std
        fit["observed_cell_count"][name] = int(len(cells))
        fit["observed_physical_zero_count"][name] = int((cells == 0).sum())

    derived_state = copy.deepcopy(state)
    old_numeric = state["schema"]["numeric_feature_names"]
    new_numeric = [*old_numeric, *names]
    derived_state["schema"]["numeric_feature_names"] = new_numeric
    derived_state["schema"]["mask_feature_order"] = [*new_numeric, *state["schema"]["categorical_feature_names"]]
    for key, values in fit.items():
        derived_state["numeric"][key] = {**state["numeric"][key], **values}
    derived_state["numeric"]["constant_features_scale_one"] = [
        *state["numeric"]["constant_features_scale_one"], *constant]

    output.mkdir(parents=True)
    try:
        write_json(output / "preprocessor.json", derived_state)
        report = {"strategy_features": names, "strategy_round": strategy_round, "stint_median_train": stint_median,
                  "scheduled_laps": {f"{k[0]}-{k[1]}": v for k, v in total_laps.items()},
                  "fit": fit, "constant_features_scale_one": constant, "splits": {},
                  "slot_rule": "slot lap L carries the value known at the decision time (start) of lap L",
                  "test_features_transformed": False}
        mean = np.array([fit["mean"][n] for n in names])
        scale = np.array([fit["scale"][n] for n in names])
        insert_at = len(old_numeric)
        for split in SPLITS:
            data = splits[split]
            values, reasons = raw[split]
            standardized = np.where(reasons == PRESENT, (np.nan_to_num(values) - mean) / scale, 0.0).astype(np.float32)
            arrays = dict(data.arrays)
            arrays["X_numeric"] = np.concatenate([arrays["X_numeric"], standardized], axis=-1)
            reason_codes = np.concatenate([arrays["mask_reason_codes"][..., :insert_at], reasons.astype(arrays["mask_reason_codes"].dtype),
                                           arrays["mask_reason_codes"][..., insert_at:]], axis=-1)
            arrays["mask_reason_codes"] = reason_codes
            arrays["temporal_unavailable_mask"] = reason_codes == TEMPORAL
            arrays["feature_missing_mask"] = np.isin(reason_codes, [2, 4, 5])
            arrays["numeric"] = np.array(new_numeric)
            np.savez_compressed(output / f"{split}_inputs.npz", **arrays)
            for suffix in ("metadata.csv.gz", "labels.csv.gz"):
                shutil.copyfile(source / f"{split}_{suffix}", output / f"{split}_{suffix}")
            derived = load_split(output, split, derived_state)
            if (not np.array_equal(derived.positions, data.positions)
                    or not np.array_equal(derived.targets, data.targets)
                    or not np.array_equal(derived.arrays["X_numeric"][..., :insert_at], data.arrays["X_numeric"])):
                raise ValueError(f"Existing inputs or population changed in {split}")
            eligible = reasons[data.positions]
            report["splits"][split] = {
                "samples": int(len(data.metadata)), "supervised_eligible": int(len(data.positions)),
                "X_numeric_shape": list(arrays["X_numeric"].shape),
                "present_share_last_slot_eligible": {n: float((eligible[:, -1, i] == PRESENT).mean())
                                                      for i, n in enumerate(names)},
                "temporal_unavailable_cells": int((reasons == TEMPORAL).sum()),
            }
        write_json(output / "report.json", report)
        shutil.copyfile(config_path, output / "config.json")
        output_files = ["preprocessor.json", "report.json", "config.json",
                        *[f"{split}_{suffix}" for split in SPLITS
                          for suffix in ("inputs.npz", "metadata.csv.gz", "labels.csv.gz")]]
        schema = derived_state["schema"]
        write_json(output / "manifest.json", {
            "artifact_version": "preprocessed_causal_inputs_v1",
            "derivation_version": "strategy_feature_extension_v1",
            "derived_from": {"package": config["source_package"], "manifest_sha256": config["source_manifest_sha256"],
                             "fit_state_sha256": source_manifest["fit_state_sha256"]},
            "fit_state_sha256": _state_hash(derived_state), "fit_split": "train",
            "fit_eligibility": source_manifest.get("fit_eligibility", "supervised_eligible"),
            "split_membership": source_manifest.get("split_membership"),
            "source_hashes": {**source_manifest["source_hashes"],
                              "strategy_config": sha256(config_path),
                              "strategy_script": sha256(Path(__file__)),
                              "strategy_module": sha256(ROOT / STRATEGY_MODULES[strategy_round]),
                              "laps": config["laps_sha256"]},
            "feature_order": {key: schema[key] for key in (
                "window_length", "numeric_feature_names", "categorical_feature_names", "static_feature_names",
                "mask_codes", "mask_feature_order", "feature_contract_version")},
            "test_features_transformed": False, "test_label_values_used": False,
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
    parser.add_argument("--config", type=Path, default=Path("configs/entradas_estrategia_v1.json"))
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
