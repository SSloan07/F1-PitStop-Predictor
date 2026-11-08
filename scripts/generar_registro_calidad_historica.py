"""Build an auditable quality register from the published historical matrix.

This process writes evidence and reports only; it never rewrites or duplicates
the matrix, labels, cached source files, or causal feature tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.auditar_calidad_matriz import EXPECTED_COLUMNS, KEY, sha256
from scripts.normalizar_compound import (
    DEFAULT_SOURCE as MASTER_SOURCE,
    load_normalization_audit,
    load_normalized_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = MASTER_SOURCE.parent / "vueltas_2022_2025_compound_normalizado_v1" / "vueltas_2022_2025_compound_normalizado.pkl"
DEFAULT_MANIFEST = DEFAULT_INPUT.parent / "manifest.json"
DEFAULT_COMPOUND_AUDIT = DEFAULT_INPUT.parent / "compound_normalization_audit.csv"
DEFAULT_COMPOUND_EVIDENCE = ROOT / "docs/aclaraciones/evidencia_compound_v1/summary.json"
DEFAULT_CACHE_ROOT = ROOT / "EDA/dataset/cache_fastf1_carreras"
DEFAULT_OUTPUT = ROOT / "EDA/dataset/auditorias_limpieza/registro_calidad_historica_v1"
RULE_VERSION = "historical_quality_contract_v2"
MATRIX_ARTIFACT_NAME = "vueltas_2022_2025_compound_normalizado.pkl"
AUDIT_ARTIFACT_NAME = "compound_normalization_audit.csv"

CACHE_CASES = {
    "2022-18-JPN-BOT": {
        "season": 2022, "round": 18, "driver": "BOT", "driver_number": "77",
        "laps": list(range(3, 29)),
        "path": "2022/2022-10-09_Japanese_Grand_Prix/2022-10-09_Race/car_data.ff1pkl",
        "channels": ["Speed", "RPM", "nGear", "Throttle"],
    },
    "2022-05-MIA-MAG": {
        "season": 2022, "round": 5, "driver": "MAG", "driver_number": "20",
        "laps": [1],
        "path": "2022/2022-05-08_Miami_Grand_Prix/2022-05-08_Race/car_data.ff1pkl",
        "channels": ["Speed", "RPM", "nGear", "Throttle"],
    },
    "2022-08-AZE-LEC": {
        "season": 2022, "round": 8, "driver": "LEC", "driver_number": "16",
        "laps": [21],
        "path": "2022/2022-06-12_Azerbaijan_Grand_Prix/2022-06-12_Race/car_data.ff1pkl",
        "channels": ["Speed", "RPM", "nGear", "Throttle"],
    },
    "2025-01-AUS-HAD": {
        "season": 2025, "round": 1, "driver": "HAD", "driver_number": "6",
        "laps": [1],
        "path": "2025/2025-03-16_Australian_Grand_Prix/2025-03-16_Race/car_data.ff1pkl",
        "channels": ["Speed", "RPM", "nGear", "Throttle", "DRS"],
    },
    "2025-16-ITA-ALB": {
        "season": 2025, "round": 16, "driver": "ALB", "driver_number": "23",
        "laps": list(range(1, 11)),
        "path": "2025/2025-09-07_Italian_Grand_Prix/2025-09-07_Race/car_data.ff1pkl",
        "channels": ["Speed", "RPM", "nGear", "Throttle", "DRS"],
    },
}

SECTOR_CACHE = "2024/2024-11-23_Las_Vegas_Grand_Prix/2024-11-23_Race/_extended_timing_data.ff1pkl"
CLOCK_CACHES = {
    (2023, 3): {
        "name": "Australian Grand Prix",
        "directory": "2023/2023-04-02_Australian_Grand_Prix/2023-04-02_Race",
    },
    (2024, 8): {
        "name": "Monaco Grand Prix",
        "directory": "2024/2024-05-26_Monaco_Grand_Prix/2024-05-26_Race",
    },
}


class QualityContractError(ValueError):
    """Input does not satisfy the published historical matrix contract."""


def validate_telemetry_contract(frame: pd.DataFrame) -> list[str]:
    """Validate deterministic accelerator coverage and value invariants."""
    required = [
        "Telemetry_Samples", "Telemetry_Throttle_valid_samples",
        "Telemetry_Throttle_valid_fraction", "Telemetry_Throttle_valid_mean",
        "Telemetry_Throttle_valid_max",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        return [f"missing throttle quality columns: {missing}"]
    errors: list[str] = []
    total = frame["Telemetry_Samples"]
    valid = frame["Telemetry_Throttle_valid_samples"]
    fraction = frame["Telemetry_Throttle_valid_fraction"]
    if total.isna().any() or total.lt(0).any():
        errors.append("Telemetry_Samples contains null or negative counts")
    if valid.isna().any() or valid.lt(0).any() or valid.gt(total).any():
        errors.append("valid throttle sample count is outside [0, Telemetry_Samples]")
    has_total = total.gt(0)
    expected_fraction = valid[has_total] / total[has_total]
    observed_fraction = fraction[has_total]
    if observed_fraction.isna().any() or not np.allclose(
        expected_fraction.to_numpy(dtype=float), observed_fraction.to_numpy(dtype=float),
        rtol=0.0, atol=1e-12,
    ):
        errors.append("valid throttle fraction does not equal valid_samples / samples")
    if fraction[~has_total].notna().any():
        errors.append("valid throttle fraction must be missing when there are no samples")
    no_valid = valid.eq(0)
    for column in ("Telemetry_Throttle_valid_mean", "Telemetry_Throttle_valid_max"):
        values = frame[column]
        if values[no_valid].notna().any() or values[~no_valid].isna().any():
            errors.append(f"{column} missingness does not match valid sample count")
        if values.dropna().lt(0).any() or values.dropna().gt(100).any():
            errors.append(f"{column} has a valid aggregate outside [0, 100]")
    return errors


def _key_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
    return set(frame[KEY].itertuples(index=False, name=None))


def _key_label(row: Any) -> tuple[Any, ...]:
    return tuple(getattr(row, name) for name in KEY)


def _value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def find_clock_overlaps(frame: pd.DataFrame) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []
    ordered = frame.sort_values(["Temporada", "Ronda", "DriverNumber", "LapNumber"])
    for _, group in ordered.groupby(["Temporada", "Ronda", "DriverNumber"], dropna=False):
        group = group.sort_values("LapNumber")
        previous = None
        for row in group.itertuples(index=False):
            if (previous is not None and pd.notna(previous.Time)
                    and pd.notna(row.LapStartTime) and row.LapStartTime < previous.Time):
                overlaps.append({
                    "key": _key_label(row), "driver": row.Driver,
                    "previous_lap": float(previous.LapNumber),
                    "current_lap": float(row.LapNumber),
                    "overlap_seconds": float((previous.Time - row.LapStartTime).total_seconds()),
                    "previous_time": str(previous.Time),
                    "current_lap_start": str(row.LapStartTime),
                    "previous_track_status": _value(previous.TrackStatus),
                    "current_track_status": _value(row.TrackStatus),
                })
            previous = row
    return overlaps


def validate_clock_race_identity(
    matrix_race_names: set[str], season: int, metadata: dict[str, str], session_info: dict[str, Any]
) -> None:
    """Reject cached timing context unless it identifies the matrix race and season."""
    meeting = session_info.get("Meeting", {})
    source_name = str(meeting.get("Name", "")).strip()
    expected_matrix_name = {name.strip().casefold() for name in matrix_race_names}
    if len(expected_matrix_name) != 1 or source_name.casefold() not in expected_matrix_name:
        raise QualityContractError(
            f"Clock source identity mismatch for {season} round: matrix races="
            f"{sorted(matrix_race_names)}, session_info meeting={source_name!r}; refusing publication."
        )
    start_date = session_info.get("StartDate")
    if not hasattr(start_date, "year") or int(start_date.year) != int(season):
        raise QualityContractError(
            f"Clock source season mismatch: expected {season}, session_info StartDate={start_date!r}."
        )
    if str(session_info.get("Name", "")).casefold() != "race":
        raise QualityContractError(
            f"Clock source is not a Race session: session_info Name={session_info.get('Name')!r}."
        )
    source_path = str(session_info.get("Path", "")).strip("/")
    if source_path != metadata["directory"]:
        raise QualityContractError(
            f"Clock source path mismatch: expected {metadata['directory']!r}, got {source_path!r}."
        )


def matching_restart_event(
    lap_start_time: Any, status_frame: pd.DataFrame, tolerance: pd.Timedelta = pd.Timedelta(milliseconds=1)
) -> dict[str, Any] | None:
    """Match this lap start only to a Started event immediately following Aborted."""
    ordered = status_frame.sort_values("Time").reset_index(drop=True)
    for index, event in ordered.iterrows():
        if str(event["Status"]).casefold() != "started" or index == 0:
            continue
        previous = ordered.iloc[index - 1]
        if str(previous["Status"]).casefold() != "aborted":
            continue
        difference = abs(pd.Timedelta(lap_start_time) - pd.Timedelta(event["Time"]))
        if difference <= tolerance:
            return {
                "aborted_time": str(previous["Time"]),
                "started_time": str(event["Time"]),
                "start_delta_ms": float(difference.total_seconds() * 1000),
                "status_before_started": str(previous["Status"]),
            }
    return None


def create_quality_register(frame: pd.DataFrame, compound_audit: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    def add(row: Any, field: str, reason: str, action: str, status: str,
            evidence: str, original_value: str = "") -> None:
        record = dict(zip(KEY, _key_label(row)))
        record.update({
            "field": field,
            "value_in_matrix": _value(getattr(row, field)) if hasattr(row, field) else "<NA>",
            "original_value": original_value,
            "reason": reason,
            "action": action,
            "review_status": status,
            "evidence": evidence,
            "rule_version": RULE_VERSION,
        })
        records.append(record)

    compound_originals = compound_audit.set_index(KEY)["original_literal"].to_dict()
    missing_columns = [
        "Compound", "TyreLife", "Stint", "LapTime", "Sector1Time", "Sector2Time",
        "Sector3Time", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
        "Position", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Telemetry_Throttle_valid_mean", "Telemetry_Throttle_valid_max",
    ]
    for row in frame.itertuples(index=False):
        key = _key_label(row)
        for field in missing_columns:
            value = getattr(row, field)
            if not pd.isna(value):
                continue
            original = ""
            if field == "Compound":
                original = _value(compound_originals.get(key, pd.NA))
                reason = "compound_not_determined_normalized"
                status = "approved_normalization_preserve_unknown"
                evidence = f"Exact source literal {original}; Compound audit links this row and records its conversion reason."
            elif field == "TyreLife" and pd.isna(row.Stint):
                reason = "missing_stint_in_cached_initial_block"
                status = "preserve_no_eligible_age_source"
                evidence = "Initial pilot-race block has no Stint/TyreLife anchor in the consolidated matrix."
            elif field == "TyreLife":
                reason = "missing_entire_identified_stint"
                status = "preserve_no_anchor_in_stint"
                evidence = "The identified driver-race-stint has TyreLife missing throughout; no interpolation or imputation."
            elif field == "Stint":
                reason = "missing_stint_in_cached_initial_block"
                status = "preserve_no_reconstructed_stint"
                evidence = "Consolidated Stint is absent in the initial cached block; do not infer a counter."
            elif field == "Position" and bool(row.FastF1Generated):
                reason = "generated_row"
                status = "explained_preserve_row"
                evidence = "All generated rows have Position absent; retain row for audit, do not impute or globally delete."
            elif field == "SpeedFL" and pd.notna(row.PitInTime):
                reason = "pit_in_lap_speedfl_unavailable"
                status = "explained_contextual_missing"
                evidence = "SpeedFL absence coincides with PitInTime; no zero substitution."
            elif bool(row.FastF1Generated):
                reason = "generated_row"
                status = "explained_preserve_row"
                evidence = "FastF1Generated row; preserve sequence and missing values, do not use as complete lap."
            elif field.startswith("Telemetry_Throttle_valid_"):
                reason = "no_valid_throttle_samples"
                status = "explained_preserve_missing"
                evidence = "Filtered throttle mean/max are absent exactly where valid sample count is zero."
            elif field.startswith("Sector"):
                reason = "sector_timing_missing_preserved"
                status = "preserve_no_reconstruction"
                evidence = "Sector timing is missing; candidate arithmetic is recorded separately and not materialized."
            elif field.startswith("Speed"):
                reason = "timing_speed_missing_preserved"
                status = "preserve_contextual_missing"
                evidence = "Timing speed can be structurally missing (pit-in, generated row, red flag); no zero fill."
            else:
                reason = "lap_time_missing_preserved"
                status = "preserve_no_reconstruction"
                evidence = "Missing lap duration retained; no imputation."
            add(row, field, reason, "preserve_value_and_row", status, evidence, original)

        if bool(row.FastF1Generated):
            add(row, "FastF1Generated", "generated_row", "preserve_row_and_flag",
                "explained_preserve_row", "Flag is context; it does not alone prove retirement or justify dropping the row.")
        if pd.isna(row.Stint) and bool(row.FreshTyre):
            add(row, "FreshTyre", "fresh_tyre_true_without_stint_state", "preserve_value_no_inference",
                "source_limitation_recorded", "Miami 2025 initial block lacks Stint/TyreLife and preceding cached tyre state.")

        if pd.notna(row.PitInTime) and pd.notna(row.SpeedFL):
            add(row, "SpeedFL", "speedfl_present_with_pit_in_context",
                "preserve_value_flag_for_case_review", "case_review_required",
                "Two PitIn rows have SpeedFL present (ZHO 2022 round 21 lap 16; SAR 2024 round 11 lap 1). Preserve both values; do not infer validity from the field alone.")

    # Raw accelerator aggregates include the source's unavailable code; keep them for audit.
    for field in ("Telemetry_Throttle_mean", "Telemetry_Throttle_max"):
        series = frame[field]
        for row in frame.loc[series.notna() & ~series.between(0, 100)].itertuples(index=False):
            add(row, field, "raw_throttle_contains_out_of_range_source_code",
                "preserve_raw_audit_value_exclude_from_baseline", "known_source_encoding",
                "Raw throttle aggregate includes values outside 0-100 (FastF1 code 104); filtered aggregate is retained separately.")

    # Gear aggregates are intentionally paused, not clipped or recalculated here.
    for field in ("Telemetry_nGear_mean", "Telemetry_nGear_max"):
        series = frame[field]
        for row in frame.loc[series.notna() & series.gt(8)].itertuples(index=False):
            add(row, field, "gear_aggregate_exceeds_physical_range_candidate",
                "preserve_raw_audit_value_pause_feature", "recalculation_pending",
                "Values above gear 8 require raw-sample validation; no clipping or aggregate correction is authorized.")

    for field in ("Telemetry_Speed_mean", "Telemetry_Speed_max", "Telemetry_RPM_mean", "Telemetry_RPM_max"):
        for row in frame.loc[frame[field].eq(0)].itertuples(index=False):
            add(row, field, "zero_telemetry_aggregate_source_review",
                "preserve_value_mark_for_context", "case_cross_checked_or_review_flagged",
                "Zero is retained; selected source-cache traces are cross-checked in source_case_review.csv.")

    sector_complete = frame[["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].notna().all(axis=1)
    sector_sum = frame["Sector1Time"] + frame["Sector2Time"] + frame["Sector3Time"]
    discrepancy = sector_complete & sector_sum.ne(frame["LapTime"])
    for row in frame.loc[discrepancy].itertuples(index=False):
        gap_ms = float((row.LapTime - row.Sector1Time - row.Sector2Time - row.Sector3Time).total_seconds() * 1000)
        for field in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
            add(row, field, "sector_sum_discrepancy", "preserve_source_value_no_correction",
                "source_cross_checked_accuracy_flag_false",
                f"2024 round 22: source extended timing record agrees with this field; LapTime minus sector sum={gap_ms:.3f} ms; IsAccurate={row.IsAccurate}.")

    candidate = (
        frame["LapTime"].notna() & frame["Sector1Time"].isna()
        & frame["Sector2Time"].notna() & frame["Sector3Time"].notna()
    )
    candidate_gap = frame["LapTime"] - frame["Sector2Time"] - frame["Sector3Time"]
    for row in frame.loc[candidate & candidate_gap.gt(pd.Timedelta(0))].itertuples(index=False):
        gap = row.LapTime - row.Sector2Time - row.Sector3Time
        add(row, "Sector1Time", "positive_algebraic_sector1_candidate",
            "do_not_reconstruct_without_source_review", "optional_manual_review",
            f"Algebraic candidate is {gap.total_seconds():.3f} s; positive difference alone does not validate a measurement.")

    for overlap in find_clock_overlaps(frame):
        key = overlap["key"]
        selector = np.ones(len(frame), dtype=bool)
        for column, value in zip(KEY, key):
            selector &= frame[column].eq(value).to_numpy()
        row = next(frame.loc[selector].itertuples(index=False))
        add(row, "LapStartTime", "lap_start_precedes_previous_lap_time",
            "preserve_timestamps_no_shift", "session_boundary_reviewed_or_unresolved",
            f"Previous lap {overlap['previous_lap']:g} Time exceeds this LapStartTime by {overlap['overlap_seconds']:.3f}s; see clock_boundary_review.csv.")

    columns = [*KEY, "field", "value_in_matrix", "original_value", "reason", "action",
               "review_status", "evidence", "rule_version"]
    return pd.DataFrame.from_records(records, columns=columns)


def variable_quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in frame.columns:
        if column in KEY:
            status, treatment = "agreed_integrity_key", "require complete unique key; no automatic row selection"
        elif column == "Compound":
            status, treatment = "approved_historical_normalization", "exact None/nan markers are NA; Compound_asof is a separate causal replay"
        elif column == "TyreLife":
            status, treatment = "preserve_missing", "no imputation from neighbors, FreshTyre, TotalLaps, or future rows"
        elif column == "Stint":
            status, treatment = "preserve_missing", "do not infer stint count or use retrospective value as as-of state"
        elif column == "FreshTyre":
            status, treatment = "preserve_with_coverage_caveat", "354 initial Miami 2025 rows lack preceding cached tyre state; no backfill"
        elif column == "Position":
            status, treatment = "preserve_generated_missingness", "135 missing positions coincide with generated rows; no imputation"
        elif column in {"PitInTime", "PitOutTime"}:
            status, treatment = "preserve_event_missingness", "missing event time is not numeric zero or proof that no event occurred"
        elif column == "FastF1Generated":
            status, treatment = "preserve_context_flag", "flag alone does not prove retirement or authorize dropping rows"
        elif column == "IsAccurate":
            status, treatment = "preserve_context_flag", "not a global filter; assess field/case evidence separately"
        elif column in {"Telemetry_Throttle_valid_mean", "Telemetry_Throttle_valid_max"}:
            status, treatment = "baseline_candidate_with_mask", "agreed RNN candidate on completed eligible laps; retain missing aggregate when zero valid samples"
        elif column in {"Telemetry_Throttle_valid_samples", "Telemetry_Throttle_valid_fraction"}:
            status, treatment = "audit_only_coverage", "coverage metadata retained for QA; excluded from the initial baseline inputs"
        elif column in {"Telemetry_Throttle_mean", "Telemetry_Throttle_max"}:
            status, treatment = "audit_only_raw_code", "contains source code 104; excluded from baseline, not rewritten"
        elif column.startswith("Telemetry_nGear_"):
            status, treatment = "paused_pending_raw_recalculation", "values above 8 retained; do not clip or use pending validation"
        elif column.startswith("Telemetry_DRS_"):
            status, treatment = "excluded_pending_semantic_validation", "DRS aggregates include state codes; do not interpret as continuous values"
        elif column == "Telemetry_Samples" or column.endswith("valid_samples") or column.endswith("valid_fraction"):
            status, treatment = "audit_only_coverage", "coverage metadata retained for QA, not a baseline predictor"
        elif column.startswith("Telemetry_"):
            status, treatment = "preserve_historical_aggregate", "zeros and missing values reported; only completed eligible laps belong to later causal windows"
        elif column.startswith("Weather_"):
            status, treatment = "preserve_retrospective_weather_join", "Weather_Time may follow lap start; as-of selection belongs to later causal feature construction"
        elif column in {"SafetyCar", "VirtualSafetyCar", "TrackStatus"}:
            status, treatment = "preserve_retrospective_lap_state", "per-lap aggregate is not live as-of state; replay race control later"
        elif column in {"SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"}:
            status, treatment = "preserve_timing_speed_missingness", "no zero fill; SpeedFL absence is contextual and two PitIn rows with values remain flagged"
        elif column in {"LapTime", "Sector1Time", "Sector2Time", "Sector3Time"}:
            status, treatment = "preserve_timing_values", "no reconstruction or correction in this phase; source discrepancies are separately registered"
        else:
            status, treatment = "preserve_historical_value", "no deletion, imputation, or semantic conversion without field-specific approval"
        records.append({
            "field": column, "dtype": str(frame[column].dtype),
            "rows": int(len(frame)), "missing_count": int(frame[column].isna().sum()),
            "distinct_non_null": int(frame[column].nunique(dropna=True)),
            "decision_status": status, "current_treatment": treatment,
            "rule_version": RULE_VERSION,
        })
    return pd.DataFrame.from_records(records)


def review_telemetry_source_cases(frame: pd.DataFrame, cache_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for case_id, case in CACHE_CASES.items():
        cache_path = cache_root / case["path"]
        if not cache_path.is_file():
            raise QualityContractError(f"Required local source cache is absent: {cache_path}")
        cache_hash = sha256(cache_path)
        hashes[case_path_key(case["path"])] = cache_hash
        payload = pd.read_pickle(cache_path)
        cache = payload.get("data", {}).get(case["driver_number"])
        if not isinstance(cache, pd.DataFrame):
            raise QualityContractError(f"No telemetry cache for driver {case['driver_number']} in {cache_path}")
        selected = frame[
            frame["Temporada"].eq(case["season"])
            & frame["Ronda"].eq(case["round"])
            & frame["Driver"].eq(case["driver"])
            & frame["LapNumber"].isin(case["laps"])
        ]
        if len(selected) != len(case["laps"]):
            raise QualityContractError(f"Matrix rows missing for source review case {case_id}")
        numeric: dict[str, pd.Series] = {}
        matrix_samples = int(selected["Telemetry_Samples"].sum())
        raw_samples = 0
        for row in selected.itertuples(index=False):
            interval = cache[cache["Time"].ge(row.LapStartTime) & cache["Time"].le(row.Time)]
            raw_samples += int(len(interval))
            for channel in case["channels"]:
                values = pd.to_numeric(interval[channel], errors="coerce")
                numeric.setdefault(channel, []).append(values)
        channel_report = {}
        for channel, chunks in numeric.items():
            values = pd.concat(chunks, ignore_index=True).dropna()
            channel_report[channel] = {
                "min": None if values.empty else float(values.min()),
                "max": None if values.empty else float(values.max()),
                "zero_samples": int(values.eq(0).sum()),
                "sample_count": int(len(values)),
                "unique_values": int(values.nunique()),
            }
        matrix_report = {
            channel: {
                "min": None if selected[f"Telemetry_{channel}_mean"].dropna().empty else float(selected[f"Telemetry_{channel}_mean"].min()),
                "max": None if selected[f"Telemetry_{channel}_max"].dropna().empty else float(selected[f"Telemetry_{channel}_max"].max()),
            }
            for channel in case["channels"]
            if f"Telemetry_{channel}_mean" in selected and f"Telemetry_{channel}_max" in selected
        }
        records.append({
            "case_id": case_id, "season": case["season"], "round": case["round"],
            "driver": case["driver"], "laps": json.dumps(case["laps"]),
            "matrix_rows": int(len(selected)), "matrix_telemetry_samples": matrix_samples,
            "raw_cache_interval_samples": raw_samples,
            "matrix_minus_raw_sample_count": matrix_samples - raw_samples,
            "cache_path": case["path"], "cache_sha256": cache_hash,
            "raw_channel_profile": json.dumps(channel_report, sort_keys=True),
            "matrix_aggregate_profile": json.dumps(matrix_report, sort_keys=True),
            "decision": "preserve_values_and_flag_context",
            "resolution": "The cached raw telemetry reproduces the zero/code pattern. It does not establish whether zeros mean physical standstill or unavailable feed; no value is changed.",
        })
    return pd.DataFrame.from_records(records), hashes


def case_path_key(path: str) -> str:
    return f"cache:{path}"


def review_sector_source(frame: pd.DataFrame, cache_root: Path) -> tuple[pd.DataFrame, str]:
    path = cache_root / SECTOR_CACHE
    if not path.is_file():
        raise QualityContractError(f"Required local timing source cache is absent: {path}")
    cache_hash = sha256(path)
    extended = pd.read_pickle(path)["data"][0]
    race = frame[frame["Temporada"].eq(2024) & frame["Ronda"].eq(22)]
    complete = race[["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].notna().all(axis=1)
    mismatch = complete & (race["Sector1Time"] + race["Sector2Time"] + race["Sector3Time"]).ne(race["LapTime"])
    checked = 0
    exact = 0
    for row in race.loc[mismatch].itertuples(index=False):
        match = extended[
            extended["Driver"].astype(str).eq(str(row.DriverNumber))
            & extended["Time"].eq(row.Time)
        ]
        checked += 1
        if len(match) == 1 and all(match.iloc[0][field] == getattr(row, field)
                                   for field in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time")):
            exact += 1
    gap_ms = (
        race.loc[mismatch, "LapTime"]
        - race.loc[mismatch, "Sector1Time"]
        - race.loc[mismatch, "Sector2Time"]
        - race.loc[mismatch, "Sector3Time"]
    ).dt.total_seconds() * 1000
    result = pd.DataFrame([{
        "season": 2024, "round": 22, "race": str(race["GranPremio"].iloc[0]),
        "mismatch_rows": int(mismatch.sum()), "matches_in_extended_timing": checked,
        "exact_field_matches": exact, "difference_ms_min": float(gap_ms.min()),
        "difference_ms_max": float(gap_ms.max()),
        "all_isaccurate_false": bool(race.loc[mismatch, "IsAccurate"].eq(False).all()),
        "track_status_values": json.dumps(sorted(race.loc[mismatch, "TrackStatus"].dropna().unique().tolist())),
        "cache_path": SECTOR_CACHE, "cache_sha256": cache_hash,
        "decision": "preserve_source_values_no_correction",
        "resolution": "The local FastF1 extended timing cache exactly matches the consolidated LapTime and all three sectors in every discrepancy. This confirms no consolidation drift, not independent FIA validation. IsAccurate is false for these rows; no repair rule is approved.",
    }])
    if checked != 14 or exact != 14:
        raise QualityContractError(
            f"Sector source cross-check changed: expected 14 exact matches, observed {checked}/{exact}; review the source before publication."
        )
    return result, cache_hash


def review_clock_boundaries(frame: pd.DataFrame, cache_root: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    overlaps = find_clock_overlaps(frame)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for case in overlaps:
        key = case["key"]
        grouped.setdefault((int(key[0]), int(key[1])), []).append(case)
    records = []
    hashes: dict[str, str] = {}
    for race_key, metadata in CLOCK_CACHES.items():
        case_rows = grouped.get(race_key, [])
        directory = cache_root / metadata["directory"]
        info_path = directory / "session_info.ff1pkl"
        status_path = directory / "session_status_data.ff1pkl"
        message_path = directory / "race_control_messages.ff1pkl"
        if not info_path.is_file() or not status_path.is_file() or not message_path.is_file():
            raise QualityContractError(f"Session identity/status/race control cache missing for {metadata['name']}")
        info_hash, status_hash, message_hash = map(sha256, (info_path, status_path, message_path))
        hashes[case_path_key(info_path.relative_to(cache_root).as_posix())] = info_hash
        hashes[case_path_key(status_path.relative_to(cache_root).as_posix())] = status_hash
        hashes[case_path_key(message_path.relative_to(cache_root).as_posix())] = message_hash
        session_info = pd.read_pickle(info_path)["data"]
        matrix_race_names = set(
            frame.loc[frame["Temporada"].eq(race_key[0]) & frame["Ronda"].eq(race_key[1]), "GranPremio"]
            .dropna().astype(str)
        )
        validate_clock_race_identity(matrix_race_names, race_key[0], metadata, session_info)
        status_frame = pd.DataFrame(pd.read_pickle(status_path)["data"])
        messages = pd.DataFrame(pd.read_pickle(message_path)["data"])
        messages["Message"] = messages["Message"].astype("string")
        for case in case_rows:
            prior_lap = int(case["previous_lap"])
            current_lap = int(case["current_lap"])
            restart = matching_restart_event(pd.Timedelta(case["current_lap_start"]), status_frame)
            selected_messages = messages[
                messages["Lap"].isin({prior_lap, current_lap})
                & messages["Message"].str.contains(r"RED FLAG|RESTART|RESUM", case=False, na=False)
            ]
            if restart:
                status = "restart_boundary_time_match_preserve_timestamps"
                finding = (
                    f"LapStartTime matches a Started event {restart['start_delta_ms']:.3f} ms after "
                    f"an Aborted event ({restart['aborted_time']} → {restart['started_time']}); "
                    "this transition is consistent with a session restart. Preserve both timestamps; "
                    "the prior lap closed after this decision and is unavailable to causal history."
                )
            else:
                status = "no_temporally_matching_restart_preserve_and_review"
                finding = (
                    "No Started event immediately following Aborted matches this LapStartTime within 1 ms. "
                    "A suspension elsewhere in the session does not explain this transition; preserve both "
                    "timestamps and exclude the overlapping prior lap from causal history."
                )
            record = dict(zip(KEY, case["key"]))
            record.update({
                "season": race_key[0], "round": race_key[1],
                "race": str(next(iter(matrix_race_names))),
                "session_meeting_name": str(session_info["Meeting"]["Name"]),
                "previous_lap": prior_lap, "current_lap": current_lap,
                "previous_time": case["previous_time"],
                "current_lap_start": case["current_lap_start"],
                "overlap_seconds": case["overlap_seconds"],
                "previous_track_status": case["previous_track_status"],
                "current_track_status": case["current_track_status"],
                "restart_match": bool(restart),
                "matched_aborted_time": restart["aborted_time"] if restart else "",
                "matched_started_time": restart["started_time"] if restart else "",
                "matched_start_delta_ms": restart["start_delta_ms"] if restart else None,
                "status_before_started": restart["status_before_started"] if restart else "",
                "race_control_context": json.dumps([
                    {"Time": str(msg.Time), "Lap": int(msg.Lap), "Message": str(msg.Message)}
                    for msg in selected_messages.itertuples(index=False)
                ], ensure_ascii=False),
                "session_info_path": info_path.relative_to(cache_root).as_posix(),
                "session_info_sha256": info_hash,
                "session_status_cache_sha256": status_hash,
                "race_control_cache_sha256": message_hash,
                "decision": "preserve_timestamps_no_shift",
                "review_status": status, "finding": finding,
            })
            records.append(record)
    if sum(len(x) for x in grouped.values()) != 22:
        raise QualityContractError(f"Expected 22 clock overlap transitions; found {len(overlaps)}")
    result = pd.DataFrame.from_records(records)
    if len(result) != len(overlaps):
        raise QualityContractError(f"Expected per-transition clock evidence for {len(overlaps)} overlaps, found {len(result)}")
    return result, hashes


def _validate_base(frame: pd.DataFrame, compound_audit: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    unexpected_columns = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing_columns:
        errors.append(f"expected schema columns missing: {missing_columns}")
    if unexpected_columns:
        errors.append(f"unexpected schema columns: {unexpected_columns}")
    if frame[KEY].isna().any(axis=1).any():
        errors.append("null rows in the full observation key")
    if frame.duplicated(KEY).any():
        errors.append("duplicate full observation keys")
    if len(frame) != 101290:
        errors.append(f"expected 101290 matrix rows, found {len(frame)}")
    if frame["Compound"].isna().sum() != len(compound_audit):
        errors.append("normalized Compound missing count does not reconcile with its audit")
    if _key_set(frame.loc[frame["Compound"].isna()]) != _key_set(compound_audit):
        errors.append("normalized Compound missing keys differ from the approved conversion audit")
    if len(compound_audit) != 780:
        errors.append(f"expected 780 Compound audit rows, found {len(compound_audit)}")
    observed_literals = compound_audit["original_literal"].value_counts().to_dict()
    if observed_literals != {"None": 426, "nan": 354}:
        errors.append(f"Compound audit literal counts differ from approved evidence: {observed_literals}")
    errors.extend(validate_telemetry_contract(frame))

    generated = frame["FastF1Generated"].astype(bool)
    if not frame["Position"].isna().equals(generated):
        errors.append("Position missingness no longer coincides exactly with FastF1Generated")
    expected_speedfl_context = frame["PitInTime"].notna() | generated
    if (frame["SpeedFL"].isna() & ~expected_speedfl_context).any():
        errors.append("SpeedFL has missing values outside PitInTime/generated-row context")
    if int(frame["TyreLife"].isna().sum()) != 989:
        errors.append(f"expected 989 TyreLife missing values; found {frame['TyreLife'].isna().sum()}")
    if int(frame["Stint"].isna().sum()) != 354:
        errors.append(f"expected 354 Stint missing values; found {frame['Stint'].isna().sum()}")
    groups = ["Temporada", "Ronda", "DriverNumber", "Stint"]
    mixed_groups = frame.groupby(groups, dropna=False)["TyreLife"].filter(
        lambda values: values.isna().any() and values.notna().any()
    )
    if not mixed_groups.empty:
        errors.append("TyreLife missingness appears inside a stint with known ages; review before using whole-stint reason")
    return errors


def load_and_validate_input(input_path: Path, manifest_path: Path, compound_audit_path: Path):
    for path in (input_path, manifest_path, compound_audit_path, DEFAULT_COMPOUND_EVIDENCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hash = sha256(input_path)
    expected_matrix_hash = manifest.get("outputs", {}).get(input_path.name)
    if not expected_matrix_hash or input_hash != expected_matrix_hash:
        raise QualityContractError(
            "Published normalized matrix hash is absent from or differs from its manifest; "
            "review the selected matrix/manifest before generating quality artifacts."
        )
    source_meta = manifest.get("source", {})
    expected_master_path = MASTER_SOURCE.resolve()
    if Path(source_meta.get("path", "")).resolve() != expected_master_path:
        raise QualityContractError("Published matrix manifest does not identify the approved master source")
    compound_evidence = json.loads(DEFAULT_COMPOUND_EVIDENCE.read_text(encoding="utf-8"))
    master_relative = expected_master_path.relative_to(ROOT).as_posix()
    expected_master_hash = compound_evidence.get("input_sha256", {}).get(master_relative)
    if (not expected_master_hash or source_meta.get("sha256_after") != expected_master_hash
            or sha256(expected_master_path) != expected_master_hash):
        raise QualityContractError("Master source hash differs from the approved Compound evidence")
    audit_hash = sha256(compound_audit_path)
    expected_audit_hash = manifest.get("outputs", {}).get(compound_audit_path.name)
    if not expected_audit_hash or audit_hash != expected_audit_hash:
        raise QualityContractError("Compound conversion audit hash is absent or differs from its manifest")
    frame = load_normalized_matrix(input_path)
    audit = load_normalization_audit(compound_audit_path)
    errors = _validate_base(frame, audit)
    if errors:
        raise QualityContractError("Historical matrix quality contract failed: " + "; ".join(errors))
    return frame, audit, manifest, input_hash, audit_hash


def run(input_path: Path, manifest_path: Path, compound_audit_path: Path,
        cache_root: Path, output_dir: Path) -> dict[str, Any]:
    input_path, manifest_path, compound_audit_path = map(
        lambda path: path.resolve(), (input_path, manifest_path, compound_audit_path)
    )
    cache_root, output_dir = cache_root.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing quality output: {output_dir}")
    frame, audit, input_manifest, input_hash, audit_hash = load_and_validate_input(
        input_path, manifest_path, compound_audit_path
    )
    register = create_quality_register(frame, audit)
    variable_summary = variable_quality_summary(frame)
    telemetry_review, cache_hashes = review_telemetry_source_cases(frame, cache_root)
    sector_review, sector_hash = review_sector_source(frame, cache_root)
    cache_hashes[case_path_key(SECTOR_CACHE)] = sector_hash
    clock_review, clock_hashes = review_clock_boundaries(frame, cache_root)
    cache_hashes.update(clock_hashes)
    register_path_names = (
        "quality_register.csv", "variable_quality_summary.csv",
        "source_case_review.csv", "sector_source_review.csv", "clock_boundary_review.csv",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        register.to_csv(temp_dir / register_path_names[0], index=False)
        variable_summary.to_csv(temp_dir / register_path_names[1], index=False)
        telemetry_review.to_csv(temp_dir / register_path_names[2], index=False)
        sector_review.to_csv(temp_dir / register_path_names[3], index=False)
        clock_review.to_csv(temp_dir / register_path_names[4], index=False)
        register_counts = {
            str(key): int(value)
            for key, value in register["reason"].value_counts().sort_index().items()
        }
        clock_summary = clock_review.groupby("race", sort=True).agg(
            transitions=("restart_match", "size"), restart_matches=("restart_match", "sum")
        )
        clock_summary_lines = [
            f"- {race}: {int(row.restart_matches)}/{int(row.transitions)} solapamientos coinciden "
            "temporalmente con Started posterior a Aborted."
            for race, row in clock_summary.iterrows()
        ]
        reconciliation = {
            "input_path": str(input_path), "input_sha256": input_hash,
            "compound_audit_sha256": audit_hash,
            "rows": int(len(frame)), "columns": int(len(frame.columns)),
            "full_key_rows_null": int(frame[KEY].isna().any(axis=1).sum()),
            "full_key_duplicates": int(frame.duplicated(KEY).sum()),
            "rows_added_or_removed": 0, "columns_added_or_removed": 0,
            "matrix_written": False,
            "compound_nulls_and_audit_rows": int(len(audit)),
            "register_rows": int(len(register)), "register_by_reason": register_counts,
            "telemetry_contract_errors": [],
            "position_missing_equals_generated": True,
            "speedfl_missing_values_within_pitin_or_generated_context": True,
            "pitin_rows_with_speedfl_present_flagged": int((frame["PitInTime"].notna() & frame["SpeedFL"].notna()).sum()),
            "throttle_contract_valid": True,
            "sector_discrepancies": int(sector_review.iloc[0]["mismatch_rows"]),
            "sector_source_exact_matches": int(sector_review.iloc[0]["exact_field_matches"]),
            "clock_overlap_transitions": int(len(clock_review)),
            "clock_restart_matches": int(clock_review["restart_match"].sum()),
            "clock_restart_unmatched": int((~clock_review["restart_match"]).sum()),
        }
        (temp_dir / "reconciliation.json").write_text(
            json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report = [
            "# Registro de calidad de la matriz histórica",
            "",
            f"- Entrada publicada: `{input_path}`",
            f"- SHA-256 entrada: `{input_hash}`",
            f"- Filas/columnas: {len(frame):,}/{len(frame.columns)}; claves nulas/duplicadas: 0/0.",
            f"- Filas de registro: {len(register):,}; razones distintas: {len(register_counts)}.",
            f"- Compound: 780 ausencias enlazadas a auditoría (426 literal `None`, 354 `nan`).",
            "- TyreLife/Stint: se conservan 989/354 faltantes; no se imputan.",
            f"- SpeedFL: los faltantes quedan dentro del contexto PitInTime/filas generadas; {(frame['PitInTime'].notna() & frame['SpeedFL'].notna()).sum()} entradas PitIn conservan SpeedFL y quedan señaladas. Position falta exactamente en filas generadas.",
            "- Acelerador filtrado: validación de conteos, fracción, NA y rango 0–100 aprobada.",
            "- Valores brutos, ceros, casos generados y excepciones temporales se registran sin modificarlos.",
            f"- Discrepancias LapTime/sectores: {int(sector_review.iloc[0]['mismatch_rows'])}; coincidencia exacta con caché extendida: {int(sector_review.iloc[0]['exact_field_matches'])}.",
            f"- Solapamientos Time/LapStartTime: {len(clock_review)} transiciones; "
            f"{int(clock_review['restart_match'].sum())} coinciden con un reinicio temporalmente pertinente.",
            *clock_summary_lines,
            "- Los timestamps se conservan. Un cierre de vuelta posterior al inicio de decisión no puede entrar en el historial causal.",
            "- No se escribió otra matriz, no se alteraron etiquetas y no se cambió ningún valor.",
            "- Los casos sin fuente independiente suficiente quedan preservados y señalados; no se aplican reparaciones por intuición.",
            "",
            "## Política temporal",
            "",
            "`Time` de FastF1 respalda replay histórico por tiempo de evento, no prueba latencia de recepción en vivo. Clima, estados agregados y resúmenes completos de vuelta siguen siendo históricos; su disponibilidad as-of se construirá después.",
            "",
        ]
        (temp_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        source_hashes_after = {str(path): sha256(path) for path in [input_path, compound_audit_path]}
        if source_hashes_after[str(input_path)] != input_hash or source_hashes_after[str(compound_audit_path)] != audit_hash:
            raise QualityContractError("Published matrix or Compound audit changed during quality review")
        for relative, initial_hash in cache_hashes.items():
            if sha256(cache_root / relative.removeprefix("cache:")) != initial_hash:
                raise QualityContractError(f"Local source cache changed during review: {relative}")
        output_hashes = {name: sha256(temp_dir / name) for name in register_path_names + ("reconciliation.json", "report.md")}
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                      text=True, check=True).stdout.strip()
        except Exception:
            revision = None
        manifest = {
            "quality_register_version": RULE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(input_path), "sha256": input_hash,
                      "normalization_manifest_path": str(manifest_path),
                      "normalization_manifest_sha256": sha256(manifest_path),
                      "compound_audit_path": str(compound_audit_path),
                      "compound_audit_sha256": audit_hash,
                      "master_source_sha256": input_manifest["source"]["sha256_after"],
                      "rows": len(frame), "columns": len(frame.columns)},
            "caches_read_only": cache_hashes,
            "matrix_output_written": False,
            "rules": {"exact_markers_only": ["None", "nan"], "no_imputation": True,
                      "no_row_deletion": True, "no_value_repair": True},
            "counts": {"register_rows": len(register), "register_by_reason": register_counts,
                       "variables_profiled": len(variable_summary)},
            "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
            "code": {"git_revision": revision, "script_sha256": sha256(Path(__file__).resolve())},
            "outputs": output_hashes,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing quality output: {output_dir}")
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--compound-audit", type=Path, default=DEFAULT_COMPOUND_AUDIT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.input, args.manifest, args.compound_audit, args.cache_root, args.output_dir)
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir.resolve()),
                      "rows": result["input"]["rows"], "register_rows": result["counts"]["register_rows"],
                      "matrix_written": result["matrix_output_written"]}, indent=2))


if __name__ == "__main__":
    main()
