"""Create a read-only, reproducible quality audit for the consolidated matrix.

No rows or values are changed. The output is a new audit directory containing
provenance, column profiling, per-driver race coverage, and review exceptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "EDA/dataset/datos_carreras_completos_2022_2025/vueltas_2022_2025.pkl"
DEFAULT_OUTPUT = ROOT / "EDA/dataset/auditorias_limpieza/limpieza_conservadora_v1_3"
KEY = ["Temporada", "Ronda", "DriverNumber", "LapNumber"]
GROUP = ["Temporada", "Ronda", "GranPremio", "DriverNumber"]
EXPECTED_COLUMNS = [
    "Temporada", "Ronda", "GranPremio", "Time", "Driver", "DriverNumber",
    "LapTime", "LapNumber", "Stint", "PitOutTime", "PitInTime", "Sector1Time",
    "Sector2Time", "Sector3Time", "Sector1SessionTime", "Sector2SessionTime",
    "Sector3SessionTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "IsPersonalBest", "Compound", "TyreLife", "FreshTyre", "Team", "LapStartTime",
    "LapStartDate", "TrackStatus", "Position", "Deleted", "DeletedReason",
    "FastF1Generated", "IsAccurate", "SafetyCar", "VirtualSafetyCar", "Weather_Time",
    "Weather_AirTemp", "Weather_Humidity", "Weather_Pressure", "Weather_Rainfall",
    "Weather_TrackTemp", "Weather_WindDirection", "Weather_WindSpeed", "Telemetry_Samples",
    "Telemetry_Speed_mean", "Telemetry_Speed_max", "Telemetry_RPM_mean", "Telemetry_RPM_max",
    "Telemetry_Throttle_mean", "Telemetry_Throttle_max", "Telemetry_Brake_mean",
    "Telemetry_Brake_max", "Telemetry_nGear_mean", "Telemetry_nGear_max",
    "Telemetry_DRS_mean", "Telemetry_DRS_max", "Telemetry_Throttle_valid_samples",
    "Telemetry_Throttle_valid_fraction", "Telemetry_Throttle_valid_mean",
    "Telemetry_Throttle_valid_max",
]
EXPECTED_CORE_DTYPES = {
    "Temporada": "int64", "Ronda": "int64", "DriverNumber": "object",
    "LapNumber": "float64", "Time": "timedelta64[ns]",
    "PitInTime": "timedelta64[ns]", "PitOutTime": "timedelta64[ns]",
    "Compound": "object", "TyreLife": "float64", "FreshTyre": "bool",
    "FastF1Generated": "bool",
}
SENTINELS = {"", "NONE", "NAN", "UNKNOWN", "<NA>"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_structure(frame: pd.DataFrame) -> dict:
    """Return structural failures without mutating the input."""
    missing_columns = sorted(set(KEY) - set(frame.columns))
    if missing_columns:
        return {"missing_key_columns": missing_columns, "null_key_rows": None,
                "duplicate_key_rows": None}
    return {
        "missing_key_columns": [],
        "null_key_rows": int(frame[KEY].isna().any(axis=1).sum()),
        "duplicate_key_rows": int(frame.duplicated(KEY, keep=False).sum()),
    }


def validate_core_dtypes(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    mismatches = {}
    for column, expected in EXPECTED_CORE_DTYPES.items():
        if column not in frame:
            continue
        observed = str(frame[column].dtype)
        if observed != expected:
            mismatches[column] = {"expected": expected, "observed": observed}
    return mismatches


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in frame.columns:
        series = frame[column]
        record = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "null_count": int(series.isna().sum()),
            "null_fraction": float(series.isna().mean()) if len(series) else 0.0,
            "distinct_non_null": int(series.nunique(dropna=True)),
            "zero_count": None,
            "negative_count": None,
            "minimum": None,
            "maximum": None,
        }
        if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            numeric = pd.to_numeric(series, errors="coerce")
            record.update({
                "zero_count": int(numeric.eq(0).sum()),
                "negative_count": int(numeric.lt(0).sum()),
                "minimum": None if numeric.dropna().empty else float(numeric.min()),
                "maximum": None if numeric.dropna().empty else float(numeric.max()),
            })
        records.append(record)
    return pd.DataFrame.from_records(records)


def coverage_by_driver(frame: pd.DataFrame) -> pd.DataFrame:
    missing_group = sorted(set(GROUP) - set(frame.columns))
    if missing_group or "LapNumber" not in frame:
        raise ValueError(f"Missing coverage columns: {missing_group or ['LapNumber']}")
    ordered = frame.sort_values(GROUP + ["LapNumber"], kind="stable").copy()
    ordered["_lap_delta"] = ordered.groupby(GROUP, dropna=False)["LapNumber"].diff()
    ordered["_gap_count"] = ordered["_lap_delta"].gt(1)
    ordered["_generated_row"] = (
        ordered["FastF1Generated"].fillna(False).astype(bool)
        if "FastF1Generated" in ordered else False
    )
    return ordered.groupby(GROUP, dropna=False, as_index=False).agg(
        row_count=("LapNumber", "size"),
        lap_min=("LapNumber", "min"),
        lap_max=("LapNumber", "max"),
        null_lap_count=("LapNumber", lambda values: int(values.isna().sum())),
        gap_count=("_gap_count", "sum"),
        max_observed_gap=("_lap_delta", "max"),
        generated_rows=("_generated_row", "sum"),
    )


def missingness_by_driver(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-race/driver null counts, keeping fields and groups intact."""
    value_columns = [column for column in frame.columns if column not in GROUP]
    grouped = frame.groupby(GROUP, dropna=False)[value_columns].agg(
        lambda values: int(values.isna().sum())
    )
    return grouped.reset_index()


def categorical_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = [column for column in ("GranPremio", "Driver", "DriverNumber", "Team",
                                         "Compound", "TrackStatus") if column in frame]
    records = []
    for column in candidates:
        values = frame[column].astype("string")
        counts = values.value_counts(dropna=False, sort=True)
        for value, count in counts.items():
            records.append({"column": column,
                            "value": "<NULL>" if pd.isna(value) else f"literal::{value}",
                            "count": int(count)})
    return pd.DataFrame.from_records(records, columns=["column", "value", "count"])


def exception_register(frame: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    records = []
    structure = validate_structure(frame)
    for kind, count in (("null_sample_key", structure["null_key_rows"]),
                        ("duplicate_sample_key", structure["duplicate_key_rows"])):
        if count:
            records.append({"exception_type": kind, "column": ",".join(KEY),
                            "affected_count": count, "status": "integrity_gate_failure",
                            "treatment": "stop_and_review; no row selected or removed"})
    gaps = int(coverage["gap_count"].sum())
    if gaps:
        records.append({"exception_type": "lap_number_gap", "column": "LapNumber",
                        "affected_count": gaps, "status": "audit_only",
                        "treatment": "preserve rows; report by race and driver; no lap invented"})
    if "Compound" in frame:
        values = frame["Compound"].astype("string").str.strip().str.upper()
        for token, count in values.loc[values.isin(SENTINELS)].value_counts().items():
            records.append({"exception_type": "compound_text_sentinel", "column": "Compound",
                            "affected_count": int(count), "status": "domain_decision_pending",
                            "treatment": f"preserve literal {token!r}; do not coerce to null"})
    for column in ("PitInTime", "PitOutTime"):
        if column in frame:
            count = int(frame[column].isna().sum())
            records.append({"exception_type": "event_field_missing", "column": column,
                            "affected_count": count, "status": "preserve_as_unrecorded",
                            "treatment": "do not impute; missingness alone does not prove event absence"})
    if "FastF1Generated" in frame:
        count = int(frame["FastF1Generated"].fillna(False).astype(bool).sum())
        records.append({"exception_type": "fastf1_generated_row", "column": "FastF1Generated",
                        "affected_count": count, "status": "preserve_audit_only",
                        "treatment": "retain; do not equate automatically with retirement or a complete lap"})
    return pd.DataFrame.from_records(records, columns=[
        "exception_type", "column", "affected_count", "status", "treatment"
    ])


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_audit(source: Path, output: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit run: {output}")

    source_hash = sha256(source)
    frame = pd.read_pickle(source)
    structure = validate_structure(frame)
    dtype_mismatches = validate_core_dtypes(frame)
    column_profile = profile_columns(frame)
    coverage = coverage_by_driver(frame)
    missingness = missingness_by_driver(frame)
    categories = categorical_distributions(frame)
    exceptions = exception_register(frame, coverage)
    missing_expected = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    unexpected = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    errors = []
    if structure["missing_key_columns"]:
        errors.append(f"missing key columns: {structure['missing_key_columns']}")
    if structure["null_key_rows"]:
        errors.append(f"rows with null sample key: {structure['null_key_rows']}")
    if structure["duplicate_key_rows"]:
        errors.append(f"rows participating in duplicate sample keys: {structure['duplicate_key_rows']}")
    if missing_expected:
        errors.append(f"missing columns from expected source schema: {missing_expected}")
    if unexpected:
        errors.append(f"unexpected columns from expected source schema: {unexpected}")
    if dtype_mismatches:
        errors.append(f"core dtype mismatches: {dtype_mismatches}")

    output.mkdir(parents=True)
    column_profile.to_csv(output / "column_profile.csv", index=False)
    coverage.to_csv(output / "coverage_by_race_driver.csv", index=False)
    missingness.to_csv(output / "missingness_by_race_driver.csv", index=False)
    categories.to_csv(output / "categorical_distributions.csv", index=False)
    exceptions.to_csv(output / "exceptions.csv", index=False)
    (output / "reconciliation.json").write_text(json.dumps({
        "input": {"path": str(source.resolve()), "sha256": source_hash,
                  "rows": int(len(frame)), "columns": int(len(frame.columns)),
                  "key_null_rows": structure.get("null_key_rows"),
                  "duplicate_key_rows": structure.get("duplicate_key_rows")},
        "output_matrix": None,
        "rows_added": 0,
        "rows_removed": 0,
        "columns_added": 0,
        "columns_removed": 0,
        "values_changed": 0,
        "reason": "No value-level transformation is authorized by the current domain decisions.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_hash_after = sha256(source)
    if source_hash_after != source_hash:
        errors.append("source file hash changed during audit")
    output_hashes = {
        name: sha256(output / name)
        for name in (
            "column_profile.csv", "coverage_by_race_driver.csv",
            "missingness_by_race_driver.csv", "categorical_distributions.csv",
            "exceptions.csv", "reconciliation.json",
        )
    }
    manifest = {
        "audit_version": "cleaning_audit_v1.3",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source.resolve()),
        "source_sha256": source_hash,
        "source_sha256_after": source_hash_after,
        "source_unchanged": source_hash_after == source_hash,
        "source_bytes": source.stat().st_size,
        "source_retrieval_time": None,
        "source_retrieval_time_note": "Not recorded; filesystem modification time is not treated as acquisition time.",
        "source_format": "pandas_pickle",
        "source_rows": int(len(frame)),
        "source_columns": int(len(frame.columns)),
        "schema_columns_in_order": list(frame.columns),
        "schema_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "expected_schema_missing_columns": missing_expected,
        "unexpected_columns": unexpected,
        "unit_of_observation": "one driver-lap record in one race",
        "expected_key": KEY,
        "structural_validation": structure,
        "core_dtype_validation": {"expected": EXPECTED_CORE_DTYPES,
                                  "mismatches": dtype_mismatches},
        "races": int(frame[["Temporada", "Ronda"]].drop_duplicates().shape[0]),
        "driver_race_sequences": int(coverage.shape[0]),
        "seasons": {str(k): int(v) for k, v in frame["Temporada"].value_counts().sort_index().items()},
        "transformations_applied": [],
        "rows_changed": 0,
        "columns_changed": 0,
        "data_matrix_written": False,
        "reason_no_cleaned_matrix": "Domain decisions remain open for text sentinels and temporal feature availability.",
        "time_assumption": "FastF1 Time supports historical event-time replay only; it does not establish live receipt latency.",
        "rules_version": "SPEC-CLEAN-001 draft; conservative audit profile only",
        "audit_script_sha256": sha256(Path(__file__).resolve()),
        "cleaning_spec_sha256": sha256(ROOT / "docs/specs/04_limpieza_candidata.md"),
        "output_artifact_sha256": output_hashes,
        "git_revision": git_revision(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "validation_errors": errors,
        "validation_status": "failed" if errors else "passed",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise ValueError("Quality audit failed: " + "; ".join(errors))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = run_audit(args.source.resolve(), args.output.resolve())
    print(json.dumps({
        "status": manifest["validation_status"],
        "source_sha256": manifest["source_sha256"],
        "rows": manifest["source_rows"],
        "columns": manifest["source_columns"],
        "audit_directory": str(args.output.resolve()),
        "transformations_applied": manifest["transformations_applied"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
