"""Create a new historical matrix with only exact Compound markers normalized.

This is a representation-only transformation, not a causal feature builder.
The master pickle is read-only; each transformed row is recorded in a reversible
audit keyed by the complete driver-lap key.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from auditar_calidad_matriz import DEFAULT_SOURCE, KEY, sha256


DEFAULT_EVIDENCE = ROOT / "docs/aclaraciones/evidencia_compound_v1/summary.json"
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent / "vueltas_2022_2025_compound_normalizado_v1"
RULE_VERSION = "compound_null_markers_v1"
RULES = {
    "None": "literal_none_in_consolidated",
    "nan": "literal_nan_in_consolidated",
}


class EvidenceValidationError(ValueError):
    """The supplied evidence does not validate the selected input matrix."""


def _evidence_hash_for_source(evidence: Any, source: Path) -> str:
    if not isinstance(evidence, dict):
        raise EvidenceValidationError("Evidence summary must be a JSON object; review or regenerate it.")
    hashes = evidence.get("input_sha256")
    if not isinstance(hashes, dict):
        raise EvidenceValidationError(
            "Evidence is missing the input_sha256 mapping; review or regenerate evidence for this source."
        )
    candidates = [str(source)]
    if source.is_relative_to(ROOT):
        candidates.insert(0, source.relative_to(ROOT).as_posix())
    expected_hash = next((hashes[key] for key in candidates if key in hashes), None)
    if expected_hash is None:
        raise EvidenceValidationError(
            f"Evidence has no SHA-256 entry for selected source {source}; "
            "review or regenerate evidence for this source."
        )
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise EvidenceValidationError(
            f"Evidence SHA-256 for {source} is invalid; review or regenerate the evidence."
        )
    return expected_hash.lower()


def _expected_marker_counts(evidence: Any) -> dict[str, int]:
    markers = evidence.get("markers") if isinstance(evidence, dict) else None
    if not isinstance(markers, dict):
        raise EvidenceValidationError(
            'Evidence is missing expected counts for "None" and "nan"; '
            "review or regenerate the evidence."
        )
    expected = {}
    for marker in RULES:
        if marker not in markers:
            raise EvidenceValidationError(
                f"Evidence is missing the expected count for exact literal {marker!r}; "
                "review or regenerate the evidence."
            )
        value = markers.get(marker)
        if type(value) is not int or value < 0:
            raise EvidenceValidationError(
                f"Evidence count for exact literal {marker!r} must be a non-negative integer; "
                "review or regenerate the evidence."
            )
        expected[marker] = value
    return expected


def normalize_compound_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a copy with exact marker strings changed to nullable string NA.

    No trimming, case-folding, Python-null coercion, reconstruction, or changes
    to other columns are performed.
    """
    missing = [column for column in [*KEY, "Compound"] if column not in frame]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if frame[KEY].isna().any(axis=1).any():
        raise ValueError("Null values in the full matrix key; refusing transformation")
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate values in the full matrix key; refusing transformation")

    original = frame["Compound"]
    non_null_non_string = original.dropna().map(lambda value: not isinstance(value, str))
    if non_null_non_string.any():
        raise TypeError("Compound contains non-string non-null values; refusing broad coercion")

    # pandas StringDtype provides nullable semantics while preserving exact text.
    normalized = original.astype("string")
    marker_mask = normalized.isin(RULES.keys())
    result = frame.copy(deep=True)
    result["Compound"] = normalized.mask(marker_mask, pd.NA)

    audit = frame.loc[marker_mask, KEY].copy().reset_index(drop=True)
    originals = original.loc[marker_mask].reset_index(drop=True)
    audit["original_literal"] = originals.astype("string")
    audit["resulting_value"] = pd.Series(pd.NA, index=audit.index, dtype="string")
    audit["reason"] = audit["original_literal"].map(RULES).astype("string")
    audit["rule_version"] = RULE_VERSION
    return result, audit


def restore_compound_from_audit(
    normalized: pd.DataFrame, audit: pd.DataFrame
) -> pd.DataFrame:
    """Restore transformed literal values using the full key in the audit."""
    required = [*KEY, "original_literal", "rule_version"]
    missing = [column for column in required if column not in audit]
    if missing:
        raise ValueError(f"Audit is missing required columns: {missing}")
    if normalized[KEY].isna().any(axis=1).any() or normalized.duplicated(KEY).any():
        raise ValueError("Normalized matrix key must be complete and unique")
    if audit[KEY].isna().any(axis=1).any() or audit.duplicated(KEY).any():
        raise ValueError("Audit keys must be complete and unique")
    if not audit["rule_version"].eq(RULE_VERSION).all():
        raise ValueError("Audit contains an unsupported rule version")
    keys = pd.MultiIndex.from_frame(normalized[KEY])
    lookup = audit.set_index(KEY)["original_literal"]
    audit_keys = pd.MultiIndex.from_frame(audit[KEY])
    positions = keys.get_indexer(audit_keys)
    if (positions < 0).any():
        raise ValueError("Audit contains keys absent from the normalized matrix")
    result = normalized.copy(deep=True)
    restored = result["Compound"].astype("string").copy()
    for pos, value in zip(positions, lookup.to_numpy()):
        if not pd.isna(restored.iloc[pos]):
            raise ValueError("Audit points to a row whose normalized Compound is not null")
        if value not in RULES:
            raise ValueError(f"Audit contains an unsupported original literal: {value!r}")
        restored.iloc[pos] = value
    # Restore the source column's conventional object representation. The
    # normalized artifact itself intentionally uses nullable StringDtype.
    result["Compound"] = restored.astype(object)
    return result


def load_normalized_matrix(path: Path) -> pd.DataFrame:
    """Load the pickle and reapply logical nullable-string dtype.

    Pandas 2.3.3 in the current Python 3.14 environment cannot unpickle its
    StringArray payload (NotImplementedError). The persisted pickle therefore
    uses an object-backed Compound column containing pd.NA; this loader restores
    the declared logical StringDtype. Other columns are read without coercion.
    """
    frame = pd.read_pickle(path)
    frame["Compound"] = frame["Compound"].astype("string")
    return frame


def load_normalization_audit(path: Path) -> pd.DataFrame:
    """Read the CSV audit without pandas' default NA parsing of literal ``None``."""
    audit = pd.read_csv(path, keep_default_na=False, dtype={"DriverNumber": "string"})
    for column in ("original_literal", "resulting_value", "reason", "rule_version"):
        if column in audit:
            audit[column] = audit[column].astype("string")
    return audit


def _profile(frame: pd.DataFrame) -> dict[str, Any]:
    compound = frame["Compound"]
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "compound_dtype": str(compound.dtype),
        "compound_nulls": int(compound.isna().sum()),
        "compound_values": {
            str(key): int(value)
            for key, value in compound.value_counts(dropna=False).items()
        },
        "key_null_rows": int(frame[KEY].isna().any(axis=1).sum()),
        "duplicate_key_rows": int(frame.duplicated(KEY, keep=False).sum()),
    }


def run_normalization(source: Path, output_dir: Path, evidence_path: Path) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    evidence_path = evidence_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    try:
        evidence_hash_before = sha256(evidence_path)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(
            f"Could not read valid evidence JSON at {evidence_path}; review or regenerate the evidence."
        ) from error
    expected_hash = _evidence_hash_for_source(evidence, source)
    actual_hash_before = sha256(source)
    hash_matches_evidence = actual_hash_before == expected_hash
    if not hash_matches_evidence:
        raise EvidenceValidationError(
            f"Input SHA-256 does not match the evidence for {source} "
            f"(expected {expected_hash}, observed {actual_hash_before}); "
            "review or regenerate evidence for this source."
        )

    expected_counts = _expected_marker_counts(evidence)

    frame = pd.read_pickle(source)
    if "Compound" not in frame:
        raise EvidenceValidationError(
            f"Selected source {source} has no Compound column; review or regenerate evidence for this source."
        )
    observed_counts = {marker: int(frame["Compound"].eq(marker).sum()) for marker in RULES}
    for marker in RULES:
        if observed_counts[marker] != expected_counts[marker]:
            raise EvidenceValidationError(
                f"Observed count for exact literal {marker!r} is {observed_counts[marker]}, "
                f"but evidence expects {expected_counts[marker]} for {source}; "
                "review or regenerate evidence for this source."
            )

    before = _profile(frame)
    transformed, audit = normalize_compound_frame(frame)
    after = _profile(transformed)
    counts = {marker: int(audit["original_literal"].eq(marker).sum()) for marker in RULES}

    # Strict before/after comparison: only Compound may differ; row order and all
    # other values (including their dtypes) must remain identical.
    if len(transformed) != len(frame) or list(transformed.columns) != list(frame.columns):
        raise AssertionError("Rows or columns changed during normalization")
    for column in frame.columns:
        if column == "Compound":
            continue
        pd.testing.assert_series_equal(
            frame[column], transformed[column], check_names=True, check_dtype=True,
            check_exact=True,
        )
    if not frame[KEY].equals(transformed[KEY]):
        raise AssertionError("Key values or row order changed")
    if counts != {"None": before["compound_values"].get("None", 0),
                  "nan": before["compound_values"].get("nan", 0)}:
        raise AssertionError("Audit counts do not reconcile with source marker counts")

    restored = restore_compound_from_audit(transformed, audit)
    pd.testing.assert_series_equal(frame["Compound"], restored["Compound"], check_dtype=False)
    second_pass, second_audit = normalize_compound_frame(transformed)
    pd.testing.assert_frame_equal(transformed, second_pass, check_dtype=True)
    if not second_audit.empty:
        raise AssertionError("Normalization is not idempotent")

    evidence_hash_after_validation = sha256(evidence_path)
    if evidence_hash_before != evidence_hash_after_validation:
        raise EvidenceValidationError(
            f"Evidence changed during validation at {evidence_path}; "
            "review or regenerate the evidence before retrying."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        matrix_path = temp_dir / "vueltas_2022_2025_compound_normalizado.pkl"
        audit_path = temp_dir / "compound_normalization_audit.csv"
        reconciliation_path = temp_dir / "reconciliation.json"
        report_path = temp_dir / "report.md"
        manifest_path = temp_dir / "manifest.json"
        # Python 3.14 + installed pandas 2.3.3 cannot unpickle StringArray state.
        # Store the nullable values as object in the pickle and provide the
        # loader above to restore the logical nullable StringDtype on read.
        storage_frame = transformed.copy(deep=True)
        storage_frame["Compound"] = storage_frame["Compound"].astype(object)
        storage_frame.to_pickle(matrix_path)
        reloaded = load_normalized_matrix(matrix_path)
        pd.testing.assert_frame_equal(transformed, reloaded, check_dtype=True)
        audit.to_csv(audit_path, index=False)

        source_hash_after = sha256(source)
        if actual_hash_before != source_hash_after:
            raise AssertionError("Source hash changed during normalization")
        if sha256(evidence_path) != evidence_hash_before:
            raise EvidenceValidationError(
                f"Evidence changed before publishing output from {evidence_path}; "
                "review or regenerate the evidence before retrying."
            )

        reconciliation = {
            "source_sha256": actual_hash_before,
            "source_sha256_after": source_hash_after,
            "source_unchanged": actual_hash_before == source_hash_after,
            "evidence_expected_source_sha256": expected_hash,
            "source_hash_matches_evidence": hash_matches_evidence,
            "expected_counts_from_evidence": expected_counts,
            "observed_counts": counts,
            "before": before,
            "after": after,
            "conversions": {
                "literal_none_in_consolidated": counts["None"],
                "literal_nan_in_consolidated": counts["nan"],
                "total": int(len(audit)),
            },
            "rows_added": 0,
            "rows_removed": 0,
            "columns_added": 0,
            "columns_removed": 0,
            "changed_columns": ["Compound"] if len(audit) else [],
            "other_columns_unchanged": True,
            "row_order_and_keys_unchanged": True,
            "normalization_idempotent": True,
            "audit_reversible": True,
        }
        reconciliation_path.write_text(
            json.dumps(reconciliation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report_path.write_text(
            "# Reporte de normalización histórica de Compound\n\n"
            f"- Fuente: `{source}`\n"
            f"- SHA-256: `{actual_hash_before}`\n"
            f"- Coincide con evidencia v1: `{hash_matches_evidence}`\n"
            f"- Fuente maestra seleccionada: `{source == DEFAULT_SOURCE.resolve()}`\n"
            f"- Filas: {len(frame):,}; columnas: {len(frame.columns)}\n"
            f"- Conversiones: `None` → NA: {counts['None']}; `nan` → NA: {counts['nan']}; "
            f"total: {len(audit)}\n"
            "- Cambió únicamente `Compound`; no se reconstruyeron compuestos ni se tocaron etiquetas.\n"
            "- Para este entorno (Python 3.14 + pandas 2.3.3), lea el pickle mediante "
            "`load_normalized_matrix()` para restaurar el dtype nullable `string`; el pickle "
            "guarda `Compound` como object con valores `pd.NA` por compatibilidad.\n"
            "- Lea el CSV de auditoría con `load_normalization_audit()` para no interpretar "
            "el literal original `None` como nulo CSV.\n"
            "- Esta es una matriz histórica normalizada, no una matriz de características aprobada "
            "para entrenar ni un reemplazo del replay causal `Compound_asof`.\n",
            encoding="utf-8",
        )
        output_hashes = {
            path.name: sha256(path)
            for path in (matrix_path, audit_path, reconciliation_path, report_path)
        }
        try:
            git_revision = __import__("subprocess").run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except Exception:
            git_revision = None
        manifest = {
            "normalization_version": RULE_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(source), "sha256_before": actual_hash_before,
                "sha256_after": source_hash_after, "unchanged": actual_hash_before == source_hash_after,
                "bytes": source.stat().st_size, "format": "pandas_pickle",
                "rows": int(len(frame)), "columns": int(len(frame.columns)),
                "schema_columns_in_order": list(frame.columns),
                "schema_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                "expected_evidence_sha256": expected_hash,
                "hash_matches_evidence": hash_matches_evidence,
                "evidence_path": str(evidence_path),
                "evidence_sha256": evidence_hash_before,
                "source_is_master_only": source == DEFAULT_SOURCE.resolve(),
            },
            "unit_of_observation": "one driver-lap record in one race",
            "key": KEY,
            "configuration": {
                "column": "Compound", "exact_literals": RULES,
                "result": "pd.NA", "logical_result_dtype": "pandas StringDtype",
                "pickle_storage_dtype": "object (pd.NA values); use load_normalized_matrix() to restore StringDtype",
                "no_trim_or_casefold": True, "no_reconstruction": True,
            },
            "counts": {"expected_from_evidence": expected_counts,
                       "observed_before": observed_counts,
                       "before": before, "after": after, "converted": counts,
                       "total_converted": int(len(audit))},
            "limitations": [
                "Historical normalized matrix; not an approved training feature matrix.",
                "Compound_asof remains a separate causal replay contract with UNKNOWN when no eligible information exists.",
                "FastF1 Time supports historical event-time replay but does not prove live receipt latency.",
                "This rule does not apply to partial timing_app_data updates; preserve known state within the active stint and never inherit across stints.",
                "Stint_asof and FreshTyre_asof remain separate proposals.",
            ],
            "software": {"python": platform.python_version(), "pandas": pd.__version__},
            "code": {"git_revision": git_revision, "script_sha256": sha256(Path(__file__).resolve()),
                     "cleaning_spec_sha256": sha256(ROOT / "docs/specs/04_limpieza_candidata.md"),
                     "decision_matrix_sha256": sha256(ROOT / "docs/analisis/20_matriz_decisiones_calidad_datos.md")},
            "outputs": output_hashes,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-summary", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    manifest = run_normalization(args.input, args.output_dir, args.evidence_summary)
    print(json.dumps({
        "status": "passed", "source_sha256": manifest["source"]["sha256_before"],
        "source_hash_matches_evidence": manifest["source"]["hash_matches_evidence"],
        "rows": manifest["source"]["rows"], "converted": manifest["counts"]["converted"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
