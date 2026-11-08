"""Focused tests for the approved Compound literal normalization contract."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.normalizar_compound import (
    EvidenceValidationError,
    RULE_VERSION,
    load_normalization_audit,
    load_normalized_matrix,
    normalize_compound_frame,
    restore_compound_from_audit,
    run_normalization,
)


def sample_frame():
    return pd.DataFrame({
        "Temporada": [2022, 2022, 2022, 2022, 2022, 2022],
        "Ronda": [1] * 6,
        "DriverNumber": ["1", "1", "1", "1", "1", "1"],
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "Compound": ["None", "nan", "SOFT", "UNKNOWN", "TEST_UNKNOWN", pd.NA],
        "Other": [7, 8, 9, 10, 11, 12],
    })


class CompoundNormalizationTests(unittest.TestCase):
    def _assert_evidence_rejected_without_output(self, evidence_mutator, message, frame=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pkl"
            source_frame = frame if frame is not None else sample_frame()
            source_frame.to_pickle(source)
            evidence = self._consistent_evidence(source)
            evidence_mutator(evidence)
            evidence_path = root / "summary.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            output_parent = root / "not-created"
            output = output_parent / "matrix-output"
            with self.assertRaisesRegex(EvidenceValidationError, message):
                run_normalization(source, output, evidence_path)
            self.assertFalse(output.exists())
            self.assertFalse(output_parent.exists())
            self.assertEqual(list(root.rglob("manifest.json")), [])

    @staticmethod
    def _consistent_evidence(source):
        source_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        frame = pd.read_pickle(source)
        return {
            "input_sha256": {str(source.resolve()): source_hash},
            "markers": {marker: int(frame["Compound"].eq(marker).sum())
                        for marker in ("None", "nan")},
        }

    def test_exact_markers_become_nullable_na_with_reason(self):
        result, audit = normalize_compound_frame(sample_frame())
        self.assertEqual(str(result["Compound"].dtype), "string")
        self.assertTrue(pd.isna(result.loc[0, "Compound"]))
        self.assertTrue(pd.isna(result.loc[1, "Compound"]))
        self.assertEqual(audit["reason"].tolist(), [
            "literal_none_in_consolidated", "literal_nan_in_consolidated"
        ])
        self.assertEqual(audit["rule_version"].unique().tolist(), [RULE_VERSION])
        self.assertEqual(audit["original_literal"].tolist(), ["None", "nan"])

    def test_valid_categories_and_preexisting_null_are_preserved(self):
        frame = sample_frame()
        result, audit = normalize_compound_frame(frame)
        self.assertEqual(result.loc[2, "Compound"], "SOFT")
        self.assertEqual(result.loc[3, "Compound"], "UNKNOWN")
        self.assertEqual(result.loc[4, "Compound"], "TEST_UNKNOWN")
        self.assertTrue(pd.isna(result.loc[5, "Compound"]))
        self.assertEqual(len(audit), 2)

    def test_only_compound_changes_and_structure_is_preserved(self):
        frame = sample_frame()
        result, _ = normalize_compound_frame(frame)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.columns.tolist(), frame.columns.tolist())
        pd.testing.assert_frame_equal(result[frame.columns.drop("Compound")],
                                      frame[frame.columns.drop("Compound")])
        pd.testing.assert_frame_equal(result[ ["Temporada", "Ronda", "DriverNumber", "LapNumber"] ],
                                      frame[ ["Temporada", "Ronda", "DriverNumber", "LapNumber"] ])

    def test_audit_reverses_transform_and_second_run_is_idempotent(self):
        frame = sample_frame()
        result, audit = normalize_compound_frame(frame)
        restored = restore_compound_from_audit(result, audit)
        pd.testing.assert_series_equal(restored["Compound"], frame["Compound"])
        second, second_audit = normalize_compound_frame(result)
        pd.testing.assert_frame_equal(second, result)
        self.assertTrue(second_audit.empty)

    def test_output_run_records_artifacts_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pkl"
            output = root / "new-output"
            evidence_path = root / "summary.json"
            source_frame = sample_frame()
            source_frame.to_pickle(source)
            source_hash = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
            evidence_path.write_text(json.dumps({
                "input_sha256": {str(source): source_hash},
                "markers": {"None": 1, "nan": 1},
            }), encoding="utf-8")
            manifest = run_normalization(source, output, evidence_path)
            self.assertTrue(manifest["source"]["hash_matches_evidence"])
            self.assertFalse(manifest["source"]["source_is_master_only"])
            self.assertEqual(manifest["counts"]["total_converted"], 2)
            for name in ("vueltas_2022_2025_compound_normalizado.pkl",
                         "compound_normalization_audit.csv", "manifest.json",
                         "reconciliation.json", "report.md"):
                self.assertTrue((output / name).is_file(), name)
            loaded = load_normalized_matrix(output / "vueltas_2022_2025_compound_normalizado.pkl")
            self.assertEqual(str(loaded["Compound"].dtype), "string")
            self.assertEqual(int(loaded["Compound"].isna().sum()), 3)
            loaded_audit = load_normalization_audit(output / "compound_normalization_audit.csv")
            self.assertEqual(loaded_audit["original_literal"].tolist(), ["None", "nan"])
            pd.testing.assert_series_equal(
                restore_compound_from_audit(loaded, loaded_audit)["Compound"],
                source_frame["Compound"], check_dtype=False,
            )
            pd.testing.assert_frame_equal(pd.read_pickle(source), source_frame)
            with self.assertRaises(FileExistsError):
                run_normalization(source, output, evidence_path)

    def test_missing_expected_hash_rejects_before_creating_output(self):
        self._assert_evidence_rejected_without_output(
            lambda evidence: evidence["input_sha256"].clear(), "no SHA-256 entry"
        )

    def test_different_expected_hash_rejects_before_creating_output(self):
        self._assert_evidence_rejected_without_output(
            lambda evidence: evidence["input_sha256"].update(
                {next(iter(evidence["input_sha256"])): "0" * 64}
            ),
            "does not match the evidence",
        )

    def test_missing_expected_marker_count_rejects_before_creating_output(self):
        self._assert_evidence_rejected_without_output(
            lambda evidence: evidence["markers"].pop("nan"), "missing the expected count"
        )

    def test_invalid_expected_marker_count_rejects_before_creating_output(self):
        self._assert_evidence_rejected_without_output(
            lambda evidence: evidence["markers"].update({"nan": "1"}),
            "non-negative integer",
        )

    def test_observed_marker_count_mismatch_rejects_before_creating_output(self):
        frame = sample_frame()
        frame.loc[2, "Compound"] = "None"
        self._assert_evidence_rejected_without_output(
            lambda evidence: evidence["markers"].update({"None": 1}),
            "Observed count.*expects",
            frame=frame,
        )

    def test_non_exact_variants_are_not_converted(self):
        frame = sample_frame()
        frame.loc[2, "Compound"] = " none "
        frame.loc[3, "Compound"] = "NAN"
        frame.loc[4, "Compound"] = "None "
        result, audit = normalize_compound_frame(frame)
        self.assertEqual(len(audit), 2)
        self.assertEqual(result.loc[2, "Compound"], " none ")
        self.assertEqual(result.loc[3, "Compound"], "NAN")
        self.assertEqual(result.loc[4, "Compound"], "None ")


if __name__ == "__main__":
    unittest.main()
