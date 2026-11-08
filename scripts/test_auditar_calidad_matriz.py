"""Focused contract tests for the conservative matrix audit."""
import unittest

import pandas as pd

from scripts.auditar_calidad_matriz import (
    coverage_by_driver,
    categorical_distributions,
    exception_register,
    missingness_by_driver,
    profile_columns,
    validate_structure,
    validate_core_dtypes,
)


def sample_frame():
    return pd.DataFrame({
        "Temporada": [2022, 2022, 2022],
        "Ronda": [1, 1, 1],
        "GranPremio": ["Test GP"] * 3,
        "DriverNumber": ["1"] * 3,
        "LapNumber": [1, 3, 4],
        "Compound": ["None", None, "SOFT"],
        "PitInTime": [pd.NaT, pd.NaT, pd.Timedelta(30, unit="s")],
        "PitOutTime": [pd.Timedelta(1, unit="s"), pd.NaT, pd.NaT],
        "FastF1Generated": [False, False, True],
        "Telemetry_Speed_mean": [0.0, 250.0, -1.0],
    })


class MatrixAuditTests(unittest.TestCase):
    def test_key_validation_counts_null_and_duplicate_rows(self):
        frame = sample_frame()
        frame.loc[1, "LapNumber"] = 1
        frame.loc[2, "DriverNumber"] = None
        result = validate_structure(frame)
        self.assertEqual(result["null_key_rows"], 1)
        self.assertEqual(result["duplicate_key_rows"], 2)

    def test_coverage_reports_gap_without_filling_laps(self):
        frame = sample_frame()
        before = frame.copy(deep=True)
        coverage = coverage_by_driver(frame)
        self.assertEqual(int(coverage.loc[0, "gap_count"]), 1)
        self.assertEqual(int(coverage.loc[0, "max_observed_gap"]), 2)
        pd.testing.assert_frame_equal(frame, before)

    def test_literal_compound_none_is_preserved_and_reported(self):
        frame = sample_frame()
        before = frame.copy(deep=True)
        coverage = coverage_by_driver(frame)
        profile = profile_columns(frame)
        exceptions = exception_register(frame, coverage)
        sentinel = exceptions.loc[exceptions["exception_type"] == "compound_text_sentinel"]
        self.assertEqual(int(sentinel["affected_count"].sum()), 1)
        self.assertEqual(sentinel.iloc[0]["status"], "domain_decision_pending")
        self.assertEqual(int(profile.loc[profile["column"] == "Compound", "null_count"].iloc[0]), 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_telemetry_extremes_are_profiled_not_changed(self):
        profile = profile_columns(sample_frame())
        speed = profile.loc[profile["column"] == "Telemetry_Speed_mean"].iloc[0]
        self.assertEqual(speed["zero_count"], 1)
        self.assertEqual(speed["negative_count"], 1)
        self.assertEqual(speed["minimum"], -1.0)

    def test_core_dtype_mismatch_is_explicit(self):
        frame = sample_frame()
        frame["Temporada"] = frame["Temporada"].astype("string")
        mismatch = validate_core_dtypes(frame)
        self.assertEqual(mismatch["Temporada"], {"expected": "int64", "observed": "string"})

    def test_missingness_and_category_reports_preserve_observations(self):
        frame = sample_frame()
        before = frame.copy(deep=True)
        missingness = missingness_by_driver(frame)
        categories = categorical_distributions(frame)
        self.assertEqual(int(missingness.loc[0, "Compound"]), 1)
        none_count = categories.loc[
            (categories["column"] == "Compound")
            & (categories["value"] == "literal::None"), "count"
        ].iloc[0]
        self.assertEqual(int(none_count), 1)
        pd.testing.assert_frame_equal(frame, before)


if __name__ == "__main__":
    unittest.main()
