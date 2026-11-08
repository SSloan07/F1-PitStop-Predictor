"""Focused tests for the historical quality contracts."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.generar_registro_calidad_historica import (
    run,
    find_clock_overlaps,
    matching_restart_event,
    validate_clock_race_identity,
    validate_telemetry_contract,
    variable_quality_summary,
)


def throttle_frame():
    return pd.DataFrame({
        "Telemetry_Samples": [4, 2, 0],
        "Telemetry_Throttle_valid_samples": [2, 0, 0],
        "Telemetry_Throttle_valid_fraction": [0.5, 0.0, pd.NA],
        "Telemetry_Throttle_valid_mean": [50.0, pd.NA, pd.NA],
        "Telemetry_Throttle_valid_max": [100.0, pd.NA, pd.NA],
    })


class HistoricalQualityContractTests(unittest.TestCase):
    def test_throttle_contract_accepts_valid_coverage_and_empty_rows(self):
        self.assertEqual(validate_telemetry_contract(throttle_frame()), [])

    def test_throttle_contract_rejects_incoherent_fraction(self):
        frame = throttle_frame()
        frame.loc[0, "Telemetry_Throttle_valid_fraction"] = 0.75
        self.assertIn(
            "valid throttle fraction does not equal valid_samples / samples",
            validate_telemetry_contract(frame),
        )

    def test_throttle_contract_rejects_out_of_range_filtered_aggregate(self):
        frame = throttle_frame()
        frame.loc[0, "Telemetry_Throttle_valid_max"] = 101.0
        self.assertTrue(any("outside [0, 100]" in error for error in validate_telemetry_contract(frame)))

    def test_clock_overlap_is_reported_without_changing_timestamps(self):
        frame = pd.DataFrame({
            "Temporada": [2023, 2023], "Ronda": [3, 3],
            "DriverNumber": ["1", "1"], "LapNumber": [8, 9],
            "Driver": ["VER", "VER"], "Time": pd.to_timedelta([100, 180], unit="s"),
            "LapStartTime": pd.to_timedelta([0, 90], unit="s"),
            "TrackStatus": ["1", "1"],
        })
        before = frame.copy(deep=True)
        overlaps = find_clock_overlaps(frame)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["overlap_seconds"], 10.0)
        pd.testing.assert_frame_equal(frame, before)

    def test_existing_output_directory_is_rejected_before_reading_inputs(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "published"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Path(temp) / "missing.pkl", Path(temp) / "missing.json",
                    Path(temp) / "missing.csv", Path(temp), output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_clock_identity_rejects_matrix_cache_race_mismatch(self):
        info = {
            "Meeting": {"Name": "Saudi Arabian Grand Prix"},
            "StartDate": pd.Timestamp("2023-03-19"),
            "Name": "Race",
            "Path": "2023/2023-04-02_Australian_Grand_Prix/2023-04-02_Race/",
        }
        metadata = {"directory": "2023/2023-04-02_Australian_Grand_Prix/2023-04-02_Race"}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_clock_race_identity({"Australian Grand Prix"}, 2023, metadata, info)

    def test_restart_match_requires_temporal_adjacency_to_aborted(self):
        status = pd.DataFrame({
            "Time": pd.to_timedelta([0, 10, 20, 30], unit="s"),
            "Status": ["Inactive", "Started", "Aborted", "Started"],
        })
        # The first start is not after Aborted; a later suspension does not explain it.
        self.assertIsNone(matching_restart_event(pd.Timedelta(seconds=10), status))
        match = matching_restart_event(pd.Timedelta(seconds=30), status)
        self.assertIsNotNone(match)
        self.assertEqual(match["status_before_started"], "Aborted")
        self.assertEqual(match["start_delta_ms"], 0.0)

    def test_accelerator_fields_follow_agreed_baseline_classification(self):
        frame = pd.DataFrame({
            "Telemetry_Throttle_valid_mean": [50.0],
            "Telemetry_Throttle_valid_max": [90.0],
            "Telemetry_Throttle_valid_samples": [10],
            "Telemetry_Throttle_valid_fraction": [0.5],
            "Telemetry_Samples": [20],
        })
        summary = variable_quality_summary(frame).set_index("field")
        for field in ("Telemetry_Throttle_valid_mean", "Telemetry_Throttle_valid_max"):
            self.assertEqual(summary.loc[field, "decision_status"], "baseline_candidate_with_mask")
        for field in ("Telemetry_Throttle_valid_samples", "Telemetry_Throttle_valid_fraction", "Telemetry_Samples"):
            self.assertEqual(summary.loc[field, "decision_status"], "audit_only_coverage")


if __name__ == "__main__":
    unittest.main()
