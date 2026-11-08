"""Focused causal alignment tests for the point-in-time builder."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1_pitstop.causal_samples import (  # noqa: E402
    ContractError, active_timing_stint, build_samples, event_frame, lap_metrics_available,
    load_fastf1_cache, matrix_stint_for_timing_stint, metric_delay_key, nominal_window_laps,
    state_asof, validate_inputs,
)


def race_row(lap: int, start: float, close: float, driver: str = "1") -> dict:
    return {
        "Temporada": 2023, "Ronda": 1, "DriverNumber": driver, "LapNumber": lap,
        "LapStartTime": pd.Timedelta(seconds=start), "Time": pd.Timedelta(seconds=close),
        "Driver": "VER", "Team": "Red Bull", "GranPremio": "Bahrain Grand Prix",
        "Compound": "SOFT", "TyreLife": float(lap), "Stint": 1, "FastF1Generated": False,
    }


class CausalContractTests(unittest.TestCase):
    def test_start_short_and_full_window_identity(self):
        self.assertEqual([-2, -1, 0, 1], nominal_window_laps(1, 4))
        self.assertEqual([-1, 0, 1, 2], nominal_window_laps(2, 4))
        self.assertEqual([1, 2, 3, 4], nominal_window_laps(4, 4))
        self.assertEqual([4, 5, 6, 7, 8, 9, 10, 11], nominal_window_laps(11, 8))
        self.assertEqual([0, 0, 0, 1], [0 if x < 1 else x for x in nominal_window_laps(1, 4)])

    def test_horizon_is_independent_of_window(self):
        for _window in (1, 4, 8, 20):
            self.assertEqual((11, 12, 13), tuple(range(11, 14)))

    def test_temporal_completion_and_isolation(self):
        a = pd.DataFrame([race_row(1, 0, 60), race_row(2, 60, 120)])
        b = pd.DataFrame([race_row(1, 0, 60, "2"), race_row(2, 60, 120, "2")])
        # At lap-2 start, lap 1 is closed and lap 2 is not; other driver rows stay outside group.
        self.assertTrue(a.iloc[0].Time <= a.iloc[1].LapStartTime)
        self.assertFalse(a.iloc[1].Time <= a.iloc[1].LapStartTime)
        self.assertNotEqual(a.iloc[0].DriverNumber, b.iloc[0].DriverNumber)
        self.assertTrue(lap_metrics_available(1, 2, 60.0, 60.0))
        self.assertFalse(lap_metrics_available(2, 2, 60.0, 60.0))
        self.assertFalse(lap_metrics_available(1, 2, 61.0, 60.0))
        self.assertEqual("matrix_position", metric_delay_key("Position"))
        self.assertEqual("matrix_telemetry", metric_delay_key("Telemetry_Speed_mean"))
        self.assertEqual("matrix_lap_metrics", metric_delay_key("LapTime"))

    def test_partial_compound_keeps_same_stint_value_and_stint_switch_resets(self):
        timing = pd.DataFrame([
            {"Driver": "1", "Time": pd.Timedelta(seconds=1), "Stint": 1, "Compound": "SOFT"},
            {"Driver": "1", "Time": pd.Timedelta(seconds=10), "Stint": 1, "Compound": None},
            {"Driver": "1", "Time": pd.Timedelta(seconds=20), "Stint": 2, "Compound": None},
        ])
        track = event_frame({"Time": [pd.Timedelta(0)], "Status": ["1"]})
        weather = pd.DataFrame()
        session = event_frame({"Time": [pd.Timedelta(0)], "Status": ["Started"]})
        rows = pd.DataFrame([race_row(1, 0, 30)])
        before_switch = race_row(1, 15, 30)
        after_switch = race_row(1, 25, 30)
        before_switch["_unused"] = 0
        state, _, _ = state_asof(pd.Series(before_switch), rows, {
            "timing_app_data": timing, "track_status_data": {"Time": [pd.Timedelta(0)], "Status": ["1"]},
            "weather_data": {}, "session_status_data": {"Time": [pd.Timedelta(0)], "Status": ["Started"]},
        }, {}, 120)
        self.assertEqual("SOFT", state["Compound_asof"])
        state2, _, _ = state_asof(pd.Series(after_switch), rows, {
            "timing_app_data": timing, "track_status_data": {"Time": [pd.Timedelta(0)], "Status": ["1"]},
            "weather_data": {}, "session_status_data": {"Time": [pd.Timedelta(0)], "Status": ["Started"]},
        }, {}, 120)
        self.assertEqual("UNKNOWN", state2["Compound_asof"])

    def test_late_old_stint_update_does_not_revert_active_compound(self):
        timing = pd.DataFrame([
            {"Driver": "11", "Time": pd.Timedelta(seconds=10), "Stint": 0, "Compound": "MEDIUM"},
            {"Driver": "11", "Time": pd.Timedelta(seconds=20), "Stint": 1, "Compound": "HARD"},
            # A delayed correction for the previous stint arrives after the new stint.
            {"Driver": "11", "Time": pd.Timedelta(seconds=30), "Stint": 0, "Compound": "MEDIUM"},
        ])
        rows = pd.DataFrame([
            {**race_row(20, 1, 5, "11"), "Stint": 2, "TyreLife": 19},
            {**race_row(21, 5, 15, "11"), "Stint": 2, "TyreLife": 20},
        ])
        latest = pd.Series(race_row(23, 35, 50, "11"))
        state, _, _ = state_asof(latest, rows, {
            "timing_app_data": timing, "track_status_data": {"Time": [pd.Timedelta(0)], "Status": ["1"]},
            "weather_data": {}, "session_status_data": {"Time": [pd.Timedelta(0)], "Status": ["Started"]},
        }, {}, 120)
        self.assertEqual("HARD", state["Compound_asof"])
        self.assertEqual(20.0, state["TyreLife_last_eligible"])
        self.assertEqual(1, active_timing_stint(timing))
        self.assertEqual(2.0, matrix_stint_for_timing_stint(1, rows))

    def test_real_bahrain_and_saudi_tyre_asof_regressions(self):
        matrix_path = ROOT / (
            "EDA/dataset/datos_carreras_completos_2022_2025/"
            "vueltas_2022_2025_compound_normalizado_v1/vueltas_2022_2025_compound_normalizado.pkl"
        )
        matrix = pd.read_pickle(matrix_path)
        cases = [
            (2023, 1, "1", 16, "Bahrain Grand Prix",
             "2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race",
             "HARD", 1.0),
            (2022, 2, "11", 23, "Saudi Arabian Grand Prix",
             "2022/2022-03-27_Saudi_Arabian_Grand_Prix/2022-03-27_Race",
             "HARD", 7.0),
        ]
        names = ["timing_app_data", "track_status_data", "weather_data", "session_status_data"]
        cache_root = ROOT / "EDA/dataset/cache_fastf1_carreras"
        for year, rnd, driver, lap, gp, relative, expected_compound, expected_age in cases:
            with self.subTest(year=year, race=gp, driver=driver, lap=lap):
                group = matrix.loc[
                    (matrix.Temporada == year) & (matrix.Ronda == rnd)
                    & (matrix.DriverNumber.astype(str) == driver)
                ].copy()
                row = group.loc[group.LapNumber == lap].iloc[0]
                cache_dir = cache_root / relative
                sources = {
                    name: load_fastf1_cache(cache_dir / f"{name}.ff1pkl")
                    for name in names
                }
                state, _, _ = state_asof(row, group, sources, {}, 120)
                self.assertEqual(expected_compound, state["Compound_asof"])
                self.assertEqual(expected_age, state["TyreLife_last_eligible"])

    def test_weather_future_expired_and_latency(self):
        row = pd.Series(race_row(3, 100, 160))
        timing = pd.DataFrame()
        track = {"Time": [pd.Timedelta(0)], "Status": ["1"]}
        session = {"Time": [pd.Timedelta(0)], "Status": ["Started"]}
        future = {"Time": [pd.Timedelta(seconds=101)], "AirTemp": [20.0]}
        state, _, _ = state_asof(row, pd.DataFrame([race_row(1,0,10)]), {
            "timing_app_data": timing, "track_status_data": track, "weather_data": future,
            "session_status_data": session,
        }, {}, 120)
        self.assertNotIn("Weather_AirTemp", state)
        stale = {"Time": [pd.Timedelta(seconds=-100)], "AirTemp": [20.0]}
        state, _, _ = state_asof(row, pd.DataFrame([race_row(1,0,10)]), {
            "timing_app_data": timing, "track_status_data": track, "weather_data": stale,
            "session_status_data": session,
        }, {}, 120)
        self.assertNotIn("Weather_AirTemp", state)
        recent = {"Time": [pd.Timedelta(seconds=70)], "AirTemp": [20.0]}
        state, _, _ = state_asof(row, pd.DataFrame([race_row(1,0,10)]), {
            "timing_app_data": timing, "track_status_data": track, "weather_data": recent,
            "session_status_data": session,
        }, {"weather_data": 40}, 120)
        self.assertNotIn("Weather_AirTemp", state)

    def test_future_updates_do_not_change_prior_state_and_prefix_matches(self):
        row = pd.Series(race_row(2, 100, 160))
        timing_prefix = pd.DataFrame([
            {"Driver": "1", "Time": pd.Timedelta(seconds=10), "Stint": 1, "Compound": "MEDIUM"}
        ])
        rows = pd.DataFrame([race_row(1, 0, 50)])
        prefix = {"timing_app_data": timing_prefix,
                  "track_status_data": {"Time": [pd.Timedelta(seconds=0)], "Status": ["1"]},
                  "weather_data": {"Time": [pd.Timedelta(seconds=90)], "AirTemp": [18.0]},
                  "session_status_data": {"Time": [pd.Timedelta(seconds=0)], "Status": ["Started"]}}
        full = {"timing_app_data": pd.concat([timing_prefix, pd.DataFrame([
                    {"Driver": "1", "Time": pd.Timedelta(seconds=101), "Stint": 1, "Compound": "HARD"}])],
                    ignore_index=True),
                "track_status_data": {"Time": [pd.Timedelta(seconds=0), pd.Timedelta(seconds=101)],
                                      "Status": ["1", "5"]},
                "weather_data": {"Time": [pd.Timedelta(seconds=90), pd.Timedelta(seconds=101)],
                                 "AirTemp": [18.0, 40.0]},
                "session_status_data": {"Time": [pd.Timedelta(seconds=0), pd.Timedelta(seconds=101)],
                                        "Status": ["Started", "Aborted"]}}
        a = state_asof(row, rows, prefix, {}, 120)[0]
        b = state_asof(row, rows, full, {}, 120)[0]
        self.assertEqual(a, b)
        self.assertEqual(a["Compound_asof"], "MEDIUM")
        self.assertEqual(a["Weather_AirTemp"], 18.0)

    def test_restart_prior_lap_closure_after_decision_not_eligible(self):
        previous = pd.Series(race_row(8, 100, 200))
        decision = 150.0
        self.assertGreater(previous.Time.total_seconds(), decision)

    def test_unknown_and_missing_labels_are_preserved_by_contract(self):
        matrix = pd.DataFrame([race_row(1, 0, 30)])
        labels = pd.DataFrame([{
            "Temporada": 2023, "Ronda": 1, "DriverNumber": "1", "LapNumber": 1,
            "PitWithin3Laps": pd.NA, "EligibleForSupervisedLearning": False,
        }])
        validate_inputs(matrix, labels)
        self.assertTrue(pd.isna(labels.iloc[0].PitWithin3Laps))
        self.assertFalse(bool(labels.iloc[0].EligibleForSupervisedLearning))

    def test_duplicate_labels_rejected(self):
        matrix = pd.DataFrame([race_row(1, 0, 30)])
        labels = pd.DataFrame([{
            "Temporada": 2023, "Ronda": 1, "DriverNumber": "1", "LapNumber": 1,
            "PitWithin3Laps": True, "EligibleForSupervisedLearning": True,
        }] * 2)
        with self.assertRaises(ContractError):
            validate_inputs(matrix, labels)

    def test_existing_output_is_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "already_published"
            output.mkdir()
            with self.assertRaises(ContractError):
                build_samples(Path("missing-config.json"), output)
            self.assertTrue(output.is_dir())

    def test_feature_whitelist_excludes_future_and_audit_fields(self):
        from f1_pitstop.causal_samples import FEATURES, FORBIDDEN
        self.assertTrue(FORBIDDEN.isdisjoint(FEATURES))
        for field in ("PitInTime", "PitOutTime", "PitWithin3Laps", "Telemetry_Samples",
                      "Telemetry_Throttle_valid_fraction", "Stint_asof", "FreshTyre_asof"):
            self.assertNotIn(field, FEATURES)


if __name__ == "__main__":
    unittest.main()
