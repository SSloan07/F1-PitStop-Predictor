"""Causality check for the GBM strategy features: data after the decision time must not matter."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "scripts"))
from f1_pitstop.strategy_features import strategy_features


def _laps(future_pit_for_rival=False, future_pace_noise=0.0):
    rows = []
    for driver, offset in (("1", 0.0), ("2", 1.5), ("3", 3.0)):
        for lap in range(1, 21):
            end = 100.0 * lap + offset
            pit = end - 1.0 if (driver == "2" and lap == 6) else None
            if future_pit_for_rival and driver == "3" and lap == 15:
                pit = end - 1.0
            stint = 1 if (driver != "2" or lap <= 6) else 2
            lap_time = 100.0 + 0.1 * lap + (future_pace_noise if lap >= 10 else 0.0)
            rows.append({"Temporada": 2022, "Ronda": 1, "GranPremio": "Test Grand Prix", "DriverNumber": driver,
                         "LapNumber": float(lap), "Time": pd.to_timedelta(end, unit="s"),
                         "PitInTime": pd.to_timedelta(pit, unit="s") if pit else pd.NaT,
                         "PitOutTime": pd.NaT, "LapTime": pd.to_timedelta(lap_time, unit="s"),
                         "TrackStatus": "1", "Position": float({"1": 1, "2": 2, "3": 3}[driver]),
                         "Stint": float(stint), "Compound": "SOFT" if stint == 1 else "HARD",
                         "TyreLife": float(lap if stint == 1 else lap - 6)})
    return pd.DataFrame(rows)


class StrategyCausalityTest(unittest.TestCase):
    def test_future_rows_do_not_change_features(self):
        samples = pd.DataFrame({"sample_id": [0, 1], "Temporada": [2022, 2022], "Ronda": [1, 1],
                                "DriverNumber": ["1", "2"], "prediction_lap": [8, 9],
                                "decision_time_seconds": [700.0, 801.5]})
        total = {(2022, 1, "2022-01-01_Test_Grand_Prix"): 20}
        medians = {"SOFT": 14.0, "HARD": 21.0}
        base = strategy_features(samples, _laps(), total, medians)
        changed = strategy_features(samples, _laps(future_pit_for_rival=True, future_pace_noise=5.0), total, medians)
        pd.testing.assert_frame_equal(base, changed)
        first = base.iloc[0]
        self.assertEqual(first.stops_done, 0)
        self.assertEqual(first.field_pit_entries_last2laps, 1)  # rival entered at lap 6, within two laps
        self.assertEqual(first.behind_pitted_last2laps, 1.0)
        self.assertAlmostEqual(first.race_fraction, 7 / 20)
        second = base.iloc[1]
        self.assertEqual(second.stops_done, 1)
        self.assertEqual(second.two_dry_compounds_used, 1.0)
        self.assertEqual(second.laps_since_last_stop, 3)


if __name__ == "__main__":
    unittest.main()
