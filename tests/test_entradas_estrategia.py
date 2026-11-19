"""Synthetic checks for per-slot strategy features: lookup by slot lap and mask reasons."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "scripts"))
from preparar_entradas_estrategia import FEATURE_MISSING, PADDING, PRESENT, ROW_MISSING, TEMPORAL, slot_values


class StrategySlotTest(unittest.TestCase):
    def test_slot_lap_lookup_and_reasons(self):
        metadata = pd.DataFrame({"sample_id": [7], "Temporada": [2022], "Ronda": [1], "DriverNumber": ["1"]})
        arrays = {
            "window_lap_numbers": np.array([[0, 2, 3, 4, 5]], dtype=np.int16),
            "padding_mask": np.array([[True, False, False, False, False]]),
            "slot_exists_mask": np.array([[False, True, True, True, True]]),
            "slot_row_missing_mask": np.array([[False, False, True, False, False]]),
        }
        # Lap 4 has a decision time later than the sample's (restart anomaly) -> temporally unavailable.
        table = pd.DataFrame({"Temporada": 2022, "Ronda": 1, "DriverNumber": "1",
                              "slot_lap": [2, 3, 4, 5], "slot_decision_time": [100.0, 200.0, 900.0, 400.0],
                              "stops_done": [0.0, 0.0, 1.0, np.nan], "laps_remaining": [9.0, 8.0, 7.0, 6.0]})
        index = pd.DataFrame({"sample_id": [7], "decision_time_seconds": [400.0]})
        values, reasons = slot_values(metadata, arrays, table, index, ["stops_done", "laps_remaining"])
        self.assertEqual(values.shape, (1, 5, 2))
        np.testing.assert_array_equal(values[0, 1], [0.0, 9.0])
        np.testing.assert_array_equal(reasons[0, :, 0], [PADDING, PRESENT, ROW_MISSING, TEMPORAL, FEATURE_MISSING])
        np.testing.assert_array_equal(reasons[0, :, 1], [PADDING, PRESENT, ROW_MISSING, TEMPORAL, PRESENT])


if __name__ == "__main__":
    unittest.main()
