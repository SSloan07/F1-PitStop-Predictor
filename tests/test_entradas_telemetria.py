"""Synthetic checks for the telemetry-only feature projection and its pair gate."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

STAGING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING / "scripts"))
from preparar_entradas_telemetria import derive_arrays, derive_state
from entrenar_comparacion_telemetria import _validate_pair_configs
from f1_pitstop.rnn_data import read_json
from f1_pitstop.rnn_model import model_spec_from_preprocessor


def _state():
    vocab = lambda values: [{"id": i + 4, "category": v} for i, v in enumerate(values)]
    numeric = ["LapNumber", "LapTime", "Weather_AirTemp"]
    return {
        "schema": {"window_length": 2, "numeric_feature_names": numeric,
                   "categorical_feature_names": ["Compound_asof", "track_phase_asof"],
                   "static_feature_names": ["Driver", "Team", "GranPremio"],
                   "mask_feature_order": [*numeric, "Compound_asof", "track_phase_asof"]},
        "numeric": {"mean": {n: float(i) for i, n in enumerate(numeric)},
                    "scale": {n: 1.0 for n in numeric}, "constant_features_scale_one": ["Weather_AirTemp"],
                    "variance_ddof": 0},
        "vocabularies": {"dynamic": {"Compound_asof": vocab(["SOFT"]), "track_phase_asof": vocab(["GREEN"])},
                         "static": {"Driver": vocab(["VER"]), "Team": vocab(["RBR"]), "GranPremio": vocab(["X"])}},
    }


CONFIG = {"keep_numeric": ["LapNumber", "LapTime"], "keep_dynamic_categorical": ["Compound_asof", "track_phase_asof"],
          "removed_numeric": ["Weather_AirTemp"], "removed_static": ["Driver", "Team", "GranPremio"],
          "static_policy": "constant_absent_placeholder", "static_placeholder_name": "SinContextoEstatico"}


class TelemetryProjectionTest(unittest.TestCase):
    def test_projection_selects_columns_and_blanks_static_context(self):
        state = _state()
        derived, numeric_index, dynamic_index, mask_index = derive_state(state, CONFIG)
        self.assertEqual(derived["schema"]["numeric_feature_names"], ["LapNumber", "LapTime"])
        self.assertEqual(derived["schema"]["static_feature_names"], ["SinContextoEstatico"])
        self.assertEqual(derived["numeric"]["mean"], {"LapNumber": 0.0, "LapTime": 1.0})
        self.assertEqual(derived["numeric"]["constant_features_scale_one"], [])
        self.assertEqual(mask_index, [0, 1, 3, 4])
        self.assertEqual(state["schema"]["static_feature_names"], ["Driver", "Team", "GranPremio"])
        spec = model_spec_from_preprocessor(derived)
        self.assertEqual((spec["num_numeric"], spec["static_cardinalities"]), (2, [4]))
        n, w = 3, 2
        arrays = {"X_numeric": np.arange(n * w * 3, dtype=np.float32).reshape(n, w, 3),
                  "X_dynamic_categorical_ids": np.full((n, w, 2), 4, dtype=np.int32),
                  "X_static_categorical_ids": np.full((n, 3), 4, dtype=np.int32),
                  "mask_reason_codes": np.arange(n * w * 5).reshape(n, w, 5) % 6,
                  "padding_mask": np.zeros((n, w), dtype=bool), "sample_id": np.arange(n)}
        out = derive_arrays(arrays, numeric_index, dynamic_index, mask_index)
        np.testing.assert_array_equal(out["X_numeric"], arrays["X_numeric"][..., :2])
        np.testing.assert_array_equal(out["mask_reason_codes"], arrays["mask_reason_codes"][..., [0, 1, 3, 4]])
        np.testing.assert_array_equal(out["X_static_categorical_ids"], np.ones((n, 1), dtype=np.int32))
        np.testing.assert_array_equal(out["sample_id"], arrays["sample_id"])

    def test_projection_rejects_undeclared_drops_and_reordering(self):
        bad = dict(CONFIG, removed_numeric=[])
        with self.assertRaises(ValueError):
            derive_state(_state(), bad)
        with self.assertRaises(ValueError):
            derive_state(_state(), dict(CONFIG, keep_numeric=["LapTime", "LapNumber"]))

    def test_pair_gate_allows_only_input_package_change(self):
        reference = read_json(STAGING / "configs/experimento_lstm32_telemetria_ref_cpu_v1.json")
        candidate = read_json(STAGING / "configs/experimento_lstm32_telemetria_candidate_cpu_v1.json")
        _validate_pair_configs(reference, candidate, root=STAGING)
        changed = copy.deepcopy(candidate)
        changed["training"]["seed"] = 43
        with self.assertRaises(ValueError):
            _validate_pair_configs(reference, changed, root=STAGING)
        changed = copy.deepcopy(candidate)
        changed["model"]["dropout_rate"] = 0.0
        with self.assertRaises(ValueError):
            _validate_pair_configs(reference, changed, root=STAGING)


if __name__ == "__main__":
    unittest.main()
