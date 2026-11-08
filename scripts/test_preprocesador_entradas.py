import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.f1_pitstop.preprocessing import (
    InputPreprocessor,
    PreprocessingContractError,
    RESERVED_CATEGORY_IDS,
)
from scripts.preparar_preprocesador import build_preprocessing


def synthetic_inputs():
    # 5 samples, W=2, 1 dynamic numeric, 1 dynamic categorical, 1 static categorical.
    numeric = np.array([
        [[0.0], [0.0]],       # first slot is padding; 0 in slot 1 is a real measurement
        [[2.0], [4.0]],
        [[0.0], [0.0]],       # both feature cells absent for different reasons
        [[100.0], [101.0]],   # validation
        [[500.0], [501.0]],   # held out synthetic test; excluded from fit
    ], dtype=np.float32)
    categorical = np.array([
        [[""], ["A"]],
        [["UNKNOWN"], ["B"]],
        [[""], [""]],
        [["NEW"], ["A"]],
        [["TEST_ONLY"], ["TEST_ONLY"]],
    ], dtype="<U32")
    static = np.array([["D1"], ["D2"], ["D3"], ["D_NEW"], ["D_TEST"]], dtype="<U32")
    reason = np.zeros((5, 2, 2), dtype=np.uint8)
    padding = np.zeros((5, 2), dtype=bool)
    exists = np.ones((5, 2), dtype=bool)
    row_missing = np.zeros((5, 2), dtype=bool)
    temporal = np.zeros((5, 2, 2), dtype=bool)
    missing = np.zeros((5, 2, 2), dtype=bool)
    # Padding position.
    padding[0, 0], exists[0, 0] = True, False
    reason[0, 0, :] = 1
    # Source UNKNOWN category remains distinct from absent and preserves mask code 5.
    reason[1, 0, 1] = 5
    missing[1, 0, 1] = True
    # A nominal lap whose source row is absent: not padding.
    row_missing[2, 0] = True
    reason[2, 0, :] = 2
    missing[2, 0, :] = True
    # A feature-level missing value in another existing slot.
    reason[2, 1, :] = 4
    missing[2, 1, :] = True
    masks = {
        "padding_mask": padding, "slot_exists_mask": exists,
        "slot_row_missing_mask": row_missing, "temporal_unavailable_mask": temporal,
        "feature_missing_mask": missing, "mask_reason_codes": reason,
    }
    schema = {
        "window_length": 2, "numeric_feature_names": ["num"],
        "categorical_feature_names": ["compound"], "static_feature_names": ["driver"],
        "mask_codes": {"0": "present", "1": "padding", "2": "nominal_lap_row_missing",
                       "3": "temporally_unavailable", "4": "feature_missing", "5": "source_event_unavailable"},
        "mask_feature_order": ["num", "compound"], "feature_contract_version": "causal_sample_contract_v1",
    }
    return numeric, categorical, static, masks, schema


def fit_synthetic(inputs=None):
    numeric, categorical, static, masks, schema = inputs or synthetic_inputs()
    fit_rows = np.array([True, True, True, False, False])
    pre = InputPreprocessor().fit(numeric, categorical, static, masks, schema, fit_rows, "fit-ids-hash")
    return pre, (numeric, categorical, static, masks, schema)


def transform_one_validation(pre, inputs):
    numeric, categorical, static, masks, schema = inputs
    selected_masks = {name: value[3:4] for name, value in masks.items()}
    return pre.transform(numeric[3:4], categorical[3:4], static[3:4], selected_masks, schema)


class TestInputPreprocessor(unittest.TestCase):
    def test_padding_is_excluded_but_physical_zero_is_fitted(self):
        pre, inputs = fit_synthetic()
        state = pre.to_dict()
        self.assertEqual(state["numeric"]["observed_cell_count"]["num"], 3)
        self.assertAlmostEqual(state["numeric"]["mean"]["num"], 2.0)
        self.assertEqual(state["numeric"]["observed_physical_zero_count"]["num"], 1)
        self.assertAlmostEqual(state["numeric"]["scale"]["num"], np.sqrt(8 / 3))
        output = pre.transform(*inputs[:4], inputs[4])
        self.assertEqual(output["X_numeric"][0, 0, 0], 0.0)  # masked technical padding
        self.assertAlmostEqual(output["X_numeric"][0, 1, 0], -2.0 / np.sqrt(8 / 3), places=6)

    def test_validation_and_test_values_do_not_change_fit_state(self):
        original = synthetic_inputs()
        changed = tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in original)
        changed[0][3:, :, :] = -999999
        changed[1][3:, :, :] = "NEW_VALIDATION_OR_TEST_CATEGORY"
        changed[2][3:, :] = "NEW_STATIC_CATEGORY"
        left, right = fit_synthetic(original)[0], fit_synthetic(changed)[0]
        self.assertEqual(left.to_dict(), right.to_dict())

    def test_unknown_missing_and_oov_ids_are_distinct_and_vocab_does_not_grow(self):
        pre, inputs = fit_synthetic()
        output = transform_one_validation(pre, inputs)
        self.assertEqual(output["X_dynamic_categorical_ids"][0, 0, 0], RESERVED_CATEGORY_IDS["out_of_vocabulary"])
        self.assertEqual(output["X_static_categorical_ids"][0, 0], RESERVED_CATEGORY_IDS["out_of_vocabulary"])
        self.assertEqual(output["_oov_dynamic_counts"].tolist(), [1])
        self.assertEqual(output["_oov_static_counts"].tolist(), [1])
        self.assertEqual(len(pre.to_dict()["vocabularies"]["dynamic"]["compound"]), 2)

    def test_different_masks_are_preserved_without_becoming_predictor_fields(self):
        pre, inputs = fit_synthetic()
        output = pre.transform(*inputs[:4], inputs[4])
        for name in inputs[3]:
            np.testing.assert_array_equal(output[name], inputs[3][name])
        self.assertNotIn("padding_mask", output["X_numeric"].dtype.names or [])
        self.assertIn("slot_row_missing_mask", output)
        self.assertIn("temporal_unavailable_mask", output)
        self.assertIn("feature_missing_mask", output)
        ids = output["X_dynamic_categorical_ids"]
        self.assertEqual(ids[0, 0, 0], RESERVED_CATEGORY_IDS["padding"])
        self.assertEqual(ids[1, 0, 0], RESERVED_CATEGORY_IDS["source_unknown"])
        self.assertEqual(ids[2, 0, 0], RESERVED_CATEGORY_IDS["absent"])

    def test_zero_variance_uses_scale_one_and_all_missing_rejects_fit(self):
        inputs = list(synthetic_inputs())
        inputs[0][:3, :, 0] = 7.0
        pre = fit_synthetic(tuple(inputs))[0]
        self.assertIn("num", pre.to_dict()["numeric"]["constant_features_scale_one"])
        self.assertEqual(pre.to_dict()["numeric"]["scale"]["num"], 1.0)
        missing_inputs = list(synthetic_inputs())
        observed_fit = missing_inputs[3]["mask_reason_codes"][:3, :, 0] == 0
        missing_inputs[3]["mask_reason_codes"][:3, :, 0][observed_fit] = 4
        missing_inputs[3]["feature_missing_mask"][:3, :, 0][observed_fit] = True
        with self.assertRaisesRegex(PreprocessingContractError, "No observed training values.*num"):
            fit_synthetic(tuple(missing_inputs))

    def test_transform_does_not_mutate_state_and_schema_order_is_enforced(self):
        pre, inputs = fit_synthetic()
        before = pre.to_dict()
        pre.transform(*inputs[:4], inputs[4])
        self.assertEqual(pre.to_dict(), before)
        bad_schema = dict(inputs[4], numeric_feature_names=["other"])
        with self.assertRaisesRegex(PreprocessingContractError, "feature dimension|schema"):
            pre.transform(*inputs[:4], bad_schema)

    def test_save_reload_reproduces_transform(self):
        pre, inputs = fit_synthetic()
        expected = pre.transform(*inputs[:4], inputs[4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            pre.save(path)
            reloaded = InputPreprocessor.load(path)
            actual = reloaded.transform(*inputs[:4], inputs[4])
        self.assertEqual(reloaded.to_dict(), pre.to_dict())
        for name in expected:
            np.testing.assert_array_equal(expected[name], actual[name])

    def test_refit_and_output_overwrite_are_rejected(self):
        pre, inputs = fit_synthetic()
        with self.assertRaisesRegex(PreprocessingContractError, "frozen"):
            pre.fit(*inputs[:4], inputs[4], np.array([1, 1, 1, 0, 0], dtype=bool), "again")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            config = Path(tmp) / "no-read.json"
            config.write_text("{}")
            with self.assertRaisesRegex(PreprocessingContractError, "refusing to overwrite"):
                build_preprocessing(config, output)

    def test_incompatible_manifest_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            output = Path(tmp) / "not-published"
            config.write_text(json.dumps({
                "preprocessing_version": "train_only_preprocessor_v1",
                "feature_package": "EDA/dataset/muestras_causales_v1_3",
                "feature_manifest_sha256": "0" * 64,
                "partition_package": "EDA/dataset/muestras_causales_v1_3_etiquetas_v3_particiones_v1_1",
                "partition_manifest_sha256": "0" * 64,
                "label_package": "EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025",
                "label_manifest_sha256": "0" * 64,
            }))
            with self.assertRaisesRegex(PreprocessingContractError, "feature package manifest hash mismatch"):
                build_preprocessing(config, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
