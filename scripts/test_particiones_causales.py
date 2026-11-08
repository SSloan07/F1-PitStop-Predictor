import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.preparar_particiones_causales import (
    PartitionContractError,
    align_labels_and_partition,
    build_package,
)


def fixtures():
    samples = pd.DataFrame({
        "sample_id": [10, 11, 12],
        "Temporada": [2022, 2022, 2024],
        "Ronda": [1, 1, 2],
        "DriverNumber": [1, 1, 7],
        "prediction_lap": [1, 2, 1],
        "decision_time_seconds": [100.0, 200.0, 300.0],
        "prediction_eligible": [True, False, True],
        "prediction_ineligible_reason": [None, "red_flag", None],
    })
    labels = pd.DataFrame({
        "Temporada": [2022, 2022], "Ronda": [1, 1],
        "DriverNumber": ["1", "1"], "LapNumber": [1, 2],
        "PitWithin3Laps": pd.array([True, None], dtype="boolean"),
        "LabelReason": ["valid_pair", "unknown_horizon"],
        "LabelBasis": ["pit_pair", "unresolved"],
        "RuleVersion": ["pit_entry_v2", "pit_entry_v2"],
        "EligibleForSupervisedLearning": [True, False],
    })
    static = pd.DataFrame({
        "sample_id": [10, 11, 12], "Temporada": [2022, 2022, 2024],
        "Ronda": [1, 1, 2], "DriverNumber": [1, 1, 7],
        "prediction_lap": [1, 2, 1],
        "GranPremio": ["A", "A", "B"],
    })
    return samples, labels, static


class TestCausalPartitions(unittest.TestCase):
    def test_left_join_preserves_ids_and_distinguishes_label_states(self):
        samples, labels, static = fixtures()
        aligned, races = align_labels_and_partition(
            samples, labels, static,
            {"train": [2022, 2023], "validation": [2024], "test_final_reserved": [2025]},
        )
        self.assertEqual(aligned.sample_id.tolist(), [10, 11, 12])
        self.assertEqual(aligned.label_status.tolist(), ["published_known", "published_unknown", "not_published"])
        self.assertEqual(aligned.supervised_eligible.tolist(), [True, False, False])
        self.assertEqual(aligned.split.tolist(), ["train", "train", "validation"])
        self.assertEqual(races.split.tolist(), ["train", "validation"])
        # Label eligibility and prediction eligibility remain separate dimensions.
        self.assertFalse(aligned.loc[1, "prediction_eligible"])
        self.assertFalse(aligned.loc[1, "label_usable"])
        # Padding or missing features are not columns in this selection contract.
        self.assertNotIn("padding_mask", aligned.columns)

    def test_duplicate_label_key_rejected(self):
        samples, labels, static = fixtures()
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(PartitionContractError, "duplicated"):
            align_labels_and_partition(samples, labels, static, {"train": [2022, 2023], "validation": [2024], "test_final_reserved": [2025]})

    def test_static_key_mismatch_rejected_even_when_sample_ids_match(self):
        samples, labels, static = fixtures()
        static.loc[0, "DriverNumber"] = 99
        with self.assertRaisesRegex(PartitionContractError, "Static context keys"):
            align_labels_and_partition(samples, labels, static, {"train": [2022, 2023], "validation": [2024], "test_final_reserved": [2025]})

    def test_race_with_multiple_grand_prix_names_rejected(self):
        samples, labels, static = fixtures()
        static.loc[1, "GranPremio"] = "Other"
        with self.assertRaisesRegex(PartitionContractError, "identity is not unique"):
            align_labels_and_partition(samples, labels, static, {"train": [2022, 2023], "validation": [2024], "test_final_reserved": [2025]})

    def test_existing_output_is_rejected_without_touching_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "already-published"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("preserve")
            config = root / "config.json"
            config.write_text("{}")
            with self.assertRaisesRegex(PartitionContractError, "refusing overwrite"):
                build_package(config, output)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_incompatible_pinned_feature_manifest_fails_before_publication(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new-package"
            config = {
                "feature_package": "EDA/dataset/muestras_causales_v1_3",
                "feature_package_manifest_sha256": "0" * 64,
                "label_package": "EDA/dataset/datos_carreras_completos_2022_2025/etiquetado_pit_entry_v3_2022_2025",
                "label_manifest_sha256": "0" * 64,
                "source_master_sha256": "0" * 64,
                "splits": {"train": [2022, 2023], "validation": [2024], "test_final_reserved": [2025]},
            }
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(PartitionContractError, "manifest hash differs"):
                build_package(config_path, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
