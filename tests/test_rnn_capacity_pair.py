"""Controlled capacity comparison checks using only the tiny synthetic fixture."""

import copy
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from f1_pitstop.rnn_data import read_json, sha256, write_json
from f1_pitstop.rnn_experiment import evaluate_bundle
from test_rnn_experiment import fixture
import entrenar_comparacion_capacidad as pair


def pair_fixture(root):
    data, original_path, _ = fixture(root)
    reference = read_json(original_path)
    reference["training"]["max_epochs"] = 1
    reference["selection"]["beta"] = 2
    candidate = copy.deepcopy(reference)
    candidate["experiment_id"] = "rnn_hidden16_f2_v1"
    candidate["model"]["hidden_size"] = 16
    candidate["question"] = "Synthetic capacity comparison"
    reference["pairing"] = {"role": "reference"}
    candidate["pairing"] = {"role": "candidate"}
    reference_path = root / "configs/reference.json"
    candidate_path = root / "configs/candidate.json"
    write_json(reference_path, reference)
    write_json(candidate_path, candidate)
    return data, reference_path, candidate_path


class CapacityPairTests(unittest.TestCase):
    def test_synthetic_pair_exports_exact_zip_and_both_models_evaluate(self):
        original_determinism = torch.are_deterministic_algorithms_enabled()
        original_threads = torch.get_num_threads()
        self.addCleanup(torch.use_deterministic_algorithms, original_determinism)
        self.addCleanup(torch.set_num_threads, original_threads)
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, reference, candidate = pair_fixture(root)
            output = root / "pair"
            with redirect_stdout(io.StringIO()):
                result = pair.train_capacity_pair(reference, candidate, output, root)
            self.assertEqual(result, output)
            self.assertEqual(read_json(output / "status.json")["status"], "complete")
            manifest = read_json(output / "pair_manifest.json")
            self.assertEqual(manifest["artifact_version"], "rnn_capacity_pair_v1")
            self.assertTrue(manifest["comparison_authorized"])
            self.assertFalse(manifest["test_used"])
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["train_eligible_samples"], 7)
            self.assertEqual(manifest["validation_eligible_samples"], 8)
            self.assertEqual(manifest["selection_metric"], "f2")
            environment = read_json(output / "pair_environment.json")
            self.assertEqual(set(environment["snapshots"]), {"before", "between", "after"})
            self.assertTrue(environment["comparison_authorized"])
            self.assertFalse(environment["snapshots"]["before"]["deterministic_algorithms"])
            self.assertEqual(environment["snapshots"]["before"]["cpu_num_threads"], 3)
            self.assertTrue(environment["snapshots"]["after"]["deterministic_algorithms"])
            self.assertEqual(environment["snapshots"]["after"]["cpu_num_threads"], 2)
            for child in environment["children"].values():
                self.assertEqual(child["device"], "cpu")
                self.assertEqual(child["seed"], 42)
                self.assertTrue(child["deterministic_algorithms"])
            summary = read_json(output / "pair_summary.json")
            self.assertEqual(summary["reference"]["hidden_size"], 32)
            self.assertEqual(summary["candidate"]["hidden_size"], 16)
            self.assertLess(summary["candidate"]["parameter_count"], summary["reference"]["parameter_count"])
            for role in ("reference", "candidate"):
                for population in ("train", "validation"):
                    for threshold in ("fixed_threshold_0_5", "selected_threshold"):
                        self.assertTrue({"f1", "f2", "average_precision", "precision", "recall"}
                                        .issubset(summary[role][population][threshold]))
            for child in ("rnn32_reference", "rnn16_candidate"):
                self.assertTrue((output / f"{child}.zip").exists())
                evaluation = root / f"evaluation_{child}"
                evaluate_bundle(output / child, data, evaluation, "cpu", root)
                self.assertTrue(read_json(evaluation / "metrics.json")["parity_passed"])
            expected = set(manifest["files_sha256"]) | {"pair_manifest.json"}
            with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
                self.assertEqual(set(archive.namelist()), expected)
                self.assertFalse(any(name.endswith(".zip") for name in archive.namelist()))
                for name, digest in manifest["files_sha256"].items():
                    self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                    self.assertEqual(sha256(output / name), digest)
            original_hash = sha256(output / "rnn32_reference/weights.pt")
            with self.assertRaises(FileExistsError):
                pair.train_capacity_pair(reference, candidate, output, root)
            self.assertEqual(sha256(output / "rnn32_reference/weights.pt"), original_hash)

    def test_material_control_differences_fail_before_any_output_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reference, candidate_path = pair_fixture(root)
            original = read_json(candidate_path)
            mutations = [
                ("learning_rate", lambda config: config["training"].update(learning_rate=0.002)),
                ("patience", lambda config: config["training"].update(patience=2)),
                ("seed", lambda config: config["training"].update(seed=7)),
                ("model", lambda config: config["model"].update(embedding_dim=8)),
                ("capacity", lambda config: config["model"].update(hidden_size=32)),
                ("grid", lambda config: config["selection"].update(threshold_grid=[0.1, 0.5])),
                ("hash", lambda config: config.update(preprocessed_manifest_sha256="changed")),
                ("mask", lambda config: config.update(mask_representation="changed")),
                ("events", lambda config: config["evaluation"].update(events_sha256="changed")),
                ("test", lambda config: config.update(test_policy="changed")),
                ("beta", lambda config: config["selection"].update(beta=1)),
            ]
            for name, mutate in mutations:
                with self.subTest(control=name):
                    candidate = copy.deepcopy(original)
                    mutate(candidate)
                    write_json(candidate_path, candidate)
                    output = root / f"rejected_{name}"
                    with patch.object(pair, "train_experiment") as training:
                        with self.assertRaises(ValueError):
                            pair.train_capacity_pair(reference, candidate_path, output, root)
                        training.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix(".zip").exists())

    def test_environment_change_between_models_blocks_comparison_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reference, candidate = pair_fixture(root)
            output = root / "failed_pair"
            capture = pair._capture_environment
            calls = 0

            def changed_environment(protocol):
                nonlocal calls
                result = capture(protocol)
                calls += 1
                if calls == 2:
                    result["platform"] = "different effective environment"
                return result

            with patch.object(pair, "_capture_environment", side_effect=changed_environment):
                with patch.object(pair, "train_experiment", wraps=pair.train_experiment) as training:
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, "Effective environment changed"):
                            pair.train_capacity_pair(reference, candidate, output, root)
                    self.assertEqual(training.call_count, 1)
            status = read_json(output / "status.json")
            self.assertEqual(status["status"], "failed")
            self.assertFalse(status["comparison_authorized"])
            self.assertEqual(status["stage"], "rnn32_reference_environment_gate")
            self.assertTrue((output / "rnn32_reference/weights.pt").exists())
            self.assertFalse((output / "rnn16_candidate").exists())
            self.assertFalse((output / "pair_summary.json").exists())
            self.assertFalse((output / "pair_manifest.json").exists())
            self.assertFalse(output.with_suffix(".zip").exists())
            environment = read_json(output / "pair_environment.json")
            self.assertEqual(environment["snapshots"]["between"]["platform"],
                             "different effective environment")
            self.assertFalse(environment["comparison_authorized"])

    def test_changed_child_population_blocks_comparison_even_with_valid_file_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reference, candidate = pair_fixture(root)
            output = root / "failed_population"
            train = pair.train_experiment

            def changed_population(config, child, project):
                result = train(config, child, project)
                if child.name == "rnn16_candidate":
                    manifest = read_json(child / "manifest.json")
                    manifest["validation_eligible_samples"] += 1
                    write_json(child / "manifest.json", manifest)
                return result

            with patch.object(pair, "train_experiment", side_effect=changed_population):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "populations or frozen controls differ"):
                        pair.train_capacity_pair(reference, candidate, output, root)
            self.assertEqual(read_json(output / "status.json")["status"], "failed")
            self.assertFalse((output / "pair_summary.json").exists())
            self.assertFalse(output.with_suffix(".zip").exists())

    def test_config_changed_after_reference_is_rejected_before_candidate_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reference, candidate = pair_fixture(root)
            output = root / "changed_config"
            train = pair.train_experiment

            def mutate_config(config, child, project):
                result = train(config, child, project)
                modified = read_json(candidate)
                modified["training"]["learning_rate"] *= 2
                write_json(candidate, modified)
                return result

            with patch.object(pair, "train_experiment", side_effect=mutate_config) as training:
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "config changed"):
                        pair.train_capacity_pair(reference, candidate, output, root)
                self.assertEqual(training.call_count, 1)
            self.assertEqual(read_json(output / "status.json")["status"], "failed")
            self.assertFalse((output / "rnn16_candidate").exists())
            self.assertFalse(output.with_suffix(".zip").exists())


if __name__ == "__main__":
    unittest.main()
