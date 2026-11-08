"""Synthetic checks for an F1 objective without changing the BCE training loss."""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from f1_pitstop import rnn_experiment
from f1_pitstop.experiment_metrics import select_metric_threshold
from f1_pitstop.rnn_data import load_split, read_json, sha256, write_json
from test_rnn_experiment import STAGING, fixture


def f1_config(config):
    result = copy.deepcopy(config)
    result["experiment_id"] = "rnn_reference_f1_v1"
    result["selection"].update({"primary_metric": "positive_class_f1", "beta": 1})
    return result


class F1ExperimentTests(unittest.TestCase):
    def test_rejects_inconsistent_metric_identifiers_beta_or_training_loss(self):
        original = read_json(STAGING / "configs/experimento_rnn_v1.json")
        self.assertEqual(rnn_experiment._selection_protocol(original), ("f2", 2))
        changed = f1_config(original)
        rnn_experiment._validate_config(changed)
        self.assertEqual(rnn_experiment._selection_protocol(changed), ("f1", 1))
        invalid = []
        unknown = copy.deepcopy(changed)
        unknown["experiment_id"] = "rnn_unknown"
        invalid.append(unknown)
        for config, primary in (
            (original, "positive_class_f1"),
            (changed, "positive_class_f2"),
        ):
            wrong = copy.deepcopy(config)
            wrong["selection"]["primary_metric"] = primary
            invalid.append(wrong)
        for beta in (0, 2, True, "1", float("nan")):
            wrong = copy.deepcopy(changed)
            wrong["selection"]["beta"] = beta
            invalid.append(wrong)
        wrong = copy.deepcopy(original)
        wrong["selection"]["beta"] = 1
        invalid.append(wrong)
        wrong = copy.deepcopy(changed)
        wrong["training"]["loss"] = "f1_loss"
        invalid.append(wrong)
        for index, config in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                rnn_experiment._validate_config(config)

    def test_f1_checkpoint_stopping_and_threshold_differ_from_f2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, config_path, state = fixture(root)
            targets = load_split(data, "validation", state).targets
            config = f1_config(read_json(config_path))
            config["training"].update({"max_epochs": 10, "patience": 2})
            config["selection"]["threshold_grid"] = [0.2, 0.5]
            write_json(config_path, config)
            positive = np.flatnonzero(targets)
            negative = np.flatnonzero(targets == 0)
            first_scores = np.full(targets.shape, 0.3, dtype=float)
            first_scores[negative] = [0.3, 0.4]
            first_scores[positive[:-1]] = 0.9
            first_scores[positive[-1]] = 0.2
            later_scores = np.full(targets.shape, 0.9)
            snapshots = []
            real_predict = rnn_experiment.predict

            def forced_validation(model, loader, device):
                if not np.array_equal(loader.dataset.tensors[-1].numpy(), targets):
                    return real_predict(model, loader, device)
                snapshot = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                snapshots.append(snapshot)
                call = len(snapshots)
                if call <= 3:
                    return (first_scores if call == 1 else later_scores), 0.4 / call
                self.assertEqual(call, 4, "F1 patience should stop after epoch 3")
                for name, expected in snapshots[0].items():
                    self.assertTrue(torch.equal(snapshot[name], expected), name)
                # Fixed scores make threshold selection independently auditable;
                # model restoration is checked above against actual trained weights.
                return first_scores, 0.4

            run = root / "run_f1"
            with patch.object(rnn_experiment, "predict", side_effect=forced_validation):
                with redirect_stdout(io.StringIO()):
                    rnn_experiment.train_experiment(config_path, run, root)

            history = pd.read_csv(run / "history.csv")
            self.assertEqual(history.epoch.tolist(), [1, 2, 3])
            self.assertEqual(history.checkpoint_selected.tolist(), [True, False, False])
            self.assertGreater(history.validation_f1_at_0_5[0], history.validation_f1_at_0_5[1])
            self.assertLess(history.validation_f2_at_0_5[0], history.validation_f2_at_0_5[1])
            self.assertIn("train_f1_at_0_5", history)
            self.assertIn("train_f2_at_0_5", history)
            self.assertEqual(len(snapshots), 4)
            manifest = read_json(run / "manifest.json")
            metrics = read_json(run / "metrics.json")
            self.assertEqual(manifest["best_epoch"], 1)
            self.assertEqual(manifest["selection_metric"], "f1")
            self.assertEqual(manifest["selection_beta"], 1)
            self.assertEqual(metrics["selection_epoch"], 1)
            self.assertIn("F1 at fixed threshold 0.5", metrics["checkpoint_selection"])
            selected = read_json(run / "threshold.json")
            counterfactual = select_metric_threshold(targets, first_scores, [0.2, 0.5], metric="f2")
            self.assertEqual(selected["metric"], "f1")
            self.assertEqual(selected["beta"], 1)
            self.assertEqual(selected["threshold"], 0.5)
            self.assertEqual(counterfactual["threshold"], 0.2)
            self.assertGreater(selected["metrics"]["f1"], counterfactual["metrics"]["f1"])
            self.assertLess(selected["metrics"]["f2"], counterfactual["metrics"]["f2"])
            exported = torch.load(run / "weights.pt", map_location="cpu", weights_only=True)
            for name, expected in snapshots[0].items():
                self.assertTrue(torch.equal(exported[name], expected), name)
            self.assertTrue(any(
                not torch.equal(snapshots[0][name], snapshots[2][name])
                for name in exported
            ))

    def test_synthetic_f1_export_and_cpu_evaluation_keep_both_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, config_path, _ = fixture(root)
            config = f1_config(read_json(config_path))
            config["training"]["max_epochs"] = 1
            write_json(config_path, config)
            run, output = root / "run_f1", root / "evaluation"
            with redirect_stdout(io.StringIO()):
                rnn_experiment.train_experiment(config_path, run, root)
            rnn_experiment.evaluate_bundle(run, data, output, "cpu", root)
            result = read_json(output / "metrics.json")
            self.assertTrue(result["parity_passed"])
            self.assertEqual(result["selection_metric"], "f1")
            self.assertEqual(result["selection_beta"], 1)
            self.assertIn("f1", result["metrics"])
            self.assertIn("f2", result["metrics"])
            self.assertEqual(read_json(run / "config.json")["training"]["loss"],
                             "unweighted_bce_with_logits")
            self.assertEqual(read_json(run / "manifest.json")["model_source_sha256"],
                             sha256(STAGING / "src/f1_pitstop/rnn_model.py"))

    def test_legacy_f2_bundle_without_explicit_beta_still_evaluates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, config_path, _ = fixture(root)
            config = read_json(config_path)
            config["training"]["max_epochs"] = 1
            write_json(config_path, config)
            run, output = root / "run_f2", root / "evaluation"
            with redirect_stdout(io.StringIO()):
                rnn_experiment.train_experiment(config_path, run, root)
            old_config = read_json(run / "config.json")
            old_config["selection"].pop("beta", None)
            write_json(run / "config.json", old_config)
            threshold = read_json(run / "threshold.json")
            threshold.pop("metric", None)
            threshold.pop("beta", None)
            write_json(run / "threshold.json", threshold)
            manifest = read_json(run / "manifest.json")
            for key in ("selection_metric", "selection_beta", "checkpoint_selection"):
                manifest.pop(key, None)
            for name in ("config.json", "threshold.json"):
                manifest["files_sha256"][name] = sha256(run / name)
            write_json(run / "manifest.json", manifest)
            rnn_experiment.evaluate_bundle(run, data, output, "cpu", root)
            result = read_json(output / "metrics.json")
            self.assertTrue(result["parity_passed"])
            self.assertEqual(result["selection_metric"], "f2")
            self.assertEqual(result["selection_beta"], 2)


if __name__ == "__main__":
    unittest.main()
