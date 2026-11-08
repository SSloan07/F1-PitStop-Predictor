"""Synthetic regression check for stopping and restoring an earlier checkpoint."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from f1_pitstop.rnn_data import load_split, read_json, write_json
from f1_pitstop import rnn_experiment
from test_rnn_experiment import fixture


class CheckpointSelectionTests(unittest.TestCase):
    def test_patience_restores_first_epoch_before_final_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, config_path, state = fixture(root)
            validation_targets = load_split(data, "validation", state).targets
            config = read_json(config_path)
            config["training"].update({"max_epochs": 10, "patience": 2})
            write_json(config_path, config)
            snapshots = []
            real_predict = rnn_experiment.predict

            def forced_validation(model, loader, device):
                targets = loader.dataset.tensors[-1].numpy()
                if not np.array_equal(targets, validation_targets):
                    return real_predict(model, loader, device)
                snapshot = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                snapshots.append(snapshot)
                call = len(snapshots)
                if call <= 3:
                    # First epoch is perfect at 0.5. Later epochs predict no
                    # positives, even though their supplied BCE is lower.
                    probabilities = (
                        np.where(targets == 1, 0.9, 0.1)
                        if call == 1 else np.full(targets.shape, 0.1)
                    )
                    return probabilities, 0.4 / call
                self.assertEqual(call, 4, "Training exceeded patience=2")
                # This call is made after stopping and before threshold fitting.
                # It must see epoch 1's restored weights, not the final epoch.
                for name, expected in snapshots[0].items():
                    self.assertTrue(torch.equal(snapshot[name], expected), name)
                return real_predict(model, loader, device)

            run = root / "run"
            with patch.object(rnn_experiment, "predict", side_effect=forced_validation):
                with redirect_stdout(io.StringIO()):
                    rnn_experiment.train_experiment(config_path, run, root)

            history = pd.read_csv(run / "history.csv")
            self.assertEqual(history.epoch.tolist(), [1, 2, 3])
            self.assertEqual(history.checkpoint_selected.tolist(), [True, False, False])
            self.assertEqual(history.validation_f2_at_0_5.tolist(), [1.0, 0.0, 0.0])
            self.assertEqual(len(snapshots), 4)
            self.assertEqual(read_json(run / "manifest.json")["best_epoch"], 1)
            self.assertEqual(read_json(run / "metrics.json")["selection_epoch"], 1)
            self.assertTrue(
                any(not torch.equal(snapshots[0][name], snapshots[2][name])
                    for name in snapshots[0]),
                "Real synthetic training must advance beyond the first checkpoint",
            )
            exported = torch.load(run / "weights.pt", map_location="cpu", weights_only=True)
            for name, expected in snapshots[0].items():
                self.assertTrue(torch.equal(exported[name], expected), name)


if __name__ == "__main__":
    unittest.main()
