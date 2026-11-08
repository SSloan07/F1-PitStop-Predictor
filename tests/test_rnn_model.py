"""Behavioral checks for the recurrent baseline; no race data is needed."""

from __future__ import annotations

import copy
import io
import json
import unittest

import torch

from f1_pitstop.rnn_model import SmallPitStopRNN, model_spec_from_preprocessor


def preprocessor_state() -> dict:
    dynamic = ["compound", "track_status"]
    static = ["driver", "team", "circuit"]
    return {
        "schema": {
            "numeric_feature_names": [f"numeric_{index}" for index in range(22)],
            "categorical_feature_names": dynamic,
            "static_feature_names": static,
        },
        "vocabularies": {
            group: {
                name: [{"id": 4, "category": "first"}, {"id": 5, "category": "second"}]
                for name in names
            }
            for group, names in (("dynamic", dynamic), ("static", static))
        },
    }


class SmallPitStopRNNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(79)
        self.spec = model_spec_from_preprocessor(preprocessor_state())
        self.model = SmallPitStopRNN(self.spec)

    def inputs(self, batch: int = 3) -> dict[str, torch.Tensor]:
        return {
            "numeric": torch.randn(batch, 8, 22),
            "dynamic_ids": torch.full((batch, 8, 2), 4, dtype=torch.long),
            "static_ids": torch.full((batch, 3), 4, dtype=torch.long),
            "mask_reason_codes": torch.zeros(batch, 8, 22, dtype=torch.long),
            "padding_mask": torch.zeros(batch, 8, dtype=torch.bool),
            "slot_exists_mask": torch.ones(batch, 8, dtype=torch.bool),
            "slot_row_missing_mask": torch.zeros(batch, 8, dtype=torch.bool),
        }

    def test_spec_uses_frozen_feature_order_and_json_values(self) -> None:
        spec = model_spec_from_preprocessor(
            preprocessor_state(),
            {"hidden_size": 11, "embedding_dim": 3, "window_length": 5},
        )
        self.assertEqual(spec["num_numeric"], 22)
        self.assertEqual(spec["dynamic_cardinalities"], [6, 6])
        self.assertEqual(spec["static_cardinalities"], [6, 6, 6])
        self.assertEqual(spec["hidden_size"], 11)
        self.assertEqual(spec["embedding_dim"], 3)
        self.assertEqual(spec["window_length"], 5)
        self.assertEqual(spec, json.loads(json.dumps(spec)))
        supplied = copy.deepcopy(spec)
        model = SmallPitStopRNN(supplied)
        supplied["dynamic_cardinalities"][0] = 100
        self.assertEqual(model.spec["dynamic_cardinalities"][0], 6)

    def test_batch_and_singleton_return_one_finite_logit_per_sample(self) -> None:
        for batch in (1, 3):
            with self.subTest(batch=batch):
                logits = self.model(**self.inputs(batch))
                self.assertEqual(tuple(logits.shape), (batch,))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_each_slot_mask_excludes_junk_without_indexing_it(self) -> None:
        inputs = self.inputs()
        inputs["padding_mask"][0, 0] = True
        inputs["slot_exists_mask"][1, 3] = False
        inputs["slot_row_missing_mask"][2, 6] = True
        expected = self.model(**inputs)
        active = (
            ~inputs["padding_mask"]
            & inputs["slot_exists_mask"]
            & ~inputs["slot_row_missing_mask"]
        )
        inputs["numeric"][~active] = float("nan")
        inputs["dynamic_ids"][~active] = -99999
        inputs["mask_reason_codes"][~active] = -99999
        actual = self.model(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_missing_numeric_values_are_sanitized_and_reason_is_only_presence(self) -> None:
        inputs = self.inputs()
        inputs["mask_reason_codes"][:, 2, 1] = 1
        expected = self.model(**inputs)
        inputs["numeric"][:, 2, 1] = float("nan")
        inputs["mask_reason_codes"][:, 2, 1] = 9
        torch.testing.assert_close(self.model(**inputs), expected, rtol=0, atol=0)

    def test_physical_zero_and_missing_zero_have_different_availability_bits(self) -> None:
        inputs = self.inputs(2)
        inputs["numeric"].zero_()
        inputs["mask_reason_codes"][1, 0, 7] = 1
        recurrent_inputs = []

        def capture(_module, args):
            recurrent_inputs.append(args[0].detach().clone())

        handle = self.model.rnn_cell.register_forward_pre_hook(capture)
        try:
            self.model(**inputs)
        finally:
            handle.remove()
        first_step = recurrent_inputs[0]
        torch.testing.assert_close(first_step[0, :22], first_step[1, :22])
        self.assertEqual(first_step[0, 22 + 7].item(), 1)
        self.assertEqual(first_step[1, 22 + 7].item(), 0)
        self.assertEqual(tuple(first_step.shape), (2, 22 + 22 + 2 * 4))

    def test_skipped_slot_does_not_update_or_shift_hidden_timeline(self) -> None:
        inputs = self.inputs(1)
        inputs["slot_row_missing_mask"][0, 2] = True
        hidden_before_each_slot = []

        def capture(_module, args):
            hidden_before_each_slot.append(args[1].detach().clone())

        handle = self.model.rnn_cell.register_forward_pre_hook(capture)
        try:
            self.model(**inputs)
        finally:
            handle.remove()
        self.assertEqual(len(hidden_before_each_slot), 8)
        torch.testing.assert_close(
            hidden_before_each_slot[2], hidden_before_each_slot[3], rtol=0, atol=0
        )
        self.assertTrue((hidden_before_each_slot[0] == 0).all().item())

    def test_hidden_state_does_not_cross_samples_batches_or_calls(self) -> None:
        inputs = self.inputs(3)
        expected = self.model(**inputs)
        separately = torch.cat(
            [self.model(**{key: value[index:index + 1] for key, value in inputs.items()})
             for index in range(3)]
        )
        torch.testing.assert_close(separately, expected, rtol=1e-5, atol=1e-6)
        permutation = torch.tensor([2, 0, 1])
        shuffled = self.model(**{key: value[permutation] for key, value in inputs.items()})
        torch.testing.assert_close(shuffled, expected[permutation], rtol=1e-5, atol=1e-6)
        self.model(**self.inputs(1))
        torch.testing.assert_close(self.model(**inputs), expected, rtol=0, atol=0)

    def test_pad_and_oov_remain_zero_with_changed_or_reloaded_weights(self) -> None:
        inputs = self.inputs()
        inputs["dynamic_ids"].fill_(3)
        inputs["static_ids"].fill_(3)
        expected = self.model(**inputs)
        with torch.no_grad():
            for embedding in (*self.model.dynamic_embeddings, *self.model.static_embeddings):
                embedding.weight[0].fill_(123)
                embedding.weight[3].fill_(-321)
        torch.testing.assert_close(self.model(**inputs), expected, rtol=0, atol=0)
        restored = SmallPitStopRNN(self.spec)
        restored.load_state_dict(self.model.state_dict())
        for embedding in (*restored.dynamic_embeddings, *restored.static_embeddings):
            vectors = embedding(torch.tensor([0, 3]))
            self.assertTrue((vectors == 0).all().item())
        torch.testing.assert_close(restored(**inputs), expected, rtol=0, atol=0)

    def test_absent_and_unknown_are_distinct_learnable_categories(self) -> None:
        embedding = self.model.dynamic_embeddings[0]
        ids = torch.tensor([0, 1, 2, 3])
        vectors = embedding(ids)
        self.assertFalse(torch.equal(vectors[1], vectors[2]))
        vectors.sum().backward()
        self.assertTrue((embedding.weight.grad[0] == 0).all().item())
        self.assertTrue((embedding.weight.grad[3] == 0).all().item())
        self.assertTrue((embedding.weight.grad[1] != 0).all().item())
        self.assertTrue((embedding.weight.grad[2] != 0).all().item())

    def test_gradients_and_portable_state_dict_roundtrip(self) -> None:
        inputs = self.inputs()
        inputs["numeric"].requires_grad_()
        inputs["mask_reason_codes"][0, 1, 2] = 1
        with torch.no_grad():
            inputs["numeric"][0, 1, 2] = float("nan")
        logits = self.model(**inputs)
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.0, 1.0, 1.0])
        ).backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        self.assertGreater(self.model.rnn_cell.weight_ih.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(inputs["numeric"].grad).all().item())
        self.assertEqual(inputs["numeric"].grad[0, 1, 2].item(), 0)
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = SmallPitStopRNN(json.loads(json.dumps(self.model.spec)))
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored(**inputs), logits, rtol=0, atol=0)

    def test_visible_invalid_inputs_fail_with_clear_errors(self) -> None:
        mutations = [
            ("dynamic_ids", lambda value: value.fill_(6), ValueError),
            ("static_ids", lambda value: value.fill_(-1), ValueError),
            ("mask_reason_codes", lambda value: value.fill_(-1), ValueError),
            ("numeric", lambda value: value.fill_(float("nan")), ValueError),
        ]
        for name, mutate, error in mutations:
            with self.subTest(name=name):
                inputs = self.inputs()
                mutate(inputs[name])
                with self.assertRaises(error):
                    self.model(**inputs)
        for name in self.inputs():
            with self.subTest(shape=name):
                inputs = self.inputs()
                inputs[name] = inputs[name][..., :-1]
                with self.assertRaises(ValueError):
                    self.model(**inputs)
        for name in ("dynamic_ids", "static_ids", "mask_reason_codes", "padding_mask"):
            with self.subTest(dtype=name):
                inputs = self.inputs()
                inputs[name] = inputs[name].float()
                with self.assertRaises(TypeError):
                    self.model(**inputs)

    def test_malformed_model_and_preprocessor_specs_fail(self) -> None:
        for patch in (
            {"num_numeric": 0}, {"hidden_size": True}, {"embedding_dim": -1},
            {"window_length": 0}, {"dynamic_cardinalities": []},
            {"static_cardinalities": [3]}, {"label": "target"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                SmallPitStopRNN({**self.spec, **patch})
        with self.assertRaises(ValueError):
            SmallPitStopRNN({})
        with self.assertRaises(ValueError):
            model_spec_from_preprocessor(preprocessor_state(), {"num_numeric": 10})
        with self.assertRaises(ValueError):
            model_spec_from_preprocessor(preprocessor_state(), {"window_length": 1.5})
        for change in ("duplicate_ids", "missing_vocabulary", "duplicate_name"):
            state = preprocessor_state()
            if change == "duplicate_ids":
                state["vocabularies"]["dynamic"]["compound"][1]["id"] = 4
            elif change == "missing_vocabulary":
                del state["vocabularies"]["static"]["team"]
            else:
                state["schema"]["numeric_feature_names"][1] = "numeric_0"
            with self.subTest(change=change), self.assertRaises(ValueError):
                model_spec_from_preprocessor(state)


if __name__ == "__main__":
    unittest.main()
