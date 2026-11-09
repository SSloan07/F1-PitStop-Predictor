"""Small recurrent baseline with explicit masks and optional fusion-head dropout.

The model receives already transformed inputs. Numeric reason code 0 means an
observed value; every other code suppresses that value. A slot updates the hidden
state only when it exists, has a row, and is not padding. Category IDs 0 (PAD)
and 3 (OOV) always embed as zero; 1 (absent) and 2 (UNKNOWN) remain learnable.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import Tensor, nn

__all__ = ["SmallPitStopRNN", "model_spec_from_preprocessor"]

_DIMENSION_DEFAULTS = {
    "hidden_size": 32,
    "embedding_dim": 4,
    "static_hidden_size": 16,
    "head_hidden_size": 16,
    "window_length": 8,
    "num_recurrent_layers": 1,
}
_DROPOUT_DEFAULT = 0.0
_COUNT_KEYS = {"num_numeric", "dynamic_cardinalities", "static_cardinalities"}
_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validated_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    unknown = set(spec) - _COUNT_KEYS - set(_DIMENSION_DEFAULTS) - {"dropout_rate"}
    if unknown:
        raise ValueError(f"Unknown model spec keys: {sorted(unknown)}")
    missing = _COUNT_KEYS - set(spec)
    if missing:
        raise ValueError(f"Missing model spec keys: {sorted(missing)}")
    result: dict[str, Any] = {
        "num_numeric": _positive_integer(spec["num_numeric"], "num_numeric")
    }
    for name in ("dynamic_cardinalities", "static_cardinalities"):
        values = spec[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        result[name] = [
            _positive_integer(value, f"{name}[{index}]", minimum=4)
            for index, value in enumerate(values)
        ]
    for name, default in _DIMENSION_DEFAULTS.items():
        result[name] = _positive_integer(spec.get(name, default), name)
    dropout_rate = spec.get("dropout_rate", _DROPOUT_DEFAULT)
    if (isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float))
            or not 0.0 <= float(dropout_rate) < 1.0):
        raise ValueError("dropout_rate must be a number in [0, 1)")
    result["dropout_rate"] = float(dropout_rate)
    return result


def _feature_names(schema: dict[str, Any], key: str) -> list[str]:
    names = schema.get(key)
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"schema.{key} must be a nonempty list of unique names")
    return names


def model_spec_from_preprocessor(
    state: dict[str, Any], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Derive input counts from the frozen preprocessor, preserving its order.

    Vocabulary entries must use consecutive IDs starting at 4, after the four
    reserved IDs. Options may change only model dimensions or window length;
    they cannot substitute feature lists or vocabulary cardinalities.
    """
    if not isinstance(state, dict) or not isinstance(state.get("schema"), dict):
        raise ValueError("Preprocessor state must contain a schema dict")
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be a dict or None")
    options = {} if options is None else options
    unknown = set(options) - set(_DIMENSION_DEFAULTS) - {"dropout_rate"}
    if unknown:
        raise ValueError(f"Unsupported model options: {sorted(unknown)}")
    schema = state["schema"]
    numeric_names = _feature_names(schema, "numeric_feature_names")
    dynamic_names = _feature_names(schema, "categorical_feature_names")
    static_names = _feature_names(schema, "static_feature_names")
    vocabularies = state.get("vocabularies")
    if not isinstance(vocabularies, dict):
        raise ValueError("Preprocessor state must contain vocabularies")
    cardinalities: dict[str, list[int]] = {}
    for group, names in (("dynamic", dynamic_names), ("static", static_names)):
        vocabularies_for_group = vocabularies.get(group)
        if not isinstance(vocabularies_for_group, dict):
            raise ValueError(f"vocabularies.{group} must be a dict")
        group_counts = []
        for name in names:
            entries = vocabularies_for_group.get(name)
            if not isinstance(entries, list):
                raise ValueError(f"Vocabulary {group}.{name} must be a list")
            ids = []
            for entry in entries:
                if not isinstance(entry, dict) or "category" not in entry:
                    raise ValueError(f"Invalid vocabulary entry for {group}.{name}")
                ids.append(_positive_integer(entry.get("id"), "category id", minimum=4))
                try:
                    json.dumps(entry["category"], allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid category for {group}.{name}") from exc
            if sorted(ids) != list(range(4, len(entries) + 4)):
                raise ValueError(f"Vocabulary IDs for {group}.{name} must be consecutive from 4")
            group_counts.append(len(entries) + 4)
        cardinalities[f"{group}_cardinalities"] = group_counts
    return _validated_spec(
        {"num_numeric": len(numeric_names), **cardinalities, **options}
    )


class _CategoryEmbedding(nn.Embedding):
    """Keep PAD and OOV outputs zero, including after loading arbitrary weights."""

    def __init__(self, cardinality: int, embedding_dim: int) -> None:
        super().__init__(cardinality, embedding_dim, padding_idx=0)
        with torch.no_grad():
            self.weight[3].zero_()

    def forward(self, ids: Tensor) -> Tensor:
        vectors = super().forward(ids)
        zero_ids = (ids == 0) | (ids == 3)
        return torch.where(zero_ids.unsqueeze(-1), torch.zeros_like(vectors), vectors)


class SmallPitStopRNN(nn.Module):
    """One tanh RNNCell, a static branch, and a binary-logit fusion head.

    Optional dropout is applied after the fusion ReLU. Its default rate is zero;
    the layer has no parameters, so existing checkpoints retain compatible keys.

    ``padding_mask=True`` marks padding. ``slot_exists_mask=True`` marks a real
    timeline slot. ``slot_row_missing_mask=True`` marks an absent source row.
    These three masks have shape (batch, window), while numeric reason codes
    have the same (batch, window, numeric features) shape as numeric values.
    No hidden state is retained between calls or shared between batch members.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__()
        self.spec = _validated_spec(spec)
        if self.spec["num_recurrent_layers"] != 1:
            raise ValueError("num_recurrent_layers is supported only by the LSTM model")
        embedding_dim = self.spec["embedding_dim"]
        self.dynamic_embeddings = nn.ModuleList(
            _CategoryEmbedding(cardinality, embedding_dim)
            for cardinality in self.spec["dynamic_cardinalities"]
        )
        self.static_embeddings = nn.ModuleList(
            _CategoryEmbedding(cardinality, embedding_dim)
            for cardinality in self.spec["static_cardinalities"]
        )
        recurrent_input_size = (
            2 * self.spec["num_numeric"]
            + len(self.dynamic_embeddings) * embedding_dim
        )
        self.rnn_cell = nn.RNNCell(
            recurrent_input_size, self.spec["hidden_size"], nonlinearity="tanh"
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(
                len(self.static_embeddings) * embedding_dim,
                self.spec["static_hidden_size"],
            ),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(
                self.spec["hidden_size"] + self.spec["static_hidden_size"],
                self.spec["head_hidden_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.spec["head_hidden_size"], 1),
        )
        # Keep the historical head parameter names (head.0/head.2) so bundles
        # trained before this factor remain loadable by the shared evaluator.
        self.head_dropout = nn.Dropout(p=self.spec["dropout_rate"])

    def _validate_inputs(
        self,
        numeric: Tensor,
        dynamic_ids: Tensor,
        static_ids: Tensor,
        mask_reason_codes: Tensor,
        padding_mask: Tensor,
        slot_exists_mask: Tensor,
        slot_row_missing_mask: Tensor,
    ) -> None:
        named = {
            "numeric": numeric,
            "dynamic_ids": dynamic_ids,
            "static_ids": static_ids,
            "mask_reason_codes": mask_reason_codes,
            "padding_mask": padding_mask,
            "slot_exists_mask": slot_exists_mask,
            "slot_row_missing_mask": slot_row_missing_mask,
        }
        for name, value in named.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if numeric.ndim != 3:
            raise ValueError("numeric must have shape (batch, window, numeric features)")
        batch, window, features = numeric.shape
        if (window, features) != (self.spec["window_length"], self.spec["num_numeric"]):
            raise ValueError("numeric window or feature count does not match model spec")
        expected_shapes = {
            "dynamic_ids": (batch, window, len(self.dynamic_embeddings)),
            "static_ids": (batch, len(self.static_embeddings)),
            "mask_reason_codes": tuple(numeric.shape),
            "padding_mask": (batch, window),
            "slot_exists_mask": (batch, window),
            "slot_row_missing_mask": (batch, window),
        }
        for name, shape in expected_shapes.items():
            if tuple(named[name].shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not numeric.is_floating_point():
            raise TypeError("numeric must have a floating-point dtype")
        if numeric.dtype != self.rnn_cell.weight_ih.dtype:
            raise TypeError("numeric dtype must match the model parameter dtype")
        for name in ("dynamic_ids", "static_ids", "mask_reason_codes"):
            if named[name].dtype not in _INTEGER_DTYPES:
                raise TypeError(f"{name} must have an integer dtype")
        for name in ("padding_mask", "slot_exists_mask", "slot_row_missing_mask"):
            if named[name].dtype != torch.bool:
                raise TypeError(f"{name} must have bool dtype")
        for name, value in named.items():
            if value.device != self.rnn_cell.weight_ih.device:
                raise ValueError(f"{name} must be on the same device as the model")

    def forward(
        self,
        numeric: Tensor,
        dynamic_ids: Tensor,
        static_ids: Tensor,
        mask_reason_codes: Tensor,
        padding_mask: Tensor,
        slot_exists_mask: Tensor,
        slot_row_missing_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            numeric, dynamic_ids, static_ids, mask_reason_codes,
            padding_mask, slot_exists_mask, slot_row_missing_mask,
        )
        active = ~padding_mask & slot_exists_mask & ~slot_row_missing_mask
        if (mask_reason_codes[active] < 0).any().item():
            raise ValueError("Visible numeric reason codes must be nonnegative")
        available = (mask_reason_codes == 0) & active.unsqueeze(-1)
        safe_numeric = torch.where(available, numeric, torch.zeros_like(numeric))
        if not torch.isfinite(safe_numeric).all().item():
            raise ValueError("Observed numeric values must be finite")

        # Mask before indexing: excluded slots can contain out-of-range sentinels.
        safe_dynamic_ids = torch.where(
            active.unsqueeze(-1), dynamic_ids, torch.zeros_like(dynamic_ids)
        ).long()
        for index, cardinality in enumerate(self.spec["dynamic_cardinalities"]):
            ids = safe_dynamic_ids[..., index]
            if ((ids < 0) | (ids >= cardinality)).any().item():
                raise ValueError(f"Visible dynamic_ids feature {index} is out of bounds")
        for index, cardinality in enumerate(self.spec["static_cardinalities"]):
            ids = static_ids[:, index]
            if ((ids < 0) | (ids >= cardinality)).any().item():
                raise ValueError(f"static_ids feature {index} is out of bounds")
        dynamic_vectors = [
            embedding(safe_dynamic_ids[..., index])
            for index, embedding in enumerate(self.dynamic_embeddings)
        ]
        recurrent_input = torch.cat(
            [safe_numeric, available.to(numeric.dtype), *dynamic_vectors], dim=-1
        )
        hidden = numeric.new_zeros((numeric.shape[0], self.spec["hidden_size"]))
        for step in range(self.spec["window_length"]):
            candidate = self.rnn_cell(recurrent_input[:, step], hidden)
            hidden = torch.where(active[:, step, None], candidate, hidden)
        static_vectors = torch.cat(
            [
                embedding(static_ids[:, index].long())
                for index, embedding in enumerate(self.static_embeddings)
            ],
            dim=-1,
        )
        static_hidden = self.static_encoder(static_vectors)
        fused = self.head[0](torch.cat([hidden, static_hidden], dim=-1))
        fused = self.head[1](fused)
        fused = self.head_dropout(fused)
        return self.head[2](fused).squeeze(-1)
