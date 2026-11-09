"""Causal LSTM model reusing the frozen RNN feature and FFNN contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .rnn_model import SmallPitStopRNN, _CategoryEmbedding, _validated_spec

__all__ = ["SmallPitStopLSTM"]


class SmallPitStopLSTM(SmallPitStopRNN):
    """One unidirectional LSTMCell, static FFNN branch and fusion head.

    Padding and unavailable source slots preserve both hidden and cell state.
    Every sample starts from zero state; no state crosses sample boundaries.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        if not isinstance(spec, dict) or spec.get("recurrent_cell") != "lstm":
            raise ValueError("SmallPitStopLSTM requires recurrent_cell=lstm")
        base_spec = {key: value for key, value in spec.items() if key != "recurrent_cell"}
        nn.Module.__init__(self)
        self.spec = _validated_spec(base_spec)
        self.spec["recurrent_cell"] = "lstm"
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
        self.lstm_cell = nn.LSTMCell(recurrent_input_size, self.spec["hidden_size"])
        self.additional_lstm_cells = nn.ModuleList(
            nn.LSTMCell(self.spec["hidden_size"], self.spec["hidden_size"])
            for _ in range(self.spec["num_recurrent_layers"] - 1)
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(len(self.static_embeddings) * embedding_dim, self.spec["static_hidden_size"]),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(self.spec["hidden_size"] + self.spec["static_hidden_size"], self.spec["head_hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.spec["head_hidden_size"], 1),
        )
        self.head_dropout = nn.Dropout(p=self.spec["dropout_rate"])

    @property
    def rnn_cell(self):
        # Reuse the inherited input validator without changing the frozen RNN module.
        return self.lstm_cell

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
        cells = (self.lstm_cell, *self.additional_lstm_cells)
        hidden_states = [
            numeric.new_zeros((numeric.shape[0], self.spec["hidden_size"]))
            for _ in cells
        ]
        cell_states = [torch.zeros_like(hidden) for hidden in hidden_states]
        for step in range(self.spec["window_length"]):
            layer_input = recurrent_input[:, step]
            active_step = active[:, step, None]
            for layer_index, cell in enumerate(cells):
                candidate_hidden, candidate_cell = cell(
                    layer_input, (hidden_states[layer_index], cell_states[layer_index])
                )
                hidden_states[layer_index] = torch.where(
                    active_step, candidate_hidden, hidden_states[layer_index]
                )
                cell_states[layer_index] = torch.where(
                    active_step, candidate_cell, cell_states[layer_index]
                )
                layer_input = hidden_states[layer_index]
        hidden = hidden_states[-1]
        static_vectors = torch.cat(
            [embedding(static_ids[:, index].long())
             for index, embedding in enumerate(self.static_embeddings)], dim=-1
        )
        static_hidden = self.static_encoder(static_vectors)
        fused = self.head_dropout(self.head[1](self.head[0](torch.cat([hidden, static_hidden], dim=-1))))
        return self.head[2](fused).squeeze(-1)
