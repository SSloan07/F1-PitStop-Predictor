"""Train-only preprocessing for causal F1 pit-stop samples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class PreprocessingContractError(ValueError):
    """Raised when a preprocessor schema or input violates the frozen contract."""


RESERVED_CATEGORY_IDS = {"padding": 0, "absent": 1, "source_unknown": 2, "out_of_vocabulary": 3}
MISSING_CATEGORY_TOKENS = {"", "nan", "none", "<na>"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def state_sha256(state: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def category_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    text = str(value)
    return None if text.casefold() in MISSING_CATEGORY_TOKENS else text


def validate_input_arrays(
    numeric: np.ndarray,
    categorical: np.ndarray,
    static: np.ndarray,
    masks: dict[str, np.ndarray],
    schema: dict[str, Any],
) -> None:
    required_masks = {
        "padding_mask", "slot_exists_mask", "slot_row_missing_mask",
        "temporal_unavailable_mask", "feature_missing_mask", "mask_reason_codes",
    }
    if set(masks) != required_masks:
        raise PreprocessingContractError(
            f"Mask schema mismatch; expected {sorted(required_masks)}, got {sorted(masks)}"
        )
    if numeric.ndim != 3 or categorical.ndim != 3 or static.ndim != 2:
        raise PreprocessingContractError("Expected numeric/categorical [samples,W,features] and static [samples,features]")
    n, width, n_numeric = numeric.shape
    if categorical.shape[:2] != (n, width) or static.shape[0] != n:
        raise PreprocessingContractError("Input arrays disagree on sample count or window length")
    if width != int(schema["window_length"]):
        raise PreprocessingContractError(f"Window length mismatch: {width} != {schema['window_length']}")
    if n_numeric != len(schema["numeric_feature_names"]):
        raise PreprocessingContractError("Numeric feature dimension does not match registry")
    if categorical.shape[2] != len(schema["categorical_feature_names"]):
        raise PreprocessingContractError("Categorical feature dimension does not match registry")
    if static.shape[1] != len(schema["static_feature_names"]):
        raise PreprocessingContractError("Static feature dimension does not match registry")
    if not np.issubdtype(numeric.dtype, np.number):
        raise PreprocessingContractError("Numeric input dtype is not numeric")
    if categorical.dtype.kind not in {"U", "S", "O"} or static.dtype.kind not in {"U", "S", "O"}:
        raise PreprocessingContractError("Categorical inputs must contain strings or nullable objects")

    expected_shapes = {
        "padding_mask": (n, width), "slot_exists_mask": (n, width),
        "slot_row_missing_mask": (n, width), "temporal_unavailable_mask": (n, width, n_numeric + categorical.shape[2]),
        "feature_missing_mask": (n, width, n_numeric + categorical.shape[2]),
        "mask_reason_codes": (n, width, n_numeric + categorical.shape[2]),
    }
    for name, shape in expected_shapes.items():
        if masks[name].shape != shape:
            raise PreprocessingContractError(f"{name} shape mismatch: {masks[name].shape} != {shape}")
    padding = masks["padding_mask"].astype(bool, copy=False)
    exists = masks["slot_exists_mask"].astype(bool, copy=False)
    if not np.array_equal(padding, ~exists):
        raise PreprocessingContractError("Padding and slot-existence masks disagree")
    if not np.array_equal(masks["mask_reason_codes"] == 1,
                          np.broadcast_to(padding[..., None], masks["mask_reason_codes"].shape)):
        raise PreprocessingContractError("Padding mask disagrees with feature reason codes")
    if not np.array_equal(
        masks["mask_reason_codes"] == 2,
        np.broadcast_to(masks["slot_row_missing_mask"].astype(bool)[..., None], masks["mask_reason_codes"].shape),
    ):
        raise PreprocessingContractError("Missing-row mask disagrees with feature reason codes")
    if not np.array_equal(masks["mask_reason_codes"] == 3, masks["temporal_unavailable_mask"].astype(bool)):
        raise PreprocessingContractError("Temporal-unavailability mask disagrees with feature reason codes")
    expected_missing = np.isin(masks["mask_reason_codes"], [2, 4, 5])
    if not np.array_equal(expected_missing, masks["feature_missing_mask"].astype(bool)):
        raise PreprocessingContractError("Feature-missing mask disagrees with feature reason codes")
    if np.any((masks["mask_reason_codes"] > 5)):
        raise PreprocessingContractError("Unknown mask reason code")
    present = masks["mask_reason_codes"] == 0
    if not np.isfinite(numeric[present[..., :n_numeric]]).all():
        raise PreprocessingContractError("A numeric value marked present is non-finite")


class InputPreprocessor:
    """Immutable-after-fit scaler and deterministic categorical vocabularies."""

    def __init__(self, state: dict[str, Any] | None = None):
        self._state = state

    @property
    def fitted(self) -> bool:
        return self._state is not None

    def to_dict(self) -> dict[str, Any]:
        if self._state is None:
            raise PreprocessingContractError("Preprocessor has not been fitted")
        return json.loads(canonical_json(self._state))

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "InputPreprocessor":
        if state.get("preprocessor_version") != "train_only_preprocessor_v1":
            raise PreprocessingContractError("Unsupported preprocessor version")
        if state.get("reserved_category_ids") != RESERVED_CATEGORY_IDS:
            raise PreprocessingContractError("Reserved category ID contract mismatch")
        obj = cls(json.loads(canonical_json(state)))
        for branch in ("dynamic", "static"):
            for feature, pairs in obj._state["vocabularies"][branch].items():
                ids = [int(pair["id"]) for pair in pairs]
                categories = [pair["category"] for pair in pairs]
                if ids != list(range(4, 4 + len(ids))) or categories != sorted(set(categories)):
                    raise PreprocessingContractError(f"Malformed deterministic vocabulary: {branch}/{feature}")
        return obj

    def fit(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        static: np.ndarray,
        masks: dict[str, np.ndarray],
        schema: dict[str, Any],
        fit_rows: np.ndarray,
        fit_sample_ids_sha256: str,
    ) -> "InputPreprocessor":
        if self._state is not None:
            raise PreprocessingContractError("Fitted preprocessor is frozen; create a new instance to refit")
        validate_input_arrays(numeric, categorical, static, masks, schema)
        fit_rows = np.asarray(fit_rows, dtype=bool)
        if fit_rows.shape != (numeric.shape[0],) or not fit_rows.any():
            raise PreprocessingContractError("Fit population must be a non-empty boolean sample mask")
        width = numeric.shape[1]
        num_names = schema["numeric_feature_names"]
        cat_names = schema["categorical_feature_names"]
        static_names = schema["static_feature_names"]
        reason = masks["mask_reason_codes"]
        exists = masks["slot_exists_mask"].astype(bool, copy=False)
        not_padding = ~masks["padding_mask"].astype(bool, copy=False)
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        counts: dict[str, int] = {}
        constants: list[str] = []
        zeros: dict[str, int] = {}
        fit_numeric = numeric[fit_rows]
        fit_reason = reason[fit_rows]
        fit_exists = exists[fit_rows] & not_padding[fit_rows]
        for feature_idx, feature in enumerate(num_names):
            observed = (fit_reason[:, :, feature_idx] == 0) & fit_exists
            values = fit_numeric[:, :, feature_idx][observed].astype(np.float64, copy=False)
            if values.size == 0:
                raise PreprocessingContractError(
                    f"No observed training values for numeric feature {feature!r}; refusing to invent statistics"
                )
            if not np.isfinite(values).all():
                raise PreprocessingContractError(f"Non-finite observed training values in {feature!r}")
            mean = float(values.mean(dtype=np.float64))
            variance = float(np.mean((values - mean) ** 2, dtype=np.float64))
            constant = variance == 0.0
            means[feature] = mean
            scales[feature] = 1.0 if constant else float(np.sqrt(variance))
            counts[feature] = int(values.size)
            zeros[feature] = int(np.count_nonzero(values == 0.0))
            if constant:
                constants.append(feature)

        vocabularies: dict[str, dict[str, list[dict[str, Any]]]] = {"static": {}, "dynamic": {}}
        cat_vocab_sets: dict[str, set[str]] = {}
        fit_cat = categorical[fit_rows]
        for feature_idx, feature in enumerate(cat_names):
            observed = (fit_reason[:, :, len(num_names) + feature_idx] == 0) & fit_exists
            values: set[str] = set()
            for value in fit_cat[:, :, feature_idx][observed].tolist():
                normalized = category_value(value)
                if normalized is None:
                    raise PreprocessingContractError(
                        f"Empty category is marked present in training feature {feature!r}"
                    )
                if normalized != "UNKNOWN":
                    values.add(normalized)
            ordered = sorted(values)
            vocabularies["dynamic"][feature] = [
                {"category": value, "id": idx + 4} for idx, value in enumerate(ordered)
            ]
            cat_vocab_sets[feature] = values
        fit_static = static[fit_rows]
        for feature_idx, feature in enumerate(static_names):
            values = set()
            for value in fit_static[:, feature_idx].tolist():
                normalized = category_value(value)
                if normalized is not None and normalized != "UNKNOWN":
                    values.add(normalized)
            ordered = sorted(values)
            vocabularies["static"][feature] = [
                {"category": value, "id": idx + 4} for idx, value in enumerate(ordered)
            ]

        self._state = {
            "preprocessor_version": "train_only_preprocessor_v1",
            "fit_split": "train",
            "fit_population": "partition == train AND supervised_eligible",
            "fit_sample_count": int(fit_rows.sum()),
            "fit_sample_ids_sha256": str(fit_sample_ids_sha256),
            "window_length": width,
            "schema": json.loads(canonical_json(schema)),
            "numeric": {
                "mean": means, "scale": scales, "observed_cell_count": counts,
                "constant_features_scale_one": constants,
                "observed_physical_zero_count": zeros,
                "variance_ddof": 0,
            },
            "vocabularies": vocabularies,
            "reserved_category_ids": RESERVED_CATEGORY_IDS,
            "missing_category_tokens_casefold": sorted(MISSING_CATEGORY_TOKENS),
            "dynamic_category_mask_priority": [
                "padding_or_nonexistent_slot", "explicit_source_UNKNOWN", "masked_or_absent", "known_vocabulary", "out_of_vocabulary"
            ],
            "mask_policy": "preserve every source mask; masks are auxiliary arrays, never appended to predictor features",
        }
        return self

    def transform(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        static: np.ndarray,
        masks: dict[str, np.ndarray],
        schema: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        if self._state is None:
            raise PreprocessingContractError("Preprocessor has not been fitted")
        validate_input_arrays(numeric, categorical, static, masks, schema)
        if canonical_json(schema) != canonical_json(self._state["schema"]):
            raise PreprocessingContractError("Feature schema, order, mask contract, or window length differs from fitted state")
        before = state_sha256(self._state)
        num_names = schema["numeric_feature_names"]
        cat_names = schema["categorical_feature_names"]
        static_names = schema["static_feature_names"]
        reason = masks["mask_reason_codes"]
        exists = masks["slot_exists_mask"].astype(bool, copy=False)
        padding = masks["padding_mask"].astype(bool, copy=False)
        valid_slots = exists & ~padding
        numeric_out = np.zeros(numeric.shape, dtype=np.float32)
        for idx, feature in enumerate(num_names):
            observed = (reason[:, :, idx] == 0) & valid_slots
            values = numeric[:, :, idx][observed].astype(np.float64, copy=False)
            if values.size and not np.isfinite(values).all():
                raise PreprocessingContractError(f"Non-finite value marked present in {feature!r}")
            scaled = (values - self._state["numeric"]["mean"][feature]) / self._state["numeric"]["scale"][feature]
            numeric_out[:, :, idx][observed] = scaled.astype(np.float32)

        categorical_out = np.empty(categorical.shape, dtype=np.int32)
        dynamic_maps = {
            feature: {item["category"]: int(item["id"]) for item in entries}
            for feature, entries in self._state["vocabularies"]["dynamic"].items()
        }
        oov_dynamic = {feature: 0 for feature in cat_names}
        unknown_dynamic = {feature: 0 for feature in cat_names}
        for row in range(categorical.shape[0]):
            for slot in range(categorical.shape[1]):
                for idx, feature in enumerate(cat_names):
                    code = int(reason[row, slot, len(num_names) + idx])
                    value = category_value(categorical[row, slot, idx])
                    if padding[row, slot]:
                        ident = RESERVED_CATEGORY_IDS["padding"]
                    elif not exists[row, slot]:
                        ident = RESERVED_CATEGORY_IDS["absent"]
                    elif value == "UNKNOWN" and code in (0, 5):
                        ident = RESERVED_CATEGORY_IDS["source_unknown"]
                        unknown_dynamic[feature] += 1
                    elif code != 0 or value is None:
                        ident = RESERVED_CATEGORY_IDS["absent"]
                    else:
                        ident = dynamic_maps[feature].get(value, RESERVED_CATEGORY_IDS["out_of_vocabulary"])
                        if ident == RESERVED_CATEGORY_IDS["out_of_vocabulary"]:
                            oov_dynamic[feature] += 1
                    categorical_out[row, slot, idx] = ident

        static_out = np.empty(static.shape, dtype=np.int32)
        static_maps = {
            feature: {item["category"]: int(item["id"]) for item in entries}
            for feature, entries in self._state["vocabularies"]["static"].items()
        }
        oov_static = {feature: 0 for feature in static_names}
        for row in range(static.shape[0]):
            for idx, feature in enumerate(static_names):
                value = category_value(static[row, idx])
                if value == "UNKNOWN":
                    ident = RESERVED_CATEGORY_IDS["source_unknown"]
                elif value is None:
                    ident = RESERVED_CATEGORY_IDS["absent"]
                else:
                    ident = static_maps[feature].get(value, RESERVED_CATEGORY_IDS["out_of_vocabulary"])
                    if ident == RESERVED_CATEGORY_IDS["out_of_vocabulary"]:
                        oov_static[feature] += 1
                static_out[row, idx] = ident

        result = {
            "X_numeric": numeric_out,
            "X_dynamic_categorical_ids": categorical_out,
            "X_static_categorical_ids": static_out,
        }
        result.update({name: np.array(value, copy=True) for name, value in masks.items()})
        if state_sha256(self._state) != before:
            raise PreprocessingContractError("transform mutated fitted preprocessor state")
        result["_oov_dynamic_counts"] = np.asarray([oov_dynamic[x] for x in cat_names], dtype=np.int64)
        result["_unknown_dynamic_counts"] = np.asarray([unknown_dynamic[x] for x in cat_names], dtype=np.int64)
        result["_oov_static_counts"] = np.asarray([oov_static[x] for x in static_names], dtype=np.int64)
        return result

    def save(self, path: Path) -> None:
        if self._state is None:
            raise PreprocessingContractError("Cannot save an unfitted preprocessor")
        path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "InputPreprocessor":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
