"""Binary decision metrics and validation-only F2 threshold selection.

The caller owns the data split: select thresholds on validation predictions,
freeze the result, and only then report metrics on the untouched test set.
This module never derives a threshold from a loss or from labels implicitly.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np


SELECTION_RULE = "maximize_f2_then_precision_then_threshold"


def _numeric_vector(values: Any, name: str) -> np.ndarray:
    """Read numeric values without silently discarding masks or missing labels."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} must not contain missing or masked values")
    try:
        vector = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if vector.dtype.kind not in "biuf":
        # Nullable pandas columns can yield object arrays even without missing
        # values. Accept actual numeric scalars, while rejecting strings, None,
        # pandas.NA, complex numbers, and arbitrary truthy objects.
        if vector.dtype.kind != "O" or not all(
            isinstance(value, (Real, np.bool_)) for value in vector
        ):
            raise ValueError(f"{name} must contain real numbers without missing values")
    try:
        numeric = vector.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite values without missing values")
    return numeric


def _validated_inputs(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = _numeric_vector(y_true, "y_true")
    scores = _numeric_vector(probabilities, "probabilities")
    if labels.shape != scores.shape:
        raise ValueError("y_true and probabilities must have equal lengths")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return labels.astype(bool), scores


def _validated_threshold(threshold: Any) -> float:
    if not isinstance(threshold, Real) or isinstance(threshold, (bool, np.bool_)):
        raise ValueError("threshold must be a finite real number in [0, 1]")
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be a finite real number in [0, 1]")
    return value


def _validated_grid(thresholds: Any) -> np.ndarray:
    grid = _numeric_vector(thresholds, "thresholds")
    if np.any((grid < 0) | (grid > 1)):
        raise ValueError("thresholds must be in [0, 1]")
    return np.unique(grid)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Non-interpolated AP, treating every equal-score group as one cutoff."""
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    cumulative_positives = np.cumsum(labels[order], dtype=np.int64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0), labels.size - 1]
    group_true_positives = cumulative_positives[group_ends]
    added_positives = np.diff(np.r_[0, group_true_positives])
    precisions = group_true_positives / (group_ends + 1)
    return float(np.sum((added_positives / positives) * precisions))


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    average_precision: float,
) -> dict[str, int | float]:
    predicted = scores >= threshold
    tp = int(np.count_nonzero(labels & predicted))
    fp = int(np.count_nonzero(~labels & predicted))
    fn = int(np.count_nonzero(labels & ~predicted))
    tn = int(np.count_nonzero(~labels & ~predicted))

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "n_samples": int(labels.size),
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "f2": divide(5 * tp, 5 * tp + 4 * fn + fp),
        "false_positives_per_100_decisions": 100.0 * fp / labels.size,
        "average_precision": average_precision,
    }


def binary_metrics(
    y_true: Any, probabilities: Any, threshold: float
) -> dict[str, int | float]:
    """Compute binary metrics using ``probability >= threshold`` as positive.

    Undefined precision, recall, F1, and F2 are reported as zero. Average
    precision is zero when there are no positive labels. Missing labels are
    rejected, never converted into negatives.
    """
    labels, scores = _validated_inputs(y_true, probabilities)
    value = _validated_threshold(threshold)
    return _metrics(labels, scores, value, _average_precision(labels, scores))


def threshold_curve(
    y_true: Any, probabilities: Any, thresholds: Any
) -> list[dict[str, int | float]]:
    """Return flat metric rows for the sorted, unique explicit candidate grid."""
    labels, scores = _validated_inputs(y_true, probabilities)
    grid = _validated_grid(thresholds)
    average_precision = _average_precision(labels, scores)
    return [
        {
            "threshold": float(threshold),
            **_metrics(labels, scores, float(threshold), average_precision),
        }
        for threshold in grid
    ]


def select_f2_threshold(
    y_true: Any, probabilities: Any, thresholds: Any
) -> dict[str, Any]:
    """Select on validation data: greatest F2, then precision, then threshold.

    The candidate grid must be provided explicitly. Its input order and repeated
    candidates have no effect on the chosen threshold.
    """
    rows = threshold_curve(y_true, probabilities, thresholds)
    best = max(rows, key=lambda row: (row["f2"], row["precision"], row["threshold"]))
    return {
        "threshold": best["threshold"],
        "metrics": {key: value for key, value in best.items() if key != "threshold"},
        "selection_rule": SELECTION_RULE,
    }
