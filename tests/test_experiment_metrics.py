"""Analytic metric checks independent of a machine-learning framework."""

import unittest

import numpy as np

from f1_pitstop.experiment_metrics import (
    binary_metrics,
    select_f1_threshold,
    select_f2_threshold,
    select_metric_threshold,
    threshold_curve,
)


class BinaryMetricsTests(unittest.TestCase):
    def test_analytic_counts_and_recall_weighted_f2(self):
        metrics = binary_metrics(
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [.9, .8, .4, .3, .7, .2, .1, .1, .1, .1],
            .5,
        )
        self.assertEqual(metrics["n_samples"], 10)
        self.assertEqual(metrics["positives"], 4)
        self.assertEqual(
            [metrics[key] for key in ("tp", "fp", "fn", "tn")], [2, 1, 2, 5]
        )
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["f1"], 4 / 7)
        self.assertAlmostEqual(metrics["f2"], 10 / 19)
        self.assertLess(metrics["f2"], metrics["f1"])
        self.assertEqual(metrics["false_positives_per_100_decisions"], 10)

    def test_all_negative_and_no_predictions_use_zero_convention(self):
        metrics = binary_metrics([0, 0], [.1, .2], .5)
        self.assertEqual(metrics["tn"], 2)
        for metric in ("precision", "recall", "f1", "f2", "average_precision"):
            self.assertEqual(metrics[metric], 0)
        false_alarms = binary_metrics([0, 0], [.9, .8], .5)
        self.assertEqual(false_alarms["f2"], 0)
        self.assertEqual(false_alarms["false_positives_per_100_decisions"], 100)

    def test_all_positive_labels_and_no_positive_predictions(self):
        metrics = binary_metrics([True, True], [.1, .2], .5)
        self.assertEqual(metrics["fn"], 2)
        self.assertEqual(metrics["average_precision"], 1)
        for metric in ("precision", "recall", "f1", "f2"):
            self.assertEqual(metrics[metric], 0)
        perfect = binary_metrics([1, 1], [.5, 1], .5)
        self.assertEqual(perfect["f2"], 1)

    def test_equality_and_boundary_thresholds(self):
        self.assertEqual(binary_metrics([1, 0], [.5, .49], .5)["tp"], 1)
        self.assertEqual(binary_metrics([1, 0], [0, 0], 0)["fp"], 1)
        self.assertEqual(binary_metrics([1, 0], [1, .99], 1)["tp"], 1)
        self.assertEqual(binary_metrics([1, 0], [1, .99], 1)["fp"], 0)

    def test_average_precision_groups_ties_without_order_dependence(self):
        # Cutoffs .9 and .5 add 1/2 recall each, at precision 1 and 2/3.
        metrics = binary_metrics([1, 0, 1, 0], [.9, .5, .5, .1], .5)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        reordered = binary_metrics([1, 1, 0, 0], [.9, .5, .5, .1], .2)
        self.assertEqual(metrics["average_precision"], reordered["average_precision"])
        self.assertEqual(binary_metrics([1, 0], [.5, .5], .5)["average_precision"], .5)

    def test_invalid_labels_and_probabilities_are_rejected(self):
        invalid_pairs = [
            ([], []),
            ([1], [.1, .2]),
            ([[1, 0]], [.1, .2]),
            ([1, 0], [[.1, .2]]),
            ([1, np.nan], [.1, .2]),
            ([1, None], [.1, .2]),
            ([1, np.inf], [.1, .2]),
            ([1, 2], [.1, .2]),
            ([1, -1], [.1, .2]),
            ([1, .5], [.1, .2]),
            (["1", "0"], [.1, .2]),
            ([1, 0], [.1, np.nan]),
            ([1, 0], [.1, np.inf]),
            ([1, 0], [.1, None]),
            ([1, 0], [-.1, .2]),
            ([1, 0], [.1, 1.1]),
            ([1, 0], [.1, .2j]),
            (np.ma.array([1, 0], mask=[False, True]), [.1, .2]),
            ([1, 0], np.ma.array([.1, .2], mask=[False, True])),
        ]
        for labels, scores in invalid_pairs:
            with self.subTest(labels=labels, scores=scores):
                with self.assertRaises(ValueError):
                    binary_metrics(labels, scores, .5)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (np.nan, np.inf, -.1, 1.1, None, "0.5", True, [.5]):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    binary_metrics([1, 0], [.9, .1], threshold)

    def test_binary_object_arrays_are_accepted_without_truthiness_cast(self):
        metrics = binary_metrics(np.array([0, 1], dtype=object), [.1, .9], .5)
        self.assertEqual(metrics["f2"], 1)


class ThresholdSelectionTests(unittest.TestCase):
    def test_declared_f1_and_f2_select_different_thresholds(self):
        labels, scores, grid = [1, 1, 0, 0, 0], [.9, .6, .6, .6, .6], [.5, .8]
        # At .8: TP=1, FP=0, FN=1. At .5: TP=2, FP=3, FN=0.
        # F1 strictly favors .8 (2/3 > 4/7); F2 favors .5 (10/13 > 5/9).
        f1 = select_metric_threshold(labels, scores, grid, metric="f1")
        f2 = select_metric_threshold(labels, scores, grid, metric="f2")
        self.assertEqual(f1["threshold"], .8)
        self.assertAlmostEqual(f1["metrics"]["f1"], 2 / 3)
        self.assertAlmostEqual(f1["metrics"]["f2"], 5 / 9)
        self.assertEqual(f1["selection_rule"], "maximize_f1_then_precision_then_threshold")
        self.assertEqual(f2["threshold"], .5)
        self.assertAlmostEqual(f2["metrics"]["f2"], 10 / 13)
        self.assertAlmostEqual(f2["metrics"]["f1"], 4 / 7)
        self.assertEqual(select_f1_threshold(labels, scores, grid), f1)

    def test_default_and_explicit_f2_preserve_original_wrapper_result(self):
        labels, scores, grid = [1, 0, 1], [.9, .5, .3], [.8, .2, .5, .2]
        expected = {
            "threshold": .2,
            "metrics": binary_metrics(labels, scores, .2),
            "selection_rule": "maximize_f2_then_precision_then_threshold",
        }
        self.assertEqual(select_f2_threshold(labels, scores, grid), expected)
        self.assertEqual(select_metric_threshold(labels, scores, grid), expected)
        self.assertEqual(select_metric_threshold(labels, scores, grid, metric="f2"), expected)

    def test_equal_f1_prefers_precision(self):
        # F1 is 2/3 at both thresholds; precision is 1 versus 1/2.
        result = select_metric_threshold([1, 1, 0, 0], [.9, .6, .6, .6], [.5, .8], metric="f1")
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["metrics"]["precision"], 1)

    def test_equal_metric_and_precision_prefers_higher_threshold_for_both_metrics(self):
        for metric in ("f1", "f2"):
            with self.subTest(metric=metric):
                result = select_metric_threshold([1, 0], [.9, .1], [.5, .8, .6], metric=metric)
                self.assertEqual(result["threshold"], .8)
                negatives = select_metric_threshold([0, 0], [.9, .1], [.8, .2, 1], metric=metric)
                self.assertEqual(negatives["threshold"], 1)

    def test_invalid_selection_metrics_are_rejected(self):
        for metric in ("accuracy", "precision", "f3", "F1", "", None, 1, True, ["f1"]):
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "metric must be 'f1' or 'f2'"):
                    select_metric_threshold([1, 0], [.9, .1], [.5], metric=metric)

    def test_f2_prefers_recovering_positives_in_analytic_example(self):
        result = select_f2_threshold(
            [1, 1, 1, 1, 0, 0, 0], [.9, .8, .4, .3, .7, .6, .2], [.75, .25]
        )
        self.assertEqual(result["threshold"], .25)
        self.assertAlmostEqual(result["metrics"]["f2"], 20 / 22)
        self.assertEqual(result["selection_rule"], "maximize_f2_then_precision_then_threshold")

    def test_equal_f2_prefers_precision(self):
        # F2 is 5/9 at both thresholds; precision is 1 versus 2/10.
        result = select_f2_threshold([1] + [0] * 8 + [1], [.9] + [.6] * 9, [.5, .8])
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["metrics"]["precision"], 1)

    def test_equal_f2_and_precision_prefers_higher_threshold(self):
        result = select_f2_threshold([1, 0], [.9, .1], [.5, .8, .6])
        self.assertEqual(result["threshold"], .8)
        negative_result = select_f2_threshold([0, 0], [.9, .1], [.2, .8, 1])
        self.assertEqual(negative_result["threshold"], 1)

    def test_curve_and_selection_ignore_grid_order_and_duplicates(self):
        labels, scores = [1, 0, 1], [.9, .5, .3]
        curve = threshold_curve(labels, scores, [.8, .2, .5, .2])
        self.assertEqual([row["threshold"] for row in curve], [.2, .5, .8])
        for row in curve:
            self.assertEqual(
                {key: value for key, value in row.items() if key != "threshold"},
                binary_metrics(labels, scores, row["threshold"]),
            )
        self.assertEqual(
            select_f2_threshold(labels, scores, [.8, .2, .5, .2]),
            select_f2_threshold(labels, scores, [.5, .8, .2]),
        )

    def test_invalid_grids_are_rejected(self):
        for grid in ([], [[.5]], [.1, np.nan], [.1, np.inf], [-.1], [1.1], [None]):
            for function in (threshold_curve, select_f2_threshold, select_f1_threshold, select_metric_threshold):
                with self.subTest(grid=grid, function=function.__name__):
                    with self.assertRaises(ValueError):
                        function([1, 0], [.9, .1], grid)


if __name__ == "__main__":
    unittest.main()
