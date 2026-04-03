from __future__ import annotations

import math
import unittest

import numpy as np

from chem_sae.eval.feature_audit import (
    aggregate_probe_coefficients,
    collect_top_activating_examples,
    summarize_single_feature,
)


class TestFeatureAudit(unittest.TestCase):
    def test_aggregate_probe_coefficients_computes_stability_metrics(self) -> None:
        coefficients = np.array(
            [
                [3.0, -0.2, 0.4],
                [2.0, -0.1, 0.1],
                [4.0, 0.3, 0.2],
                [5.0, -0.4, 0.5],
                [1.0, -0.3, 0.3],
            ],
            dtype=np.float32,
        )

        rows = aggregate_probe_coefficients(coefficients)

        self.assertEqual([row["feature_idx"] for row in rows], [0, 2, 1])

        feature0 = rows[0]
        self.assertAlmostEqual(feature0["coef_mean"], 3.0)
        self.assertAlmostEqual(feature0["abs_coef_mean"], 3.0)
        self.assertAlmostEqual(feature0["sign_consistency"], 1.0)
        self.assertAlmostEqual(feature0["rank_mean"], 1.0)
        self.assertAlmostEqual(feature0["rank_std"], 0.0)

        feature1 = rows[2]
        self.assertAlmostEqual(feature1["sign_consistency"], 0.8)
        self.assertGreater(feature1["rank_mean"], 2.0)

    def test_collect_top_activating_examples_orders_descending(self) -> None:
        smiles = ["A", "B", "C", "D"]
        activations = np.array([0.1, 0.9, 0.4, 0.8], dtype=np.float32)
        labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        label_mask = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)

        rows = collect_top_activating_examples(
            smiles=smiles,
            activations=activations,
            labels=labels,
            label_mask=label_mask,
            split_name="test",
            top_k=3,
        )

        self.assertEqual([row["smiles"] for row in rows], ["B", "D", "C"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["label"], 1.0)
        self.assertIsNone(rows[2]["label"])

    def test_summarize_single_feature_returns_expected_shape(self) -> None:
        train_activations = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        train_labels = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        train_mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        test_activations = np.array([0.7, 0.6, 0.3, 0.2], dtype=np.float32)
        test_labels = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        test_mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        row = summarize_single_feature(
            task="bbbp",
            feature_idx=7,
            train_activations=train_activations,
            train_labels=train_labels,
            train_label_mask=train_mask,
            test_activations=test_activations,
            test_labels=test_labels,
            test_label_mask=test_mask,
            coefficient_stats={
                "coef_mean": 0.25,
                "coef_std": 0.05,
                "abs_coef_mean": 0.25,
                "sign_consistency": 1.0,
                "rank_mean": 2.0,
                "rank_std": 0.0,
            },
        )

        self.assertEqual(row["task"], "bbbp")
        self.assertEqual(row["feature_idx"], 7)
        self.assertIn("single_feature_roc_auc", row)
        self.assertIn("positive_mean_activation", row)
        self.assertIn("negative_mean_activation", row)
        self.assertIn("activation_frequency", row)
        self.assertTrue(math.isfinite(float(row["single_feature_roc_auc"])))
        self.assertGreater(row["positive_mean_activation"], row["negative_mean_activation"])

    def test_summarize_single_feature_accepts_column_shaped_labels(self) -> None:
        row = summarize_single_feature(
            task="bbbp",
            feature_idx=3,
            train_activations=np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
            train_labels=np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32),
            train_label_mask=np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
            test_activations=np.array([0.7, 0.6, 0.3, 0.2], dtype=np.float32),
            test_labels=np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32),
            test_label_mask=np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32),
            coefficient_stats={
                "coef_mean": 0.4,
                "coef_std": 0.1,
                "abs_coef_mean": 0.4,
                "sign_consistency": 1.0,
                "rank_mean": 1.0,
                "rank_std": 0.0,
            },
        )

        self.assertEqual(row["feature_idx"], 3)
        self.assertTrue(math.isfinite(float(row["single_feature_roc_auc"])))


if __name__ == "__main__":
    unittest.main()
