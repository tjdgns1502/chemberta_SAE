from __future__ import annotations

import unittest

import torch

from chem_sae.eval.feature_intervention import (
    build_feature_intervention_result_row,
    parse_feature_indices,
    sample_matched_control_features,
)
from chem_sae.modeling.roberta_mlm import (
    apply_feature_intervention,
    apply_sae_latent_intervention,
)


class _DummySae:
    def encode(self, x: torch.Tensor):
        return x + 1.0, {"dummy": True}

    def decode(self, latents: torch.Tensor, info=None):
        del info
        return latents - 1.0


class TestFeatureIntervention(unittest.TestCase):
    def test_parse_feature_indices_reads_csv(self) -> None:
        self.assertEqual(parse_feature_indices("1, 4,7"), [1, 4, 7])

    def test_sample_matched_control_features_is_reproducible(self) -> None:
        first = sample_matched_control_features(
            num_features=16,
            group_size=3,
            seed=42,
            exclude={1, 2, 3},
        )
        second = sample_matched_control_features(
            num_features=16,
            group_size=3,
            seed=42,
            exclude={1, 2, 3},
        )
        self.assertEqual(first, second)
        self.assertTrue(set(first).isdisjoint({1, 2, 3}))

    def test_result_row_schema_for_group_intervention(self) -> None:
        row = build_feature_intervention_result_row(
            run_id="feature_intervention_contract",
            task="bbbp",
            layer=0,
            checkpoint_path="checkpoint.pt",
            condition="target",
            feature_indices=[4, 9],
            mode="zero",
            baseline_roc_auc=0.71,
            intervened_roc_auc=0.66,
            mean_logit_shift=-0.12,
            mean_probability_shift=-0.08,
            control_kind="matched_random",
        )
        self.assertEqual(row["feature_count"], 2)
        self.assertEqual(row["feature_indices"], "4,9")
        self.assertEqual(row["mode"], "zero")
        self.assertAlmostEqual(row["roc_auc_delta"], -0.05)
        self.assertIn("mean_probability_shift", row)

    def test_zero_only_edits_requested_indices(self) -> None:
        latents = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        edited = apply_feature_intervention(
            latents,
            feature_indices=[1, 3],
            mode="zero",
        )
        expected = torch.tensor([[1.0, 0.0, 3.0, 0.0]])
        self.assertTrue(torch.equal(edited, expected))

    def test_mean_clamp_only_edits_requested_indices(self) -> None:
        latents = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        edited = apply_feature_intervention(
            latents,
            feature_indices=[0, 2],
            mode="mean_clamp",
            feature_values=[-0.5, 9.0],
        )
        expected = torch.tensor([[-0.5, 2.0, 9.0, 4.0]])
        self.assertTrue(torch.equal(edited, expected))

    def test_force_on_only_edits_requested_indices(self) -> None:
        latents = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        edited = apply_feature_intervention(
            latents,
            feature_indices=[2],
            mode="force_on",
            feature_values=7.5,
        )
        expected = torch.tensor([[1.0, 2.0, 7.5, 4.0]])
        self.assertTrue(torch.equal(edited, expected))

    def test_noop_sae_intervention_matches_existing_reconstruction_path(self) -> None:
        attn_output = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        sae = _DummySae()

        untouched = apply_sae_latent_intervention(attn_output, sae, latent_intervention=None)

        self.assertTrue(torch.equal(untouched, attn_output))


if __name__ == "__main__":
    unittest.main()
