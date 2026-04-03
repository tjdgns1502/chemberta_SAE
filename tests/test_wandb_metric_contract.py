from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from chem_sae.utils.wandb_logging import (
    WandbRunLogger,
    build_sparsity_log_dict,
    build_train_step_log_dict,
)


class TestWandbMetricContract(unittest.TestCase):
    def test_step_logs_include_sae_lens_core_metrics(self) -> None:
        sae_in = torch.randn(8, 16)
        sae_out = sae_in + 0.05 * torch.randn(8, 16)
        feature_acts = torch.relu(torch.randn(8, 32))
        overall_loss = torch.tensor(1.2345)
        losses = {
            "mse_loss": torch.tensor(0.8),
            "l1_loss": torch.tensor(0.2),
            "l0_loss": torch.tensor(0.2),
            "pre_act_loss": torch.tensor(0.0345),
        }
        n_forward = torch.arange(32, dtype=torch.long)

        payload = build_train_step_log_dict(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            overall_loss=overall_loss,
            losses=losses,
            metrics={},
            current_learning_rate=1e-4,
            n_training_samples=1024,
            n_forward_passes_since_fired=n_forward,
            dead_feature_window=10,
            coefficients={"l0": 0.1, "l1": 0.0},
            global_step=200,
        )
        self.assertIn("metrics/explained_variance", payload)
        self.assertIn("metrics/explained_variance_legacy", payload)
        self.assertIn("metrics/explained_variance_legacy_std", payload)
        self.assertIn("metrics/l0", payload)
        self.assertIn("sparsity/mean_passes_since_fired", payload)
        self.assertIn("sparsity/dead_features", payload)
        self.assertIn("details/n_training_samples", payload)
        self.assertIn("details/l0_coefficient", payload)
        self.assertIn("details/l1_coefficient", payload)
        self.assertIn("losses/mse_loss", payload)
        self.assertIn("losses/l0_loss", payload)

    def test_sparsity_logs_include_sae_lens_feature_density_metrics(self) -> None:
        act_freq_scores = torch.rand(64)
        payload = build_sparsity_log_dict(
            act_freq_scores=act_freq_scores,
            n_frac_active_samples=128,
        )
        self.assertIn("metrics/mean_log10_feature_sparsity", payload)
        self.assertIn("plots/feature_density_line_chart", payload)
        self.assertIn("sparsity/below_1e-5", payload)
        self.assertIn("sparsity/below_1e-6", payload)

    def test_wandb_frequency_contract_is_step_aligned(self) -> None:
        cfg = SimpleNamespace(
            log_to_wandb=True,
            wandb_project="x",
            wandb_entity=None,
            wandb_group="probe_layer_02",
            run_name=None,
            wandb_id=None,
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=100,
        )
        logger = WandbRunLogger(
            logger_cfg=cfg,
            run_root=Path("."),
            run_id="run",
            config_payload={},
            job_type="test",
        )
        self.assertTrue(logger.should_log(9))
        self.assertFalse(logger.should_log(10))
        self.assertTrue(logger.should_eval_log(999))
        self.assertFalse(logger.should_eval_log(1000))

    def test_wandb_run_metadata_persists_group(self) -> None:
        cfg = SimpleNamespace(
            log_to_wandb=True,
            wandb_project="x",
            wandb_entity="entity",
            wandb_group="probe_layer_02",
            run_name="L02 | l0=0.0800 | n=1536 | probe-run",
            wandb_id="abc123",
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=100,
        )
        with unittest.mock.patch("pathlib.Path.write_text") as mock_write:
            logger = WandbRunLogger(
                logger_cfg=cfg,
                run_root=Path("/tmp/probe-run"),
                run_id="probe-run",
                config_payload={},
                job_type="sae_probe",
                tags=["probe", "jumprelu", "layer:02"],
            )
            logger._save_run_metadata()

        payload = mock_write.call_args.args[0]
        self.assertIn('"wandb_group": "probe_layer_02"', payload)


if __name__ == "__main__":
    unittest.main()
