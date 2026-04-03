import tempfile
import unittest
from pathlib import Path

import torch

from chem_sae.config import SaeExperimentConfig
from chem_sae.train.sae_training import train_sae_for_layer
from chem_sae.utils import set_seed


class TestTrainResumeContract(unittest.TestCase):
    def test_checkpoint_stores_resume_state_and_resume_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            cfg = SaeExperimentConfig(run_id="resume_contract")
            cfg.runs_dir = tmp_root / "runs"
            cfg.acts_dir = cfg.runs_dir / "acts"
            cfg.ckpt_dir = cfg.runs_dir / "checkpoints"
            cfg.log_path = cfg.runs_dir / "downstream_records.csv"
            cfg.n_latents = 8
            cfg.sae_batch_size = 8
            cfg.sae_lr = 1e-3
            cfg.early_stopping_patience = 10
            cfg.val_fraction = 0.5
            cfg.layers_spec = "0"
            cfg.ensure_dirs()

            layer_dir = cfg.acts_dir / "layer_0"
            layer_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(16, 4), layer_dir / "chunk_00000.pt")
            torch.save(torch.randn(16, 4), layer_dir / "chunk_00001.pt")

            checkpoint_root = tmp_root / "probe_ckpts"
            plot_root = tmp_root / "plots"
            set_seed(7)
            _, first_result = train_sae_for_layer(
                cfg=cfg,
                layer=0,
                device=torch.device("cpu"),
                arch="jumprelu",
                epochs=1,
                checkpoint_root=checkpoint_root,
                plot_root=plot_root,
                l0_coefficient=0.001,
                resume=False,
                trial_seed=7,
            )

            latest_path = checkpoint_root / "layer_0" / "latest.pt"
            self.assertTrue(latest_path.exists())
            latest_state = torch.load(latest_path, map_location="cpu", weights_only=False)
            self.assertIn("patience_counter", latest_state)
            self.assertIn("rng_state", latest_state)
            self.assertIn("python", latest_state["rng_state"])
            self.assertIn("numpy", latest_state["rng_state"])
            self.assertIn("torch", latest_state["rng_state"])
            self.assertIn("threshold", latest_state["model"])
            self.assertNotIn("log_threshold", latest_state["model"])

            _, resumed_result = train_sae_for_layer(
                cfg=cfg,
                layer=0,
                device=torch.device("cpu"),
                arch="jumprelu",
                epochs=2,
                checkpoint_root=checkpoint_root,
                plot_root=plot_root,
                l0_coefficient=0.001,
                resume=True,
                trial_seed=7,
            )
            self.assertGreater(resumed_result.global_step, first_result.global_step)


if __name__ == "__main__":
    unittest.main()
