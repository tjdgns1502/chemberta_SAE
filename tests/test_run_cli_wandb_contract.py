import subprocess
import sys
import unittest
from pathlib import Path


class TestRunCliWandbContract(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.script_path = self.project_root / "scripts" / "run.py"

    def test_wandb_flags_in_sae_dry_run(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "sae",
                "--run-id",
                "run_cli_wandb_contract",
                "--layers",
                "0",
                "--arch",
                "jumprelu",
                "--dry-run",
                "--log-to-wandb",
                "--wandb-project",
                "sae_lens_training",
                "--wandb-entity",
                "unit-test",
                "--wandb-id",
                "xyz789",
                "--wandb-run-name",
                "runpy-wandb",
                "--wandb-log-frequency",
                "3",
                "--eval-every-n-wandb-logs",
                "9",
                "--disable-wandb-log-weights",
                "--log-optimizer-state-to-wandb",
                "--log-activations-store-to-wandb",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("log_to_wandb=True", proc.stdout)
        self.assertIn("wandb_project=sae_lens_training", proc.stdout)
        self.assertIn("wandb_entity=unit-test", proc.stdout)
        self.assertIn("wandb_id=xyz789", proc.stdout)
        self.assertIn("wandb_run_name=runpy-wandb", proc.stdout)
        self.assertIn("wandb_log_frequency=3", proc.stdout)
        self.assertIn("eval_every_n_wandb_logs=9", proc.stdout)
        self.assertIn("log_weights_to_wandb=False", proc.stdout)
        self.assertIn("log_optimizer_state_to_wandb=True", proc.stdout)
        self.assertIn("log_activations_store_to_wandb=True", proc.stdout)

    def test_wandb_frequency_validation(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "sae",
                "--run-id",
                "run_cli_wandb_contract_invalid",
                "--layers",
                "0",
                "--arch",
                "jumprelu",
                "--dry-run",
                "--wandb-log-frequency",
                "0",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--wandb-log-frequency must be >= 1", proc.stderr + proc.stdout)


if __name__ == "__main__":
    unittest.main()
