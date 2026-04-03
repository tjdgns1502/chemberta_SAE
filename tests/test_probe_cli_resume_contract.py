import subprocess
import sys
import unittest
from pathlib import Path


class TestProbeCliResumeContract(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.script_path = self.project_root / "scripts" / "run_sae_probe.py"

    def test_resume_requires_run_id(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--layers",
                "0",
                "--epochs",
                "1",
                "--base-l0",
                "0.001",
                "--schedule",
                "none",
                "--dry-run",
                "--resume",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--resume requires --run-id", proc.stderr + proc.stdout)

    def test_resume_with_run_id_succeeds_in_dry_run(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--run-id",
                "resume_cli_contract",
                "--layers",
                "0",
                "--epochs",
                "1",
                "--base-l0",
                "0.001",
                "--schedule",
                "none",
                "--dry-run",
                "--resume",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("resume=True", proc.stdout)

    def test_dashboard_flags_in_dry_run(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--run-id",
                "dashboard_cli_contract",
                "--layers",
                "0",
                "--epochs",
                "1",
                "--base-l0",
                "0.001",
                "--schedule",
                "none",
                "--dry-run",
                "--dashboard-every-epochs",
                "2",
                "--disable-dashboard",
            ],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("dashboard_enabled=False", proc.stdout)
        self.assertIn("dashboard_every_epochs=2", proc.stdout)

    def test_wandb_flags_in_dry_run(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(self.script_path),
                "--run-id",
                "wandb_cli_contract",
                "--layers",
                "0",
                "--epochs",
                "1",
                "--base-l0",
                "0.001",
                "--schedule",
                "none",
                "--dry-run",
                "--log-to-wandb",
                "--wandb-project",
                "sae_lens_training",
                "--wandb-entity",
                "unit-test",
                "--wandb-id",
                "abc123",
                "--wandb-run-name",
                "probe-wandb",
                "--wandb-log-frequency",
                "5",
                "--eval-every-n-wandb-logs",
                "7",
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
        self.assertIn("wandb_id=abc123", proc.stdout)
        self.assertIn("wandb_run_name=probe-wandb", proc.stdout)
        self.assertIn("wandb_log_frequency=5", proc.stdout)
        self.assertIn("eval_every_n_wandb_logs=7", proc.stdout)
        self.assertIn("log_weights_to_wandb=False", proc.stdout)
        self.assertIn("log_optimizer_state_to_wandb=True", proc.stdout)
        self.assertIn("log_activations_store_to_wandb=True", proc.stdout)


if __name__ == "__main__":
    unittest.main()
