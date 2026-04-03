from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestFeatureInterventionCli(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.script_path = self.project_root / "scripts" / "run_feature_intervention.py"

    def test_dry_run_prints_intervention_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "dummy.pt"
            checkpoint_path.write_bytes(b"dummy")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--run-id",
                    "feature_intervention_cli_contract",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--layer",
                    "0",
                    "--task",
                    "bbbp",
                    "--features",
                    "2,5,7",
                    "--mode",
                    "zero",
                    "--control",
                    "matched_random",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                check=False,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertIn("run_id=feature_intervention_cli_contract", proc.stdout)
        self.assertIn(f"checkpoint_path={checkpoint_path}", proc.stdout)
        self.assertIn("feature_indices=[2, 5, 7]", proc.stdout)
        self.assertIn("control=matched_random", proc.stdout)


if __name__ == "__main__":
    unittest.main()
