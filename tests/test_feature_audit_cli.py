from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestFeatureAuditCli(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.script_path = self.project_root / "scripts" / "run_feature_audit.py"

    def test_dry_run_prints_resolved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "dummy.pt"
            checkpoint_path.write_bytes(b"dummy")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--run-id",
                    "feature_audit_cli_contract",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--layer",
                    "0",
                    "--tasks",
                    "bbbp,bace_classification",
                    "--top-k",
                    "7",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                check=False,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertIn("run_id=feature_audit_cli_contract", proc.stdout)
        self.assertIn(f"checkpoint_path={checkpoint_path}", proc.stdout)
        self.assertIn("tasks=('bbbp', 'bace_classification')", proc.stdout)
        self.assertIn("top_k=7", proc.stdout)


if __name__ == "__main__":
    unittest.main()
