from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestFeatureSubstructureCli(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.script_path = self.project_root / "scripts" / "run_feature_substructure.py"

    def test_dry_run_prints_resolved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "audit_reports"
            cards_dir = reports_dir / "feature_cards" / "bbbp"
            cards_dir.mkdir(parents=True)
            (cards_dir / "feature_1237.json").write_text(
                json.dumps(
                    {
                        "task": "bbbp",
                        "summary": {"feature_idx": 1237},
                        "top_train_examples": [],
                        "top_test_examples": [],
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--run-id",
                    "feature_substructure_cli_contract",
                    "--audit-reports-dir",
                    str(reports_dir),
                    "--tasks",
                    "bbbp",
                    "--features",
                    "1237,1492",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                check=False,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertIn("run_id=feature_substructure_cli_contract", proc.stdout)
        self.assertIn(f"audit_reports_dir={reports_dir}", proc.stdout)
        self.assertIn("tasks=('bbbp',)", proc.stdout)
        self.assertIn("feature_ids=(1237, 1492)", proc.stdout)


if __name__ == "__main__":
    unittest.main()
