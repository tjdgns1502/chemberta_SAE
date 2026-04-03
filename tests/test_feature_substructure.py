from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestFeatureSubstructure(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.python = self.project_root.parents[1] / "micromamba" / "envs" / "chemberta-repro" / "bin" / "python"

    def _run_python_json(self, code: str) -> dict:
        proc = subprocess.run(
            [str(self.python), "-c", code],
            capture_output=True,
            text=True,
            cwd=str(self.project_root),
            env={**os.environ, "PYTHONPATH": "src"},
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_murcko_scaffold_smiles_extracts_aromatic_core(self) -> None:
        payload = self._run_python_json(
            """
import json
from chem_sae.analysis.feature_substructure import murcko_scaffold_smiles
print(json.dumps({
    "a": murcko_scaffold_smiles("Cc1ccccc1"),
    "b": murcko_scaffold_smiles("CCc1ccccc1"),
    "c": murcko_scaffold_smiles("not-a-smiles"),
}))
"""
        )
        self.assertEqual(payload["a"], "c1ccccc1")
        self.assertEqual(payload["b"], "c1ccccc1")
        self.assertIsNone(payload["c"])

    def test_summarize_molecule_set_substructures_reports_mcs_and_scaffolds(self) -> None:
        summary = self._run_python_json(
            """
import json
from chem_sae.analysis.feature_substructure import summarize_molecule_set_substructures
summary = summarize_molecule_set_substructures(
    ["Cc1ccccc1", "CCc1ccccc1", "Clc1ccccc1", "Oc1ccccc1"],
    top_n_scaffolds=3,
    mcs_threshold=1.0,
    mcs_timeout_seconds=2,
)
print(json.dumps(summary))
"""
        )
        self.assertEqual(summary["valid_molecule_count"], 4)
        self.assertEqual(summary["invalid_smiles_count"], 0)
        self.assertEqual(summary["top_scaffolds"][0]["scaffold_smiles"], "c1ccccc1")
        self.assertEqual(summary["top_scaffolds"][0]["count"], 4)
        self.assertIn("mcs_smarts", summary)
        self.assertIn("mcs_canceled", summary)
        self.assertFalse(summary["mcs_canceled"])
        self.assertGreaterEqual(summary["mcs_num_atoms"], 6)

    def test_summarize_feature_card_substructures_uses_top_examples(self) -> None:
        feature_card = {
            "task": "bbbp",
            "summary": {"feature_idx": 7, "coef_mean": 1.2},
            "top_train_examples": [
                {"smiles": "Cc1ccccc1", "activation": 2.0, "label": 1.0, "has_label": True},
                {"smiles": "CCc1ccccc1", "activation": 1.8, "label": 1.0, "has_label": True},
            ],
            "top_test_examples": [
                {"smiles": "Clc1ccccc1", "activation": 1.7, "label": 0.0, "has_label": True},
                {"smiles": "not-a-smiles", "activation": 1.5, "label": 0.0, "has_label": True},
            ],
        }

        summary = self._run_python_json(
            f"""
import json
from chem_sae.analysis.feature_substructure import summarize_feature_card_substructures
feature_card = json.loads({json.dumps(json.dumps(feature_card))})
summary = summarize_feature_card_substructures(feature_card, top_n_examples=3)
print(json.dumps(summary))
"""
        )

        self.assertEqual(summary["task"], "bbbp")
        self.assertEqual(summary["feature_idx"], 7)
        self.assertEqual(summary["top_example_count"], 3)
        self.assertEqual(summary["substructure_summary"]["valid_molecule_count"], 3)
        self.assertEqual(
            summary["substructure_summary"]["top_scaffolds"][0]["scaffold_smiles"],
            "c1ccccc1",
        )

    def test_run_feature_substructure_analysis_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir) / "audit_reports"
            cards_dir = reports_dir / "feature_cards" / "bbbp"
            cards_dir.mkdir(parents=True)
            feature_card = {
                "task": "bbbp",
                "summary": {"feature_idx": 1237, "coef_mean": 2.0},
                "top_train_examples": [
                    {"smiles": "Cc1ccccc1", "activation": 2.1, "label": 1.0, "has_label": True},
                    {"smiles": "CCc1ccccc1", "activation": 1.9, "label": 1.0, "has_label": True},
                ],
                "top_test_examples": [
                    {"smiles": "Clc1ccccc1", "activation": 1.8, "label": 0.0, "has_label": True}
                ],
            }
            card_path = cards_dir / "feature_1237.json"
            card_path.write_text(json.dumps(feature_card), encoding="utf-8")

            output_dir = Path(tmpdir) / "out"
            result = self._run_python_json(
                f"""
import json
from pathlib import Path
from chem_sae.analysis.feature_substructure import run_feature_substructure_analysis
result = run_feature_substructure_analysis(
    audit_reports_dir=Path({json.dumps(str(reports_dir))}),
    output_dir=Path({json.dumps(str(output_dir))}),
    tasks=("bbbp",),
    feature_ids=None,
    top_n_examples=3,
)
print(json.dumps(result))
"""
            )
            self.assertTrue((output_dir / "feature_substructure_summary.json").exists())
            self.assertTrue((output_dir / "feature_substructure_report.md").exists())
            self.assertEqual(result["feature_count"], 1)
            self.assertEqual(result["tasks"], ["bbbp"])


if __name__ == "__main__":
    unittest.main()
