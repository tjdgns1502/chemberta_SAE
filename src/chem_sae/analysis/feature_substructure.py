from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

from chem_sae.utils import save_json


def murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold)


def summarize_molecule_set_substructures(
    smiles_list: list[str],
    *,
    top_n_scaffolds: int = 5,
    mcs_threshold: float = 0.6,
    mcs_timeout_seconds: int = 3,
) -> dict[str, Any]:
    valid_mols = []
    scaffold_counter: Counter[str] = Counter()
    invalid_smiles = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_smiles += 1
            continue
        valid_mols.append(mol)
        scaffold = murcko_scaffold_smiles(smiles)
        if scaffold is not None:
            scaffold_counter[scaffold] += 1

    valid_count = len(valid_mols)
    top_scaffolds = []
    for scaffold_smiles, count in scaffold_counter.most_common(top_n_scaffolds):
        top_scaffolds.append(
            {
                "scaffold_smiles": scaffold_smiles,
                "count": int(count),
                "fraction": float(count / valid_count) if valid_count else 0.0,
            }
        )

    threshold = max(0.1, min(float(mcs_threshold), 1.0))
    mcs_smarts = None
    mcs_num_atoms = 0
    mcs_num_bonds = 0
    mcs_canceled = False
    if valid_count >= 2:
        mcs_result = rdFMCS.FindMCS(
            valid_mols,
            threshold=threshold,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=max(1, int(mcs_timeout_seconds)),
        )
        if mcs_result and mcs_result.numAtoms > 0:
            mcs_smarts = mcs_result.smartsString
            mcs_num_atoms = int(mcs_result.numAtoms)
            mcs_num_bonds = int(mcs_result.numBonds)
            mcs_canceled = bool(getattr(mcs_result, "canceled", False))

    return {
        "valid_molecule_count": valid_count,
        "invalid_smiles_count": int(invalid_smiles),
        "top_scaffolds": top_scaffolds,
        "mcs_smarts": mcs_smarts,
        "mcs_num_atoms": mcs_num_atoms,
        "mcs_num_bonds": mcs_num_bonds,
        "mcs_canceled": mcs_canceled,
    }


def _combine_top_examples(
    feature_card: dict[str, Any],
    *,
    top_n_examples: int,
) -> list[dict[str, Any]]:
    combined = list(feature_card.get("top_train_examples", [])) + list(
        feature_card.get("top_test_examples", [])
    )
    combined.sort(
        key=lambda row: float(row.get("activation", float("-inf"))),
        reverse=True,
    )
    return combined[:top_n_examples]


def summarize_feature_card_substructures(
    feature_card: dict[str, Any],
    *,
    top_n_examples: int = 10,
    top_n_scaffolds: int = 5,
    mcs_threshold: float = 0.6,
    mcs_timeout_seconds: int = 3,
) -> dict[str, Any]:
    summary = dict(feature_card.get("summary", {}))
    top_examples = _combine_top_examples(feature_card, top_n_examples=top_n_examples)
    top_smiles = [str(row["smiles"]) for row in top_examples if row.get("smiles")]

    return {
        "task": feature_card.get("task"),
        "feature_idx": int(summary["feature_idx"]),
        "coef_mean": float(summary.get("coef_mean", 0.0)),
        "single_feature_roc_auc": summary.get("single_feature_roc_auc"),
        "top_example_count": len(top_examples),
        "top_examples": top_examples,
        "substructure_summary": summarize_molecule_set_substructures(
            top_smiles,
            top_n_scaffolds=top_n_scaffolds,
            mcs_threshold=mcs_threshold,
            mcs_timeout_seconds=mcs_timeout_seconds,
        ),
    }


def _load_feature_cards(
    audit_reports_dir: Path,
    *,
    tasks: tuple[str, ...] | None,
    feature_ids: tuple[int, ...] | None,
) -> list[dict[str, Any]]:
    cards_root = audit_reports_dir / "feature_cards"
    if not cards_root.exists():
        raise FileNotFoundError(f"feature_cards directory not found: {cards_root}")

    task_dirs = sorted(path for path in cards_root.iterdir() if path.is_dir())
    selected_tasks = set(tasks) if tasks else None
    selected_features = set(feature_ids) if feature_ids else None

    cards: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        if selected_tasks is not None and task_dir.name not in selected_tasks:
            continue
        for card_path in sorted(task_dir.glob("feature_*.json")):
            card = json.loads(card_path.read_text(encoding="utf-8"))
            feature_idx = int(card["summary"]["feature_idx"])
            if selected_features is not None and feature_idx not in selected_features:
                continue
            cards.append(card)
    return cards


def _render_feature_substructure_report_markdown(
    *,
    summaries: list[dict[str, Any]],
    output_path: Path,
    audit_reports_dir: Path,
) -> None:
    lines = [
        "# Feature Substructure Grounding",
        "",
        f"Source audit reports: `{audit_reports_dir}`",
        "",
    ]
    for summary in summaries:
        sub = summary["substructure_summary"]
        lines.append(f"## {summary['task']} feature {int(summary['feature_idx'])}")
        lines.append("")
        lines.append(f"- coef_mean: `{float(summary['coef_mean']):.4f}`")
        if summary.get("single_feature_roc_auc") is not None:
            lines.append(
                f"- single_feature_roc_auc: `{float(summary['single_feature_roc_auc']):.4f}`"
            )
        lines.append(f"- top_example_count: `{int(summary['top_example_count'])}`")
        lines.append(f"- valid_molecule_count: `{int(sub['valid_molecule_count'])}`")
        lines.append(f"- invalid_smiles_count: `{int(sub['invalid_smiles_count'])}`")
        if sub.get("mcs_smarts"):
            lines.append(f"- mcs_smarts: `{sub['mcs_smarts']}`")
            lines.append(f"- mcs_num_atoms: `{int(sub['mcs_num_atoms'])}`")
        else:
            lines.append("- mcs_smarts: `None`")
        if sub["top_scaffolds"]:
            top_scaffold = sub["top_scaffolds"][0]
            lines.append(
                f"- dominant_scaffold: `{top_scaffold['scaffold_smiles']}`"
                f" ({int(top_scaffold['count'])}/{int(sub['valid_molecule_count'])})"
            )
        else:
            lines.append("- dominant_scaffold: `None`")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_substructure_analysis(
    *,
    audit_reports_dir: Path,
    output_dir: Path,
    tasks: tuple[str, ...] | None = None,
    feature_ids: tuple[int, ...] | None = None,
    top_n_examples: int = 10,
    top_n_scaffolds: int = 5,
    mcs_threshold: float = 0.6,
    mcs_timeout_seconds: int = 3,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = _load_feature_cards(
        audit_reports_dir,
        tasks=tasks,
        feature_ids=feature_ids,
    )
    summaries = [
        summarize_feature_card_substructures(
            card,
            top_n_examples=top_n_examples,
            top_n_scaffolds=top_n_scaffolds,
            mcs_threshold=mcs_threshold,
            mcs_timeout_seconds=mcs_timeout_seconds,
        )
        for card in cards
    ]
    summaries.sort(key=lambda row: (str(row["task"]), int(row["feature_idx"])))

    summary_payload = {
        "audit_reports_dir": str(audit_reports_dir),
        "tasks": sorted({str(summary["task"]) for summary in summaries}),
        "feature_count": len(summaries),
        "top_n_examples": int(top_n_examples),
        "top_n_scaffolds": int(top_n_scaffolds),
        "mcs_threshold": float(mcs_threshold),
        "mcs_timeout_seconds": int(mcs_timeout_seconds),
        "feature_summaries": summaries,
    }
    summary_path = output_dir / "feature_substructure_summary.json"
    report_path = output_dir / "feature_substructure_report.md"
    save_json(summary_path, summary_payload)
    _render_feature_substructure_report_markdown(
        summaries=summaries,
        output_path=report_path,
        audit_reports_dir=audit_reports_dir,
    )
    return {
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "feature_count": len(summaries),
        "tasks": summary_payload["tasks"],
    }
