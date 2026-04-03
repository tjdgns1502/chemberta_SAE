#!/home/yoo122333/micromamba/envs/chemberta-repro/bin/python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORTS_DIR = Path("/home/yoo122333/capstone/chemberta_SAE/artifacts/reports")
FEATURE_1419_PATH = Path(
    "/home/yoo122333/capstone/chemberta_SAE/artifacts/runs/sae/"
    "layer0_audit_bace_top100_20260319/reports/feature_cards/"
    "bace_classification/feature_1419.json"
)

DOWNSTREAM_PNG = REPORTS_DIR / "layer0_downstream_choice_chart_20260319.png"
CAUSAL_PNG = REPORTS_DIR / "layer0_causal_effect_chart_20260319.png"
ACTIVATION_PNG = REPORTS_DIR / "bace1419_top_activation_chart_20260319.png"


def _base_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_downstream_chart() -> None:
    tasks = ["BBBP", "BACE", "ClinTox"]
    baseline = np.array([0.7190, 0.7637, 0.9993])
    sae1536 = np.array([0.6918, 0.7620, 0.9978])
    sae2048 = np.array([0.7041, 0.7726, 1.0000])

    x = np.arange(len(tasks))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(x - width, baseline, width, label="Baseline", color="#8A9199")
    ax.bar(x, sae1536, width, label="SAE 1536 / 0.05", color="#E39B2E")
    ax.bar(x + width, sae2048, width, label="SAE 2048 / 0.05", color="#2F6B3B")

    ax.set_title("Why Layer 0 Candidate Was Chosen")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.65, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    for idx, value in enumerate(sae2048):
        ax.text(x[idx] + width, value + 0.006, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(DOWNSTREAM_PNG, bbox_inches="tight")
    plt.close(fig)


def make_causal_chart() -> None:
    labels = ["BBBP zero", "BBBP force_on", "BACE zero", "BACE force_on"]
    target = np.array([-0.00333, -0.00491, 0.00063, -0.00424])
    control = np.array([-0.00057, -0.00020, -0.00007, 0.00017])

    y = np.arange(len(labels))
    height = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.barh(y + height / 2, target, height, label="Target feature/group", color="#B4442A")
    ax.barh(y - height / 2, control, height, label="Matched random control", color="#4B88A2")
    ax.axvline(0, color="#444444", linewidth=1)

    ax.set_title("Task-Linked Features Move AUC More Than Random Controls")
    ax.set_xlabel("Delta ROC-AUC")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")

    for ypos, value in zip(y + height / 2, target):
        ax.text(value + (0.00015 if value >= 0 else -0.00015), ypos, f"{value:+.4f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)
    for ypos, value in zip(y - height / 2, control):
        ax.text(value + (0.00015 if value >= 0 else -0.00015), ypos, f"{value:+.4f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)

    fig.tight_layout()
    fig.savefig(CAUSAL_PNG, bbox_inches="tight")
    plt.close(fig)


def make_activation_chart() -> None:
    payload = json.loads(FEATURE_1419_PATH.read_text(encoding="utf-8"))
    top_train = payload["top_train_examples"][:4]
    top_test = payload["top_test_examples"][:4]
    rows = top_train + top_test

    labels = [f"{row['split'][0].upper()}{row['rank']}" for row in rows]
    values = np.array([row["activation"] for row in rows], dtype=float)
    colors = ["#2F6B3B"] * len(top_train) + ["#4B88A2"] * len(top_test)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("BACE Feature 1419: Top Example Activations")
    ax.set_ylabel("Activation")
    ax.set_xlabel("T = train top examples, E = test top examples")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.18)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.text(0.01, 0.98, "Detailed SMILES are listed in the table / spreadsheet", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color="#444444")

    fig.tight_layout()
    fig.savefig(ACTIVATION_PNG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _base_style()
    make_downstream_chart()
    make_causal_chart()
    make_activation_chart()
    print(DOWNSTREAM_PNG)
    print(CAUSAL_PNG)
    print(ACTIVATION_PNG)


if __name__ == "__main__":
    main()
