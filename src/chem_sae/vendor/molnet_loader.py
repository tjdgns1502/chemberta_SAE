"""MolNet downstream loader using Hugging Face Hub-hosted CSV + split files."""

from __future__ import annotations

import json
from typing import List

import pandas as pd
from huggingface_hub import hf_hub_download


MOLNET_DIRECTORY = {
    "bace_classification": {
        "dataset_type": "classification",
        "split": "scaffold",
        "repo_id": "scikit-fingerprints/MoleculeNet_BACE",
        "csv_filename": "bace.csv",
        "split_filename": "ogb_splits_bace.json",
        "tasks": ["label"],
    },
    "bbbp": {
        "dataset_type": "classification",
        "split": "scaffold",
        "repo_id": "scikit-fingerprints/MoleculeNet_BBBP",
        "csv_filename": "bbbp.csv",
        "split_filename": "ogb_splits_bbbp.json",
        "tasks": ["label"],
    },
    "clintox": {
        "dataset_type": "classification",
        "split": "scaffold",
        "repo_id": "scikit-fingerprints/MoleculeNet_ClinTox",
        "csv_filename": "clintox.csv",
        "split_filename": "ogb_splits_clintox.json",
        "tasks": ["FDA_APPROVED", "CT_TOX"],
        "tasks_wanted": ["CT_TOX"],
    },
}


def get_dataset_info(name: str):
    if name not in MOLNET_DIRECTORY:
        raise ValueError(f"Unsupported MolNet dataset: {name}")
    return MOLNET_DIRECTORY[name]


def _download_hf_dataset_files(
    repo_id: str,
    csv_filename: str,
    split_filename: str,
    local_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    csv_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=csv_filename,
        local_files_only=local_only,
    )
    split_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=split_filename,
        local_files_only=local_only,
    )

    df = pd.read_csv(csv_path)
    with open(split_path) as f:
        split_indices = json.load(f)
    return df, split_indices


def _slice_by_split_indices(
    df: pd.DataFrame, split_indices: dict[str, list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.iloc[split_indices["train"]].reset_index(drop=True)
    valid_df = df.iloc[split_indices["valid"]].reset_index(drop=True)
    test_df = df.iloc[split_indices["test"]].reset_index(drop=True)
    return train_df, valid_df, test_df


def load_molnet_dataset(
    name: str,
    split: str | None = None,
    tasks_wanted: List[str] | None = None,
    df_format: str = "chemberta",
    local_only: bool = False,
):
    """Load selected MolNet dataset from Hugging Face Hub.

    Returns:
        tasks_wanted, (train_df, valid_df, test_df), transformers
    """
    info = get_dataset_info(name)
    requested_split = split or info["split"]
    if requested_split != info["split"]:
        print(
            f"[load_molnet_dataset] '{name}' supports '{info['split']}' only via OGB split file. "
            f"Requested '{requested_split}' ignored."
        )

    raw_df, split_indices = _download_hf_dataset_files(
        repo_id=info["repo_id"],
        csv_filename=info["csv_filename"],
        split_filename=info["split_filename"],
        local_only=local_only,
    )

    # Normalize SMILES column name for downstream code.
    if "SMILES" not in raw_df.columns:
        raise KeyError(f"Expected 'SMILES' column in {name} dataset.")
    raw_df = raw_df.rename(columns={"SMILES": "smiles"})

    selected_tasks = tasks_wanted or info.get("tasks_wanted", info["tasks"])
    missing = [task for task in selected_tasks if task not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing task columns for {name}: {missing}")

    split_df = raw_df[["smiles"] + selected_tasks]
    train_df, valid_df, test_df = _slice_by_split_indices(split_df, split_indices)
    return (
        selected_tasks,
        [
            make_dataframe(train_df, info["dataset_type"], selected_tasks, df_format=df_format),
            make_dataframe(valid_df, info["dataset_type"], selected_tasks, df_format=df_format),
            make_dataframe(test_df, info["dataset_type"], selected_tasks, df_format=df_format),
        ],
        [],
    )


def make_dataframe(
    df: pd.DataFrame,
    dataset_type: str,
    tasks_wanted: list[str],
    df_format: str = "chemberta",
):
    labels = df[tasks_wanted]

    if dataset_type == "classification":
        labels = labels.astype(float)
    elif dataset_type == "regression":
        labels = labels.astype(float)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    if df_format == "chemberta":
        if len(tasks_wanted) == 1:
            labels_out = labels.values.flatten()
        else:
            labels_out = labels.values.tolist()
        return pd.DataFrame({"text": df["smiles"].values.tolist(), "labels": labels_out})

    if df_format == "chemprop":
        out = pd.DataFrame({"smiles": df["smiles"].values.tolist()})
        for task in tasks_wanted:
            out[task] = labels[task].values
        return out

    raise ValueError(f"Unsupported df_format: {df_format}")

