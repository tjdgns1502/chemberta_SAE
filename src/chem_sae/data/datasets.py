from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import RobertaTokenizerFast


class MLMSmilesDataset(Dataset):
    def __init__(self, path: Path, tokenizer: RobertaTokenizerFast, max_len: int = 128):
        with open(path) as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.lines[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


class SmilesClassificationDataset(Dataset):
    def __init__(self, df, tokenizer: RobertaTokenizerFast, label_cols: list[str], max_len: int = 128):
        self.smiles = df["smiles"].astype(str).tolist()
        labels = df[label_cols].to_numpy(dtype=np.float32)
        self.label_mask = ~np.isnan(labels)
        self.labels = np.nan_to_num(labels, nan=0.0)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        item["label_mask"] = torch.tensor(self.label_mask[idx], dtype=torch.float32)
        return item


class ActivationChunkDataset(IterableDataset):
    def __init__(
        self,
        chunk_paths: list[Path],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.chunk_paths = chunk_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch_count = 0

    def __iter__(self):
        epoch_seed = self.seed + self._epoch_count
        self._epoch_count += 1
        rng = random.Random(epoch_seed)
        paths = list(self.chunk_paths)
        if self.shuffle:
            rng.shuffle(paths)
        for path in paths:
            acts = torch.load(path, map_location="cpu", weights_only=True)
            if self.shuffle:
                idx = torch.randperm(acts.shape[0], generator=torch.Generator().manual_seed(epoch_seed))
                acts = acts[idx]
            for i in range(0, acts.shape[0], self.batch_size):
                yield acts[i : i + self.batch_size]
