"""
Dataset loader for sequential recommendation with leave-one-out split.

Expected input format: a tab-separated file where each line is
    user_id<TAB>item_id<TAB>...<TAB>item_id
Or directly a pickle file with format {user_id: [item_ids in chronological order]}.

We implement raw loading + 5-core filtering + leave-one-out split.
For Amazon Beauty / Toys: download from https://nijianmo.github.io/amazon/index.html
or use a local preprocessed file.

For ML-100K: download from https://files.grouplens.org/datasets/movielens/ml-100k.zip
File format: u.data has lines "user_id\titem_id\trating\ttimestamp".
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_ml100k(path):
    """Parse u.data file. Returns dict {user_id: [(item_id, timestamp), ...]}."""
    interactions = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            u, i, _r, t = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            interactions[u].append((i, t))
    return interactions


def parse_amazon(path):
    """
    Parse Amazon review file (one JSON per line, or a TSV with cols
    user_id, item_id, rating, timestamp).
    We support a simple TSV here for robustness.
    """
    interactions = defaultdict(list)
    with open(path, "r") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if first:
                first = False
                # Try to detect header
                try:
                    int(parts[0])
                except ValueError:
                    continue
            u = parts[0]
            i = parts[1]
            t = int(parts[-1])
            interactions[u].append((i, t))
    return interactions


def five_core_filter(interactions, min_user_inter=5, min_item_inter=5, max_iter=10):
    """Iterative 5-core filtering."""
    # First, sort interactions per user by timestamp
    for u in list(interactions.keys()):
        interactions[u].sort(key=lambda x: x[1])

    for _ in range(max_iter):
        item_counts = defaultdict(int)
        for u, lst in interactions.items():
            for i, _ in lst:
                item_counts[i] += 1
        # Keep items with >= min_item_inter
        new_interactions = {}
        for u, lst in interactions.items():
            kept = [(i, t) for i, t in lst if item_counts[i] >= min_item_inter]
            if len(kept) >= min_user_inter:
                new_interactions[u] = kept

        if len(new_interactions) == len(interactions) and all(
            len(v) == len(interactions[k]) for k, v in new_interactions.items()
        ):
            interactions = new_interactions
            break
        interactions = new_interactions

    return interactions


def remap_ids(interactions):
    """Remap raw item ids to dense integers starting from 1 (0 reserved for PAD)."""
    item_id_map = {}
    next_id = 1
    sequences = {}
    for u, lst in interactions.items():
        seq = []
        for i, _ in lst:
            if i not in item_id_map:
                item_id_map[i] = next_id
                next_id += 1
            seq.append(item_id_map[i])
        sequences[u] = seq
    num_items = next_id - 1
    return sequences, num_items


def leave_one_out_split(sequences, min_len=3):
    """
    Standard leave-one-out:
      train: seq[:-2]
      valid: seq[:-1], target = seq[-2]
      test:  seq,      target = seq[-1]
    Only keep users with >= min_len interactions.
    """
    train, valid, test = {}, {}, {}
    for u, seq in sequences.items():
        if len(seq) < min_len:
            continue
        train[u] = seq[:-2]
        valid[u] = (seq[:-2], seq[-2])
        test[u] = (seq[:-1], seq[-1])
    return train, valid, test


# ----------------------------------------------------------------------
# Torch Datasets
# ----------------------------------------------------------------------
class TrainDataset(Dataset):
    """
    For each user training sequence of length L,
    we yield (L-1) (history, target) pairs - per-step prediction.
    Equivalently: each item (except first) is a target with preceding items as history.
    """

    def __init__(self, train_sequences, max_len=50):
        self.max_len = max_len
        self.samples = []
        for u, seq in train_sequences.items():
            for k in range(1, len(seq)):
                history = seq[:k]
                target = seq[k]
                self.samples.append((history, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        # Truncate to max_len, left-pad with zeros
        history = history[-self.max_len:]
        pad_len = self.max_len - len(history)
        history_padded = [0] * pad_len + list(history)
        return (
            torch.tensor(history_padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class EvalDataset(Dataset):
    """For validation and test - one sample per user."""

    def __init__(self, eval_sequences, max_len=50):
        self.max_len = max_len
        self.samples = [(history, target) for u, (history, target) in eval_sequences.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        history = history[-self.max_len:]
        pad_len = self.max_len - len(history)
        history_padded = [0] * pad_len + list(history)
        return (
            torch.tensor(history_padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


# ----------------------------------------------------------------------
# Top-level loader
# ----------------------------------------------------------------------
def load_dataset(dataset_name, data_root, max_len=50, cache=True):
    """
    Returns: train_ds, valid_ds, test_ds, num_items.

    Supported dataset_name: 'ml-100k', 'amazon_beauty', 'amazon_toys'.
    """
    data_root = Path(data_root)
    cache_path = data_root / f"{dataset_name}_cache.pkl"

    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return (
            TrainDataset(data["train"], max_len=max_len),
            EvalDataset(data["valid"], max_len=max_len),
            EvalDataset(data["test"], max_len=max_len),
            data["num_items"],
        )

    if dataset_name == "ml-100k":
        raw = parse_ml100k(data_root / "ml-100k" / "u.data")
    elif dataset_name in ("amazon_beauty", "amazon_toys"):
        # Expects a TSV file at data_root/<name>/interactions.tsv
        raw = parse_amazon(data_root / dataset_name / "interactions.tsv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    raw = five_core_filter(raw, min_user_inter=5, min_item_inter=5)
    sequences, num_items = remap_ids(raw)
    train, valid, test = leave_one_out_split(sequences, min_len=3)

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "train": train,
                    "valid": valid,
                    "test": test,
                    "num_items": num_items,
                },
                f,
            )

    print(
        f"[Data] {dataset_name}: users={len(train)}, items={num_items}, "
        f"train pairs={sum(len(s)-1 for s in train.values())}"
    )

    return (
        TrainDataset(train, max_len=max_len),
        EvalDataset(valid, max_len=max_len),
        EvalDataset(test, max_len=max_len),
        num_items,
    )