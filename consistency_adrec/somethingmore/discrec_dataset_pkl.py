"""
Dataset loader for pre-processed .pkl files from ADRec/DiffuRec pipeline.

Expected format:
    pickle file with dict keys ['train', 'val', 'test']
    train: list of length num_users; train[u] is the user's training sequence (list of item ids)
    val:   list of length num_users; val[u]   is [target_item_id] for validation
    test:  list of length num_users; test[u]  is [target_item_id] for test

Item IDs are integers starting from 1; 0 is reserved for PAD.

Evaluation convention (matching SASRec/DiffuRec/ADRec):
    - For validation: history = train[u], target = val[u][0]
    - For test:       history = train[u] + val[u], target = test[u][0]
"""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Train: per-step prediction (every prefix -> next item)
# ----------------------------------------------------------------------
class TrainDatasetPKL(Dataset):
    """
    For user training sequence of length L, yields (L-1) (history, target) pairs:
        history = seq[:k], target = seq[k], for k = 1..L-1
    This is standard per-step training as in DiffuRec / ADRec.
    """

    def __init__(self, train_list, max_len=50):
        self.max_len = max_len
        self.samples = []
        for seq in train_list:
            if len(seq) < 2:
                continue
            for k in range(1, len(seq)):
                history = seq[:k]
                target = seq[k]
                self.samples.append((history, target))

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
# Eval: one sample per user
# ----------------------------------------------------------------------
class EvalDatasetPKL(Dataset):
    """
    For validation: history = train[u], target = val[u][0]
    For test:       history = train[u] + val[u], target = test[u][0]
    """

    def __init__(self, histories, targets, max_len=50):
        assert len(histories) == len(targets)
        self.histories = histories
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        history = self.histories[idx]
        target = self.targets[idx]
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
def load_pkl_dataset(pkl_path, max_len=50, min_history=1):
    """
    Returns: train_ds, valid_ds, test_ds, num_items.

    min_history: minimum history length to include user in eval splits.
                 Users with shorter history are skipped from eval (but kept in train).
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    train_list = data["train"]
    val_list = data["val"]
    test_list = data["test"]
    num_users = len(train_list)
    assert len(val_list) == num_users and len(test_list) == num_users, \
        f"Mismatched split sizes: train={num_users}, val={len(val_list)}, test={len(test_list)}"

    # Compute num_items: max id observed across all splits
    max_id = 0
    for split in (train_list, val_list, test_list):
        for seq in split:
            if len(seq) > 0:
                max_id = max(max_id, max(seq))
    num_items = max_id  # since ids are in [1, max_id], num_items == max_id

    # Build validation: history = train[u], target = val[u][0]
    val_histories, val_targets = [], []
    for u in range(num_users):
        if len(val_list[u]) == 0:
            continue
        if len(train_list[u]) < min_history:
            continue
        val_histories.append(train_list[u])
        val_targets.append(val_list[u][0])

    # Build test: history = train[u] + val[u], target = test[u][0]
    test_histories, test_targets = [], []
    for u in range(num_users):
        if len(test_list[u]) == 0:
            continue
        combined_history = list(train_list[u]) + list(val_list[u])
        if len(combined_history) < min_history:
            continue
        test_histories.append(combined_history)
        test_targets.append(test_list[u][0])

    train_ds = TrainDatasetPKL(train_list, max_len=max_len)
    valid_ds = EvalDatasetPKL(val_histories, val_targets, max_len=max_len)
    test_ds = EvalDatasetPKL(test_histories, test_targets, max_len=max_len)

    print(f"[Data] Loaded {pkl_path.name}")
    print(f"[Data] num_users={num_users}, num_items={num_items}")
    print(f"[Data] train pairs={len(train_ds)}, valid users={len(valid_ds)}, test users={len(test_ds)}")

    return train_ds, valid_ds, test_ds, num_items