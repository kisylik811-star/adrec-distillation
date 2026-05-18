"""
Evaluation utilities: HR@k, NDCG@k computed via full-catalog ranking.
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, eval_dataset, k_list=(5, 10, 20), batch_size=512, device="cuda", num_steps=1):
    """
    Standard full-catalog evaluation.
    For each user: rank all N items by predicted probability, compute HR and NDCG @ k.

    num_steps: number of inference steps (1 for one-step, can be > 1).
    """
    model.eval()
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    hits = {k: 0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    total = 0

    for history, target in loader:
        history = history.to(device)
        target = target.to(device)

        if num_steps == 1:
            scores = model.predict_one_step(history)            # (B, N)
        else:
            scores = model.predict_multi_step(history, num_steps=num_steps)

        # target ids are in [1, N]; we compare to argsort indices in [0, N-1]
        target_idx = target - 1                                  # (B,)

        # Get rank of true target for each user
        # rank = number of items with strictly higher score + 1
        true_scores = scores.gather(1, target_idx.unsqueeze(1))  # (B, 1)
        rank = (scores > true_scores).sum(dim=1) + 1             # (B,)

        for k in k_list:
            hit = (rank <= k).float()
            hits[k] += hit.sum().item()
            # NDCG: 1 / log2(rank + 1) if rank <= k else 0
            log_rank = torch.log2(rank.float() + 1.0)
            ndcg = (1.0 / log_rank) * hit
            ndcgs[k] += ndcg.sum().item()

        total += history.size(0)

    metrics = {}
    for k in k_list:
        metrics[f"HR@{k}"] = 100.0 * hits[k] / total
        metrics[f"NDCG@{k}"] = 100.0 * ndcgs[k] / total
    return metrics


@torch.no_grad()
def measure_latency(model, eval_dataset, batch_size=1, num_warmup=10, num_runs=100, device="cuda"):
    """Per-sample inference latency (ms)."""
    model.eval()
    sample = eval_dataset[0]
    history = sample[0].unsqueeze(0).to(device)

    # Warmup
    for _ in range(num_warmup):
        _ = model.predict_one_step(history)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = model.predict_one_step(history)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs * 1000.0  # ms
    return elapsed