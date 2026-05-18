"""
Evaluation utilities for ADRec consistency distillation.

- teacher_truncated_predict / evaluate_teacher_truncated:
    The "naive truncation" baseline. Runs the teacher's native
    denoise_sample loop with `num_steps` reverse steps instead of T.
    Only the last sequence position is denoised iteratively (matches
    ADRec's eval protocol). Stochastic ancestral sampling — same
    operator as the teacher's real inference, just fewer iterations.

- measure_latency_grid:
    Per-sample latency in ms for teacher (truncated) and student at each
    NFE, and for the teacher with the full T-step schedule.

Both functions are deliberately written to use the teacher's own modules
end-to-end (no re-implemented DDIM) so the baseline reflects what someone
would actually deploy if they "just turned down the step count".
"""
import time

import numpy as np
import torch

# From the adrec_original tree.
from metrics import hrs_and_ndcgs_k


@torch.no_grad()
def teacher_truncated_predict(teacher, sequence, target, num_steps):
    """
    Run the teacher with `num_steps` reverse steps and return per-item scores
    for the last sequence position.

    Mirrors Att_Diffuse_model.forward(train_flag=False) -> diffu.denoise_sample
    but replaces the full `range(T)[::-1]` iteration with `linspace(T-1, 0, K)`
    where K = num_steps.

    num_steps = 1 means a single reverse step at t = T-1. num_steps = T
    recovers the teacher's native inference exactly.
    """
    device = sequence.device

    # ----- Preprocessing identical to Att_Diffuse_model.forward (eval) -----
    item_emb = teacher.item_embedding(sequence)
    item_emb = teacher.embed_dropout(item_emb)
    item_emb = teacher.hist_norm(item_emb)
    mask_seq = (sequence > 0).float()

    tgt_emb = teacher.item_embedding(target)
    mask_tag = (target > 0).float().view(target.shape[0], -1)

    diffu = teacher.diffu
    T = diffu.num_timesteps

    # Encode history ONCE.
    item_rep = diffu.ag_encoder(item_emb, mask_seq)
    B, L, H = item_rep.shape

    # Last position starts as Gaussian noise, others get clean past targets
    # (autoregressive teacher forcing; ADRec native protocol).
    noise_x_t = torch.randn_like(tgt_emb)

    if num_steps == 1:
        ts = [int(T - 1)]
    else:
        # T-1 -> 0 inclusive, K points (matches range(T)[::-1] when K=T).
        ts = np.linspace(T - 1, 0, num_steps).round().astype(int).tolist()

    for t_val in ts:
        t = torch.tensor(
            [0] * (L - 1) + [int(t_val)],
            device=device, dtype=torch.long,
        ).unsqueeze(0).repeat(B, 1)
        # Re-anchor positions 0..L-2 to the clean target context every step
        # (this is what the teacher's denoise_sample does).
        noise_x_t = torch.cat([tgt_emb[:, :-1], noise_x_t[:, -1:]], dim=1)
        noise_x_t = diffu.p_sample(item_rep, noise_x_t, t, mask_seq, mask_tag)

    last_item = noise_x_t[:, -1, :]
    return teacher.calculate_score(last_item)


@torch.no_grad()
def evaluate_teacher_truncated(teacher, loader, num_steps, device,
                               ks=(5, 10, 20)):
    """HR / NDCG for the teacher with `num_steps` reverse iterations."""
    teacher.eval()
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        scores = teacher_truncated_predict(teacher, seq, target, num_steps)
        m = hrs_and_ndcgs_k(scores, target[:, -1:], list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def measure_latency_grid(teacher, student, sample_batch, device,
                         nfe_grid, n_warmup=10, n_runs=50):
    """ms/sample for {student, teacher_truncated} at each NFE plus teacher_full.

    Uses a single fixed batch repeated `n_runs` times after `n_warmup` warmup
    iterations. CUDA syncs are taken before/after timing windows.
    """
    seq, target = [x.to(device) for x in sample_batch]
    bs = seq.size(0)
    out = {'student': {}, 'teacher_truncated': {}, 'teacher_full': None}

    teacher.eval()
    student.eval()

    def _time(fn):
        for _ in range(n_warmup):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_runs):
            fn()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        return (time.time() - t0) * 1000.0 / n_runs / bs

    for nfe in nfe_grid:
        out['student'][str(nfe)] = _time(
            lambda n=nfe: student.predict_scores(seq, target, num_steps=n)
        )
        out['teacher_truncated'][str(nfe)] = _time(
            lambda n=nfe: teacher_truncated_predict(teacher, seq, target,
                                                    num_steps=n)
        )

    def _teacher_full():
        out_seq, last_item, *_ = teacher(seq, target, train_flag=False)
        teacher.calculate_score(last_item)

    out['teacher_full'] = _time(_teacher_full)

    return out