"""Training loop and evaluation utilities for ADRec consistency distillation.

Supports single runs, sweep mode, and multi-seed mode. Every successful run
saves:
  artifact_dir/student_final.pt          (model weights)
  artifact_dir/test_predictions_nfe1.npz (per-user predictions for analysis)
  artifact_dir/config.json               (full configuration snapshot)
  artifact_dir/summary.json              (metrics + paths to all artifacts)

Per-epoch loss components and val metrics go to CSV files for downstream
plots (later, in analyze.py).
"""
import copy
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# From the adrec_original tree (caller arranges PYTHONPATH).
from metrics import hrs_and_ndcgs_k


# ===================================================================
# Evaluation helpers
# ===================================================================

@torch.no_grad()
def evaluate_at_nfe(student, loader, num_steps, device, ks=(5, 10, 20)):
    """Evaluate the student at a given NFE on a full data loader.

    Returns HR@k / NDCG@k as percentages (matching ADRec's convention).
    """
    student.eval()
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        scores = student.predict_scores(seq, target, num_steps=num_steps)
        # ADRec convention: HR / NDCG against the last token (the held-out
        # answer). target[:, -1:] has shape (B, 1).
        m = hrs_and_ndcgs_k(scores, target[:, -1:], list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def evaluate_teacher_full_nfe(teacher_model, loader, device, ks=(5, 10, 20)):
    """Evaluate the (frozen) teacher with all T DDIM steps."""
    teacher_model.eval()
    acc = {f'HR@{k}': [] for k in ks}
    acc.update({f'NDCG@{k}': [] for k in ks})
    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        out_seq, last_item, *_ = teacher_model(seq, target, train_flag=False)
        scores = teacher_model.calculate_score(last_item)
        m = hrs_and_ndcgs_k(scores, target[:, -1:], list(ks))
        for k, v in m.items():
            acc[k].append(v)
    return {k: round(float(np.mean(v)) * 100, 4) for k, v in acc.items()}


@torch.no_grad()
def measure_inference_latency(model, sample_batch, device, num_steps=None,
                              n_warmup=10, n_runs=50, mode='student'):
    """Latency per sample in ms. Uses a single fixed batch for stability.

    mode='student'  -> model is ConsistencyADRecStudent; uses predict_scores.
    mode='teacher'  -> model is Att_Diffuse_model; runs forward(train_flag=False)
                       which iterates the full T-step denoise_sample.
    """
    seq, target = [x.to(device) for x in sample_batch]
    model.eval()

    def _run_once():
        if mode == 'student':
            return model.predict_scores(seq, target, num_steps=num_steps)
        else:
            out_seq, last_item, *_ = model(seq, target, train_flag=False)
            return model.calculate_score(last_item)

    # Warmup.
    for _ in range(n_warmup):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        _ = _run_once()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - t0) * 1000.0 / n_runs
    return elapsed_ms / seq.size(0)


@torch.no_grad()
def dump_per_user_predictions(student, loader, num_steps, device,
                              out_path, ks=(5, 10, 20)):
    """Save per-user predictions for length-aware / outlier analysis.

    For ADRec, the "history length" is the count of non-padding tokens in the
    input sequence (target is the LAST token, target[:, -1]).
    """
    student.eval()
    max_k = max(ks)
    all_lengths, all_targets, all_ranks, all_topk, all_hits = [], [], [], [], []

    for batch in loader:
        seq, target = [x.to(device) for x in batch]
        scores = student.predict_scores(seq, target, num_steps=num_steps)

        lengths = (seq > 0).sum(dim=1).cpu().numpy()
        target_last = target[:, -1].cpu().numpy()

        _, topk = torch.topk(scores, k=max_k, dim=-1)
        topk_np = topk.cpu().numpy()

        sorted_idx = scores.argsort(dim=-1, descending=True)
        ranks = []
        for i in range(seq.size(0)):
            pos = (sorted_idx[i] == target[i, -1]).nonzero(as_tuple=True)[0]
            ranks.append(pos.item() if len(pos) > 0 else -1)
        ranks_np = np.array(ranks)

        hits_per_k = []
        for k in ks:
            hit = (topk_np[:, :k] == target_last[:, None]).any(axis=1)
            hits_per_k.append(hit)
        hits_np = np.stack(hits_per_k, axis=1)

        all_lengths.append(lengths)
        all_targets.append(target_last)
        all_ranks.append(ranks_np)
        all_topk.append(topk_np)
        all_hits.append(hits_np)

    np.savez_compressed(
        out_path,
        hist_lengths=np.concatenate(all_lengths),
        target_items=np.concatenate(all_targets),
        target_rank=np.concatenate(all_ranks),
        top_k_items=np.concatenate(all_topk, axis=0),
        hit_at_k=np.concatenate(all_hits, axis=0),
        ks=np.array(list(ks)),
    )


# ===================================================================
# CSV helpers
# ===================================================================

def _open_csv(path, header):
    Path(os.path.dirname(path) or '.').mkdir(parents=True, exist_ok=True)
    f = open(path, 'w', newline='')
    w = csv.writer(f)
    w.writerow(header)
    return f, w


# ===================================================================
# Main training loop
# ===================================================================

def distill_train(student, teacher_diffu, train_loader, val_loader,
                  test_loader, args, logger,
                  log_csv_path=None, run_name=None):
    """Train the student via consistency distillation.

    Args contract:
        args.distill_lr             — Adam LR for the student.
        args.distill_epochs         — max epochs.
        args.distill_eval_interval  — validate every N epochs.
        args.distill_patience       — early-stop patience (# eval intervals).
        args.cons_weight            — weight on L_cons.
        args.ce_weight              — weight on L_ce.
        args.contrast_weight        — beta on L_contrast (effective only when
                                       use_rccd=True).
        args.contrast_temperature   — tau for InfoNCE.
        args.use_rccd               — bool, enable RCCD term.
        args.use_adaptive_weighting — bool, enable min-SNR per-token weights
                                       on L_cons.
        args.dataset                — used in artifact path.
        args.random_seed            — used in default run_name and artifacts.
    """
    device = torch.device(args.device) if isinstance(args.device, str) \
                                       else args.device
    student = student.to(device)

    # Freeze teacher just in case.
    teacher_diffu.eval()
    for p in teacher_diffu.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=args.distill_lr)

    contrast_weight = float(getattr(args, 'contrast_weight', 0.5))
    contrast_temperature = float(getattr(args, 'contrast_temperature', 0.1))
    use_rccd = bool(getattr(args, 'use_rccd', True))
    use_aw = bool(getattr(args, 'use_adaptive_weighting', True))

    # ----- Run name & artifact directory -----
    if run_name is None:
        rccd_tag = 'rccd' if use_rccd else 'norccd'
        aw_tag = 'aw' if use_aw else 'noaw'
        run_name = (f"seed{args.random_seed}_{rccd_tag}_{aw_tag}"
                    f"_beta{contrast_weight}_tau{contrast_temperature}")

    artifact_dir = os.path.join('artifacts_adrec', args.dataset, run_name)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    # ----- Config snapshot for reproducibility -----
    config_snapshot = {
        k: (v if isinstance(v, (int, float, str, bool, list, type(None)))
            else str(v))
        for k, v in vars(args).items()
    }
    config_snapshot['run_name'] = run_name
    with open(os.path.join(artifact_dir, 'config.json'), 'w') as f:
        json.dump(config_snapshot, f, indent=2)

    # ----- CSV logs -----
    loss_writer = val_writer = None
    loss_f = val_f = None
    if log_csv_path is not None:
        loss_f, loss_writer = _open_csv(
            log_csv_path,
            ['epoch', 'cons_loss', 'ce_loss', 'contrast_loss', 'total_loss'],
        )
        val_path = log_csv_path + '.val.csv'
        val_f, val_writer = _open_csv(
            val_path,
            ['epoch', 'HR@5', 'HR@10', 'HR@20',
             'NDCG@5', 'NDCG@10', 'NDCG@20'],
        )

    best_score = -1.0
    best_student = None
    best_epoch = -1
    bad_count = 0

    print(f'[Distill] Starting run "{run_name}" '
          f'(use_rccd={use_rccd}, use_adaptive_weighting={use_aw}, '
          f'beta={contrast_weight}, tau={contrast_temperature})')
    logger.info(f'[Distill] run="{run_name}" use_rccd={use_rccd} '
                f'use_adaptive_weighting={use_aw} '
                f'beta={contrast_weight} tau={contrast_temperature}')

    try:
        for epoch in range(args.distill_epochs):
            student.train()
            running_cons, running_ce, running_contrast, n_b = 0.0, 0.0, 0.0, 0

            for batch in train_loader:
                seq, target = [x.to(device) for x in batch]
                optimizer.zero_grad()

                cons_loss, ce_loss, contrast_loss = student.consistency_loss(
                    seq, target, teacher_diffu,
                    use_rccd=use_rccd,
                    use_adaptive_weighting=use_aw,
                    contrast_temperature=contrast_temperature,
                )

                loss = (
                    args.cons_weight * cons_loss
                    + args.ce_weight * ce_loss
                    + (contrast_weight * contrast_loss if use_rccd
                       else 0.0 * contrast_loss)
                )
                loss.backward()
                optimizer.step()
                student.update_ema()

                running_cons += cons_loss.item()
                running_ce += ce_loss.item()
                running_contrast += contrast_loss.item()
                n_b += 1

            avg_cons = running_cons / max(n_b, 1)
            avg_ce = running_ce / max(n_b, 1)
            avg_contrast = running_contrast / max(n_b, 1)
            avg_total = (
                args.cons_weight * avg_cons
                + args.ce_weight * avg_ce
                + (contrast_weight * avg_contrast if use_rccd else 0.0)
            )

            msg = (f'[Distill][Epoch {epoch}] cons={avg_cons:.4f} '
                   f'ce={avg_ce:.4f} contrast={avg_contrast:.4f} '
                   f'total={avg_total:.4f}')
            print(msg)
            logger.info(msg)

            if loss_writer is not None:
                loss_writer.writerow([epoch, avg_cons, avg_ce,
                                      avg_contrast, avg_total])
                loss_f.flush()

            # ----- Validation -----
            if epoch % args.distill_eval_interval == 0:
                val_metrics = evaluate_at_nfe(student, val_loader,
                                              num_steps=1, device=device)
                msg = f'[Val NFE=1] {val_metrics}'
                print(msg); logger.info(msg)

                if val_writer is not None:
                    val_writer.writerow([
                        epoch,
                        val_metrics['HR@5'], val_metrics['HR@10'],
                        val_metrics['HR@20'],
                        val_metrics['NDCG@5'], val_metrics['NDCG@10'],
                        val_metrics['NDCG@20'],
                    ])
                    val_f.flush()

                if val_metrics['HR@10'] > best_score:
                    best_score = val_metrics['HR@10']
                    best_student = copy.deepcopy(student)
                    best_epoch = epoch
                    bad_count = 0
                else:
                    bad_count += 1
                    if bad_count >= args.distill_patience:
                        print('Early stop')
                        logger.info('Early stop')
                        break
    finally:
        if loss_f is not None:
            loss_f.close()
        if val_f is not None:
            val_f.close()

    if best_student is None:
        best_student = student
        best_epoch = -1

    # ----- Final test across NFEs -----
    print('\n=== Final Test (Student, varying NFE) ===')
    logger.info('=== Final Test (Student, varying NFE) ===')
    test_metrics_per_nfe = {}
    for nfe in [1, 2, 4, 8]:
        m = evaluate_at_nfe(best_student, test_loader, num_steps=nfe,
                            device=device)
        test_metrics_per_nfe[str(nfe)] = m
        line = f'  NFE={nfe} {m}'
        print(line); logger.info(line)

    # ----- Save student checkpoint -----
    checkpoint_path = os.path.join(artifact_dir, 'student_final.pt')
    torch.save(best_student.state_dict(), checkpoint_path)
    print(f'[Save] student checkpoint -> {checkpoint_path}')
    logger.info(f'[Save] student checkpoint -> {checkpoint_path}')

    # ----- Save per-user predictions on test set -----
    pred_path = os.path.join(artifact_dir, 'test_predictions_nfe1.npz')
    print(f'[Save] per-user test predictions -> {pred_path}')
    logger.info(f'[Save] per-user test predictions -> {pred_path}')
    dump_per_user_predictions(best_student, test_loader, num_steps=1,
                              device=device, out_path=pred_path)

    # ----- Latency per NFE on a fixed batch -----
    print('\n=== Inference Latency (ms/sample) ===')
    logger.info('=== Inference Latency (ms/sample) ===')
    sample_batch = next(iter(test_loader))
    latency_per_nfe = {}
    for nfe in [1, 2, 4, 8]:
        ms = measure_inference_latency(best_student, sample_batch, device,
                                       num_steps=nfe, mode='student')
        latency_per_nfe[str(nfe)] = ms
        line = f'  Student NFE={nfe}: {ms:.4f}'
        print(line); logger.info(line)

    # ----- Summary JSON -----
    summary = {
        'run_name': run_name,
        'dataset': args.dataset,
        'random_seed': args.random_seed,
        'use_rccd': use_rccd,
        'use_adaptive_weighting': use_aw,
        'contrast_weight': contrast_weight,
        'contrast_temperature': contrast_temperature,
        'best_val_HR10': float(best_score),
        'best_epoch': int(best_epoch),
        'test_metrics_per_nfe': test_metrics_per_nfe,
        'latency_per_nfe_ms': latency_per_nfe,
        'artifact_paths': {
            'student_checkpoint': os.path.abspath(checkpoint_path),
            'test_predictions':   os.path.abspath(pred_path),
            'config':             os.path.abspath(
                os.path.join(artifact_dir, 'config.json')),
            'loss_csv':           (os.path.abspath(log_csv_path)
                                   if log_csv_path else None),
            'val_csv':            (os.path.abspath(log_csv_path + '.val.csv')
                                   if log_csv_path else None),
        },
    }
    summary_path = os.path.join(artifact_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[Save] summary -> {summary_path}')
    logger.info(f'[Save] summary -> {summary_path}')

    return best_student