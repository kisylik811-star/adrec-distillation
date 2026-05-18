"""
Train DiscRec on preprocessed .pkl datasets (ADRec/DiffuRec format).

Example usage:
    python train_discrec_pkl.py --pkl_path /path/to/ml-100k.pkl --epochs 100
    python train_discrec_pkl.py --pkl_path /path/to/amazon_beauty.pkl --epochs 100
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from discrec_model import DiscRec
from discrec_dataset_pkl import load_pkl_dataset
from discrec_eval import evaluate, measure_latency


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path", type=str, required=True,
                   help="Path to .pkl file with keys ['train', 'val', 'test']")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Optional name for output directory. Defaults to .pkl stem.")
    p.add_argument("--out_dir", type=str, default="./artifacts_discrec")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--num_layers_history", type=int, default=2)
    p.add_argument("--num_layers_score", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--T", type=int, default=50)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    # Evaluation
    p.add_argument("--eval_nfe", type=int, nargs="+", default=[1, 2, 4, 8])

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    name = args.dataset_name or Path(args.pkl_path).stem
    out_dir = Path(args.out_dir) / name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[Setup] Loading {args.pkl_path}")
    train_ds, valid_ds, test_ds, num_items = load_pkl_dataset(
        args.pkl_path, max_len=args.max_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    model = DiscRec(
        num_items=num_items,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers_history=args.num_layers_history,
        num_layers_score=args.num_layers_score,
        dropout=args.dropout,
        max_len=args.max_len,
        T=args.T,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] DiscRec: {n_params/1e6:.2f}M params, num_items={num_items}, T={args.T}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_hr10 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    history_log = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for history, target in train_loader:
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            loss = model.compute_loss(history, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(losses))
        epoch_time = time.time() - t0

        log_entry = {"epoch": epoch, "train_loss": avg_loss, "time": epoch_time}
        print(f"[Epoch {epoch:3d}] loss={avg_loss:.4f}  time={epoch_time:.1f}s")

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate(
                model, valid_ds, k_list=(5, 10, 20),
                batch_size=args.batch_size, device=device, num_steps=1,
            )
            print(f"  [Val NFE=1] {val_metrics}")
            log_entry["val"] = val_metrics

            if val_metrics["HR@10"] > best_val_hr10:
                best_val_hr10 = val_metrics["HR@10"]
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), out_dir / "best.pt")
                print(f"  [Save] new best (HR@10={best_val_hr10:.4f})")
            else:
                epochs_without_improvement += args.eval_interval

            if epochs_without_improvement >= args.patience:
                print(f"[Early stop] no improvement for {args.patience} epochs")
                break

        history_log.append(log_entry)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))

    print(f"\n=== Final Test Evaluation (best epoch={best_epoch}) ===")
    test_results = {}
    for nfe in args.eval_nfe:
        m = evaluate(
            model, test_ds, k_list=(5, 10, 20),
            batch_size=args.batch_size, device=device, num_steps=nfe,
        )
        test_results[f"NFE={nfe}"] = m
        print(f"  NFE={nfe} {m}")

    latency = measure_latency(model, test_ds, batch_size=1, device=device)
    print(f"\n=== Inference Latency: {latency:.4f} ms/sample (NFE=1, batch=1) ===")

    summary = {
        "dataset": name,
        "pkl_path": str(args.pkl_path),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_HR@10": best_val_hr10,
        "test_results": test_results,
        "latency_ms_per_sample": latency,
        "num_params_M": n_params / 1e6,
        "num_items": num_items,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(history_log, f, indent=2)

    print(f"\n[Done] Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()