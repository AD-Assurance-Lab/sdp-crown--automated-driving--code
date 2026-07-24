#!/usr/bin/env python3
"""
Behavior-cloning / DAgger training for the steering teacher (CarlaSteeringNet).

train_model() takes a LIST of manifests so DAgger can retrain from scratch on the
aggregated dataset (base BC + all rounds). Baseline CLI defaults: no balancing,
no augmentation, plain MSE. val MSE is an optimistic convergence check (correlated
expert frames) — closed-loop CTE is the real metric.

    python train.py --dataset clear --epochs 120 --out steering_bc_baseline
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from model import CarlaSteeringNet
from dataset import load_manifests, block_split, balance_straight, SteeringDataset


def _evaluate(model, loader, device):
    model.eval()
    se, n, preds, tgts = 0.0, 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            se += nn.functional.mse_loss(out, y, reduction="sum").item()
            n += y.numel()
            preds.append(out.cpu().numpy()); tgts.append(y.cpu().numpy())
    return se / n, np.concatenate(preds).ravel(), np.concatenate(tgts).ravel()


def train_model(manifest_paths, out, epochs=120, batch_size=64, lr=1e-3,
                weight_decay=1e-5, patience=20, balance=False, augment=False,
                shift_max_px=40, shift_k=0.0015, device=None, quiet=False):
    """Train from scratch on the aggregated manifests. Returns best val MSE."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    base, rows = load_manifests(manifest_paths)
    tr_idx, va_idx = block_split(len(rows), val_frac=0.15, block=50, seed=0)
    if balance:
        tr_idx = balance_straight(rows, tr_idx)
    if not quiet:
        print(f"manifests={len(manifest_paths)} total={len(rows)} train={len(tr_idx)} "
              f"val={len(va_idx)} | balance={balance} augment={augment} | device={device}")

    preload = not augment
    tr = SteeringDataset(base, rows, tr_idx, augment=augment,
                         shift_max_px=shift_max_px, shift_k=shift_k, preload=preload)
    va = SteeringDataset(base, rows, va_idx, preload=True)
    nw = 0 if preload else 6
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = CarlaSteeringNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    ckpt = os.path.join(C.CHECKPOINT_DIR, f"{out}.pth")
    hist = {"train": [], "val": []}
    best_val, bad = float("inf"), 0
    for ep in range(epochs):
        model.train()
        run, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            loss.backward(); opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        tr_mse = run / n
        va_mse, _, _ = _evaluate(model, vl, device)
        sched.step(va_mse)
        hist["train"].append(tr_mse); hist["val"].append(va_mse)
        improved = va_mse < best_val
        if improved:
            best_val, bad = va_mse, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if not quiet and (ep % 10 == 0 or improved):
            print(f"  epoch {ep:3d} | train {tr_mse:.3e} | val {va_mse:.3e} "
                  f"| rmse {np.sqrt(va_mse):.4f}{' *' if improved else ''}")
        if bad >= patience:
            if not quiet:
                print(f"  early stop at epoch {ep}")
            break

    model.load_state_dict(torch.load(ckpt))
    va_mse, preds, tgts = _evaluate(model, vl, device)
    if not quiet:
        print(f"BEST val MSE={best_val:.3e} RMSE={np.sqrt(best_val):.4f} -> {ckpt}")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot(hist["train"], label="train"); a1.plot(hist["val"], label="val")
        a1.set_yscale("log"); a1.set_xlabel("epoch"); a1.set_ylabel("MSE"); a1.legend()
        a1.set_title(f"{out}: loss")
        lim = max(abs(tgts).max(), abs(preds).max()) * 1.1
        a2.scatter(tgts, preds, s=4, alpha=0.3); a2.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        a2.set_xlabel("expert steer"); a2.set_ylabel("pred steer"); a2.set_aspect("equal")
        a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim)
        a2.set_title(f"val pred vs expert (RMSE={np.sqrt(va_mse):.4f})")
        fig.tight_layout()
        fig.savefig(os.path.join(C.RESULTS_DIR, f"{out}_training.png"), dpi=110)
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clear", help="dataset name under data/ (single)")
    ap.add_argument("--manifests", nargs="*", default=None, help="explicit manifest paths")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out", default="steering_bc_baseline")
    args = ap.parse_args()

    manifests = args.manifests or [os.path.join(C.DATASET_DIR, args.dataset, "manifest.csv")]
    train_model(manifests, args.out, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
                balance=args.balance, augment=args.augment)


if __name__ == "__main__":
    main()
