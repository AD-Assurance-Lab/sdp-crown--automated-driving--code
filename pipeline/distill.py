#!/usr/bin/env python3
"""
Knowledge distillation: train a small verifiable StudentNet to match the DAgger
teacher's steering output, over the aggregated DAgger dataset (which already
contains recovery states, so the student inherits recovery behavior).

Teacher targets are resolution-independent, so they're computed once and cached;
the resolution sweep then reuses them for each student input size.

    python distill.py --in-w 96 --in-h 42 --out student_096x42
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config as C
from model import CarlaSteeringNet
from student import StudentNet, student_preprocess
from imaging import preprocess_for_model
from dataset import load_manifests, block_split


def aggregated_manifests(base="clear"):
    """Base BC manifest + every DAgger round manifest present on disk."""
    paths = [os.path.join(C.DATASET_DIR, base, "manifest.csv")]
    dagger = os.path.join(C.DATASET_DIR, "dagger")
    if os.path.isdir(dagger):
        for d in sorted(os.listdir(dagger)):
            m = os.path.join(dagger, d, "manifest.csv")
            if os.path.isfile(m):
                paths.append(m)
    return paths


def teacher_targets(rows, teacher_name, device):
    """dict abs_image_path -> teacher steering. Cached to an .npz keyed by teacher."""
    cache = os.path.join(C.DATASET_DIR, "dagger", f"teacher_targets_{teacher_name}.npz")
    if os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        d = dict(zip(z["paths"], z["steer"]))
        if all(r["image"] in d for r in rows):
            print(f"loaded cached teacher targets ({len(d)}) from {cache}")
            return d
    print(f"computing teacher targets for {len(rows)} frames with '{teacher_name}'...")
    teacher = CarlaSteeringNet().to(device)
    teacher.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{teacher_name}.pth"),
                                       map_location=device))
    teacher.eval()
    paths, steers = [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            bgr = cv2.imread(r["image"])
            x = torch.from_numpy(preprocess_for_model(bgr)).unsqueeze(0).to(device)
            paths.append(r["image"]); steers.append(float(teacher(x).item()))
            if i % 2000 == 0:
                print(f"  {i}/{len(rows)}")
    np.savez(cache, paths=np.array(paths), steer=np.array(steers, dtype=np.float32))
    return dict(zip(paths, steers))


class KDDataset(Dataset):
    def __init__(self, rows, indices, targets, in_w, in_h, preload=True):
        self.rows, self.indices, self.targets = rows, indices, targets
        self.in_w, self.in_h = in_w, in_h
        self.cache = None
        if preload:
            self.cache = [student_preprocess(cv2.imread(rows[i]["image"]), in_w, in_h)
                          for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = self.cache[k] if self.cache is not None else \
            student_preprocess(cv2.imread(self.rows[i]["image"]), self.in_w, self.in_h)
        return torch.from_numpy(x), torch.tensor([self.targets[self.rows[i]["image"]]],
                                                 dtype=torch.float32)


def _mse(model, loader, device):
    model.eval(); se = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            se += nn.functional.mse_loss(model(x), y, reduction="sum").item(); n += y.numel()
    return se / n


def distill_student(in_w, in_h, out_name, teacher_name="steering_dagger_r02",
                    base="clear", epochs=120, batch_size=64, lr=1e-3, patience=20,
                    device=None, quiet=False):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    _, rows = load_manifests(aggregated_manifests(base))
    targets = teacher_targets(rows, teacher_name, device)
    tr_idx, va_idx = block_split(len(rows), val_frac=0.15, block=50, seed=0)

    tr = KDDataset(rows, tr_idx, targets, in_w, in_h)
    va = KDDataset(rows, va_idx, targets, in_w, in_h)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    student = StudentNet(in_h, in_w).to(device)
    nrelu = student.num_relu_neurons()
    if not quiet:
        print(f"student {in_w}x{in_h}: {nrelu} ReLU neurons, {sum(p.numel() for p in student.parameters())} params "
              f"| train={len(tr_idx)} val={len(va_idx)}")
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)

    ckpt = os.path.join(C.CHECKPOINT_DIR, f"{out_name}.pth")
    best, bad = float("inf"), 0
    for ep in range(epochs):
        student.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = nn.functional.mse_loss(student(x), y)
            loss.backward(); opt.step()
        v = _mse(student, vl, device); sched.step(v)
        if v < best:
            best, bad = v, 0; torch.save(student.state_dict(), ckpt)
        else:
            bad += 1
        if not quiet and (ep % 10 == 0 or bad == 0):
            print(f"  epoch {ep:3d} | val KD-MSE {v:.3e} | rmse {np.sqrt(v):.4f}{' *' if bad==0 else ''}")
        if bad >= patience:
            break
    if not quiet:
        print(f"BEST KD val MSE={best:.3e} RMSE={np.sqrt(best):.4f} ({nrelu} neurons) -> {ckpt}")
    return {"best_val": best, "neurons": nrelu, "ckpt": ckpt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-w", type=int, required=True)
    ap.add_argument("--in-h", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="steering_dagger_r02")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    distill_student(args.in_w, args.in_h, args.out, teacher_name=args.teacher, epochs=args.epochs)


if __name__ == "__main__":
    main()
