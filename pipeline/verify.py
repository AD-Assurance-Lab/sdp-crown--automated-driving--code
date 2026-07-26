#!/usr/bin/env python3
"""
Milestone 3, Prong B — SDP-CROWN verification of the student under ACDC weather
perturbations. Prepends a semantic perturbation layer (2-D [eps_c, eps_b] -> image)
to the student, then bounds the steering output over the ACDC epsilon-box with
CROWN / SDP-CROWN. Certifies a frame SAFE if the worst-case steering stays within
the +/- corridor of the nominal (clear) steering. Reports the certified-safe rate
per condition -- to be compared against the Prong-A empirical failures.

    python verify.py --student student_84x28_dagger_r02 --w 84 --h 28 \
                     --condition night --method CROWN --frames 20
"""
import os
import sys
import gc
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # pipeline/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for auto_LiRPA

import numpy as np
import cv2
import torch
import torch.nn as nn

import config as C
from student import StudentNet, student_preprocess
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from weather_bounds import ACDC_BOUNDS, CARLA_BOUNDS, BOUNDS_SETS  # noqa: F401


class SemanticPerturbationLayer(nn.Module):
    """Maps 2-D [eps_c, eps_b] to a degraded image: x' = x0 + M*(x0*eps_c + eps_b).
    Implemented as nn.Linear(2, n_pixels) with Weight=[x0*M | M], Bias=x0."""

    def __init__(self, x0, mask=None):
        super().__init__()
        _, c, h, w = x0.shape
        flat = x0.reshape(-1)
        n = flat.numel()
        m = torch.ones_like(flat) if mask is None else mask.reshape(-1)
        self.fc = nn.Linear(2, n)
        self.fc.weight.data.copy_(torch.stack([flat * m, m], dim=1))
        self.fc.bias.data.copy_(flat.clone())
        self.shape = (1, c, h, w)

    def forward(self, eps):
        return self.fc(eps).view(self.shape)


class SemanticVerifiedNetwork(nn.Module):
    def __init__(self, base, x0, mask=None):
        super().__init__()
        self.sem = SemanticPerturbationLayer(x0, mask)
        self.base = base

    def forward(self, eps):
        return self.base(self.sem(eps))


def verify_frame(base, x0, b, corridor, device, method="CROWN",
                 mask=None, iterations=100):
    """Verify one nominal frame against eps-bounds dict b={'c':(lo,hi),'b':(lo,hi)}.
    Returns dict(nominal, lb, ub, safe, vacuous)."""
    with torch.no_grad():
        nominal = float(base(x0).item())
    wrapped = SemanticVerifiedNetwork(base, x0, mask).to(device).eval()

    eps0 = torch.zeros(1, 2, device=device)
    eL = torch.tensor([[b["c"][0], b["b"][0]]], device=device)
    eU = torch.tensor([[b["c"][1], b["b"][1]]], device=device)
    beps = BoundedTensor(eps0, PerturbationLpNorm(norm=float("inf"), eps=None, x_L=eL, x_U=eU))

    if method in ("SDP-CROWN", "alpha-CROWN"):
        bound_opts = {"conv_mode": "patches", "optimize_bound_args": {
            "iteration": iterations, "lr_alpha": 0.5, "lr_lambda": 0.05,
            "early_stop_patience": 20, "fix_interm_bounds": True,
            "enable_SDP_crown": (method == "SDP-CROWN")}}
        m = "CROWN-Optimized"
    elif method == "IBP":
        bound_opts = {"conv_mode": "patches"}
        m = "IBP"
    else:  # CROWN
        bound_opts = {"conv_mode": "patches"}
        m = "CROWN"

    lm = BoundedModule(wrapped, beps, device=device, verbose=0, bound_opts=bound_opts)
    lb, ub = lm.compute_bounds(x=(beps,), method=m, bound_lower=True, bound_upper=True)
    lo, hi = min(lb.item(), ub.item()), max(lb.item(), ub.item())
    vacuous = abs(lo) > 100 or abs(hi) > 100
    safe = (not vacuous) and (lo >= nominal - corridor) and (hi <= nominal + corridor)

    del lm, wrapped, beps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(nominal=nominal, lb=lo, ub=hi, safe=safe, vacuous=vacuous)


def load_frames(dataset, w, h, n, device, seed=0):
    """Sample n clear frames spanning curvature (straights AND curves), so the
    verification isn't biased toward easy straight frames. Returns (x0, steer)."""
    rows = list(csv.DictReader(open(os.path.join(C.DATASET_DIR, dataset, "manifest.csv"))))
    rows.sort(key=lambda r: abs(float(r["steer"])))          # straightest -> sharpest
    idx = [int(round(k * (len(rows) - 1) / (n - 1))) for k in range(n)]  # even spread over curvature
    base = os.path.join(C.DATASET_DIR, dataset)
    out = []
    for i in idx:
        bgr = cv2.imread(os.path.join(base, rows[i]["image"]))
        x = torch.from_numpy(student_preprocess(bgr, w, h)).unsqueeze(0).to(device)
        out.append((x, float(rows[i]["steer"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--condition", default="night", choices=list(ACDC_BOUNDS))
    ap.add_argument("--bounds", default="acdc", choices=["acdc", "carla"])
    ap.add_argument("--method", default="CROWN",
                    choices=["IBP", "CROWN", "alpha-CROWN", "SDP-CROWN"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--dataset", default="clear")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corridor = C.STEER_CORRIDOR_NORM
    base = StudentNet(args.h, args.w).to(device)
    base.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{args.student}.pth"),
                                    map_location=device))
    base.eval()

    b = BOUNDS_SETS[args.bounds][args.condition]
    frames = load_frames(args.dataset, args.w, args.h, args.frames, device)
    print(f"{args.student} | {args.condition} | bounds={args.bounds.upper()} | {args.method} "
          f"| corridor +/-{corridor:.4f} | eps_c{b['c']} eps_b{b['b']}")
    safe = total = vac = 0
    for i, (x0, steer) in enumerate(frames):
        r = verify_frame(base, x0, b, corridor, device, args.method)
        total += 1
        safe += r["safe"]
        vac += r["vacuous"]
        tag = "VACUOUS" if r["vacuous"] else ("SAFE" if r["safe"] else "UNSAFE")
        kind = "curve " if abs(steer) > 0.02 else "straight"
        print(f"  [{kind}] nominal={r['nominal']:+.4f}  bounds=[{r['lb']:+.4f},{r['ub']:+.4f}]  {tag}")
    print(f"\n== {args.condition} @ {args.bounds.upper()}: certified-safe {100*safe/total:.1f}% "
          f"({safe}/{total}) | vacuous {vac} ==")


if __name__ == "__main__":
    main()
