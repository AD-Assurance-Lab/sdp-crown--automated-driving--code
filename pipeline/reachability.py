#!/usr/bin/env python3
"""
M3 Level-2 (v2): position-resolved offline certificate.

Along the route, compare the verified weather-induced steering deviation
  delta(position, eps) = max over the eps-box of |student(perturbed) - student(clean)|
against the closed-loop stability tolerance tau (measured cliff ~0.0125, tightest at
the sharp curve). Where delta(position) > tau, the weather can push steering beyond
what the closed loop absorbs -> predicted departure. This localizes the failure
offline (no per-condition sim) and should coincide with the sim's departure point.

delta is computed with the CONCRETE oracle (grid the eps-box, student's true min/max
-- no relaxation looseness, fast). Frames are the clear eastbound lap, ordered by step
(i.e. by route position).

    python reachability.py --student student_84x28_dagger_r02 --w 84 --h 28 --tau 0.0125
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from student import StudentNet, student_preprocess
from weather_bounds import ACDC_BOUNDS

# sim departure step for the eastbound sharp curve (from closed-loop runs)
SIM_DEPART_STEP = 442


def concrete_deviation(model, x0, crange, brange, device, grid=7):
    """max over the eps-box of |steer(perturbed) - steer(clean)| (true worst-case)."""
    with torch.no_grad():
        nom = float(model(x0).item())
        dev = 0.0
        for ec in np.linspace(crange[0], crange[1], grid):
            for eb in np.linspace(brange[0], brange[1], grid):
                s = float(model(x0 * (1.0 + ec) + eb).item())
                dev = max(dev, abs(s - nom))
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--dataset", default="clear")
    ap.add_argument("--lap", default="eastbound_lap00")
    ap.add_argument("--tau", type=float, default=0.0125, help="measured stability tolerance")
    ap.add_argument("--stride", type=int, default=5, help="frame subsampling")
    ap.add_argument("--conditions", default="fog,rain,night")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentNet(args.h, args.w).to(device)
    model.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{args.student}.pth"),
                                     map_location=device))
    model.eval()

    base = os.path.join(C.DATASET_DIR, args.dataset)
    rows = list(csv.DictReader(open(os.path.join(base, "manifest.csv"))))
    rows = [r for r in rows if args.lap in r["image"]]
    rows = rows[::args.stride]
    steps = [int(r["step"]) for r in rows]
    conds = args.conditions.split(",")

    print(f"route frames: {len(rows)} (stride {args.stride}) | tau={args.tau} | sim departs @ step {SIM_DEPART_STEP}")
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in conds:
        b = ACDC_BOUNDS[cond]
        dev = []
        for r in rows:
            bgr = cv2.imread(os.path.join(base, r["image"]))
            x0 = torch.from_numpy(student_preprocess(bgr, args.w, args.h)).unsqueeze(0).to(device)
            dev.append(concrete_deviation(model, x0, b["c"], b["b"], device))
        dev = np.array(dev)
        n_exceed = int((dev > args.tau).sum())
        ax.plot(steps, dev, lw=1.2, label=f"{cond} (delta>tau at {n_exceed}/{len(dev)} frames)")
        # first exceedance step
        ex = [s for s, d in zip(steps, dev) if d > args.tau]
        print(f"  {cond:5s}: max delta={dev.max():.4f} @ step {steps[int(dev.argmax())]} | "
              f"delta>tau at {n_exceed} frames" + (f", first @ step {ex[0]}" if ex else ""))

    ax.axhline(args.tau, ls="--", c="k", label=f"tau (stability tolerance) = {args.tau}")
    ax.axvline(SIM_DEPART_STEP, ls=":", c="r", label=f"sim departure @ step {SIM_DEPART_STEP}")
    ax.set_xlabel("route position (step)"); ax.set_ylabel("verified steering deviation delta")
    ax.set_title("v2: verified weather steering deviation vs stability tolerance (eastbound)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(C.RESULTS_DIR, "reachability_v2_delta_vs_tau.png")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
