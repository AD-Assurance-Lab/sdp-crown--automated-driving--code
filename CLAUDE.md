# CLAUDE.md — SDP-CROWN Automated Driving Verification

Guidance for Claude Code (and any agent) working in this repository. This is the
single source of truth for project rules; it replaces the former `AGENTS.md`.

**The working code is the `pipeline/` package.** `README.md` (reproduction guide)
and the paper supplement `working_methodology.md` (in the itsc-paper repo) are the
authoritative descriptions of what is actually implemented and validated. If this
file and the pipeline ever disagree, the pipeline + README win.

## Project objective
Quantify the effect of environmental disturbances on end-to-end AI steering models
using **SDP-CROWN**, and show the formal results *reconcile with* closed-loop
**CARLA** simulation. Weather is modeled as an affine pixel transform (contrast
ε_c, brightness ε_b) calibrated from the real-world **ACDC** dataset. The controller
must simultaneously (1) pass closed-loop validation (drive the Town04 loop within
the CTE budget) and (2) formally certify safe steering under the ACDC ε-box.

**Weather-model scope (important).** The affine model is valid for *photometric*
disturbances — **fog, night**. **Rain and snow are non-affine** (localized /
composite; snow is also unrenderable in CARLA) and are **future work**. Do not
report rain/snow as validated results (supplement §13).

## Core guideline: the performance–verifiability tradeoff
- **Do not over-optimize simulation.** No BatchNorm, Dropout, or high-capacity /
  complex architectures (ResNet etc.) in the verifiable net — they cause
  interval-bounds explosion and make verification intractable.
- **Do not over-optimize verifiability.** Don't collapse to a trivial linear
  controller for tight bounds — it must still drive the Town04 curves closed-loop.
- **Width, not resolution, is the capacity lever** for the verifiable student:
  width adds parameters at a fixed input-perturbation dimension (cheaper to verify);
  resolution enlarges the ε-box the verifier must bound. Sweep width when capacity
  is short (supplement §14).

## Networks
- **Teacher (`model.py`, PilotNet-class):** 5 conv + 4 FC, ReLU-only, ~107k ReLU,
  input 200×66. Used only as a distillation source (too large to verify).
- **Verifiable student (`student.py`, `StudentNet`):** small ReLU-only CNN
  (3 strided conv + 2 FC), **no BatchNorm/Dropout**. Input resolution and conv/FC
  **width** are constructor parameters. Clear model: 84×28, 5,152 ReLU. Mixed
  (photometric) model: 84×28 at 2× width (`channels=(16,32,32), fc=64`), 10,304 ReLU.
- **ROI crop from ground-truth segmentation:** road occupies rows [240:450], full
  width (measured over 3,390 seg frames). Crop tight to this so lanes survive
  downsampling. Inputs normalized to `[0,1]`. **No steering multiplier** —
  `steer = pred_steer`.

## Closed-loop data collection & simulation
- **Environment:** always CARLA **Town04 (Epic Mode)** (`-quality-level=Epic`).
- **Longitudinal speed:** fixed **20 mph** (8.94 m/s) via the physics-honest PI
  throttle/brake controller (`carla_env.SpeedController`) — never a velocity
  override (that corrupts the lateral dynamics the CTE measures).
- **Labels & CTE:** **pure pursuit on a fixed reference centerline**
  (`route.py`/`build_routes.py`) — *not* the autopilot PID (it oscillates and gets
  cloned). CTE is the perpendicular distance to that centerline (immune to CARLA's
  lane-snapping).
- **Collection:** `collect_data.py --weathers …` drives the oracle over the loop
  both directions under each rendered condition. Weather presets are
  **order-independent** — each sets all confounding fields (precip/deposits/
  wetness/fog) explicitly, else e.g. night-after-rain inherits rain's wet road.

## Training recipe (mirror it exactly across models)
Behavior cloning → **teacher-DAgger** → **knowledge distillation** →
**student-DAgger**. DAgger uses the pure-pursuit recovery expert (every visited
frame relabeled), aggregates, and retrains.
- **Multi-condition DAgger needs warm-start.** From-scratch retraining each round
  (fine for single-condition/clear) **diverges** on multi-weather data. Fine-tune
  from the previous round's checkpoint at reduced LR (`dagger*.py --lr 5e-4`, which
  `train.py`/`distill.py` honor via `init_from`). Supplement §14.
- **Turn balancing:** downsample straights (`|steer| ≤ 0.01`) via `balance_straight`.

## Safety criteria — derived, not assumed
- **CTE budget** = (lane 3.500 − vehicle 2.164)/2 = **0.668 m = 2.19 ft**.
- **Per-frame steering corridor** = 2·L·CTE_budget/(v²T²) = **0.050 rad = 2.88° =
  0.041 normalized** (T=1 s). This replaces the old arbitrary ±0.1 rad. Note the
  *closed-loop* stability tolerance is tighter (~0.012, a stability cliff) — the
  per-frame corridor bounds a single frame; a systematic bias compounds.

## Verification
- **Semantic perturbation layer** (`nn.Linear`, `Weight=[x₀⊙M | M]`, `Bias=x₀`) maps
  `[ε_c, ε_b]` → image; bound the steering output under IBP/CROWN/α-CROWN/SDP-CROWN.
- **auto_LiRPA `patches`-mode fix (keep):** the elementwise form
  `x' = x·(1+ε_c)+ε_b` triggers a stride-2 `as_strided` `RuntimeError`; the
  `nn.Linear` reformulation above avoids it (~2.2 GB VRAM, seconds/frame).
- **Cross-checks:** zero-ε returns nominal; SDP-CROWN bounds always contain the
  concrete grid; tightness order IBP ⊇ CROWN ⊇ α-CROWN ⊇ SDP-CROWN.
- **Reconciliation** = per-frame certificate + concrete grid + worst-case-ε in
  closed loop (`eval_student.py --affine`). ε-bounds calibrated to **ACDC**
  (real-world); CARLA presets are ~4× harsher (supplement §13, §15).

## Execution discipline
- **Never trade experimental quality for speed** — no CPU fallback, no lowered
  CARLA quality, no cut epochs. Long runs are expected.
- **Warn on long runs (>1 h):** estimate ETC and warn before starting.
- **Stop for feedback:** after collection / closed-loop / verification sweeps,
  present metrics and **stop for user feedback** before proceeding.
- **CARLA hygiene:** don't leave a server running when done; cycle it between long
  phases (a server up for many hours can stall). To kill it without the pattern
  matching this shell, use a variable: `P=Carla; pkill -9 -f "${P}UE4"`.

## Environment setup
Deps pinned in `requirements.txt` (Python 3.10, CUDA 12.1, RTX-class GPU). The
CARLA 0.9.16 client wheel and CUDA PyTorch are installed separately — see the
header comments in `requirements.txt`.

## Authoritative docs
`README.md` (reproduction) and the itsc-paper `working_methodology.md` (full method,
results, future work §17) are the source of truth. Keep them in sync with the
pipeline when you change it.
