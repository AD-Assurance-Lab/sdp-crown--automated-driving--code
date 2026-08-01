# CLAUDE.md — SDP-CROWN Automated Driving Verification

Guidance for Claude Code (and any agent) working in this repository. This is the
single source of truth for project rules; it replaces the former `AGENTS.md`.

## Project objective
Mathematically quantify the effect of environmental disturbances (rain, fog,
night, snow) on end-to-end AI steering models using **SDP-CROWN**, *without*
running physical simulations. Weather disturbances are modeled as linear pixel
modifications (affine contrast/brightness transforms) calibrated from the
real-world **ACDC dataset**, then validated against closed-loop **CARLA** runs.

Two properties the trained controller must satisfy simultaneously:
1. Pass closed-loop validation (drive ≥ a full half-lap of the Town04 highway
   with no lane departures).
2. Formally certify safe steering bounds under worst-case weather perturbations
   (steering output deviation ≤ ±0.1 rad).

## Core guideline: the performance–verifiability tradeoff
- **Do not over-optimize simulation.** No high-capacity layers, BatchNorm,
  Dropout, or complex architectures (e.g. ResNet). They cause interval-bounds
  explosion and make formal verification intractable.
- **Do not over-optimize verifiability.** Do not collapse to a trivial/shallow
  linear controller just for tight bounds — the model must still navigate the
  Town04 highway curves in closed loop.
- The small `CarlaSteeringNet` (120×90, or 160×120 for high-res experiments) is
  the target sweet spot.

## Network architecture constraints (`CarlaSteeringNet`)
- **Verifier-friendly only:** no Dropout, no BatchNorm. (BatchNorm fusion is
  allowed only for expert baseline models, not verifier-friendly nets.)
- **Architecture:** 4 conv layers + 3 fully-connected layers.
- **Resolution:** base 120×90 (or 160×120 for high-res). Inputs normalized to
  `[0.0, 1.0]`.
- **Cropping:** crop raw frames to remove sky and vehicle hood (top 180px,
  bottom 80px) before resizing.
- **No steering multiplier:** predicted steering maps directly to CARLA controls
  (`steer = pred_steer`), no scaling gain.

## Closed-loop data collection & simulation
- **Environment:** always CARLA **Town04 (Epic Mode)** (`-quality-level=Epic`)
  for both data collection and testing.
- **Longitudinal speed:** strictly fixed at **20 mph** (8.94 m/s target) across
  all collection, DAgger recovery, and testing runs — removes velocity as a
  variable.
- **Lane:** ego drives the **second-to-left lane**, avoiding urban intersections
  and bridges to keep background scene variation low.
- **Testing/eval:** spawn right after the intersection at the start of the
  highway loop (`--start-frame 0`, Location `x=-357.1, y=30.0, z=0.5`, Yaw
  `0.0`); evaluate the full lap. Mid-track / curve-only evaluations are removed.
- **Training collection:** exactly **2 × half laps**, collected sequentially in
  one run — Stage 1 CW around the East curve (spawn `x=-357.1, y=30.0, z=0.5,
  yaw=0.0`); Stage 2 CCW around the West curve (spawn `x=-396.8, y=12.8, z=0.5,
  yaw=180.0`).

## Training & dataset volumes
- **DAgger-Lite** (interactive rollouts with autopilot-derived recovery labels)
  is the primary strategy for learning curve navigation.
- **Turn balancing:** downsample straight frames (`|steer| ≤ 0.01`) to
  `Target Straight = 0.6 × max(N_left, N_right)`.
- **Volume thresholds:** Clear-weather model ≥ **5,000** balanced frames;
  Mixed-weather model ≥ **30,000** balanced frames (supplement clear DAgger with
  collection in CARLA Fog, Rain, and Night).

## Known technical memory: auto_LiRPA `patches`-mode stride bug
On GPU in `'patches'` mode, custom elementwise weather-perturbation layers
(`x'_ij = x_ij·(1+ε_c) + ε_b`) trigger a `RuntimeError` in `as_strided` /
`patches_to_matrix` when bounds propagate backward past a stride-2 conv.

**Fix:** reformulate the perturbation layer as a standard `nn.Linear` followed by
a reshape, with `Weight = [x0 ⊙ M | M]`, `Bias = x0` (x0 = nominal image,
M = spatial road mask). Because `nn.Linear` is native, auto_LiRPA handles it in
`'patches'` mode without stride errors (~2.2 GB VRAM, runs in seconds).

## Fallback strategies (use ONLY if DAgger-Lite stalls)
1. **Shift augmentation:** horizontal translation of clear-weather frames.
   ⚠️ Sign correction MUST be additive: `steer = steer + dx * 0.003`.
   Subtraction teaches positive feedback for errors (steering away from center).
2. **Knowledge distillation:** if `CarlaSteeringNet` can't learn a half-lap,
   train a larger `CarlaSteeringExpertNet` (with BatchNorm/Dropout) and distill
   its policy (MSE against expert steering logits) into the verifiable net.

## Execution discipline
- **Never trade experimental quality for speed** — do not switch to CPU, lower
  CARLA graphics quality, or cut training epochs just to finish faster. Long
  runs are expected.
- **Warn on long runs:** if a task (training / batch verification) will take
  > 1 hour, estimate the ETC and warn the user before starting.
- **Stop for feedback:** after data collection, closed-loop testing, or offline
  verification sweeps, run the corresponding plotting/post-processing scripts,
  present the graphs and metrics, and **stop to wait for user feedback** before
  proceeding.
- **CARLA resource hygiene:** do not leave the CARLA server running when done.
  Terminate any server you launched; if it was already running, remind the user
  to close it (`pkill -f CarlaUE4`).

## Reference baseline (read-only)
`ROADMAP.md` holds expected outcomes and experimental baselines. **Do not edit**
its expected-outcomes table or the roadmap.

## Environment setup
Dependencies are pinned in `requirements.txt` (Python 3.10, CUDA 12.1, RTX 4070).
The CARLA 0.9.16 client wheel and the CUDA build of PyTorch are installed
separately — see the header comments in `requirements.txt`.

## Procedural skills
Step-by-step procedures live in `.claude/skills/`:
- `sdp_crown_verification` — configuring disturbance bounds and running formal
  CROWN / SDP-CROWN sweeps on GPU.
- `carla_data_collection` — collecting training data in CARLA.
- `carla_closed_loop_testing` — closed-loop evaluation runs.
