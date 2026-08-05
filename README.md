# SDP-CROWN — Formal Verification meets Closed-Loop Validation for E2E Steering

This repository verifies and validates end-to-end camera→steering controllers under
adverse weather, using the **SDP-CROWN** neural-network verifier
([auto_LiRPA](auto_LiRPA/)) *together with* closed-loop **CARLA** simulation. The
central question: **can formal verification, run outside the simulator, predict
closed-loop safety outcomes?** The answer we demonstrate is *yes* — and, for one
condition, verification catches a worst-case failure that nominal simulation
declares safe.

The full methodology, results, and tables live in the paper supplement
[`working_methodology.md`](../sdp-crown--automated-driving--itsc-paper/working_methodology.md)
(itsc-paper repo). This README is the standalone reproduction guide.

---

## What we show

- **Clear-only model fails weather, and verification predicts it.** A steering model
  trained only on clear weather departs the lane under fog/rain/night; SDP-CROWN,
  using ACDC-calibrated perturbation bounds, flags those failures *without
  simulation*.
- **A weather-trained model reconciles verification with simulation.** A
  mixed/photometric model (trained on clear+fog+night) drives those conditions in
  closed loop, and per-condition:
  - **fog** — high certificate, concretely robust, drives under worst-case ε →
    verification and simulation **agree**.
  - **night** — drives *nominally*, but verification (per-frame + concrete grid +
    worst-case-ε-in-closed-loop) shows genuine fragility, and the car **actually
    departs** under worst-case ε → verification catches a risk nominal simulation
    **misses**.
- **Scope (honest).** The affine perturbation model is valid for *photometric*
  disturbances (**fog, night**). **Rain and snow are non-affine** (localized /
  composite; snow is also unrenderable in CARLA) and are **future work** — see the
  supplement §13, §17.

---

## Repository layout

```
pipeline/            the reproducible pipeline (self-contained; see below)
auto_LiRPA/          the SDP-CROWN verifier (dependency; imported by verify.py)
datasets/ACDC/       real-world clear/adverse pairs used to calibrate ε-bounds
requirements.txt     pinned deps (Python 3.10, CUDA 12.1, RTX-class GPU)
ROADMAP.md           expected-outcomes baseline (read-only reference)
```
Regenerable artifacts — frame data (`pipeline/data/`), model weights
(`pipeline/checkpoints/`), and results (`pipeline/results/`) — are gitignored.
Reference routes rebuild from `pipeline/build_routes.py`.

### The `pipeline/` package
| File | Role |
|------|------|
| `config.py` | all constants (speed, dt, camera, **derived** safety budgets) |
| `carla_env.py` | CARLA connect/sync/spawn, physics-honest PI speed control, weather presets |
| `route.py`, `build_routes.py` | fixed reference centerline + signed-CTE + pure-pursuit |
| `collect_data.py` | oracle-driven data collection (`--weathers`) |
| `model.py` / `expert.py` | PilotNet teacher; `student.py` verifiable StudentNet |
| `train.py` | behavior cloning / aggregated retrain (`--weathers`, warm-start `init_from`) |
| `dagger.py`, `dagger_student.py` | teacher- and student-DAgger (`--weathers`, `--lr` warm-start) |
| `distill.py` | knowledge distillation teacher→student (`--channels/--fc` width, `--weathers`) |
| `eval_student.py` | closed-loop test (rendered weather, or `--affine` ε-in-loop) |
| `verify.py` | SDP-CROWN per-frame certificate (semantic perturbation layer) |
| `reachability.py`, `rollout.py` | offline δ(position,ε) vs τ; verifier-in-the-loop rollout |
| `weather_bounds.py` | ACDC + CARLA ε-bounds (shared by verifier and closed-loop test) |

---

## Method in brief

- **Task.** Town04 highway loop (~3.04 km, both directions), Tesla Model 3, fixed
  **20 mph**, 5 Hz. Only lateral (steering) control is learned.
- **Labels.** Pure pursuit on a **fixed reference centerline** — not the CARLA
  autopilot PID (which oscillates and gets cloned). CTE is measured to that
  centerline (immune to lane-snapping).
- **Safety budgets are *derived*, not assumed.** CTE budget
  `(3.500−2.164)/2 = 0.668 m = 2.19 ft`; per-frame steering corridor
  `Δδ = 2·L·CTE_budget/(v²T²) = 0.050 rad = 2.88° = 0.041` normalized (T=1 s).
- **Models.** PilotNet teacher (~107k ReLU) → distilled **verifier-friendly**
  StudentNet (ReLU-only, no BatchNorm/Dropout), sized for SDP-CROWN. The mixed
  model uses a 2×-width student (10,304 ReLU).
- **Weather = affine ε.** A semantic perturbation layer (`nn.Linear`,
  `Weight=[x₀⊙M | M]`, `Bias=x₀`) maps `[ε_c, ε_b]` (contrast, brightness) to the
  image; bounds propagate under IBP/CROWN/α-CROWN/**SDP-CROWN**. ε-bounds are
  calibrated from the real-world **ACDC** dataset; CARLA's default presets are
  deliberately noted as *harsher* than real weather (supplement §13).

---

## Setup

- CARLA **0.9.16** at Town04, `-quality-level=Epic` (launch on its RPC port 2000).
- Python 3.10, CUDA 12.1; install `requirements.txt` (the CARLA client wheel and
  CUDA PyTorch are installed separately — see the file header).
- Verify the environment: `python -c "import carla, torch; print(torch.cuda.is_available())"`.

## Reproduce

Run from the repo root with CARLA up on Town04. Close CARLA during pure-training
steps (BC/distill) to free the GPU; keep it up for the interleaved DAgger loops.

```bash
# 0. Reference routes (once)
python pipeline/build_routes.py

# 1. Mixed-weather data (oracle drives; ~13.5k frames)
python pipeline/collect_data.py --dataset mixed --weathers clear,fog,rain,night --laps 1 --direction both

# 2. Photometric teacher: BC (clear/fog/night) then warm-start DAgger -> _r04
python pipeline/train.py  --dataset mixed --weathers clear,fog,night --out steering_bc_photometric
python pipeline/dagger.py --base mixed --init steering_bc_photometric --weathers clear,fog,night \
       --dagger-dir dagger_photometric --out-prefix steering_dagger_photometric --rounds 6 --lr 5e-4

# 3. Verifiable student: distill at 2x width, then warm-start student-DAgger -> _r04 (PASS)
python pipeline/distill.py --in-w 84 --in-h 28 --out student_photometric_w20_84x28 \
       --teacher steering_dagger_photometric_r04 --base mixed --dagger-dirs dagger_photometric \
       --weathers clear,fog,night --channels 16,32,32 --fc 64
python pipeline/dagger_student.py --student student_photometric_w20_84x28 --w 84 --h 28 \
       --channels 16,32,32 --fc 64 --weathers clear,fog,night --dagger-dir dagger_student_w20 \
       --teacher steering_dagger_photometric_r04 --base mixed \
       --distill-dirs dagger_photometric,dagger_student_w20 --rounds 5 --lr 5e-4

# 4. Closed-loop (nominal rendered weather)
python pipeline/eval_student.py --student student_photometric_w20_84x28_dagger_r04 \
       --w 84 --h 28 --channels 16,32,32 --fc 64 --weather night --direction both

# 5. Formal verification (SDP-CROWN, ACDC eps)
python pipeline/verify.py --student student_photometric_w20_84x28_dagger_r04 \
       --w 84 --h 28 --channels 16,32,32 --fc 64 --condition night --bounds acdc \
       --method SDP-CROWN --corridor 0.0411

# 6. Reconciliation: worst-case affine eps in closed loop (clear render + injected eps)
python pipeline/eval_student.py --student student_photometric_w20_84x28_dagger_r04 \
       --w 84 --h 28 --channels 16,32,32 --fc 64 --affine acdc --acond night --direction both
```

Expected (2× student, supplement §15): **fog** certifies ~95% and drives under
worst-case ε; **night** drives nominally but certifies ~50%, is concretely
non-robust (34/212 frames), and **departs** under worst-case ε in closed loop.

The clear-only model and its M1–M3 story reproduce with `--dataset clear` and the
`--weather {fog,rain,night}` closed-loop / `--bounds {acdc,carla}` verification
paths (supplement §§4–11).

---

## Notes for reproduction

- **Weather presets are order-independent** — each preset sets all confounding
  fields explicitly, so collecting "night after rain" does not inherit rain's wet
  road (supplement §12).
- **Multi-condition DAgger needs warm-start** (`--lr 5e-4`, fine-tune from the prior
  round). From-scratch retraining diverges on multi-weather data (supplement §14).
- **Width, not resolution, is the capacity lever** for the verifiable net — same
  perturbation dimension, cheaper certification (supplement §14).
- **CARLA hygiene.** Don't leave a server running when done; a single server that
  runs for many hours can stall — cycle it between long phases.

## Citation

Accompanies the IEEE ITSC paper on certifying steering safety margins under weather.
See the itsc-paper repository for the manuscript and full methodology.
