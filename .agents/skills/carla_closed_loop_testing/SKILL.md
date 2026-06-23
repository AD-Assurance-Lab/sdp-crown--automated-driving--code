---
name: carla_closed_loop_testing
description: Procedural guide for running closed-loop simulator evaluations in CARLA and parsing Cross-Track Error (CTE) and speed telemetry.
---

# Procedural Skill: CARLA Closed-Loop Simulator Evaluation

This skill outlines how to run closed-loop evaluation simulations and parse telemetry metrics.

## Critical Parameters & Environmental Variables
*   **CARLA Quality Mode**: The CARLA simulator must **always** be launched and run in **Epic mode** (`-quality-level=Epic`) for all training and testing evaluations to ensure consistency in graphics, shadows, and lane visibility.
*   **Longitudinal Speed**: The ego vehicle's longitudinal speed must be strictly fixed at **20 mph** (PID throttle/brake control, 8.94 m/s target) across all simulation runs. This is a strict constraint to completely eliminate velocity as a variable.
*   **Spawn Point**: The vehicle must always spawn right after the intersection at the beginning of the highway loop (`--start-frame 0`, Location `x=-357.1, y=30.0, z=0.5`, Yaw `0.0`). This segment begins straight and features a sharp curve at the end.
*   **Removal of Mid-Track curve evaluation**: We have **removed the Mid-Track curve evaluation entirely**. Evaluating a full lap (960 frames at 5 Hz) starting from frame 0 does not take long and is much more fun and comprehensive. Do **not** use mid-track spawn points (e.g. frame 750).

## 1. Full Lap Closed-Loop Simulation Command
To evaluate the full lap:
```bash
./venv_sdp/bin/python tools/test_carla_model.py \
    --model-type CarlaSteeringNet \
    --model-path models/carla_small_mixed_aug.pth \
    --map Town04 \
    --weather rain \
    --num-frames 960 \
    --start-frame 0
```

## 2. Parsing Telemetry & Evaluation Outcomes
The vehicle must successfully drive at least a **full half-lap** (approx. 500 frames) of the Town04 highway to be considered a PASS.
*   **Telemetry CSV**: Telemetry is written to `results/carla_ai_model_testing/town04_*.csv`.
*   **Lane Departure Threshold**: If max Cross-Track Error (CTE) exceeds 5.74 feet (1.75 meters), it is flagged as FAILED (Crashed).
*   **Stall Threshold**: If speed drops and stays below 2.0 mph for the last 50 frames, it is flagged as FAILED (Stalled).

## 3. Validation Plot & User Feedback Loop
The closed-loop evaluation script `test_carla_model.py` automatically generates a combined steering control and Cross-Track Error (CTE) plot.
*   **Plot Location**: The path is specified by the `--save-plot` argument (e.g. `results/carla_ai_model_testing/town04_small_mixed_aug_rain.png`).
*   **Present Results**: Show the user the validation plot and key performance metrics (MAE, RMSE, Max CTE, and PASS/FAIL status).
*   **CARLA Server Clean-up**: Terminate the CARLA simulator to free up CPU/GPU resources and let the host machine sleep. If you launched CARLA in the background, kill the process. If it was already running, remind the user to shut it down or run:
    ```bash
    pkill -f CarlaUE4
    ```
*   **Wait for Feedback**: **Stop execution** and ask the user for feedback on the plot and what was observed during the simulation run in CARLA (e.g. smoothness of cornering, lane drift patterns) before continuing to verification or model tuning. Do **not** proceed with other tasks until the user has reviewed the performance and given approval.
