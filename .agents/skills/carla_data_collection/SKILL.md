---
name: carla_data_collection
description: Procedural guide to start Traffic Manager, collect initial clear autopilot demonstrations, and execute interactive DAgger-Lite recovery loops in CARLA.
---

# Procedural Skill: CARLA Data Collection & DAgger-Lite

This skill outlines the commands and requirements for gathering training data in CARLA.

## Critical Parameters & Environmental Variables
*   **CARLA Quality Mode**: The CARLA simulator must **always** be launched and run in **Epic mode** (`-quality-level=Epic`) for both training data collection and testing/evaluation to prevent visual/sensory representation mismatches.
*   **Longitudinal Speed**: The longitudinal speed of the vehicle must be strictly fixed at **20 mph** (PID throttle/brake control, 8.94 m/s target) across all collection and recovery runs to remove velocity as a variable.
*   **Track Selection**: We collect data strictly on the **second-to-left lane** of the Town04 highway, explicitly avoiding urban intersections and highway bridges to keep background scene variations to a minimum.
*   **Data Volume Protocol (2 x Half Laps)**: We collect exactly **2 x half laps** of data. 
    *   The collector script (`tools/carla_data_collector.py`) automatically runs Stage 1 and Stage 2 in sequence during a single execution.
    *   **Stage 1 (Eastbound half-lap)**: Spawns the vehicle right after the West intersection/curve (`x=-357.1, y=30.0, z=0.5`, Yaw `0.0`), which starts straight and loops around the East curve. Once complete, the script teleports the vehicle to the Stage 2 start point.
    *   **Stage 2 (Westbound half-lap)**: Spawns the vehicle right after the East intersection/curve (`x=-396.8, y=12.8, z=0.5`, Yaw `180.0`), which starts straight and loops around the West curve.
    *   The `--num-frames` argument is ignored for early stopping because the script terminates automatically when the Stage 2 end condition is reached, naturally yielding a total of ~1000 frames (~500 frames per half-lap). Do **not** run the script twice to get the two half-laps.

## Step 1: Initial Autopilot Collection
Collect the initial clear-weather demonstrations (this single run will automatically collect Stage 1 and Stage 2 in sequence, yielding ~1000 frames total representing 2 x half laps):
```bash
./venv_sdp/bin/python tools/carla_data_collector.py \
    --map Town04 \
    --weather clear
```

## Step 2: DAgger-Lite Interactive Recovery Loop
Run the interactive DAgger loop to gather recovery maneuvers from states visited by the steering policy (runs for 1000 steps of simulation at 5 Hz, starting from the Stage 1 spawn point):
```bash
# Policy drives, expert labels recovery command when drift exceeds threshold
./venv_sdp/bin/python tools/run_dagger_lite.py \
    --model-type CarlaSteeringNet \
    --model-path models/carla_small_clear.pth \
    --map Town04 \
    --weather clear \
    --max-steps 1000
```

## Step 3: Mixed-Weather Supplementing
To train the mixed-weather model (target 30,000+ frames), repeat the DAgger collection (1000 steps per run) inside CARLA's adverse weather conditions:
```bash
# Run DAgger in Rain
./venv_sdp/bin/python tools/run_dagger_lite.py --model-type CarlaSteeringNet --model-path models/carla_small_clear.pth --map Town04 --weather rain --max-steps 1000
# Run DAgger in Fog
./venv_sdp/bin/python tools/run_dagger_lite.py --model-type CarlaSteeringNet --model-path models/carla_small_clear.pth --map Town04 --weather fog --max-steps 1000
# Run DAgger in Night
./venv_sdp/bin/python tools/run_dagger_lite.py --model-type CarlaSteeringNet --model-path models/carla_small_clear.pth --map Town04 --weather night --max-steps 1000
```

## Step 4: Dataset Validation & User Feedback Loop
After collecting demonstrations or running DAgger-Lite interactive recovery loops:
1. **Analyze Dataset Balance**: Inspect the collected image directories and verify the dataset balance by checking the `index.csv` using:
   ```bash
   ./venv_sdp/bin/python -c "import pandas as pd; df=pd.read_csv('datasets/carla_steering_e2e/clear/index.csv'); print('Frames:', len(df)); print('Straight count:', len(df[df.steering.abs()<=0.01])); print('Left/Right turns:', len(df[df.steering<-0.01]), len(df[df.steering>0.01]))"
   ```
2. **Compile Summary**: Present the final frame count and turn-distribution statistics to the user.
3. **CARLA Server Clean-up**: Terminate the CARLA simulator to free up CPU/GPU resources and let the host machine sleep while waiting for review. If you launched CARLA in the background, kill the process. If it was already running, remind the user to shut it down or run:
   ```bash
   pkill -f CarlaUE4
   ```
4. **Wait for Feedback**: **Stop execution** and ask the user for explicit feedback on the quality of driving they observed in CARLA during the collection/takeover phase. Do **not** proceed to train any models until the user reviews the collection results and gives approval.
