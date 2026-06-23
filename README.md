# SDP-CROWN: Autonomous Steering Robustness Verification & Validation

This repository extends the **SDP-CROWN** (Semidefinite Programming based CROWN) formal neural network verifier to safety-critical autonomous driving regression tasks. Specifically, it enables physical perturbation verification and closed-loop validation for end-to-end steering controllers (`CarlaSteeringNet`) under adverse weather conditions (rain, fog, night, snow).

Semantic weather perturbations are modeled as parameterized **Semantic Perturbation Layers** (controlling contrast scaling $\epsilon_c$ and brightness bias $\epsilon_b$), calibrated directly from real-world GPS-synchronized driving scenes in the **ACDC dataset**.

---

## 🎯 Active Project Objective: Safe Closed-Loop & Formal Verification
Our goal is to train a single, verifier-friendly end-to-end steering controller that simultaneously:
1.  **Passes closed-loop simulator validation** (driving at least a full half-lap on the Town04 highway without lane departures).
2.  **Formally certifies safe steering bounds** under worst-case weather perturbations using **SDP-CROWN** (steering output deviation $\le \pm 0.1$ radians).

---

## 🛑 Coding Constraints & Controller Specifications
To balance verifiability and closed-loop control stability, all verifier-friendly controllers must adhere to these specifications:
*   **Architecture (`CarlaSteeringNet`)**: Features 4 Convolutional layers followed by 3 Fully-Connected layers.
*   **No Dropout, No BatchNorm**: These layers introduce interval bounds explosion or verification stochasticity. Fusing BatchNorm is allowed for expert baseline models, but verifier-friendly networks must omit them natively.
*   **Resolution & Input Prep**: Inputs are $120 \times 90$ pixels (or $160 \times 120$ for high-detail tests), cropped to remove the sky and hood, and normalized to `[0.0, 1.0]`.
*   **No Steering Multiplier**: Steering angle predictions map directly to CARLA control inputs without steering gain multipliers.

---

## 🧪 Data Collection & DAgger-Lite Protocol
To solve the covariate shift problem (compounding errors causing off-road drift) while keeping scene entropy low:
1.  **Map & Lane Trajectory**: Both data collection and closed-loop testing must always be run in **CARLA Town04 (Epic Mode)**. The ego vehicle drives in the **second-to-left lane** of the highway.
2.  **Longitudinal Speed**: The ego vehicle's longitudinal speed must be strictly fixed at **20 mph** always (8.94 m/s target using Traffic Manager or PID throttle/brake control) across all data collection, DAgger recovery, and closed-loop testing to remove velocity as a variable.
3.  **Entropy Reduction (2 x Half Laps)**: We collect exactly **2 x half laps** of data (Stage 1 CW loop around East curve, spawning at `x=-357.1, y=30.0, z=0.5, yaw=0.0`; and Stage 2 CCW loop around West curve, spawning at `x=-396.8, y=12.8, z=0.5, yaw=180.0`), collected sequentially in a single run. Urban intersections and bridges are excluded to keep background scene complexity low.
4.  **DAgger-Lite Loop**: Interactive rollouts are run where the policy drives, a background autopilot calculates recovery steering corrections, and recovery frames are aggregated back into the training dataset.
5.  **Turn Balancing**: Autopilot straight-driving frames ($|\text{steer}| \le 0.01$) are downsampled during dataset collection to:
    $$\text{Target Straight} = 0.6 \times \max(N_{left}, N_{right})$$
6.  **Required Frame Volumes**:
    *   **Clear-Weather Model**: 5,000+ balanced training frames.
    *   **Mixed-Weather Model**: 30,000+ balanced training frames. (Collect clear DAgger-Lite data, then supplement by repeating collection in CARLA Fog, Rain, and Night).

---

## 🛠️ The Verification Engine: Patches-Mode SDP-CROWN
Custom elementwise weather layers trigger stride-2 `RuntimeError`s in auto_LiRPA's memory-efficient `'patches'` mode.
*   **The Solution**: We reformulate the weather perturbation layer as a standard **Linear Layer** (`nn.Linear`) mapping the 2D perturbation parameters $[\epsilon_c, \epsilon_b]$ to the flattened $14,400$ image pixels:
    $$\text{Weight} = \begin{bmatrix} x_0 \odot M & M \end{bmatrix}, \quad \text{Bias} = x_0$$
    where $x_0$ is the nominal image and $M$ is the spatial road mask.
*   **The Result**: Because `nn.Linear` is standard, auto_LiRPA propagates bounds backward past convolutions in `'patches'` mode natively. This bypasses the $14400 \times 14400$ dense matrix flattening, dropping VRAM usage from **>12GB to ~2.2GB** and allowing GPU-accelerated SDP-CROWN sweeps to run in seconds.

---

## ⚙️ Alternative Strategy: Knowledge Distillation (KD)
If a verifier-friendly network struggles to learn stable closed-loop steering on the Town04 loop:
1.  Train a high-capacity "Expert" model (`CarlaSteeringExpertNet` with BatchNorm and Dropout) until it achieves stable closed-loop driving.
2.  Use **Knowledge Distillation** to transfer the expert's control logic into the lightweight `CarlaSteeringNet` by forcing it to minimize MSE loss against the expert's steering logit predictions.
3.  Verify the distilled `CarlaSteeringNet` natively.

---

## 💻 Running Commands Reference

### 1. Model Training with Weather Augmentation:
```bash
# Trains on clear dataset, dynamically applying ACDC weather scaling during training
./venv_sdp/bin/python tools/train_carla_model.py \
    --mode clear \
    --weather-aug \
    --model-type CarlaSteeringNet \
    --epochs 30
```

### 2. Full Lap Closed-Loop Simulator Evaluation:
```bash
# Spawns vehicle at frame 0 (right after intersection) and runs a full lap (960 frames)
./venv_sdp/bin/python tools/test_carla_model.py \
    --model-type CarlaSteeringNet \
    --model-path models/carla_small_mixed_aug.pth \
    --map Town04 \
    --weather rain \
    --num-frames 960 \
    --start-frame 0
```

### 3. SDP-CROWN Formal Verification Sweep:
```bash
# Verifies safety bounds on a sequence of 50 continuous frames
./venv_sdp/bin/python verify_steering.py \
    --model_type CarlaSteeringNet \
    --weights_path models/carla_small_mixed_aug.pth \
    --weather rain \
    --method SDP-CROWN \
    --device cuda \
    --num_frames 50 \
    --iterations 20
```