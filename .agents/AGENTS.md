# AD-Assurance Project Workspace Rules & Guidelines (AGENTS.md)

This document contains rules, guidelines, and technical memory for Antigravity agents working on this codebase. It establishes project alignment, active model configurations, known bug workarounds, and iteration protocols.

---

## 🎯 Active Project Objective: Scientific Verification Proof-of-Concept
The goal of this work is to mathematically quantify the effect of environmental disturbances (rain, fog, night, snow) on AI driving models using **SDP-CROWN**. By modeling these disturbances as linear pixel modifications (affine transformations) calibrated from the real-world **ACDC dataset**, we evaluate robustness *without running physical simulations*. We then validate these predictions against closed-loop simulation runs in **CARLA**.

---

## ⚖️ Core Guideline: The Performance-Verifiability Tradeoff
A primary goal of this research is balancing **closed-loop simulation performance** against **formal verifiability**. 
*   **Avoid Over-optimizing Simulation**: Do not introduce high-capacity layers, Batch Normalization, Dropout, or complex architectures (like ResNet) to improve simulator driving. These make formal verification computationally intractable or impossible due to interval bounds explosion.
*   **Avoid Over-optimizing Verifiability**: Do not make the model too shallow or simple (e.g. standard linear controllers) just to get tight verification bounds. The model must retain enough capacity to successfully navigate the Town04 highway curves in closed loop.
*   The small `CarlaSteeringNet` model ($120 \times 90$ or $160 \times 120$ inputs) is our target sweet spot for this tradeoff.

---

## 🛑 Critical System Rules & Guidelines

### 1. Network Architecture Constraints (`CarlaSteeringNet`)
*   **Verifier-Friendly Only**: Steering models must strictly use the lightweight `CarlaSteeringNet` architecture (no Dropout, no Batch Normalization). 
*   **Resolution**: Base image size must be **$120 \times 90$** (or **$160 \times 120$** for high-resolution experiments).
*   **Cropping**: Raw frames must be cropped to remove the sky and vehicle hood (cropping top 180px and bottom 80px before resizing).
*   **No Steering Multiplier**: Predicted steering values must map directly to CARLA steering controls (`steer = pred_steer`) without any scaling multipliers.

### 2. Closed-Loop Data Collection & Simulation
*   **Environment**: Both training data collection and testing/evaluation must always be run in **CARLA Town04 (Epic Mode)** (`-quality-level=Epic`).
*   **Longitudinal Speed**: The ego vehicle's longitudinal speed must be strictly fixed at **20 mph** always (8.94 m/s target using Traffic Manager or PID throttle/brake controllers) across all data collection, DAgger recovery, and closed-loop testing runs to remove velocity as a variable.
*   **Spawn Points & Trajectory**: 
    *   During testing/evaluation, the ego vehicle must always spawn right after the intersection at the beginning of the highway loop (`--start-frame 0`, Location `x=-357.1, y=30.0, z=0.5`, Yaw `0.0`), which starts straight and has a sharp curve at the end.
    *   Mid-track spawning and curve evaluations are **removed entirely**; we evaluate the full lap starting from frame 0 since full-lap runs are fast and comprehensive.
    *   During training data collection, we collect exactly **2 x half laps** of data (Stage 1 CW loop around East curve, spawning at `x=-357.1, y=30.0, z=0.5, yaw=0.0`; and Stage 2 CCW loop around West curve, spawning at `x=-396.8, y=12.8, z=0.5, yaw=180.0`), which are collected sequentially in a single run.
    *   Ego vehicle must drive in the **second-to-left lane** of the highway, explicitly avoiding urban intersections and highway bridges to keep background scene variations to a minimum.
*   **Plotting & User Feedback Loops**: After executing data collection, closed-loop testing, or offline verification sweeps, you must run the corresponding post-processing/plotting scripts, present the resulting graphs and metrics to the user, and **stop execution to wait for user feedback** on the plots and simulation quality before proceeding to subsequent tasks.

### 3. Training & Dataset Volumes
*   **DAgger-Lite**: Interactive closed-loop rollouts with autopilot-derived recovery labeling must be used to collect training recovery frames. This is our primary strategy for learning curve navigation.
*   **Turn Balancing**: Straight frames ($|\text{steer}| \le 0.01$) must be downsampled to prevent straight-line bias under weather. Use the formula:
    $$\text{Target Straight} = 0.6 \times \max(N_{left}, N_{right})$$
*   **Volume Thresholds**:
    *   **Clear-Weather Model**: Requires **5,000+** balanced training frames.
    *   **Mixed-Weather Model**: Requires **30,000+** balanced training frames. (Supplement by repeating clear DAgger collection inside CARLA Fog, Rain, and Night).

### 4. Reference Baseline (Read-Only)
*   For expected outcomes and experimental baseline targets, reference the read-only [ROADMAP.md](file:///home/za/ad-assurance--workspace/sdp-crown--automated-driving--code/ROADMAP.md) file. Do **NOT** edit the expected outcomes table or the roadmap file.

### 5. Execution Patience, Time Estimates, & Resource Hygiene
*   **Patience with Long Runs**: Simulations, model training, and verification sweeps naturally take a long time to execute. This is expected and normal for this codebase. You must **never** compromise experimental quality or rules (e.g. by switching to CPU, running CARLA in low graphics quality, or reducing training epochs) just to make runs faster.
*   **Estimated Completion Warning**: If a task (e.g. training or batch verification) is estimated to take **more than 1 hour** to complete, calculate the estimated time to completion (ETC) and explicitly **warn the user** before starting.
*   **CARLA Simulator Resource Hygiene**: Do **not** leave the CARLA simulator server running in the background when your tasks are finished. Leaving CARLA running wastes GPU/CPU power and prevents the host computer from sleeping. 
    *   If you launched CARLA in the background, terminate the server process when complete.
    *   If CARLA was already running on the host, remind the user to close it or run a cleanup command (e.g. `pkill -f CarlaUE4`) to free up resources.

---

## 🛠️ Known Technical Memory: Auto_LiRPA Patches Stride Bug
During verification on GPU using auto_LiRPA's memory-efficient `'patches'` mode, custom weather perturbation layers (representing $x_{ij}' = x_{ij} \cdot (1 + \epsilon_c) + \epsilon_b$ via custom elementwise scaling) trigger a `RuntimeError` in `as_strided` within `patches_to_matrix` when bounds are propagated backward past a stride-2 convolution.

### The Solution:
We reformulate the weather perturbation layer as a standard **Linear Layer** (`nn.Linear`) followed by a reshape:
$$\text{Weight} = \begin{bmatrix} x_0 \odot M & M \end{bmatrix}, \quad \text{Bias} = x_0$$
where $x_0$ is the nominal image and $M$ is the spatial road mask. Because `nn.Linear` is standard, auto_LiRPA handles it natively without stride errors, enabling GPU-accelerated verification in `'patches'` mode (using ~2.2GB VRAM and running in seconds).

---

## 🔄 Fallback & Alternative Strategies (Use ONLY if DAgger-Lite hits a brick wall)

### 1. Shift Data Augmentation (Tuning)
If DAgger-Lite alone is not enough to prevent lane drift, apply horizontal translation augmentation to clear-weather training frames. 
*   **Sign Correction Warning**: The steering correction target must use addition: `steer = steer + dx * 0.003`. Subtraction teaches the model positive feedback for errors (steering away from the lane center).

### 2. Knowledge Distillation (KD)
If `CarlaSteeringNet` fails a full half-lap of closed-loop simulation, train a larger expert network (`CarlaSteeringExpertNet` with BatchNorm/Dropout) and use **Knowledge Distillation** to transfer the expert steering policy into the smaller, verifiable model.
