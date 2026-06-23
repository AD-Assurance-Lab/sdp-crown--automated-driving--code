# Driving Datasets for Autonomous Steering Verification

This directory contains the driving datasets used for adverse weather characterization, neural network steering model training, and closed-loop verification.

---

## 1. CARLA Simulation Datasets (Steering Behavioral Cloning)

Used to train the custom `CarlaSteeringNet` behavioral cloning controllers. The dataset contains front RGB camera frames and corresponding vehicle telemetry (steering, throttle, brake, speed) collected under various weather conditions on the Town04 highway.

### Dataset Directory Structure
```text
carla_steering_e2e/
├── clear/
│   ├── index.csv
│   └── images/
│       ├── frame_000000.png
│       └── ...
├── rain/
│   ├── index.csv
│   └── images/
├── fog/
│   ├── index.csv
│   └── images/
└── night/
    ├── index.csv
    └── images/
```

### Telemetry Index Format (`index.csv`)
```csv
frame,image_path,steering,throttle,brake,speed_mph,x,y
```
*   `frame`: 0-indexed frame number.
*   `image_path`: Path relative to the weather folder.
*   `steering`: Autopilot steering command in CARLA range `[-1.0, 1.0]` (negative is left, positive is right).
*   `speed_mph`: Vehicle speed in miles per hour.
*   `x`, `y`: Global coordinates of the vehicle (used for spawn alignment and cross-track error tracking).

### Dataset Collection Protocol
To ensure high closed-loop success rates and low scene entropy:
1.  **Map & Environment:** All data is collected in **CARLA Town04** in **Epic** graphics quality mode.
2.  **Specific Lane Trajectory:** Driving is restricted to the **second-to-left lane** of the multi-lane highway.
3.  **Low Entropy Half-Laps:** The dataset consists of **2 x half laps** on the highway. We explicitly avoid the highway bridges and the urban intersections to minimize background scene changes, allowing a lightweight model to learn stable lane-keeping.
4.  **No Steering Multiplier:** Steering angles must remain unscaled (raw prediction directly mapping to CARLA steering control).

### DAgger-Lite & Data Balancing
1.  **DAgger-Lite Loop:** Standard behavioral cloning is supplemented with recovery frames collected by running the current model policy in closed loop, querying the CARLA expert autopilot for corrections when the policy drifts, and aggregating those recovery frames.
2.  **Turn Balancing:** To prevent straight-line dataset bias (the model learning to steer straight when weather artifacts appear), straight frames ($|\text{steer}| \le 0.01$) are downsampled to a target count of:
    $$\text{Target Straight} = 0.6 \times \max(N_{left}, N_{right})$$
    This ensures left turns, right turns, and straightaways are balanced.
3.  **Required Volumes:**
    *   **Clear-Weather Training:** Requires **5,000+ frames** of balanced DAgger data.
    *   **Mixed-Weather Training:** Requires **30,000+ frames** in total. This is achieved by running the DAgger loop under Clear, Rain, Fog, and Night conditions, and aggregating them.

---

## 2. ACDC Dataset (Adverse Conditions Dataset with Correspondences)

The **ACDC** dataset is used to extract empirical contrast drop ($\epsilon_c$) and brightness bias ($\epsilon_b$) bounds under real-world weather conditions.

*   **Fog & Night:** Global atmospheric perturbations affecting the entire image uniformly.
*   **Rain & Snow:** Localized perturbations affecting only the drivable road asphalt due to reflection/accumulation.
*   **Spatial Masking:** cityscapes semantic segmentation masks are used to isolate pixels belonging to the **Road category (TrainID 0)**. Standard deviation and mean calculations are restricted to this road mask for Rain and Snow.

---

## 3. Image Preprocessing and Cropping
Before entering the neural network, all raw `640x480` camera frames undergo the following preprocessing pipeline:
1.  **Cropping:** The top 180 pixels (sky, buildings) and bottom 80 pixels (vehicle hood) are cropped to yield a $220 \times 640$ road-only view.
2.  **Resizing:** The cropped image is resized to a verifier-friendly resolution of **$120 \times 90$** (our baseline resolution) or **$160 \times 120$** (for high-detail tests).
3.  **Normalization:** Normalized to `[0.0, 1.0]` floating-point values.
