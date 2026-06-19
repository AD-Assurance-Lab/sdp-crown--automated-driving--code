# Driving Datasets for Autonomous Steering Verification

This directory contains the driving datasets used for adverse weather characterization, neural network steering model training, and closed-loop verification.

---

## 1. ACDC Dataset (Adverse Conditions Dataset with Correspondences)

The **ACDC** dataset is used to extract empirical contrast drop ($\epsilon_c$) and brightness bias ($\epsilon_b$) coefficients. It provides GPS-synchronized image pairs: an adverse weather image and its corresponding clear-weather reference image.

### Expected Directory Hierarchy
```text
ACDC/
├── rgb_anon/
│   ├── [fog|night|snow|rain]/
│   │   ├── train/
│   │   │   └── <sequence_folder>/ (e.g., GOPR0476/)
│   │   │       └── <sequence_frame>_rgb_anon.png
│   │   ├── val/
│   │   │   └── <sequence_folder>/
│   │   │       └── <sequence_frame>_rgb_anon.png
│   │   ├── train_ref/
│   │   │   └── <sequence_folder>/
│   │   │       └── <sequence_frame>_rgb_ref_anon.png
│   │   └── val_ref/
│   │       └── <sequence_folder>/
│   │           └── <sequence_frame>_rgb_ref_anon.png
└── gt/
    ├── [fog|night|snow|rain]/
    │   ├── train/
    │   │   └── <sequence_folder>/
    │   │       └── <sequence_frame>_gt_labelTrainIds.png
    │   └── val/
    │       └── <sequence_folder>/
    │           └── <sequence_frame>_gt_labelTrainIds.png
```

### Reference Frame Mapping
For each adverse frame (e.g., `val/GOPR0402/GOPR0402_frame_000120_rgb_anon.png`), the characterizer maps it to:
1.  **Clear Reference Frame:** `val_ref/GOPR0402/GOPR0402_frame_000120_rgb_ref_anon.png`
2.  **Semantic Mask Frame:** `gt/rain/val/GOPR0402/GOPR0402_frame_000120_gt_labelTrainIds.png`

### Spatial Road Masking (Snow & Rain)
Global atmospheric changes like Fog or Night affect the entire image uniformly. Localized road conditions (like Snow on asphalt or wet Rain reflection mirrors) only affect the road surface.
*   The ground truth masks use the **Cityscapes TrainID format**.
*   **TrainID 0 represents the Road category.**
*   Our characterization script isolates pixels where `labelTrainIds == 0` to compute standard deviation and mean only on the drivable road corridor, preventing sky or building details from corrupting the coefficients.

---

## 2. CARLA Simulation Datasets (Steering Behavioral Cloning)

Used to train the custom `CarlaSteeringNet` behavioral cloning controllers. The dataset contains front RGB camera frames and corresponding vehicle telemetry (steering, throttle, brake, speed) collected under various weather conditions.

### Directory Hierarchy
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
│       └── ...
├── fog/
│   ├── index.csv
│   └── images/
│       └── ...
└── night/
    ├── index.csv
    └── images/
        └── ...
```

### Telemetry Index Format
The `index.csv` for each weather condition includes a header and is formatted as:
```csv
frame,image_path,steering,throttle,brake,speed
```
*   `frame`: 0-indexed frame number.
*   `image_path`: Path relative to the weather folder (e.g., `images/frame_000000.png`).
*   `steering`: Floating point autopilot steering command in range `[-1.0, 1.0]` (where negative is left and positive is right).
*   `throttle`: Floating point throttle command in range `[0.0, 1.0]`.
*   `brake`: Floating point brake command in range `[0.0, 1.0]`.
*   `speed`: Vehicle speed in km/h.

### Dataset Collection Protocol
To ensure experiment reproducibility and remove trajectory variation:
1.  **Deterministic Pathing:** The CARLA autopilot route is made fully deterministic by setting Python, NumPy, and CARLA Traffic Manager seeds to `42`, and starting the vehicle at spawn point index `0` on the `Town01` map.
2.  **Ignored Intersections:** To prevent the autopilot from stopping at traffic lights or taking random turns, we set Traffic Manager settings to bypass traffic light logic (`ignore_lights_percentage(100.0)`), allowing the vehicle to continuously drive the same loops.
3.  **Target Downsampling:** During training, images are resized from raw `640x480` down to `60x80` (retaining the 4:3 aspect ratio) to fit the input size of `CarlaSteeringNet`.

---

## 3. Debugging and Reference Datasets

These datasets are stored under the `debugging/` directory and are used for offline verification testing:

### Udacity Driving Dataset
*   **Location:** `debugging/Udacity/self_driving_car_dataset_jungle/`
*   **Overview:** Contains continuous steering sequences recorded from a virtual simulator (used as the regression verification baseline).
*   **Processing:** Cropped to exclude the sky and hood (top 60px, bottom 25px) and resized to $37 \times 117$ pixels to fit the `MicroPilotNet` architecture.

### Toy Datasets (MNIST & CIFAR-10)
*   **Location:** `debugging/mnist/` and `debugging/cifar/`
*   **Overview:** Standard digit classification (MNIST) and object classification (CIFAR-10) datasets stored as `.npy` matrices. Used for initial verifier checks and pipeline sanity testing.
