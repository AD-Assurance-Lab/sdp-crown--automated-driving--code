# SDP-CROWN: Neural Network Verification for Automated Driving

This repository extends the **SDP-CROWN** (Semidefinite Programming based CROWN) neural network verifier to safety-critical autonomous driving regression tasks. Specifically, it enables physical perturbation verification for end-to-end steering controllers (like NVIDIA's PilotNet) under adverse weather conditions (fog, night, snow, rain).

The physical perturbations are modeled as parameterized **Semantic Perturbation Layers** (controlling contrast scaling $\epsilon_c$ and brightness bias $\epsilon_b$), calibrated directly from real-world GPS-synchronized clear vs. adverse weather driving scenes in the **ACDC dataset**.

---

## 1. Quickstart & Walkthrough

Here is a step-by-step guide to running the automated driving verification pipeline.

### Step 1: Activate the Python Environment
The pipeline relies on PyTorch, auto_LiRPA, OpenCV, and Pandas. You can run all scripts using the pre-configured virtual environment:
```bash
# From the repository root (sdp-crown-automated-driving)
./venv_sdp/bin/python <script_name>.py [arguments]
```

**VS Code Users:** You can also run these scripts by clicking the **"Play"** button in the top right corner of the editor. Each script contains a **CONTROL PANEL** block at the top of the file where you can modify default parameters (like weather condition, number of frames, or device) without using the command line.

### Step 2: Characterize Adverse Weather (Generate $\epsilon$ Bounds)
Run the standalone calibration script to calculate physical contrast drop and brightness bias bounds from the **ACDC dataset**.

You can run this on a single drive split sequence to test a specific image grouping:
```bash
./venv_sdp/bin/python tools/extract_physics_bounds.py \
    --condition rain \
    --sequence GOPR0402 \
    --max_images 5
```
*   **What to expect:** The script will pair the rain images from `GOPR0402` with clear-weather reference frames, isolate road pixels using ground truth semantic masks, and output the derived bounds:
    ```text
    ==================================================
    ========= RAIN CHARACTERIZATION RESULTS ==========
    Total processed image pairs: 5
    Contrast Drop (eps_c) range:    [0.3623, 1.4435]
    Brightness Bias (eps_b) range:  [-0.2614, 0.0369]
    ==================================================
    ```
*   **Output:** The recommended mathematically valid intervals (containing `0.0` as the nominal baseline) are saved to `results/physics_bounds.json`.

### Step 3: Run CROWN Steering Verification
Verify the pre-trained `MicroPilotNet` steering network on the Udacity dataset using the derived bounds.

*Because dense matrix bounds tracking requires significant memory, we recommend using the `--device cpu` flag (or setting it in the script's CONTROL PANEL) to run verification on host memory and avoid GPU Out-Of-Memory (OOM) errors.*

#### Test A: Using Calibration Bounds (Sequence specific)
```bash
./venv_sdp/bin/python verify_steering.py \
    --weather rain \
    --bounds_file results/physics_bounds.json \
    --num_frames 5 \
    --device cpu
```

#### Test B: Using Paper-Calibrated Bounds (Recommended)
Verify safety under the exact bounds reported in the study (which use tighter margins for global stability):
```bash
./venv_sdp/bin/python verify_steering.py \
    --weather rain \
    --eps_c_min -0.0279 \
    --eps_c_max 0.0 \
    --eps_b_min 0.0 \
    --eps_b_max 0.1003 \
    --num_frames 5 \
    --device cpu
```
*   **What to expect:** The verifier computes worst-case steering output bounds ($\theta_{min}, \theta_{max}$) for each frame under the weather disturbance, and checks if they stay within a safety corridor of $\pm 0.1$ radians (approx. $2.5^\circ$) around the nominal clear steering path:
    ```text
    Verifying 5 frames of sequence...
    =======================================================
    ======== Starting RAIN AV Steering Verification =======
    Safety Corridor: +/- 0.1 rad deviation from nominal path
    =======================================================
    Frame 00: Nominal: -0.0100 | Bounds: [-0.0147, -0.0083] | Corridor: [-0.1100, +0.0900] | SAFE
    Frame 01: Nominal: -0.0048 | Bounds: [-0.0099, -0.0013] | Corridor: [-0.1048, +0.0952] | SAFE
    ...
    =======================================================
    ========= Final RAIN Certified Safety: 100.0% =========
    =======================================================
    ```
*   **Output:** Saves verification results to `results/verification_results.json`.

### Step 4: Plot and Visualize Results
Generate a professional graph plotting nominal steering, the safety corridor, CROWN worst-case steering bounds, and frame status:
```bash
./venv_sdp/bin/python tools/plot_verification.py
```
*   **Output:** Creates a high-resolution chart at `results/verification_plot.png`.

---

## 2. Certified Safety Results (50-Frame Stress Test)

When verified over a 50-frame continuous sequence of the Udacity Lake Track with a safety corridor of $\pm 0.1$ radians ($\approx 5.7^\circ$), the pre-trained `MicroPilotNet` controller yields the following certified safety rates under ACDC-calibrated physical bounds:

| Weather Condition | Parameter Limits | Spatial Masking | Certified Safety (%) |
| :--- | :--- | :--- | :--- |
| **Rain (Wet Road)** | $\epsilon_c \in [-0.0279, 0.0]$, $\epsilon_b \in [0.0, 0.1003]$ | Bottom 50% (Road) | **100.0%** |
| **Snow (Road Wash)** | $\epsilon_c \in [-0.2037, 0.0]$, $\epsilon_b \in [0.0, 0.2297]$ | Bottom 50% (Road) | **84.0%** |
| **Fog (Global)** | $\epsilon_c \in [-0.1625, 0.0]$, $\epsilon_b \in [0.0, 0.1237]$ | None | **72.0%** |
| **Night (Global)** | $\epsilon_c \in [-0.6275, 0.0]$, $\epsilon_b \in [0.0, 0.0472]$ | None | **0.0%** |

### VRAM & The Memory Wall
Because semantic perturbations globally couple all pixels in a convolutional network, the solver must bypass CROWN's sparse `patch-mode` and run in **dense matrix-mode** (`conv_mode: matrix`). Under standard PilotNet inputs, this requires a $39k \times 39k$ matrix, exceeding 12GB VRAM. We resolve this by downsampling the input to $37 \times 117$ pixels (`MicroPilotNet`), dropping VRAM usage to $\approx 600$MB.

---

## 3. Directory Structure

*   `tools/`
    *   `extract_physics_bounds.py`: Stands alone to run ACDC contrast and brightness statistical characterizations.
    *   `plot_verification.py`: Stands alone to plot safety corridor vs. CROWN boundaries.
*   `auto_LiRPA/`: Core Linear Bound Propagation library.
*   `datasets/`: Directory containing ACDC and Udacity driving datasets.
*   `models_weights/`: Pretrained model weights (e.g., `pilotnet_udacity.pth`).
*   `models.py`: Network definitions (including classification baselines and `MicroPilotNet`).
*   `semantic_layers.py`: Implements custom PyTorch weather layers and network wrapper module.
*   `udacity_dataset.py`: Udacity image loader, crops the top (sky) and bottom (hood) before resizing.
*   `verify_steering.py`: Runs regression verifier over continuous driving frames.
*   `ROADMAP.md`: [Roadmap & Feature Checklist](file:///home/za/ad_assurance/sdp-crown-automated-driving/ROADMAP.md) for future verification features (glare, lens blinding, depth-fog).

---


## 3. Core Baseline Verification (MNIST & CIFAR)

To run baseline classification verification (L2-norm radius perturbations) to reproduce original academic benchmarks:

```bash
# Reproducing results in the SDP-CROWN paper (re-scales and runs SDP-CROWN)
./run_sdp_crown.sh
```
Or execute individual models:
```bash
./venv_sdp/bin/python sdp_crown.py --model mnist_mlp --radius 1.0
./venv_sdp/bin/python sdp_crown.py --model cifar10_cnn_a --radius 24/255
```

---

## 4. Code Implementation Examples

Below are the code snippets illustrating the verifier-friendly `MicroPilotNet` architecture and the continuous verification/garbage collection loop:

### A. MicroPilotNet Architecture
```python
class MicroPilotNet(nn.Module):
    def __init__(self):
        super(MicroPilotNet, self).__init__()
        # Input: 3 channels (RGB), 37 height, 117 width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        # Flattened size: 64 channels * 2 height * 12 width = 1536
        self.linear_layers = nn.Sequential(
            nn.Linear(1536, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1) # Single regression output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        return self.linear_layers(x)
```

### B. CROWN/SDP-CROWN Execution and Memory Management Loop
```python
# Initialize BoundedModule (use 'conv_mode': 'matrix' for semantic perturbations)
lirpa_model = BoundedModule(
    wrapped_model, bounded_eps, device=device, verbose=0, bound_opts={'conv_mode': 'matrix'}
)

# Compute regression bounds using CROWN
crown_lb, crown_ub = lirpa_model.compute_bounds(
    x=(bounded_eps,), method='CROWN', bound_lower=True, bound_upper=True
)

# Evaluate safety corridor (+/- 0.1 rad deviation)
is_safe = (crown_lb.item() >= nominal_steering - 0.1) and (crown_ub.item() <= nominal_steering + 0.1)

# Clear computational graph and clear CUDA VRAM cache
del crown_lb, crown_ub, lirpa_model, wrapped_model, bounded_eps
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

---

## References

*   **SDP-CROWN Paper:** [ICML 2025 PDF](https://arxiv.org/pdf/2506.06665)
*   **ACDC Dataset:** [ETH Zurich](https://acdc.vision.ee.ethz.ch/)
*   **auto_LiRPA Engine:** [GitHub](https://github.com/KaidiXu/auto_LiRPA)