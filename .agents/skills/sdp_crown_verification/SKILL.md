---
name: sdp_crown_verification
description: Procedural guide for setting contrast/brightness disturbance bounds and executing offline formal sweeps via CROWN and SDP-CROWN on GPU.
---

# Procedural Skill: SDP-CROWN Formal Bounds Verification

This skill outlines how to configure weather disturbance bounds and run formal sweeps.

## CPU vs. GPU Execution Guidelines & Troubleshooting
To keep agent bound computations aligned and prevent performance blockages:
*   **Mandatory GPU (CUDA) Execution** (`--device cuda`): Verification sweeps (CROWN, SDP-CROWN) must strictly be run on the GPU. Because the weather perturbation layer is formulated as an `nn.Linear` layer and verification runs in `'patches'` mode (`conv_mode: 'patches'`), active VRAM consumption is constrained to **~2.2GB**. Running on GPU is highly recommended for sweeps and optimization iterations because it completes in seconds (~2.5s per frame, or ~2 minutes for a full 50-frame sweep).
*   **Strict Avoidance of CPU Execution** (`--device cpu`): Do **not** run batch sweeps on CPU. Computing dense bounds propagation on CPU takes **~1 minute per frame** (50 minutes for a 50-frame sweep), which will exceed agent execution timeouts and block progress.
*   **CUDA Setup & Diagnostic Commands**: If an agent encounters a CUDA/PyTorch error or `torch.cuda.is_available()` returns `False`, do not silently fall back to CPU. Instead, run these diagnostics and troubleshooting steps:
    1.  **Check CUDA Availability**:
        ```bash
        ./venv_sdp/bin/python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
        ```
    2.  **Verify GPU Hardware**: Check if the GPU is visible to the system:
        ```bash
        nvidia-smi
        ```
    3.  **Reinstall PyTorch with CUDA support** if `torch.cuda.is_available()` is `False`:
        ```bash
        ./venv_sdp/bin/pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
*   **Memory Management**: To prevent PyTorch memory leaks over continuous frames during verification sweeps, always perform explicit garbage collection inside batch loops:
    ```python
    del crown_lb, crown_ub, lirpa_model, wrapped_model, bounded_eps
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ```

## Step 1: Characterize Adverse Weather (Calibrate Bounds)
Calculate physical contrast drop ($\epsilon_c$) and brightness bias ($\epsilon_b$) bounds from the ACDC dataset:
```bash
./venv_sdp/bin/python tools/extract_physics_bounds.py \
    --condition rain \
    --sequence GOPR0402 \
    --max_images 5
```
*   *Note:* The recommended bounds are saved in `results/disturbance_characterization/acdc_physics_bounds.json`.

## Step 2: Run CROWN/SDP-CROWN Verification Sweep

Verify mathematical steering safety (steering output deviation $\le \pm 0.1$ radians) over a sequence of 50 continuous frames.

### Verification Sweep Command:
```bash
./venv_sdp/bin/python verify_steering.py \
    --model_type CarlaSteeringNet \
    --weights_path models/carla_small_mixed_aug.pth \
    --weather rain \
    --bounds_file results/disturbance_characterization/acdc_physics_bounds.json \
    --method SDP-CROWN \
    --device cuda \
    --num_frames 50 \
    --iterations 20
```

## Step 3: Run Batch Verification
To run sweeps across all models, weather conditions, and methods (CROWN vs. SDP-CROWN) automatically:
```bash
./venv_sdp/bin/python run_batch_verification.py
```
This script saves the consolidated results to `results/steering_verification/batch_verification_summary.md` and `results/steering_verification/batch_verification_summary.json`.

## Step 4: Generate Verification Plots
After the batch verification completes, you must run the plotting scripts to generate safety visualizations:

### 1. Safety Rates Bar Chart (Grouped Summary)
Generates a bar chart comparing certified safety rate percentages across models, methods, and weather conditions:
```bash
./venv_sdp/bin/python results/steering_verification/plot_safety_bar_chart.py
```
*   **Output Path**: `results/steering_verification/verification_safety_rates_bar_chart.png`

### 2. Frame-by-Frame Steering Bounds Plot (Detailed Deviations)
Consolidates individual frame bounds and plots worst-case steering deviations from nominal:
```bash
./venv_sdp/bin/python results/steering_verification/plot_verification.py
```
*   **Output Path**: `results/steering_verification/carla_steering_verification_results.png`

## Step 5: User Feedback Loop
After compiling the results and generating the plots:
1. **Present Results**: Embed or link to the markdown summary table (`results/steering_verification/batch_verification_summary.md`), the safety rates bar chart, and the frame-by-frame steering bounds plot.
2. **Wait for Feedback**: **Stop execution** and ask the user for feedback on the certified safety rates (e.g. comparing CROWN vs SDP-CROWN bounds tightness, or Clear-Only vs Mixed-Weather robustness) and the visual appeal of the plots. Do **not** finish the task or proceed to write papers/reports until the user has reviewed the outputs and provided feedback.
