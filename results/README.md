# Project Verification & Telemetry Results

This directory contains the outputs generated from our adverse weather characterization, offline formal verification sweeps, and CARLA closed-loop simulator validation.

---

## 1. Directory Structure & Key Files

### `disturbance_characterization/`
Contains the characterized mathematical contrast and brightness perturbation bounds derived from the real-world ACDC dataset.
*   **`acdc_physics_bounds.json`**: The canonical bounds file containing contrast drops ($\epsilon_c$) and brightness offsets ($\epsilon_b$) for Rain, Fog, Night, and Snow. Used as the formal inputs for verification bounds propagation.

### `steering_verification/`
Contains the formal verification bounds, safety rate calculations, and sweep logs.
*   **`batch_verification_summary.json` / `.md`**: The consolidated results of our CROWN vs. SDP-CROWN sweeps over 50 continuous frames.
*   **`<model>_<weather>_<method>.json`**: Individual verification run telemetry containing per-frame upper/lower steering bounds, nominal outputs, and safety status.
*   **`plot_verification.py`**: Visualizes the verification boundaries (safety corridor limits vs. model output bounds).

### `carla_ai_model_testing/`
Contains the telemetry logs and trajectory plots from closed-loop evaluations in the CARLA simulator.
*   **`town04_small_mixed_clear.csv` / `.png`**: Telemetry and CTE plots for the clear-weather baseline runs.
*   **`town04_small_mixed_aug_<weather>.csv` / `.png`**: Closed-loop evaluation logs for our weather-augmented model under Rain, Fog, or Night.
*   **`town04_closed_loop_summary.md`**: Consolidated closed-loop simulation validation results (steering MAE, RMSE, mean/max Cross-Track Error, and lane-keeping PASS/FAIL status).
