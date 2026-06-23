# Helper Tools & Utilities

This directory contains developer utilities and scripts to collect data, train networks, run closed-loop simulation tests, and extract environmental characterizations.

---

## 1. Tool Directory & Script Registry

### `train_carla_model.py`
Trains the `CarlaSteeringNet` (`small`) or `CarlaSteeringExpertNet` (`expert`) model on our balanced datasets.
*   **Key Flag `--weather-aug`**: Dynamically applies ACDC-calibrated weather contrast drops and brightness biases to clear-weather training frames.
*   **Key Flag `--mode`**: Selects training splits (e.g., `clear` or `mixed`).

### `test_carla_model.py`
Runs closed-loop simulation evaluations inside the CARLA environment.
*   **Key Flag `--start-frame`**: Spawns the ego vehicle at a specific frame index along the highway trajectory (e.g. frame `750` right before the Town04 curve). Warmup velocity override is automatically bypassed when starting mid-track to prevent overshooting.
*   **Key Flag `--weather`**: Configures the weather profile (rain, fog, night, clear) and turns headlight sensors ON/OFF.

### `run_dagger_lite.py`
Implements the interactive **DAgger-Lite** loop. It executes the current steering policy, queries the autopilot in the background to calculate correct recovery commands on the states visited, and aggregates the recovery frames.

### `carla_data_collector.py`
Coordinates the collection of initial autopilot training data. Configures target lane selection, Traffic Manager logic (ignoring traffic lights), and telemetry recording.

### `extract_physics_bounds.py`
Calibrates mathematical weather bounds by pairing GPS-synchronized clear vs. adverse weather image sets from the ACDC dataset.

### `extract_carla_physics_bounds.py`
Evaluates contrast and brightness drops inside the CARLA simulator (used to characterization-check CARLA's visual fidelity against real-world ACDC parameters).

### `weather_config.py`
Defines parameters for setting simulator weather conditions (sun angle, cloudness, precipitation, wind, fog density).
