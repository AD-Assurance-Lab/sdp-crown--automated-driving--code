import subprocess
import os

weathers = ["fog", "night", "rain", "snow"]
models = {
    "clear_only": "models/carla_steering_net_clear.pth",
    "mixed_weather": "models/carla_steering_net_mixed.pth"
}
RESULTS_DIR = "results/steering_verification"
os.makedirs(RESULTS_DIR, exist_ok=True)

for model_name, weights_path in models.items():
    for weather in weathers:
        output_file = f"{RESULTS_DIR}/{model_name}_{weather}_CROWN.json"
        # Skip if already exists and not empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Skipping {output_file} as it already exists.")
            continue
        print(f"Running CROWN on GPU for model={model_name}, weather={weather}...")
        cmd = [
            "./venv_sdp/bin/python", "verify_steering.py",
            "--model_type", "CarlaSteeringNet",
            "--weights_path", weights_path,
            "--weather", weather,
            "--method", "CROWN",
            "--num_frames", "10",
            "--device", "cuda",
            "--bounds_file", "results/acdc_physics_bounds.json",
            "--output_results", output_file
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Finished {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing CROWN for {model_name} in {weather}: {e}")
