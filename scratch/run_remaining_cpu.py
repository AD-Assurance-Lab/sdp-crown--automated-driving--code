import os
import subprocess

CODE_DIR = "/home/za/ad-assurance--workspace/sdp-crown--automated-driving--code"
RESULTS_DIR = os.path.join(CODE_DIR, "results/steering_verification")

# 1. Clear-only Night CROWN on CPU
clear_out = os.path.join(RESULTS_DIR, "clear_only_night_CROWN.json")
print("[RUNNING] CROWN CPU: clear_only, weather=night")
cmd_clear = [
    "./venv_sdp/bin/python", "verify_steering.py",
    "--model_type", "CarlaSteeringNet",
    "--weights_path", "models/carla_steering_net_clear.pth",
    "--weather", "night",
    "--method", "CROWN",
    "--num_frames", "10",
    "--device", "cpu",
    "--bounds_file", "results/acdc_physics_bounds.json",
    "--output_results", clear_out
]
try:
    subprocess.run(cmd_clear, check=True)
    print(f"[SUCCESS] Saved to {clear_out}")
except Exception as e:
    print(f"[ERROR] clear_only night failed: {e}")

# 2. Mixed-weather Night CROWN on CPU
mixed_out = os.path.join(RESULTS_DIR, "mixed_weather_night_CROWN.json")
print("\n[RUNNING] CROWN CPU: mixed_weather, weather=night")
cmd_mixed = [
    "./venv_sdp/bin/python", "verify_steering.py",
    "--model_type", "CarlaSteeringNet",
    "--weights_path", "models/carla_steering_net_mixed.pth",
    "--weather", "night",
    "--method", "CROWN",
    "--num_frames", "10",
    "--device", "cpu",
    "--bounds_file", "results/acdc_physics_bounds.json",
    "--output_results", mixed_out
]
try:
    subprocess.run(cmd_mixed, check=True)
    print(f"[SUCCESS] Saved to {mixed_out}")
except Exception as e:
    print(f"[ERROR] mixed_weather night failed: {e}")
