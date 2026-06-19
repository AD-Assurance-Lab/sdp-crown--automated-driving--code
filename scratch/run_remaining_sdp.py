import os
import subprocess

CODE_DIR = "/home/za/ad-assurance--workspace/sdp-crown--automated-driving--code"
RESULTS_DIR = os.path.join(CODE_DIR, "results/steering_verification")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. CarlaSteeringNet SDP runs
carla_models = {
    "clear_only": "models/carla_steering_net_clear.pth",
    "mixed_weather": "models/carla_steering_net_mixed.pth"
}
weathers = ["fog", "night", "rain", "snow"]

for model_name, weights_path in carla_models.items():
    for weather in weathers:
        sdp_out = os.path.join(RESULTS_DIR, f"{model_name}_{weather}_SDP.json")
        if os.path.exists(sdp_out) and os.path.getsize(sdp_out) > 0:
            print(f"[EXISTS] {sdp_out}")
            continue
        print(f"\n[RUNNING] SDP-CROWN: CarlaSteeringNet, Model={model_name}, Weather={weather}")
        cmd = [
            "./venv_sdp/bin/python", "verify_steering.py",
            "--model_type", "CarlaSteeringNet",
            "--weights_path", weights_path,
            "--weather", weather,
            "--method", "SDP-CROWN",
            "--iterations", "5",
            "--num_frames", "2",
            "--device", "cpu",
            "--bounds_file", "results/acdc_physics_bounds.json",
            "--output_results", sdp_out
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"[SUCCESS] Saved to {sdp_out}")
        except Exception as e:
            print(f"[ERROR] Failed for {model_name} {weather}: {e}")

# 2. MicroPilotNet SDP runs
pilotnet_weathers = ["fog", "snow"] # rain and night are already done
for weather in pilotnet_weathers:
    sdp_out = os.path.join(RESULTS_DIR, f"pilotnet_udacity_{weather}_SDP.json")
    if os.path.exists(sdp_out) and os.path.getsize(sdp_out) > 0:
        print(f"[EXISTS] {sdp_out}")
        continue
    print(f"\n[RUNNING] SDP-CROWN: MicroPilotNet, Weather={weather}")
    cmd = [
        "./venv_sdp/bin/python", "verify_steering.py",
        "--model_type", "MicroPilotNet",
        "--weights_path", "models/pilotnet_udacity.pth",
        "--weather", weather,
        "--method", "SDP-CROWN",
        "--iterations", "5",
        "--num_frames", "2",
        "--device", "cpu",
        "--bounds_file", "results/acdc_physics_bounds.json",
        "--output_results", sdp_out
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Saved to {sdp_out}")
    except Exception as e:
        print(f"[ERROR] Failed for MicroPilotNet {weather}: {e}")
