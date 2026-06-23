#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import pandas as pd

# Directories
CODE_DIR = "/home/za/ad-assurance--workspace/sdp-crown--automated-driving--code"
RESULTS_DIR = os.path.join(CODE_DIR, "results/steering_verification")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configs
num_frames = 50
weathers = ["rain", "fog", "night", "snow"]
models = {
    "small_clear": {
        "weights": "models/carla_small_clear.pth",
        "model_type": "CarlaSteeringNet",
        "csv": "datasets/carla_steering_e2e/clear/index.csv"
    },
    "small_mixed": {
        "weights": "models/carla_small_mixed.pth",
        "model_type": "CarlaSteeringNet",
        "csv": "datasets/carla_steering_e2e/clear/index.csv"
    }
}
methods = ["CROWN", "SDP-CROWN"]

# Results storage
# Structure: results[model][weather][method] = float
results = {m: {w: {meth: None for meth in methods} for w in weathers} for m in models}

print("=" * 70)
print("Starting Batch Verification for DAgger-Lite CarlaSteeringNet Models")
print(f"Weather disturbance bounds: results/disturbance_characterization/acdc_physics_bounds.json")
print("=" * 70)

for model_name, config in models.items():
    weights_path = os.path.join(CODE_DIR, config["weights"])
    csv_path = os.path.join(CODE_DIR, config["csv"])
    
    for weather in weathers:
        for method in methods:
            output_file = os.path.join(RESULTS_DIR, f"{model_name}_{weather}_{method}.json")
            
            print(f"\n[RUNNING] Model: {model_name} | Weather: {weather} | Method: {method}")
            
            cmd = [
                "./venv_sdp/bin/python", "verify_steering.py",
                "--model_type", config["model_type"],
                "--weights_path", weights_path,
                "--csv_path", csv_path,
                "--weather", weather,
                "--bounds_file", "results/disturbance_characterization/acdc_physics_bounds.json",
                "--num_frames", str(num_frames),
                "--method", method,
                "--device", "cuda",
                "--iterations", "20",
                "--output_results", output_file
            ]
            
            try:
                subprocess.run(cmd, check=True)
                with open(output_file, "r") as f:
                    res = json.load(f)
                safety_rate = res.get("safety_rate", 0.0)
                results[model_name][weather][method] = safety_rate
                print(f"[SUCCESS] Safety Rate: {safety_rate:.1f}%")
            except Exception as e:
                print(f"[FAILED] Verification failed for {model_name} in {weather} with {method}: {e}")
                results[model_name][weather][method] = "ERR"

# Print Consolidated Markdown Summary
print("\n" + "=" * 70)
print("CONSOLIDATED BATCH VERIFICATION RESULTS SUMMARY".center(70))
print("=" * 70)

markdown_table = []
markdown_table.append("| Model | Weather | CROWN (50 frames) | SDP-CROWN (50 frames, 20 iter) |")
markdown_table.append("| :--- | :--- | :--- | :--- |")

for model_name in models:
    for weather in weathers:
        rate_crown = results[model_name][weather]["CROWN"]
        rate_sdp = results[model_name][weather]["SDP-CROWN"]
        
        crown_str = f"{rate_crown:.1f}%" if isinstance(rate_crown, float) else str(rate_crown)
        sdp_str = f"{rate_sdp:.1f}%" if isinstance(rate_sdp, float) else str(rate_sdp)
        
        if model_name == "small_clear":
            model_label = "Model 4 (Small Clear-Only)"
        elif model_name == "small_mixed":
            model_label = "Model 3 (Small Mixed-Weather)"
        else:
            model_label = f"Model ({model_name})"
        markdown_table.append(f"| {model_label} | {weather.upper()} | {crown_str} | {sdp_str} |")

for line in markdown_table:
    print(line)

# Save markdown summary and json summary
summary_md_path = os.path.join(RESULTS_DIR, "batch_verification_summary.md")
with open(summary_md_path, "w") as f:
    f.write("# DAgger-Lite Small Model Batch Verification Summary\n\n")
    f.write("\n".join(markdown_table))
    f.write("\n")

summary_json_path = os.path.join(RESULTS_DIR, "batch_verification_summary.json")
with open(summary_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved consolidated summary markdown to: {summary_md_path}")
print(f"Saved consolidated summary JSON to: {summary_json_path}")
print("=" * 70)
