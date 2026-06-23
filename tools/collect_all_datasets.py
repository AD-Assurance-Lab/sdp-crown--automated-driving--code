#!/usr/bin/env python3
import os
import subprocess
import sys

def main():
    # Target directory of data collection
    data_root = "datasets/carla_steering_e2e"
    weathers = ["clear", "rain", "fog", "night"]
    
    for w in weathers:
        csv_path = os.path.join(data_root, w, "index.csv")
        if os.path.exists(csv_path):
            try:
                # Count lines in CSV to avoid pandas dependency here if pandas isn't needed
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                # Subtract 1 for the header
                num_frames = len(lines) - 1
                if num_frames >= 1000:
                    print(f"Weather '{w}' already has {num_frames} frames in {csv_path}. Skipping collection.")
                    continue
                else:
                    print(f"Weather '{w}' has incomplete collection ({num_frames}/1000 frames). Re-running...")
            except Exception as e:
                print(f"Error reading {csv_path}: {e}. Re-running...")
                
        print("\n" + "="*50)
        print(f"Starting data collection for weather: {w.upper()}")
        print("="*50)
        
        # Run collector script
        cmd = [
            "./venv_sdp/bin/python",
            "tools/carla_data_collector.py",
            "--weather", w,
            "--num-frames", "2000",
            "--seed", "42",
            "--spawn-point-idx", "12"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully finished data collection for weather: {w.upper()}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error during collection for weather {w.upper()}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
