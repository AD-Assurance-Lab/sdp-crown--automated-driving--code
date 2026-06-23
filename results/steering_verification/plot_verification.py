import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# File Paths
CONSOLIDATED_JSON = os.path.join(script_dir, "verification_results.json")
OUTPUT_PNG = os.path.join(script_dir, "carla_steering_verification_results.png")

def consolidate_files():
    """Scans the steering_verification folder for individual model/weather json files and combines them."""
    weathers = ["fog", "night", "rain", "snow"]
    models = ["clear_only", "mixed_weather", "pilotnet_udacity"]
    methods = ["CROWN", "SDP"] # CROWN = 10 frames, SDP = 2 frames SDP-CROWN
    
    consolidated = {}
    
    for model in models:
        consolidated[model] = {}
        for weather in weathers:
            consolidated[model][weather] = {}
            for method in methods:
                filename = f"{model}_{weather}_{method}.json"
                if model == "pilotnet_udacity":
                    filepath = os.path.join(script_dir, "debugging", filename)
                else:
                    filepath = os.path.join(script_dir, "json", filename)
                
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    try:
                        with open(filepath, "r") as f:
                            res_data = json.load(f)
                        method_name = "CROWN" if method == "CROWN" else "SDP-CROWN"
                        consolidated[model][weather][method_name] = res_data
                    except Exception as e:
                        print(f"Warning: Failed to parse {filename}: {e}")
                    
    # Write consolidated JSON
    with open(CONSOLIDATED_JSON, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=4)
    print(f"Consolidated all verification outputs into single file: {CONSOLIDATED_JSON}")
    return consolidated

def main():
    parser = argparse.ArgumentParser(description="Plot SDP-CROWN and CROWN Verification Bounds")
    parser.add_argument(
        "--results_json",
        default=CONSOLIDATED_JSON,
        help="Path to consolidated verification results JSON"
    )
    parser.add_argument(
        "--output_png",
        default=OUTPUT_PNG,
        help="Path to save the generated plot image"
    )
    args = parser.parse_args()
    
    # 1. Consolidate files first
    data = consolidate_files()
    
    # 2. Plotting
    weathers = ["fog", "night", "rain", "snow"]
    models = ["clear_only", "mixed_weather"]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axs = axs.ravel()
    
    colors = {
        "clear_only": {"nominal": "#007acc", "crown": "#66b2ff", "sdp": "#004080"},
        "mixed_weather": {"nominal": "#228b22", "crown": "#7fdf7f", "sdp": "#006400"}
    }
    
    for idx, weather in enumerate(weathers):
        ax = axs[idx]
        
        # Plot safety corridor relative to nominal (which is ±0.1 rad deviation)
        ax.axhspan(-0.1, 0.1, color='#e2f0d9' if weather != "snow" else '#f2f2f2', alpha=0.4, label='Safety Corridor ($\pm$0.1 rad)')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        for model in models:
            if model == "clear_only":
                label_model = "Clear-Only Model"
            else:
                label_model = "Mixed-Weather Model"
            
            # A. Plot CROWN (10 frames)
            if "CROWN" in data[model][weather]:
                crown_res = data[model][weather]["CROWN"]
                frames = crown_res.get("frames", [])
                x = [f["frame_idx"] for f in frames]
                
                # Deviations from nominal steering angle
                lb_dev = [f["lower_bound"] - f["nominal_steering"] for f in frames]
                ub_dev = [f["upper_bound"] - f["nominal_steering"] for f in frames]
                
                # Clip values for cleaner visualization in case of explosion
                lb_dev_clipped = np.clip(lb_dev, -2.5, 2.5)
                ub_dev_clipped = np.clip(ub_dev, -2.5, 2.5)
                
                ax.fill_between(
                    x, lb_dev_clipped, ub_dev_clipped, 
                    color=colors[model]["crown"], alpha=0.25,
                    label=f"{label_model} (CROWN worst-case)"
                )
                
            # B. Plot SDP-CROWN (2 frames)
            if "SDP-CROWN" in data[model][weather]:
                sdp_res = data[model][weather]["SDP-CROWN"]
                frames = sdp_res.get("frames", [])
                x_sdp = [f["frame_idx"] for f in frames]
                
                lb_dev = [f["lower_bound"] - f["nominal_steering"] for f in frames]
                ub_dev = [f["upper_bound"] - f["nominal_steering"] for f in frames]
                
                lb_dev_clipped = np.clip(lb_dev, -2.5, 2.5)
                ub_dev_clipped = np.clip(ub_dev, -2.5, 2.5)
                
                ax.errorbar(
                    x_sdp, [0]*len(x_sdp), 
                    yerr=[[-l for l in lb_dev_clipped], ub_dev_clipped], 
                    fmt='o', color=colors[model]["sdp"], elinewidth=2.5, capsize=5,
                    label=f"{label_model} (SDP-CROWN)"
                )
                
        ax.set_title(f"Weather Disturbance: {weather.upper()}", fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel("Steering Deviation from Nominal (rad)", fontsize=11)
        
        # Scale Y axis with data: tight scaling for non-exploding conditions, wider scaling for night
        if weather == "night":
            ax.set_ylim(-2.0, 2.0)
        else:
            ax.set_ylim(-0.15, 0.15)
            
        ax.set_xlim(-0.5, 9.5)
        ax.set_xticks(range(10))
        
        if idx >= 2:
            ax.set_xlabel("Frame Index", fontsize=11)
            
        # Format legends without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        
    plt.suptitle("E2E Steering Verification under Adverse Weather Constraints\n(Calibrated ACDC Epsilon Envelopes | Safety Corridor = $\pm$0.1 rad)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(args.output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Successfully generated verification comparison plot at: {args.output_png}")

if __name__ == "__main__":
    main()
