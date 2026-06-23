#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(script_dir, "batch_verification_summary.json")
    default_output = os.path.join(script_dir, "verification_safety_rates_bar_chart.png")

    parser = argparse.ArgumentParser(description="Plot certified safety rates bar chart from batch summary JSON")
    parser.add_argument("--json_path", default=default_json, help="Path to batch_verification_summary.json")
    parser.add_argument("--output_path", default=default_output, help="Path to save the generated bar chart image")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at: {args.json_path}")
        return

    with open(args.json_path, "r") as f:
        data = json.load(f)

    weathers = ["rain", "fog", "night", "snow"]
    labels_weather = [w.upper() for w in weathers]
    
    # Extract rates
    clear_crown = [data["small_clear"][w]["CROWN"] for w in weathers]
    clear_sdp = [data["small_clear"][w]["SDP-CROWN"] for w in weathers]
    mixed_crown = [data["small_mixed"][w]["CROWN"] for w in weathers]
    mixed_sdp = [data["small_mixed"][w]["SDP-CROWN"] for w in weathers]

    # Setup matplotlib styling for a premium aesthetic
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=300)
    
    x = np.arange(len(weathers))
    width = 0.35  # width of the bars

    # Colors: Sleek blue for clear-only, vibrant green for mixed-weather
    color_clear = "#1f77b4"  # Steel Blue
    color_mixed = "#2ca02c"  # Forest Green

    # Subplot 1: CROWN Safety Rates
    rects1_clear = ax1.bar(x - width/2, clear_crown, width, label="Clear-Only Model", color=color_clear, alpha=0.85, edgecolor='black', linewidth=0.7)
    rects1_mixed = ax1.bar(x + width/2, mixed_crown, width, label="Mixed-Weather Model", color=color_mixed, alpha=0.85, edgecolor='black', linewidth=0.7)
    
    ax1.set_title("CROWN Certified Safety Rates\n(50 Frames)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_weather, fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Safety Rate (%)", fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    # Add labels on top of bars for Subplot 1
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

    autolabel(rects1_clear, ax1)
    autolabel(rects1_mixed, ax1)

    # Subplot 2: SDP-CROWN Safety Rates
    rects2_clear = ax2.bar(x - width/2, clear_sdp, width, label="Clear-Only Model", color=color_clear, alpha=0.85, edgecolor='black', linewidth=0.7)
    rects2_mixed = ax2.bar(x + width/2, mixed_sdp, width, label="Mixed-Weather Model", color=color_mixed, alpha=0.85, edgecolor='black', linewidth=0.7)
    
    ax2.set_title("SDP-CROWN Certified Safety Rates\n(50 Frames, 20 Iterations)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_weather, fontsize=11, fontweight='semibold')
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

    autolabel(rects2_clear, ax2)
    autolabel(rects2_mixed, ax2)

    plt.suptitle("Certified Safety Rate Comparison under Adverse Weather Disturbances\n(Safety Corridor = $\pm$0.1 rad Steering Deviation Limit)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Successfully generated verification safety rates bar chart at: {args.output_path}")

if __name__ == "__main__":
    main()
