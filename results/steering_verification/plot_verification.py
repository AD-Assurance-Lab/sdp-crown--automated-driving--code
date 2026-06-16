import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "results_json": os.path.join(script_dir, "verification_results.json"), # Path to verification results JSON
    "output_png": os.path.join(script_dir, "verification_plot.png")        # Path to save the generated plot image
}

def parse_args():
    parser = argparse.ArgumentParser(description="Plot SDP-CROWN Verification Bounds")
    parser.add_argument(
        "--results_json",
        default=CONFIG["results_json"],
        help="Path to verification results JSON"
    )
    parser.add_argument(
        "--output_png",
        default=CONFIG["output_png"],
        help="Path to save the generated plot image"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.results_json):
        print(f"Error: Results file not found at {args.results_json}")
        print("Please run verify_steering.py first to generate the verification data.")
        return

    with open(args.results_json, "r") as f:
        data = json.load(f)

    weather = data.get("weather", "Unknown")
    eps_c_min = data.get("eps_c_min", 0.0)
    eps_b_max = data.get("eps_b_max", 0.0)
    safe_deviation = data.get("safe_deviation", 0.1)
    safety_rate = data.get("safety_rate", 0.0)
    frames = data.get("frames", [])

    if not frames:
        print("Error: No frames data found in results JSON.")
        return

    frame_idxs = [f["frame_idx"] for f in frames]
    nominal = [f["nominal_steering"] for f in frames]
    lb = [f["lower_bound"] for f in frames]
    ub = [f["upper_bound"] for f in frames]
    corridor_l = [f["lower_corridor"] for f in frames]
    corridor_u = [f["upper_corridor"] for f in frames]
    statuses = [f["status"] for f in frames]

    # Set up styling for a clean, professional aesthetic
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Convert to numpy arrays for elementwise calculations
    frame_idxs = np.array(frame_idxs)
    nominal = np.array(nominal)
    lb = np.array(lb)
    ub = np.array(ub)
    corridor_l = np.array(corridor_l)
    corridor_u = np.array(corridor_u)

    # Plot Safety Corridor (Shaded Area around Nominal)
    ax.fill_between(
        frame_idxs, corridor_l, corridor_u, 
        color="#e0f2f1", alpha=0.6, label=f"Safety Corridor (Nominal ±{safe_deviation} rad)"
    )
    
    # Plot CROWN Bounds as a shaded region (Worst-Case envelope)
    ax.fill_between(
        frame_idxs, lb, ub, 
        color="#ffc107", alpha=0.3, label="CROWN Worst-Case Bounds"
    )

    # Plot Nominal Steering line
    ax.plot(frame_idxs, nominal, color="#00796b", linewidth=2.5, linestyle="--", label="Nominal Steering (Clear)")

    # Plot Bound outlines
    ax.plot(frame_idxs, lb, color="#ffb300", linewidth=1, linestyle="-")
    ax.plot(frame_idxs, ub, color="#ffb300", linewidth=1, linestyle="-")

    # Mark frames color-coded by safety status
    safe_mask = np.array([s == "SAFE" for s in statuses])
    fail_mask = np.array([s == "FAILED" for s in statuses])
    vacuous_mask = np.array([s == "VACUOUS" for s in statuses])

    if np.any(safe_mask):
        ax.scatter(
            frame_idxs[safe_mask], nominal[safe_mask], 
            color="#2e7d32", s=50, zorder=5, label="Verified Safe Frame"
        )
    if np.any(fail_mask):
        ax.scatter(
            frame_idxs[fail_mask], nominal[fail_mask], 
            color="#c62828", s=60, marker="X", zorder=5, label="Safety Violated (Failed)"
        )
    if np.any(vacuous_mask):
        ax.scatter(
            frame_idxs[vacuous_mask], nominal[vacuous_mask], 
            color="#757575", s=60, marker="o", facecolors='none', edgecolors='#757575', 
            zorder=5, label="Vacuous/Exploded Bounds"
        )

    # Labels and Title
    ax.set_title(
        f"SDP-CROWN Steering Verification under {weather.upper()} Perturbation\n"
        f"Calibrated Constraints: $\epsilon_c \in [{eps_c_min:.4f}, 0]$, $\epsilon_b \in [0, {eps_b_max:.4f}]$ | "
        f"Certified Robustness: {safety_rate:.1f}%",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Time Frame", fontsize=12)
    ax.set_ylabel("Steering Angle (Radians)", fontsize=12)
    
    ax.set_xlim(frame_idxs[0] - 0.5, frame_idxs[-1] + 0.5)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    
    plt.tight_layout()

    # Save the output image
    output_dir = os.path.dirname(args.output_png)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output_png, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated verification plot at: {args.output_png}")

if __name__ == "__main__":
    main()
