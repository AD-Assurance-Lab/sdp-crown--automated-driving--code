import os
import cv2
import json
import argparse
import numpy as np

# ==============================================================================
# CONTROL PANEL
# Modify these values to configure the default behavior when running the script
# directly (e.g. by pressing the "Play" button in VS Code).
# ==============================================================================
CONFIG = {
    "condition": "fog",                   # Options: "fog", "night", "snow", "rain"
    "split": "val",                       # Options: "train", "val", "test"
    "sequence": "GP020475",               # Specific folder (e.g., "GOPR0476") for stable ODD calibration
    "dataset_dir": "datasets/ACDC",       # Path to ACDC dataset root
    "output_json": "results/fog_bounds1.json", # Output JSON path
    "max_images": 10                      # Aggregating too many images causes bound explosion
}
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone ACDC Physical Threat Characterization")
    parser.add_argument(
        "--condition",
        choices=["fog", "night", "snow", "rain"],
        default=CONFIG["condition"],
        help="Weather condition to characterize"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=CONFIG["split"],
        help="ACDC dataset split"
    )
    parser.add_argument(
        "--sequence",
        default=CONFIG["sequence"],
        help="Specific sequence folder name (e.g., GOPR0476) or 'all' to aggregate all folders in the split"
    )
    parser.add_argument(
        "--dataset_dir",
        default=CONFIG["dataset_dir"],
        help="Path to the ACDC dataset root"
    )
    parser.add_argument(
        "--output_json",
        default=CONFIG["output_json"],
        help="Path to save the generated bounds as a JSON file"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=CONFIG["max_images"],
        help="Maximum number of image pairs to process (useful for quick checks)"
    )
    return parser.parse_args()


def extract_bounds(condition, split, sequence, dataset_dir, max_images=None):
    dataset_dir = os.path.abspath(os.path.expanduser(dataset_dir))
    
    # 1. Establish the directories
    # ACDC structure: <dataset_dir>/rgb_anon/<condition>/<split>/<sequence_folder>
    rgb_base = os.path.join(dataset_dir, "rgb_anon", condition)
    dist_dir_root = os.path.join(rgb_base, split)
    ref_dir_root = os.path.join(rgb_base, f"{split}_ref")
    
    # For snow/rain, spatial masking uses ground truth labelTrainIds
    # gt structure: <dataset_dir>/gt/<condition>/<split>/<sequence_folder>
    use_spatial_mask = condition in ["snow", "rain"]
    gt_dir_root = os.path.join(dataset_dir, "gt", condition, split) if use_spatial_mask else None
    
    if not os.path.exists(dist_dir_root) or not os.path.exists(ref_dir_root):
        raise FileNotFoundError(f"Adverse weather or reference directory not found at: {dist_dir_root} or {ref_dir_root}")

    # 2. Determine sequence folders to process
    if sequence.lower() == "all":
        seq_folders = sorted(os.listdir(dist_dir_root))
    else:
        # Support comma-separated sequence names
        seq_folders = [s.strip() for s in sequence.split(",")]
        
    print(f"Weather condition: {condition.upper()}")
    print(f"Dataset split:     {split}")
    print(f"Sequence folders:  {seq_folders}")
    print(f"Spatial masking:   {use_spatial_mask}")
    
    eps_c_list = []
    eps_b_list = []
    processed_count = 0
    
    # Iterate through sequence folders
    for folder in seq_folders:
        dist_seq_dir = os.path.join(dist_dir_root, folder)
        ref_seq_dir = os.path.join(ref_dir_root, folder)
        gt_seq_dir = os.path.join(gt_dir_root, folder) if use_spatial_mask else None
        
        if not os.path.exists(dist_seq_dir) or not os.path.exists(ref_seq_dir):
            print(f"Warning: Skipping sequence '{folder}' (directory not found).")
            continue
            
        filenames = sorted([f for f in os.listdir(dist_seq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for f in filenames:
            if max_images is not None and processed_count >= max_images:
                break
                
            dist_path = os.path.join(dist_seq_dir, f)
            
            # Map filename from rgb_anon to rgb_ref_anon
            # Example: GP010476_2020-02-15_12-25-32_0000000000000000_rgb_anon.png 
            # -> GP010476_2020-02-15_12-25-32_0000000000000000_rgb_ref_anon.png
            ref_file = f.replace("_rgb_anon.png", "_rgb_ref_anon.png").replace("_rgb_anon.jpg", "_rgb_ref_anon.jpg")
            ref_path = os.path.join(ref_seq_dir, ref_file)
            
            if not os.path.exists(ref_path):
                continue
                
            # Read grayscale images (ACDC is color, but we analyze luminance channel)
            img_clear = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            img_dist = cv2.imread(dist_path, cv2.IMREAD_GRAYSCALE)
            
            if img_clear is None or img_dist is None:
                continue
                
            x_clear = img_clear.astype(np.float32) / 255.0
            x_dist = img_dist.astype(np.float32) / 255.0
            
            # Extract road-masked pixels or flatten full image
            if use_spatial_mask and gt_seq_dir is not None:
                # Map label filename: rgb_anon.png -> gt_labelTrainIds.png
                mask_file = f.replace("_rgb_anon.png", "_gt_labelTrainIds.png").replace("_rgb_anon.jpg", "_gt_labelTrainIds.png")
                mask_path = os.path.join(gt_seq_dir, mask_file)
                
                if not os.path.exists(mask_path):
                    continue
                    
                img_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if img_mask is None:
                    continue
                
                # Cityscapes TrainID 0 represents the Road category
                road_mask = (img_mask == 0)
                
                pixels_clear = x_clear[road_mask]
                pixels_dist = x_dist[road_mask]
                
                if len(pixels_clear) == 0:
                    continue
            else:
                pixels_clear = x_clear.flatten()
                pixels_dist = x_dist.flatten()
                
            # Calculate standard deviation and mean
            mu_clear = np.mean(pixels_clear)
            sigma_clear = np.std(pixels_clear)
            mu_dist = np.mean(pixels_dist)
            sigma_dist = np.std(pixels_dist)
            
            # Avoid division by zero
            if sigma_clear < 1e-6:
                continue
                
            # Calculate eps_c (contrast scaling factor) and eps_b (brightness bias)
            eps_c = (sigma_dist / sigma_clear) - 1.0
            eps_b = mu_dist - (mu_clear * (1.0 + eps_c))
            
            eps_c_list.append(float(eps_c))
            eps_b_list.append(float(eps_b))
            processed_count += 1
            
    if processed_count == 0:
        raise ValueError("No valid matching image pairs were found and processed.")
        
    min_eps_c, max_eps_c = min(eps_c_list), max(eps_c_list)
    min_eps_b, max_eps_b = min(eps_b_list), max(eps_b_list)
    
    # To be mathematically valid, bounds must contain 0.0 (nominal) and satisfy min <= max.
    rec_eps_c_min = min(min_eps_c, 0.0)
    rec_eps_c_max = max(max_eps_c, 0.0)
    rec_eps_b_min = min(min_eps_b, 0.0)
    rec_eps_b_max = max(max_eps_b, 0.0)

    # Print statistics
    print("\n" + "="*50)
    print(f" {condition.upper()} CHARACTERIZATION RESULTS ".center(50, "="))
    print(f"Total processed image pairs: {processed_count}")
    print(f"Contrast Drop (eps_c) range:    [{min_eps_c:.4f}, {max_eps_c:.4f}]")
    print(f"Brightness Bias (eps_b) range:  [{min_eps_b:.4f}, {max_eps_b:.4f}]")
    print("="*50)
    print(f"RECOMMENDED BOUNDS FOR SDP-CROWN:")
    print(f"  epsilon_c: [{rec_eps_c_min:.4f}, {rec_eps_c_max:.4f}]")
    print(f"  epsilon_b: [{rec_eps_b_min:.4f}, {rec_eps_b_max:.4f}]")
    print("="*50 + "\n")
    
    return {
        "condition": condition,
        "split": split,
        "sequences_evaluated": seq_folders,
        "total_pairs": processed_count,
        "eps_c_min": min_eps_c,
        "eps_c_max": max_eps_c,
        "eps_b_min": min_eps_b,
        "eps_b_max": max_eps_b,
        "recommended_eps_c_min": rec_eps_c_min,
        "recommended_eps_c_max": rec_eps_c_max,
        "recommended_eps_b_min": rec_eps_b_min,
        "recommended_eps_b_max": rec_eps_b_max
    }

def main():
    args = parse_args()
    
    try:
        results = extract_bounds(
            condition=args.condition,
            split=args.split,
            sequence=args.sequence,
            dataset_dir=args.dataset_dir,
            max_images=args.max_images
        )
        
        # Save results to output JSON
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        print(f"Successfully saved characterized bounds to: {args.output_json}")
        
    except Exception as e:
        print(f"Error during characterization: {e}")
        exit(1)

if __name__ == "__main__":
    main()
