#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
import pandas as pd

def extract_carla_bounds(data_root="datasets/carla_steering_e2e", output_json="results/carla_physics_bounds.json", max_images=1000):
    weathers = ["rain", "fog", "night"]
    results = {}
    
    # Verify clear directory exists
    clear_csv = os.path.join(data_root, "clear", "index.csv")
    if not os.path.exists(clear_csv):
        raise FileNotFoundError(f"Clear weather index not found at: {clear_csv}")
        
    print("Reading clear weather metadata...")
    clear_df = pd.read_csv(clear_csv)
    # Map frame number to image path
    clear_map = {row['frame']: os.path.join(data_root, "clear", row['image_path']) for _, row in clear_df.iterrows()}
    
    for w in weathers:
        csv_path = os.path.join(data_root, w, "index.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Index CSV not found for weather '{w}' at {csv_path}. Skipping.")
            continue
            
        print(f"\nCharacterizing CARLA weather condition: {w.upper()}")
        df = pd.read_csv(csv_path)
        
        eps_c_list = []
        eps_b_list = []
        processed_count = 0
        
        for _, row in df.iterrows():
            if max_images is not None and processed_count >= max_images:
                break
                
            frame = row['frame']
            if frame not in clear_map:
                continue
                
            clear_path = clear_map[frame]
            dist_path = os.path.join(data_root, w, row['image_path'])
            
            if not os.path.exists(clear_path) or not os.path.exists(dist_path):
                continue
                
            # Read grayscale images
            img_clear = cv2.imread(clear_path, cv2.IMREAD_GRAYSCALE)
            img_dist = cv2.imread(dist_path, cv2.IMREAD_GRAYSCALE)
            
            if img_clear is None or img_dist is None:
                continue
                
            # Convert to [0.0, 1.0] scale
            x_clear = img_clear.astype(np.float32) / 255.0
            x_dist = img_dist.astype(np.float32) / 255.0
            
            pixels_clear = x_clear.flatten()
            pixels_dist = x_dist.flatten()
            
            # Calculate standard deviation and mean
            mu_clear = np.mean(pixels_clear)
            sigma_clear = np.std(pixels_clear)
            mu_dist = np.mean(pixels_dist)
            sigma_dist = np.std(pixels_dist)
            
            if sigma_clear < 1e-6:
                continue
                
            # Calculate contrast drop eps_c and brightness bias eps_b
            eps_c = (sigma_dist / sigma_clear) - 1.0
            eps_b = mu_dist - (mu_clear * (1.0 + eps_c))
            
            eps_c_list.append(float(eps_c))
            eps_b_list.append(float(eps_b))
            processed_count += 1
            
        if processed_count == 0:
            print(f"No matching image pairs found for weather '{w}'.")
            continue
            
        min_eps_c, max_eps_c = min(eps_c_list), max(eps_c_list)
        min_eps_b, max_eps_b = min(eps_b_list), max(eps_b_list)
        
        rec_eps_c_min = min(min_eps_c, 0.0)
        rec_eps_c_max = max(max_eps_c, 0.0)
        rec_eps_b_min = min(min_eps_b, 0.0)
        rec_eps_b_max = max(max_eps_b, 0.0)
        
        print(f"Total processed image pairs: {processed_count}")
        print(f"Contrast Drop (eps_c) range:    [{min_eps_c:.4f}, {max_eps_c:.4f}]")
        print(f"Brightness Bias (eps_b) range:  [{min_eps_b:.4f}, {max_eps_b:.4f}]")
        print(f"Recommended epsilon_c bounds:   [{rec_eps_c_min:.4f}, {rec_eps_c_max:.4f}]")
        print(f"Recommended epsilon_b bounds:   [{rec_eps_b_min:.4f}, {rec_eps_b_max:.4f}]")
        
        results[w] = {
            "total_pairs_evaluated": processed_count,
            "eps_c_min": min_eps_c,
            "eps_c_max": max_eps_c,
            "eps_b_min": min_eps_b,
            "eps_b_max": max_eps_b,
            "recommended_eps_c_min": rec_eps_c_min,
            "recommended_eps_c_max": rec_eps_c_max,
            "recommended_eps_b_min": rec_eps_b_min,
            "recommended_eps_b_max": rec_eps_b_max
        }
        
    if results:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nSuccessfully saved CARLA bounds to: {output_json}")
    else:
        print("No bounds were extracted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract physical threat bounds for CARLA datasets")
    parser.add_argument("--data-root", default="datasets/carla_steering_e2e", help="Root directory of collected CARLA data")
    parser.add_argument("--output-json", default="results/carla_physics_bounds.json", help="Path to output JSON file")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum image pairs to process per weather condition")
    args = parser.parse_args()
    
    extract_carla_bounds(
        data_root=args.data_root,
        output_json=args.output_json,
        max_images=args.max_images
    )
