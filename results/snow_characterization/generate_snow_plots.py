import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    condition = "snow"
    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "datasets", "ACDC"))
    
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_expansion_all = os.path.join(output_dir, "snow_epsilon_expansion_plot.png")
    output_comparison_grid = os.path.join(output_dir, "snow_comparison_grid.png")
    output_json_all = os.path.join(output_dir, "snow_epsilon_expansion_analysis_combined.json")
    
    # -------------------------------------------------------------------------
    # 1. LOAD DATA ACROSS ALL SPLITS (train -> test -> val)
    # -------------------------------------------------------------------------
    combined_records = []
    splits = ["train", "test", "val"]
    
    for split in splits:
        dist_root = os.path.join(dataset_dir, "rgb_anon", condition, split)
        ref_root = os.path.join(dataset_dir, "rgb_anon", condition, f"{split}_ref")
        gt_root = os.path.join(dataset_dir, "gt", condition, split)
        
        if not os.path.exists(dist_root) or not os.path.exists(ref_root):
            continue
            
        folders = sorted(os.listdir(dist_root))
        for folder in folders:
            dist_seq_dir = os.path.join(dist_root, folder)
            ref_seq_dir = os.path.join(ref_root, folder)
            gt_seq_dir = os.path.join(gt_root, folder)
            
            if not os.path.exists(dist_seq_dir) or not os.path.exists(ref_seq_dir) or not os.path.exists(gt_seq_dir):
                continue
                
            filenames = sorted([f for f in os.listdir(dist_seq_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for f in filenames:
                dist_path = os.path.join(dist_seq_dir, f)
                ref_file = f.replace("_rgb_anon.png", "_rgb_ref_anon.png").replace("_rgb_anon.jpg", "_rgb_ref_anon.jpg")
                ref_path = os.path.join(ref_seq_dir, ref_file)
                
                if not os.path.exists(ref_path):
                    continue
                
                mask_file = f.replace("_rgb_anon.png", "_gt_labelTrainIds.png").replace("_rgb_anon.jpg", "_gt_labelTrainIds.png")
                mask_path = os.path.join(gt_seq_dir, mask_file)
                if not os.path.exists(mask_path):
                    continue
                    
                img_clear = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
                img_dist = cv2.imread(dist_path, cv2.IMREAD_GRAYSCALE)
                img_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                
                if img_clear is None or img_dist is None or img_mask is None:
                    continue
                
                x_clear = img_clear.astype(np.float32) / 255.0
                x_dist = img_dist.astype(np.float32) / 255.0
                
                # Apply road masking (trainId 0 is road)
                road_mask = (img_mask == 0)
                if np.sum(road_mask) == 0:
                    continue
                    
                pixels_clear = x_clear[road_mask]
                pixels_dist = x_dist[road_mask]
                
                eps_c = (np.std(pixels_dist) / np.std(pixels_clear)) - 1.0
                eps_b = np.mean(pixels_dist) - (np.mean(pixels_clear) * (1.0 + eps_c))
                
                record = {
                    "split": split,
                    "sequence": folder,
                    "file": f,
                    "eps_c": float(eps_c),
                    "eps_b": float(eps_b)
                }
                combined_records.append(record)

    total_combined = len(combined_records)
    print(f"Loaded {total_combined} images total in split order (train -> test -> val) with road masking.")
    
    eps_c_comb = np.array([r["eps_c"] for r in combined_records])
    eps_b_comb = np.array([r["eps_b"] for r in combined_records])
    
    # Calculate dataset-wide percentiles
    pct_c_5 = np.percentile(eps_c_comb, 5)
    pct_b_95 = np.percentile(eps_b_comb, 95)
    print(f"Snow bounds: eps_c 5th = {pct_c_5:.4f}, eps_b 95th = {pct_b_95:.4f}")
    
    # -------------------------------------------------------------------------
    # PLOT 1: ALL SNOW IMAGES EXPANSION PLOT (with percentile lines)
    # -------------------------------------------------------------------------
    run_eps_c_min = []
    run_eps_b_max = []
    
    for i in range(1, total_combined + 1):
        c_sub = eps_c_comb[:i]
        b_sub = eps_b_comb[:i]
        run_eps_c_min.append(min(min(c_sub), 0.0))
        run_eps_b_max.append(max(max(b_sub), 0.0))
        
    indices_comb = np.arange(1, total_combined + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Epsilon C
    plt.subplot(2, 1, 1)
    plt.scatter(indices_comb, eps_c_comb, color='#3b82f6', alpha=0.5, s=15, label='All ACDC Snow Images (Road Masked)')
    plt.step(indices_comb, run_eps_c_min, where='mid', color='#ef4444', linewidth=2, label='Running Lower Bound')
    plt.axhline(pct_c_5, color='#10b981', linestyle='--', linewidth=2, label=f'Calibrated Robust Bound ($\epsilon_c^{{5\%}} = {pct_c_5:.4f}$)')
    plt.axhline(0.0, color='black', linestyle=':', linewidth=1.5, label='Physical Limit ($\epsilon_c \leq 0.0$)')
    
    plt.title('(a) Road-Masked Contrast Scaling Bound Expansion ($\epsilon_c$) - All ACDC Snow Images', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Images Aggregated (Sorted Chronologically by Split)', fontsize=11)
    plt.ylabel('Epsilon C ($\epsilon_c$)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    
    # Subplot 2: Epsilon B
    plt.subplot(2, 1, 2)
    plt.scatter(indices_comb, eps_b_comb, color='#3b82f6', alpha=0.5, s=15, label='All ACDC Snow Images (Road Masked)')
    plt.step(indices_comb, run_eps_b_max, where='mid', color='#ef4444', linewidth=2, label='Running Upper Bound')
    plt.axhline(pct_b_95, color='#10b981', linestyle='--', linewidth=2, label=f'Calibrated Robust Bound ($\epsilon_b^{{95\%}} = {pct_b_95:.4f}$)')
    plt.axhline(0.0, color='black', linestyle=':', linewidth=1.5, label='Physical Limit ($\epsilon_b \geq 0.0$)')
    
    plt.title('(b) Road-Masked Brightness Bias Bound Expansion ($\epsilon_b$) - All ACDC Snow Images', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Images Aggregated (Sorted Chronologically by Split)', fontsize=11)
    plt.ylabel('Epsilon B ($\epsilon_b$)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_expansion_all, dpi=150)
    plt.close()
    print(f"Saved Plot 1 (Snow Expansion) to: {output_expansion_all}")
    
    # -------------------------------------------------------------------------
    # FIND REPRESENTATIVE FRAMES FOR GRID
    # -------------------------------------------------------------------------
    # 1. Worst Contrast Drop (min eps_c)
    worst_c_idx = np.argmin(eps_c_comb)
    worst_c_rec = combined_records[worst_c_idx]
    
    # 2. Worst Brightness Bias (max eps_b)
    worst_b_idx = np.argmax(eps_b_comb)
    worst_b_rec = combined_records[worst_b_idx]
    
    # 3. Calibrated Boundary Frame (eps_c closest to pct_c_5)
    diffs = np.abs(eps_c_comb - pct_c_5)
    calib_idx = np.argmin(diffs)
    calib_rec = combined_records[calib_idx]
    
    rep_frames = [
        {"rec": worst_c_rec, "title": f"Worst Contrast Drop (Frame {worst_c_rec['file'].split('_')[2]})", "color": "#ef4444"},
        {"rec": worst_b_rec, "title": f"Worst Brightness Bias (Frame {worst_b_rec['file'].split('_')[2]})", "color": "#f59e0b"},
        {"rec": calib_rec, "title": f"Calibrated Boundary Frame (Frame {calib_rec['file'].split('_')[2]})", "color": "#10b981"}
    ]
    
    # -------------------------------------------------------------------------
    # PLOT 2: SNOW COMPARISON GRID (results/snow_characterization/snow_comparison_grid.png)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    labels = [("a", "b"), ("c", "d"), ("e", "f")]
    
    for idx, rf in enumerate(rep_frames):
        rec = rf["rec"]
        split = rec["split"]
        seq = rec["sequence"]
        f_name = rec["file"]
        
        path_dist = os.path.join(dataset_dir, "rgb_anon", condition, split, seq, f_name)
        ref_file = f_name.replace("_rgb_anon.png", "_rgb_ref_anon.png").replace("_rgb_anon.jpg", "_rgb_ref_anon.jpg")
        path_ref = os.path.join(dataset_dir, "rgb_anon", condition, f"{split}_ref", seq, ref_file)
        
        img_dist_bgr = cv2.imread(path_dist)
        img_clear_bgr = cv2.imread(path_ref)
        
        lbl_c, lbl_f = labels[idx]
        
        if img_dist_bgr is not None and img_clear_bgr is not None:
            img_dist_rgb = cv2.cvtColor(img_dist_bgr, cv2.COLOR_BGR2RGB)
            img_clear_rgb = cv2.cvtColor(img_clear_bgr, cv2.COLOR_BGR2RGB)
            
            # Clear reference
            axes[idx, 0].imshow(img_clear_rgb)
            axes[idx, 0].axis('off')
            axes[idx, 0].text(0.5, -0.06, f"({lbl_c}) Clear Reference (Sequence {seq}, Frame {f_name.split('_')[2]})", 
                              transform=axes[idx, 0].transAxes, ha='center', va='top', 
                              fontsize=11, fontweight='bold')
            
            # Snowy image
            axes[idx, 1].imshow(img_dist_rgb)
            axes[idx, 1].axis('off')
            axes[idx, 1].text(0.5, -0.06, f"({lbl_f}) {rf['title']}\n$\epsilon_c$: {rec['eps_c']:.4f}, $\epsilon_b$: {rec['eps_b']:.4f}", 
                              transform=axes[idx, 1].transAxes, ha='center', va='top', 
                              fontsize=11, fontweight='bold', color=rf["color"])
        else:
            print(f"Warning: Could not load images for sequence {seq}, frame {f_name}")
            axes[idx, 0].text(0.5, 0.5, "Image Missing", ha='center', va='center')
            axes[idx, 1].text(0.5, 0.5, "Image Missing", ha='center', va='center')
            
    plt.suptitle("ACDC Snow Threat Characterization: Spatial Road-Masked Qualitative & Quantitative Comparison", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(wspace=0.02, hspace=0.26)
    plt.savefig(output_comparison_grid, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Plot 2 (Snow Comparison Grid) to: {output_comparison_grid}")
    
    # Save combined JSON data
    with open(output_json_all, "w", encoding="utf-8") as f_json:
        json.dump({
            "total_count": total_combined,
            "calibrated_eps_c_5th": pct_c_5,
            "calibrated_eps_b_95th": pct_b_95,
            "worst_contrast_drop": worst_c_rec,
            "worst_brightness_bias": worst_b_rec,
            "calibrated_boundary_frame": calib_rec,
            "records": combined_records
        }, f_json, indent=4)

if __name__ == "__main__":
    main()
