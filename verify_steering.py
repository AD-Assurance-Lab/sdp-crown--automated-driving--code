import os
import gc
import json
import torch
import argparse
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

from models import CarlaSteeringNet, MicroPilotNet
from udacity_dataset import UdacityDataset
from semantic_layers import SemanticVerifiedNetwork
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# ==============================================================================
# CONTROL PANEL
# Modify these values to configure the default behavior when running the script
# directly (e.g. by pressing the "Play" button in VS Code).
# ==============================================================================
CONFIG = {
    "weather": "fog",                     # Options: "fog", "night", "snow", "rain"
    "num_frames": 10,                      # Number of continuous frames to verify
    "safe_deviation": 0.1,                 # Maximum allowed deviation (radians)
    "bounds_file": "results/carla_physics_bounds.json", # Path to physical bounds JSON
    
    # Overrides (set to a float to override, or None to use loaded bounds)
    "eps_c_min": None,
    "eps_c_max": None,
    "eps_b_min": None,
    "eps_b_max": None,
    
    # File Paths
    "model_type": "CarlaSteeringNet",      # Options: "CarlaSteeringNet", "MicroPilotNet"
    "csv_path": None,                      # Auto-selected based on model_type
    "img_dir": None,                       # Only used for Udacity
    "weights_path": None,                  # Auto-selected based on model_type
    "output_results": "results/steering_verification/verification_results.json",
    
    # Runtime settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Use "cuda" for speed, "cpu" if you hit OOM
    "method": "CROWN",                    # Options: "CROWN", "SDP-CROWN", "IBP", "CROWN-IBP", "alpha-CROWN"
    "iterations": 100,                    # Optimization iterations for SDP-CROWN / alpha-CROWN
    "lr_alpha": 0.5,
    "lr_lambda": 0.05
}
# ==============================================================================

class CarlaVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.data_list = []
        base_dir = os.path.dirname(csv_file)
        for _, row in df.iterrows():
            img_path = os.path.join(base_dir, row["image_path"])
            steer = row["steering"]
            self.data_list.append((img_path, steer))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, steer = self.data_list[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop top 180px and bottom 80px (original size is 480x640)
        img = img[180:400, :]
        
        img = cv2.resize(img, (80, 60))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor, torch.tensor([steer], dtype=torch.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="SDP-CROWN Autonomous Driving Steering Verification")
    parser.add_argument(
        "--model_type",
        choices=["CarlaSteeringNet", "MicroPilotNet"],
        default=CONFIG["model_type"],
        help="Target model type for verification"
    )
    parser.add_argument(
        "--csv_path",
        default=CONFIG["csv_path"],
        help="Path to dataset log CSV (auto-selected if None)"
    )
    parser.add_argument(
        "--img_dir",
        default=CONFIG["img_dir"],
        help="Path to Udacity images directory (only used for Udacity)"
    )
    parser.add_argument(
        "--weights_path",
        default=CONFIG["weights_path"],
        help="Path to pre-trained weights (auto-selected if None)"
    )
    parser.add_argument(
        "--weather",
        choices=["fog", "night", "snow", "rain"],
        default=CONFIG["weather"],
        help="Weather condition to verify against"
    )
    parser.add_argument(
        "--bounds_file",
        default=CONFIG["bounds_file"],
        help="Path to load characterized physical bounds JSON from"
    )
    parser.add_argument(
        "--eps_c_min", type=float, default=CONFIG["eps_c_min"],
        help="Override minimum contrast drop (epsilon_c). If None, loaded from bounds_file."
    )
    parser.add_argument(
        "--eps_c_max", type=float, default=CONFIG["eps_c_max"],
        help="Override maximum contrast drop (epsilon_c). If None, loaded from bounds_file."
    )
    parser.add_argument(
        "--eps_b_min", type=float, default=CONFIG["eps_b_min"],
        help="Override minimum brightness bias (epsilon_b). If None, loaded from bounds_file."
    )
    parser.add_argument(
        "--eps_b_max", type=float, default=CONFIG["eps_b_max"],
        help="Override maximum brightness bias (epsilon_b). If None, loaded from bounds_file."
    )
    parser.add_argument(
        "--safe_deviation",
        type=float,
        default=CONFIG["safe_deviation"],
        help="Maximum allowed deviation (radians) from nominal steering path"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=CONFIG["num_frames"],
        help="Number of continuous frames to verify"
    )
    parser.add_argument(
        "--output_results",
        default=CONFIG["output_results"],
        help="Path to save the verification results as a JSON file"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=CONFIG["device"],
        help="Device to run verification on (use cpu if GPU memory is insufficient)"
    )
    parser.add_argument(
        "--method",
        choices=["CROWN", "SDP-CROWN", "IBP", "CROWN-IBP", "alpha-CROWN"],
        default=CONFIG["method"],
        help="Verification method to use (CROWN, SDP-CROWN, IBP, CROWN-IBP, alpha-CROWN)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=CONFIG["iterations"],
        help="Number of optimization iterations for SDP-CROWN / alpha-CROWN"
    )
    parser.add_argument(
        "--lr_alpha",
        type=float,
        default=CONFIG["lr_alpha"],
        help="Learning rate for alpha parameters in optimized bounds"
    )
    parser.add_argument(
        "--lr_lambda",
        type=float,
        default=CONFIG["lr_lambda"],
        help="Learning rate for lambda parameters in SDP-CROWN"
    )
    parser.add_argument(
        "--opt_interm",
        action="store_true",
        help="Enable intermediate bound optimization (greatly increases VRAM/RAM usage)"
    )
    return parser.parse_args()


def load_bounds(args):
    # Default fallback values (e.g. Rain reference values)
    eps_c_min, eps_c_max = -0.0279, 0.0
    eps_b_min, eps_b_max = 0.0, 0.1003
    
    loaded_from_file = False
    
    # 1. Try to load from bounds_file
    if os.path.exists(args.bounds_file):
        try:
            with open(args.bounds_file, "r") as f:
                data = json.load(f)
            
            # Check if it's a multi-condition dictionary or single-condition
            weather_data = None
            if args.weather in data:
                weather_data = data[args.weather]
            elif data.get("condition", "").lower() == args.weather.lower():
                weather_data = data
                
            if weather_data is not None:
                eps_c_min = weather_data.get("recommended_eps_c_min", weather_data.get("eps_c_min", eps_c_min))
                eps_c_max = weather_data.get("recommended_eps_c_max", weather_data.get("eps_c_max", eps_c_max))
                eps_b_min = weather_data.get("recommended_eps_b_min", weather_data.get("eps_b_min", eps_b_min))
                eps_b_max = weather_data.get("recommended_eps_b_max", weather_data.get("eps_b_max", eps_b_max))
                loaded_from_file = True
                print(f"Loaded bounds for '{args.weather}' from {args.bounds_file}:")
            else:
                print(f"Warning: Bounds file does not contain key/condition '{args.weather}'. Using default fallback values.")
        except Exception as e:
            print(f"Warning: Failed to load bounds file: {e}. Using default fallback values.")
    else:
        print(f"Warning: Bounds file '{args.bounds_file}' not found. Using default fallback values.")
        
    # 2. Command line overrides
    if args.eps_c_min is not None:
        eps_c_min = args.eps_c_min
    if args.eps_c_max is not None:
        eps_c_max = args.eps_c_max
    if args.eps_b_min is not None:
        eps_b_min = args.eps_b_min
    if args.eps_b_max is not None:
        eps_b_max = args.eps_b_max
        
    print(f"Epsilon contrast drop: [{eps_c_min:.4f}, {eps_c_max:.4f}]")
    print(f"Epsilon brightness bias: [{eps_b_min:.4f}, {eps_b_max:.4f}]")
    
    return eps_c_min, eps_c_max, eps_b_min, eps_b_max
 
def verify_regression():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Resolve default paths based on model_type
    model_type = args.model_type
    if model_type == "CarlaSteeringNet":
        weights_path = args.weights_path if args.weights_path is not None else "models/carla_steering_net_clear.pth"
        csv_path = args.csv_path if args.csv_path is not None else "datasets/carla_steering_e2e/clear/index.csv"
    else:
        weights_path = args.weights_path if args.weights_path is not None else "models/pilotnet_udacity.pth"
        csv_path = args.csv_path if args.csv_path is not None else "datasets/debugging/Udacity/self_driving_car_dataset_jungle/driving_log.csv"
        img_dir = args.img_dir if args.img_dir is not None else "datasets/debugging/Udacity/self_driving_car_dataset_jungle/IMG/"

    # Load physical bounds
    eps_c_min, eps_c_max, eps_b_min, eps_b_max = load_bounds(args)

    # 1. Load the Pre-Trained Steering Baseline
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        
    if model_type == "CarlaSteeringNet":
        base_model = CarlaSteeringNet().to(device)
    else:
        base_model = MicroPilotNet().to(device)
        
    # Use weights_only=True to resolve security warning (requires torch >= 2.4, fallback for older)
    try:
        base_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except (TypeError, RuntimeError):
        base_model.load_state_dict(torch.load(weights_path, map_location=device))
    base_model.eval()

    # 2. Load the Dataset
    csv_path = os.path.abspath(os.path.expanduser(csv_path))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset log CSV not found at: {csv_path}")
        
    if model_type == "CarlaSteeringNet":
        dataset = CarlaVerificationDataset(csv_file=csv_path)
    else:
        img_dir = os.path.abspath(os.path.expanduser(img_dir))
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Udacity images directory not found: {img_dir}")
        dataset = UdacityDataset(csv_file=csv_path, img_dir=img_dir)
    
    # Limit number of frames
    num_frames = min(args.num_frames, len(dataset))
    print(f"Verifying {num_frames} frames of sequence...")
    # NOTE: batch_size=1 is required for the semantic layers + dense matrix bounds tracking
    test_loader = DataLoader(Subset(dataset, range(num_frames)), batch_size=1, shuffle=False)

    total_frames = 0
    safe_frames = 0
    frame_results = []

    print(f"\n{'='*55}")
    print(f" Starting {args.weather.upper()} AV Steering Verification ({args.method}) ".center(55, "="))
    print(f"Safety Corridor: +/- {args.safe_deviation} rad deviation from nominal path")
    print(f"{'='*55}\n")

    for i, (image, label) in enumerate(test_loader):
        image = image.to(device)
        
        # Calculate nominal steering angle in clear weather (without any perturbation)
        with torch.no_grad():
            nominal_steering = base_model(image).item() 

        # 3. Wrap model with custom SemanticWeather layer
        wrapped_model = SemanticVerifiedNetwork(base_model, image, condition_name=args.weather).to(device)

        # 4. Define the disturbance hyperbox (contrast drop, brightness bias)
        eps_nominal = torch.zeros(1, 2).to(device)
        eps_L = torch.tensor([[eps_c_min, eps_b_min]]).to(device)
        eps_U = torch.tensor([[eps_c_max, eps_b_max]]).to(device)

        ptb = PerturbationLpNorm(norm=float("inf"), eps=None, x_L=eps_L, x_U=eps_U)
        bounded_eps = BoundedTensor(eps_nominal, ptb)

        # Initialize the auto_LiRPA BoundedModule in dense matrix mode
        if args.method in ["SDP-CROWN", "alpha-CROWN"]:
            bound_opts = {
                'conv_mode': 'matrix',
                'optimize_bound_args': {
                    'iteration': args.iterations,
                    'lr_alpha': args.lr_alpha,
                    'early_stop_patience': 20,
                    'fix_interm_bounds': True,
                    'enable_opt_interm_bounds': args.opt_interm,
                    'enable_SDP_crown': (args.method == "SDP-CROWN"),
                    'lr_lambda': args.lr_lambda
                }
            }
            method = 'CROWN-Optimized'
        elif args.method == "CROWN-IBP":
            bound_opts = {'conv_mode': 'matrix'}
            method = 'CROWN-IBP'
        elif args.method == "IBP":
            bound_opts = {'conv_mode': 'matrix'}
            method = 'IBP'
        else:
            bound_opts = {'conv_mode': 'matrix'}
            method = 'CROWN'

        lirpa_model = BoundedModule(
            wrapped_model, 
            bounded_eps, 
            device=device, 
            verbose=0, 
            bound_opts=bound_opts
        )

        # 5. Compute regression bounds
        crown_lb, crown_ub = lirpa_model.compute_bounds(
            x=(bounded_eps,), 
            method=method, 
            bound_lower=True, 
            bound_upper=True
        )

        lb_val = crown_lb.item()
        ub_val = crown_ub.item()

        # 6. Evaluate safety corridor
        lower_limit = nominal_steering - args.safe_deviation
        upper_limit = nominal_steering + args.safe_deviation

        # Check for vacuous bounds (numerical explosion)
        if abs(lb_val) > 100 or abs(ub_val) > 100:
            status = "VACUOUS"
            print(f"Frame {i:02d}: WARNING: Numerical bounds exploded. Try tighter input epsilon.")
        else:
            # Safe if the worst-case weather bounds do not exceed safety corridor limits
            is_safe = (lb_val >= lower_limit) and (ub_val <= upper_limit)

            total_frames += 1
            if is_safe:
                safe_frames += 1
                status = "SAFE"
            else:
                status = "FAILED"

        print(f"Frame {i:02d}: Nominal: {nominal_steering:+.4f} | Bounds: [{lb_val:+.4f}, {ub_val:+.4f}] | Status: {status}")

        frame_results.append({
            "frame_idx": i,
            "nominal_steering": nominal_steering,
            "lower_bound": lb_val,
            "upper_bound": ub_val,
            "lower_corridor": lower_limit,
            "upper_corridor": upper_limit,
            "status": status
        })

        # Explicit garbage collection to prevent memory leaks (Mandated by DIS.pdf)
        del crown_lb
        del crown_ub
        del lirpa_model
        del wrapped_model
        del bounded_eps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    safety_rate = (safe_frames / total_frames) * 100 if total_frames > 0 else 0.0
    print("\n" + "="*55)
    print(f" Final {args.weather.upper()} Certified Safety: {safety_rate:.1f}% ".center(55, "="))
    print("="*55)

    # Save results to output file
    output_dir = os.path.dirname(args.output_results)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    summary = {
        "weather": args.weather,
        "eps_c_min": eps_c_min,
        "eps_c_max": eps_c_max,
        "eps_b_min": eps_b_min,
        "eps_b_max": eps_b_max,
        "safe_deviation": args.safe_deviation,
        "total_frames": total_frames,
        "safe_frames": safe_frames,
        "safety_rate": safety_rate,
        "frames": frame_results
    }
    
    with open(args.output_results, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Successfully saved verification results to: {args.output_results}")

if __name__ == "__main__":
    verify_regression()
