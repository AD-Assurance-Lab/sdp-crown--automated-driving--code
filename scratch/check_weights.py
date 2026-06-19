import os
import sys
import torch
import numpy as np

# Add root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CarlaSteeringNet, MicroPilotNet

def analyze_weights(model, name):
    print(f"\n=========================================")
    print(f" Analyzing Weights: {name}")
    print(f"=========================================")
    
    total_params = 0
    conv_w_norms = []
    fc_w_norms = []
    
    for layer_name, param in model.named_parameters():
        if "weight" in layer_name:
            w = param.detach().cpu().numpy()
            total_params += w.size
            # L1 norm of elements, max abs value, std dev
            w_abs = np.abs(w)
            w_max = np.max(w_abs)
            w_mean = np.mean(w_abs)
            w_std = np.std(w)
            
            if "conv" in layer_name:
                conv_w_norms.append((layer_name, w.shape, w_max, w_mean, w_std))
            elif "linear" in layer_name or "fc" in layer_name:
                # For linear layers, row L1 norms determine Lipschitz bound (maximum input-to-output gain)
                row_l1_norms = np.sum(w_abs, axis=1)
                max_row_l1 = np.max(row_l1_norms)
                fc_w_norms.append((layer_name, w.shape, w_max, w_mean, w_std, max_row_l1))

    print(f"Total parameters: {total_params:,}")
    print("\n--- Convolutional Layers ---")
    for item in conv_w_norms:
        print(f"{item[0]} | Shape: {item[1]} | Max: {item[2]:.4f} | Mean: {item[3]:.4f} | Std: {item[4]:.4f}")
        
    print("\n--- Fully Connected Layers ---")
    for item in fc_w_norms:
        print(f"{item[0]} | Shape: {item[1]} | Max: {item[2]:.4f} | Mean: {item[3]:.4f} | Std: {item[4]:.4f} | Max Row L1 (Lipschitz): {item[5]:.4f}")

# 1. Load MicroPilotNet
pn = MicroPilotNet()
pn.load_state_dict(torch.load("models/pilotnet_udacity.pth", map_location="cpu"))
analyze_weights(pn, "MicroPilotNet (Udacity)")

# 2. Load CarlaSteeringNet Clear
carla_clear = CarlaSteeringNet()
carla_clear.load_state_dict(torch.load("models/carla_steering_net_clear.pth", map_location="cpu"))
analyze_weights(carla_clear, "CarlaSteeringNet (Clear-Only)")

# 3. Load CarlaSteeringNet Mixed
carla_mixed = CarlaSteeringNet()
carla_mixed.load_state_dict(torch.load("models/carla_steering_net_mixed.pth", map_location="cpu"))
analyze_weights(carla_mixed, "CarlaSteeringNet (Mixed-Weather)")
