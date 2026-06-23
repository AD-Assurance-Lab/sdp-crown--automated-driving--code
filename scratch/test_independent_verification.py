import os
import gc
import json
import torch
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

import sys
CODE_DIR = "/home/za/ad-assurance--workspace/sdp-crown--automated-driving--code"
sys.path.append(CODE_DIR)

from models import CarlaSteeringExpertNet
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# Folder config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load model
weights_path = os.path.join(CODE_DIR, "models/carla_steering_net_mixed.pth")
base_model = CarlaSteeringExpertNet().to(device)
base_model.load_state_dict(torch.load(weights_path, map_location=device))
base_model.eval()

# Replace Dropout
def replace_dropout(m):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.Dropout):
            setattr(m, name, torch.nn.Identity())
        else:
            replace_dropout(child)
replace_dropout(base_model)

# Fold BatchNorm
def fuse_conv_bn_eval(conv, bn):
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias
    scale = gamma / torch.sqrt(var + eps)
    w_new = w * scale.view(-1, 1, 1, 1)
    b_new = (b - mean) * scale + beta
    fused_conv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    ).to(w.device)
    fused_conv.weight.data.copy_(w_new)
    fused_conv.bias.data.copy_(b_new)
    return fused_conv

def fuse_model_conv_bn(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Sequential):
            new_layers = []
            i = 0
            while i < len(child):
                layer = child[i]
                if i + 1 < len(child) and isinstance(layer, torch.nn.Conv2d) and isinstance(child[i+1], torch.nn.BatchNorm2d):
                    new_layers.append(fuse_conv_bn_eval(layer, child[i+1]))
                    new_layers.append(torch.nn.Identity())
                    i += 2
                else:
                    new_layers.append(layer)
                    i += 1
            for idx, m in enumerate(new_layers):
                child[idx] = m
        else:
            fuse_model_conv_bn(child)

fuse_model_conv_bn(base_model)

# 2. Dataset Setup
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[180:400, :]
        img = cv2.resize(img, (80, 60))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor, torch.tensor([steer], dtype=torch.float32)

csv_path = os.path.join(CODE_DIR, "datasets/carla_steering_e2e/clear/index.csv")
dataset = CarlaVerificationDataset(csv_path)
test_loader = DataLoader(Subset(dataset, range(5)), batch_size=1, shuffle=False)

# Weather bounds (ACDC Rain)
eps_c_min, eps_c_max = -0.4337, 0.0
eps_b_min, eps_b_max = 0.0, 0.1013
use_mask = True  # Rain uses spatial masking

for i, (image, label) in enumerate(test_loader):
    image = image.to(device)
    with torch.no_grad():
        nominal_steering = base_model(image).item()
        
    # Calculate pixel-level lower and upper bounds
    lower_image = image * (1.0 + eps_c_min) + eps_b_min
    upper_image = image * (1.0 + eps_c_max) + eps_b_max
    
    if use_mask:
        _, c, h, w = image.shape
        mask = torch.zeros((1, c, h, w), device=image.device)
        mask[:, :, h//2:, :] = 1.0  # Bottom half (road region) is perturbed
        lower_image = image * (1.0 - mask) + lower_image * mask
        upper_image = image * (1.0 - mask) + upper_image * mask
        
    x_L = torch.clamp(lower_image, 0.0, 1.0)
    x_U = torch.clamp(upper_image, 0.0, 1.0)
    
    # Define perturbation
    ptb = PerturbationLpNorm(norm=float("inf"), eps=None, x_L=x_L, x_U=x_U)
    bounded_image = BoundedTensor(image, ptb)
    
    # auto_LiRPA in patches mode
    bound_opts = {'conv_mode': 'patches'}
    lirpa_model = BoundedModule(base_model, bounded_image, device=device, verbose=0, bound_opts=bound_opts)
    
    # Compute bounds
    crown_lb, crown_ub = lirpa_model.compute_bounds(
        x=(bounded_image,),
        method='CROWN',
        bound_lower=True,
        bound_upper=True
    )
    
    lb_val = crown_lb.item()
    ub_val = crown_ub.item()
    print(f"Frame {i:02d}: Nominal: {nominal_steering:+.4f} | Bounds: [{lb_val:+.4f}, {ub_val:+.4f}]")
