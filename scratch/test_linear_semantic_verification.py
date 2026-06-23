import os
import gc
import sys
import json
import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Folder config
CODE_DIR = "/home/za/ad-assurance--workspace/sdp-crown--automated-driving--code"
sys.path.append(CODE_DIR)

from models import CarlaSteeringNet
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load model
weights_path = os.path.join(CODE_DIR, "models/archive/carla_steering_net_mixed.pth")
base_model = CarlaSteeringNet().to(device)
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

# 2. Linear Semantic Perturbation layer
class LinearSemanticPerturbationLayer(nn.Module):
    def __init__(self, nominal_image, spatial_mask=False):
        super(LinearSemanticPerturbationLayer, self).__init__()
        self.spatial_mask = spatial_mask
        
        # Flatten the nominal image to (14400,)
        _, c, h, w = nominal_image.shape
        flat_image = nominal_image.clone().view(-1)  # 14400
        num_pixels = flat_image.numel()
        
        # Generate the spatial Mask M (1 for road, 0 for sky)
        mask = torch.zeros((1, c, h, w), device=nominal_image.device)
        if spatial_mask:
            mask[:, :, h//2:, :] = 1.0  # Bottom half = 1.0
        else:
            mask[:, :, :, :] = 1.0      # Global weather (all pixels = 1.0)
            
        flat_mask = mask.view(-1)  # 14400
        
        # Create weights of shape (14400, 2):
        # Column 0: flat_image * flat_mask (contrast change eps_c applies only to road)
        # Column 1: flat_mask (brightness change eps_b applies only to road)
        weight = torch.stack([flat_image * flat_mask, flat_mask], dim=1)  # (14400, 2)
        bias = flat_image.clone()  # (14400,)
        
        # Create the Linear layer
        self.fc = nn.Linear(2, num_pixels)
        self.fc.weight.data.copy_(weight)
        self.fc.bias.data.copy_(bias)
        
        self.out_shape = (1, c, h, w)

    def forward(self, eps):
        # eps shape: (1, 2)
        out_flat = self.fc(eps)  # (1, 14400)
        out_img = out_flat.view(self.out_shape)  # (1, 3, 60, 80)
        return torch.clamp(out_img, 0.0, 1.0)

class LinearSemanticVerifiedNetwork(nn.Module):
    def __init__(self, base_model, nominal_image, condition_name=""):
        super(LinearSemanticVerifiedNetwork, self).__init__()
        use_mask = condition_name.lower() in ["snow", "rain"]
        self.semantic_layer = LinearSemanticPerturbationLayer(nominal_image, spatial_mask=use_mask)
        self.base_model = base_model

    def forward(self, eps):
        degraded_image = self.semantic_layer(eps)
        return self.base_model(degraded_image)

# 3. Dataset Setup
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
weather = "rain"

for i, (image, label) in enumerate(test_loader):
    image = image.to(device)
    with torch.no_grad():
        nominal_steering = base_model(image).item()
        
    wrapped_model = LinearSemanticVerifiedNetwork(base_model, image, condition_name=weather).to(device)
    wrapped_model.eval()
    
    eps_nominal = torch.zeros(1, 2).to(device)
    eps_L = torch.tensor([[eps_c_min, eps_b_min]]).to(device)
    eps_U = torch.tensor([[eps_c_max, eps_b_max]]).to(device)
    
    ptb = PerturbationLpNorm(norm=float("inf"), eps=None, x_L=eps_L, x_U=eps_U)
    bounded_eps = BoundedTensor(eps_nominal, ptb)
    
    # auto_LiRPA in patches mode with alpha-CROWN optimization
    bound_opts = {
        'conv_mode': 'patches',
        'optimize_bound_args': {
            'iteration': 50,
            'lr_alpha': 0.5,
            'early_stop_patience': 10,
            'fix_interm_bounds': True,
            'enable_opt_interm_bounds': False,
            'enable_SDP_crown': False
        }
    }
    lirpa_model = BoundedModule(wrapped_model, bounded_eps, device=device, verbose=0, bound_opts=bound_opts)
    
    # Compute bounds using alpha-CROWN
    crown_lb, crown_ub = lirpa_model.compute_bounds(
        x=(bounded_eps,),
        method='CROWN-Optimized',
        bound_lower=True,
        bound_upper=True
    )
    
    lb_val = crown_lb.item()
    ub_val = crown_ub.item()
    print(f"Frame {i:02d}: Nominal: {nominal_steering:+.4f} | Bounds: [{lb_val:+.4f}, {ub_val:+.4f}]")
