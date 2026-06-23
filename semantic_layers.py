import torch
import torch.nn as nn

class SemanticPerturbationLayer(nn.Module):
    def __init__(self, nominal_image, spatial_mask=False):
        super(SemanticPerturbationLayer, self).__init__()
        self.spatial_mask = spatial_mask
        
        # Flatten the nominal image to (14400,)
        _, c, h, w = nominal_image.shape
        flat_image = nominal_image.clone().view(-1)  # 14400
        num_pixels = flat_image.numel()
        
        # Generate the spatial Mask M (1 for road, 0 for sky)
        mask = torch.zeros((1, c, h, w), device=nominal_image.device)
        if spatial_mask:
            mask[:, :, h//2:, :] = 1.0  # Bottom half (road region) = 1.0
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

class SemanticVerifiedNetwork(nn.Module):
    def __init__(self, base_model, nominal_image, condition_name=""):
        super(SemanticVerifiedNetwork, self).__init__()
        # Automatically toggle spatial masking based on the weather condition
        use_mask = condition_name.lower() in ["snow", "rain"]
        self.semantic_layer = SemanticPerturbationLayer(nominal_image, spatial_mask=use_mask)
        self.base_model = base_model

    def forward(self, eps):
        degraded_image = self.semantic_layer(eps)
        return self.base_model(degraded_image)
