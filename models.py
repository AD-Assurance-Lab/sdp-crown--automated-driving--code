import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class CarlaSteeringNet(nn.Module):
    def __init__(self):
        super(CarlaSteeringNet, self).__init__()
        # Input: 3 channels (RGB), 60 height, 80 width (preserves 4:3 aspect ratio)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 60x80 -> 30x40
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 30x40 -> 15x20
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 15x20 -> 8x10
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()  # 8x10 -> 4x5
        )
        # Flattened size: 64 channels * 4 height * 5 width = 1280
        self.linear_layers = nn.Sequential(
            nn.Linear(1280, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1) # Single regression output for steering angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)

class CarlaSteeringExpertNet(nn.Module):
    def __init__(self):
        super(CarlaSteeringExpertNet, self).__init__()
        # Input: 3 channels (RGB), 60 height, 80 width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),  # 60x80 -> 30x40
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 30x40 -> 15x20
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 15x20 -> 8x10
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU()  # 8x10 -> 4x5
        )
        # Flattened size: 128 channels * 4 height * 5 width = 2560
        self.linear_layers = nn.Sequential(
            nn.Linear(2560, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 1) # Single regression output for steering angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)