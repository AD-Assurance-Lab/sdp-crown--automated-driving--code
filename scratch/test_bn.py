import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2

class CarlaSteeringNetBN(nn.Module):
    def __init__(self):
        super(CarlaSteeringNetBN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1280, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.train_carla_model import CarlaDataset, collect_data_from_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_list = collect_data_from_csv("datasets/carla_steering_e2e", ["clear"])
train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)

train_dataset = CarlaDataset(train_data, is_training=True)
val_dataset = CarlaDataset(val_data, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

print("\n--- Training CarlaSteeringNetBN with LR = 5e-4 ---")
model = CarlaSteeringNetBN().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for epoch in range(10):
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        weights = 1.0 + 35.0 * torch.abs(targets)
        loss = torch.mean(weights * (outputs - targets) ** 2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            weights = 1.0 + 35.0 * torch.abs(targets)
            loss = torch.mean(weights * (outputs - targets) ** 2)
            val_loss += loss.item() * images.size(0)
        val_loss /= len(val_dataset)
        
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
