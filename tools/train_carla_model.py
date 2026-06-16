#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add parent directory to path so we can import models.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CarlaSteeringNet

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is required to run the training script.")
    sys.exit(1)

class CarlaDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: List of tuples (image_absolute_path, steering_angle)
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, steer = self.data_list[idx]
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to CarlaSteeringNet input shape (60 height, 80 width)
        img = cv2.resize(img, (80, 60))
        
        # Normalize to [0.0, 1.0] and convert to float32
        img = img.astype(np.float32) / 255.0
        
        # Permute to (Channels, Height, Width)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        return img_tensor, torch.tensor([steer], dtype=torch.float32)

def collect_data_from_csv(root_dir, weather_folders):
    data_list = []
    
    for weather in weather_folders:
        folder_path = os.path.join(root_dir, weather)
        csv_path = os.path.join(folder_path, "index.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: Index CSV not found at {csv_path}. Skipping folder.")
            continue
            
        print(f"Reading metadata from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            img_rel_path = row['image_path']
            # Resolve relative image path to absolute/full path
            img_abs_path = os.path.join(folder_path, img_rel_path)
            steer = row['steering']
            
            if os.path.exists(img_abs_path):
                data_list.append((img_abs_path, steer))
            else:
                print(f"Warning: Image file not found at {img_abs_path}")
                
    print(f"Total valid image samples aggregated: {len(data_list)}")
    return data_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train CarlaSteeringNet on CARLA steering dataset")
    parser.add_argument("--data-root", default="datasets/carla_testing", help="Root directory of collected CARLA data")
    parser.add_argument("--mode", default="clear", choices=["clear", "mixed"], help="Dataset split mode (clear only, or all weather combined)")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--val-split", default=0.2, type=float, help="Validation split ratio")
    parser.add_argument("--save-dir", default="models", help="Directory for saving model checkpoints")
    args = parser.parse_args()

    # Determine weather folders to aggregate
    if args.mode == "clear":
        weather_folders = ["clear"]
    else:
        weather_folders = ["clear", "rain", "fog", "night"]

    # Collect data list
    data_list = collect_data_from_csv(args.data_root, weather_folders)
    if not data_list:
        print("Error: No training data found. Make sure to run the data collection script first.")
        sys.exit(1)
        
    # Split into train and validation sets
    train_data, val_data = train_test_split(data_list, test_size=args.val_split, random_state=42)
    print(f"Training samples: {len(train_data)} | Validation samples: {len(val_data)}")
    
    # Initialize Datasets and Loaders
    train_dataset = CarlaDataset(train_data)
    val_dataset = CarlaDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # Initialize model
    model = CarlaSteeringNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create models directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    save_filename = f"carla_steering_net_{args.mode}.pth"
    save_path = os.path.join(args.save_dir, save_filename)
    
    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model checkpoint to {save_path}")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
