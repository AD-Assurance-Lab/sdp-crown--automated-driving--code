import torch
import cv2
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CarlaSteeringExpertNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CarlaSteeringExpertNet().to(device)
model.load_state_dict(torch.load("models/carla_expert_clear.pth", map_location=device))
model.eval()

csv_path = "datasets/carla_steering_e2e/clear/index.csv"
df = pd.read_csv(csv_path)

# Look at 20 frames from different parts of the dataset
for i in range(0, len(df), len(df)//20):
    row = df.iloc[i]
    img_path = os.path.join("datasets/carla_steering_e2e/clear", row['image_path'])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[180:400, :]
    img = cv2.resize(img, (80, 60))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_steer = float(model(img_tensor).squeeze().cpu().item())
    
    print(f"Frame {row['frame']} | True: {row['steering']:.4f} | Pred: {pred_steer:.4f} | Diff: {abs(row['steering'] - pred_steer):.4f}")
