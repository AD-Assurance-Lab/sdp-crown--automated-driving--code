import pandas as pd
import numpy as np
import os

for mode in ['clear', 'rain', 'fog', 'night']:
    csv_path = f"datasets/carla_steering_e2e/{mode}/index.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        steer = df['steering'].values
        # Balance like in train_carla_model.py
        left = df[df['steering'] < -0.01]
        right = df[df['steering'] > 0.01]
        straight = df[(df['steering'] >= -0.01) & (df['steering'] <= 0.01)]
        target_straight = int(0.6 * max(len(left), len(right)))
        straight_sampled = straight.sample(n=min(len(straight), target_straight), random_state=42)
        balanced_df = pd.concat([left, right, straight_sampled])
        steer_balanced = balanced_df['steering'].values
        
        loss_zero = np.mean((1.0 + 35.0 * np.abs(steer_balanced)) * (0.0 - steer_balanced) ** 2)
        print(f"[{mode}] count: {len(steer_balanced)} | steer mean: {np.mean(steer_balanced):.6f} | steer std: {np.std(steer_balanced):.6f} | baseline loss: {loss_zero:.6f}")
