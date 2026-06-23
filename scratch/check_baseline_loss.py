import pandas as pd
import numpy as np

csv_path = "datasets/carla_steering_e2e/clear/index.csv"
df = pd.read_csv(csv_path)

# Filter/balance like in train_carla_model.py
left = df[df['steering'] < -0.01]
right = df[df['steering'] > 0.01]
straight = df[(df['steering'] >= -0.01) & (df['steering'] <= 0.01)]

# Downsample straight
target_straight = int(0.6 * max(len(left), len(right)))
straight_sampled = straight.sample(n=target_straight, random_state=42)

balanced_df = pd.concat([left, right, straight_sampled])
steer = balanced_df['steering'].values

# Weighted MSE if prediction is 0.0
loss_zero = np.mean((1.0 + 35.0 * np.abs(steer)) * (0.0 - steer) ** 2)
print(f"Loss if predicting constant 0.0: {loss_zero:.6f}")

# Weighted MSE if prediction is mean of steer
mean_steer = np.mean(steer)
loss_mean = np.mean((1.0 + 35.0 * np.abs(steer)) * (mean_steer - steer) ** 2)
print(f"Mean steer value: {mean_steer:.6f}")
print(f"Loss if predicting constant mean: {loss_mean:.6f}")
