import pandas as pd
import numpy as np

csv_path = "datasets/carla_steering_e2e/clear/index.csv"
df = pd.read_csv(csv_path)

# Filter for DAgger frames
dagger_df = df[df['image_path'].str.contains('frame_dagger')]
normal_df = df[~df['image_path'].str.contains('frame_dagger')]

print(f"Total normal frames: {len(normal_df)}")
print(f"Total DAgger frames: {len(dagger_df)}")

print("\nNormal steering stats:")
print(normal_df['steering'].describe())

print("\nDAgger steering stats:")
print(dagger_df['steering'].describe())

# Check how many dagger frames have significant steer
print("\nDAgger frames with steer > 0.05:")
print(len(dagger_df[dagger_df['steering'].abs() > 0.05]))
print("DAgger frames with steer > 0.1:")
print(len(dagger_df[dagger_df['steering'].abs() > 0.1]))
