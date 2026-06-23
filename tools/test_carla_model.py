#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import random
import queue
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt

# Add parent directory and CARLA PythonAPI to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CarlaSteeringNet, CarlaSteeringExpertNet
from weather_config import set_weather_profile, apply_vehicle_lights

carla_root = "/home/za/carla"
try:
    sys.path.append(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla'))[0])
    sys.path.append(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    )))[0])
except IndexError:
    pass

try:
    import carla
except ImportError:
    print("Error: Could not import CARLA Python library. Ensure CARLA is installed.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Closed-Loop Evaluation for Steering Neural Networks")
    parser.add_argument("--host", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to")
    parser.add_argument("--map", default="Town01", help="Name of the CARLA map/town to load")
    parser.add_argument("--weather", default="clear", choices=["clear", "rain", "fog", "night"], help="Weather profile")
    parser.add_argument("--model-path", default="models/carla_expert_clear.pth", help="Path to trained PyTorch steering model")
    parser.add_argument("--num-frames", default=1000, type=int, help="Number of simulation frames to run evaluation")
    parser.add_argument("--width", default=640, type=int, help="Image width")
    parser.add_argument("--height", default=480, type=int, help="Image height")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--spawn-point-idx", default=12, type=int, help="Index of spawn point for the ego vehicle")
    parser.add_argument("--save-plot", default="results/carla_ai_model_testing/carla_closed_loop_evaluation.png", help="Path to save the validation plot")
    parser.add_argument("--save-csv", default="results/carla_ai_model_testing/carla_closed_loop_evaluation.csv", help="Path to save log CSV data")
    parser.add_argument("--model-type", default="CarlaSteeringExpertNet", choices=["CarlaSteeringNet", "CarlaSteeringExpertNet"], help="Model architecture type")
    parser.add_argument("--start-frame", default=0, type=int, help="Frame index on reference trajectory to spawn ego vehicle at")
    return parser.parse_args()

def set_weather(world, profile):
    set_weather_profile(world, profile)

def sensor_callback(image, image_queue):
    image_queue.put(image)

def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} onto device: {device}...")
    if args.model_type == "CarlaSteeringNet":
        model = CarlaSteeringNet().to(device)
    else:
        model = CarlaSteeringExpertNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    actor_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    
    eval_records = []
    
    # Load reference steering dataset to get nominal steering values and coordinates at each time step
    ref_csv_path = None
    if args.map == "Town04":
        small_baseline_path = os.path.join("results/carla_ai_model_testing", "town04_small_mixed_clear.csv")
        if os.path.exists(small_baseline_path):
            ref_csv_path = small_baseline_path
        else:
            town04_baseline_path = os.path.join("results/carla_ai_model_testing", "town04_clear_only_clear.csv")
            if os.path.exists(town04_baseline_path):
                ref_csv_path = town04_baseline_path
            
    if ref_csv_path is None:
        ref_csv_path = os.path.join("datasets/carla_steering_e2e", args.weather, "index.csv")

    if os.path.exists(ref_csv_path):
        print(f"Loading reference nominal dataset from: {ref_csv_path}")
        ref_df = pd.read_csv(ref_csv_path)
        # Handle column naming variations (nominal_steer vs steering)
        if "steering" in ref_df.columns:
            nominal_steer_list = ref_df["steering"].tolist()
        elif "nominal_steer" in ref_df.columns:
            nominal_steer_list = ref_df["nominal_steer"].tolist()
        else:
            nominal_steer_list = [0.0] * len(ref_df)
            
        has_ref_coords = "x" in ref_df.columns and "y" in ref_df.columns
        if has_ref_coords:
            ref_x_list = ref_df["x"].tolist()
            ref_y_list = ref_df["y"].tolist()
        else:
            ref_x_list = [0.0] * args.num_frames
            ref_y_list = [0.0] * args.num_frames
    else:
        print(f"Warning: Reference CSV not found at {ref_csv_path}. Defaulting nominal steering to 0.0.")
        nominal_steer_list = [0.0] * args.num_frames
        ref_x_list = [0.0] * args.num_frames
        ref_y_list = [0.0] * args.num_frames
        has_ref_coords = False
    
    try:
        # Load map (always reload to ensure a clean slate and delete previous actors)
        print(f"Loading map {args.map}...")
        world = client.load_world(args.map)
        set_weather(world, args.weather)
        
        # Set synchronous mode
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        
        # Align frequency: Town04 models were trained at 5 Hz, Town01 at 10 Hz
        fixed_dt = 0.2 if args.map == "Town04" else 0.1
        settings.fixed_delta_seconds = fixed_dt
        world.apply_settings(settings)
        print(f"Synchronous mode enabled at {1.0/fixed_dt:.1f} Hz.")
        
        blueprint_library = world.get_blueprint_library()
        
        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        if has_ref_coords and args.map == "Town04":
            # Spawn ego vehicle at the specified start frame of reference trajectory
            idx = min(args.start_frame, len(ref_x_list) - 1)
            loc_x = ref_x_list[idx]
            loc_y = ref_y_list[idx]
            
            # Estimate heading direction (yaw) using forward differences
            if idx + 1 < len(ref_x_list):
                dy = ref_y_list[idx + 1] - ref_y_list[idx]
                dx = ref_x_list[idx + 1] - ref_x_list[idx]
                yaw = float(np.degrees(np.arctan2(dy, dx)))
            else:
                yaw = 0.0
                
            spawn_point = carla.Transform(
                carla.Location(x=loc_x, y=loc_y, z=1.0),
                carla.Rotation(yaw=yaw)
            )
            print(f"Spawning ego vehicle at reference trajectory frame {idx}: {spawn_point.location} with yaw {yaw:.2f} degrees")
        else:
            spawn_points = world.get_map().get_spawn_points()
            if not spawn_points:
                spawn_point = carla.Transform()
                print("Warning: No spawn points found on map. Using default Transform.")
            else:
                spawn_point_idx = args.spawn_point_idx % len(spawn_points)
                spawn_point = spawn_points[spawn_point_idx]
                spawn_point.location.z += 0.5
                print(f"Spawn point index {spawn_point_idx} at location {spawn_point.location}")
            
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned ego vehicle: {ego_vehicle.type_id}")
        
        # Turn headlights ON/OFF based on weather (shared config)
        apply_vehicle_lights(ego_vehicle, args.weather)
        
        
        # Spawn front camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '90')
        
        # Hood camera placement (hood height ~1.2m, x=1.6m forward)
        camera_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(camera)
        print("Spawned front RGB camera.")
        
        # Setup queue
        image_queue = queue.Queue()
        camera.listen(lambda img: sensor_callback(img, image_queue))
        
        # Get spectator actor to track vehicle position
        spectator = world.get_spectator()
        
        # Warmup ticks for auto-exposure & vehicle stabilization
        print("Warming up auto-exposure and vehicle stabilization...")
        # Determine target speed in m/s based on map (Town04 uses 20 mph, Town01 uses 10 mph)
        target_speed_ms = 8.9408 if args.map == "Town04" else 4.4704
        print(f"Target speed set to {20.0 if args.map == 'Town04' else 10.0} mph ({target_speed_ms} m/s)")
        
        for _ in range(30):
            if args.start_frame == 0:
                try:
                    # Direct velocity override
                    transform = ego_vehicle.get_transform()
                    forward_vec = transform.get_forward_vector()
                    ego_vehicle.set_target_velocity(forward_vec * target_speed_ms)
                except Exception:
                    pass
            world.tick()
            try:
                # Update spectator to follow the ego vehicle
                transform = ego_vehicle.get_transform()
                forward_vec = transform.get_forward_vector()
                spectator_loc = transform.location - 6.0 * forward_vec + carla.Location(z=3.5)
                spectator_rot = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
                spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))
                
                image_queue.get(timeout=1.0)
            except queue.Empty:
                pass
                
        print("Warmup complete. Starting closed-loop evaluation...")
        
        for step in range(args.num_frames):
            world.tick()
            
            # Update spectator to follow the ego vehicle
            try:
                transform = ego_vehicle.get_transform()
                forward_vec = transform.get_forward_vector()
                spectator_loc = transform.location - 6.0 * forward_vec + carla.Location(z=3.5)
                spectator_rot = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
                spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))
            except Exception:
                pass
            
            try:
                # Retrieve sensor frame
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"Warning: Step {step} timed out waiting for camera frame.")
                continue
                
            # Retrieve speed for controller in mph (1 m/s = 2.23694 mph)
            vel = ego_vehicle.get_velocity()
            speed_mph = 2.23694 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            loc = ego_vehicle.get_location()
            
            # 1. Retrieve the nominal autopilot steering and CTE from spatial nearest neighbor
            if has_ref_coords:
                dists = np.sqrt((np.array(ref_x_list) - loc.x)**2 + (np.array(ref_y_list) - loc.y)**2)
                nearest_idx = int(np.argmin(dists))
                cte = float(dists[nearest_idx])
                nominal_steer = nominal_steer_list[nearest_idx]
            else:
                cte = 0.0
                nominal_steer = 0.0
            
            # 2. Extract image and preprocess
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]
            
            # Preprocessing matching train_carla_model.py
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            cropped = rgb_image[180:400, :]
            resized = cv2.resize(cropped, (80, 60))
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensor
            img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 3. Model prediction (predicted steering angle)
            with torch.no_grad():
                pred_steer_tensor = model(img_tensor)
                # Extract scalar value and clamp directly (no 2.0x multiplier needed with weighted MSE training)
                pred_steer = float(pred_steer_tensor.squeeze().cpu().item())
                pred_steer = max(-1.0, min(1.0, pred_steer))
                
            # 4. Apply closed-loop speed control and neural steer
            target_speed = 20.0 if args.map == "Town04" else 10.0  # mph
            speed_error = target_speed - speed_mph
            if speed_error > 0:
                throttle = min(0.8, 0.3 + speed_error * 0.15)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(0.5, -speed_error * 0.1)
                
            # We bypass the autopilot brake to prevent the vehicle from getting stuck at red lights.
                
            ego_vehicle.apply_control(carla.VehicleControl(
                throttle=throttle,
                steer=pred_steer,
                brake=brake
            ))
                
            eval_records.append({
                "step": step,
                "nominal_steer": nominal_steer,
                "predicted_steer": pred_steer,
                "throttle": throttle,
                "brake": brake,
                "speed_mph": speed_mph,
                "x": loc.x,
                "y": loc.y,
                "z": loc.z,
                "cte": cte
            })
            
            if (step + 1) % 100 == 0:
                mae = np.mean([abs(r["predicted_steer"] - r["nominal_steer"]) for r in eval_records[-100:]])
                cte_mean = np.mean([r["cte"] for r in eval_records[-100:]])
                print(f"Step {step+1:04d}/{args.num_frames:04d} | Speed: {speed_mph:.1f} mph | Steer MAE (last 100): {mae:.4f} | CTE (last 100): {cte_mean:.2f}m")
                
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Revert settings and destroy actors
        print("Cleaning up simulation actors...")
        
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                ego_vehicle.set_autopilot(False)
            except Exception:
                pass

        if 'camera' in locals() and camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.destroy()
            except Exception:
                pass
                
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass
        
        try:
            world.apply_settings(original_settings)
            print("Reverted world settings to original.")
        except Exception:
            pass
            
        print("Cleanup done.")
        
    # Analyze and save results
    if eval_records:
        df = pd.DataFrame(eval_records)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        
        df.to_csv(args.save_csv, index=False)
        print(f"Saved evaluation telemetry to: {args.save_csv}")
        
        # Calculate overall metrics
        errors = df["predicted_steer"] - df["nominal_steer"]
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        max_dev = np.max(np.abs(errors))
        
        mean_cte_meters = df["cte"].mean() if "cte" in df.columns else 0.0
        max_cte_meters = df["cte"].max() if "cte" in df.columns else 0.0
        
        # Convert CTE to feet (1 meter = 3.28084 feet)
        df["cte_feet"] = df["cte"] * 3.28084 if "cte" in df.columns else 0.0
        mean_cte_feet = df["cte_feet"].mean()
        max_cte_feet = df["cte_feet"].max()
        
        # Convert steps to time (10 Hz -> 0.1s per step)
        df["time_sec"] = df["step"] * 0.1
        
        print("\n" + "="*50)
        print("CLOSED-LOOP EVALUATION SUMMARY")
        print(f"Total evaluated steps: {len(df)}")
        print(f"Mean Absolute Error (MAE):      {mae:.6f}")
        print(f"Root Mean Squared Error (RMSE):    {rmse:.6f}")
        print(f"Maximum Steering Deviation:     {max_dev:.6f}")
        print(f"Mean Cross-Track Error (CTE):   {mean_cte_meters:.4f} m ({mean_cte_feet:.2f} ft)")
        print(f"Maximum Cross-Track Error (CTE): {max_cte_meters:.4f} m ({max_cte_feet:.2f} ft)")
        
        # 1.75 meters standard half-lane width threshold in Town01 = 5.74 feet
        lane_boundary_feet = 5.74
        failed_lane_keeping = max_cte_feet > lane_boundary_feet
        print(f"Lane Boundary Violation (> 5.74 ft): {'FAILED/CRASHED' if failed_lane_keeping else 'PASSED'}")
        print("="*50 + "\n")
        
        # Generate combined plot with two subfigures sharing x-axis (Time in seconds)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        
        # Subplot 1: Steering Angle Controls
        ax1.plot(df["time_sec"], df["nominal_steer"], label="Nominal Autopilot Steering", color="#1f77b4", linewidth=1.5)
        ax1.plot(df["time_sec"], df["predicted_steer"], label="Model Predicted Steering (Closed-Loop)", color="#d62728", linestyle="--", linewidth=1.5)
        ax1.set_ylabel("Steering Angle Control ([-1.0, 1.0])")
        ax1.set_title(f"Closed-Loop Steering & Cross-Track Error: {args.weather.upper()} Weather\nSteering MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")
        
        # Subplot 2: Cross-Track Error (CTE) in feet
        if "cte" in df.columns and has_ref_coords:
            ax2.fill_between(df["time_sec"], df["cte_feet"], color="#ff7f0e", alpha=0.3, label="Cross-Track Error (CTE)")
            ax2.plot(df["time_sec"], df["cte_feet"], color="#ff7f0e", linewidth=1.5)
            # Add Lane Boundary line (5.74 ft)
            ax2.axhline(y=lane_boundary_feet, color="#d62728", linestyle=":", linewidth=2, label="Lane Boundary Limit (5.74 ft)")
            
            # Annotate if failed or passed
            if failed_lane_keeping:
                ax2.text(0.02, 0.85, "LANE DEVIATION / CRASH DETECTED", transform=ax2.transAxes, color="#d62728", weight="bold", fontsize=10)
            else:
                ax2.text(0.02, 0.85, "STABLE LANE KEEPING", transform=ax2.transAxes, color="#2ca02c", weight="bold", fontsize=10)
                
            ax2.set_ylabel("Cross-Track Error (feet)")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_title(f"Cross-Track Error (CTE) in Feet | Avg: {mean_cte_feet:.2f} ft | Max: {max_cte_feet:.2f} ft")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=300)
        plt.close()
        print(f"Combined validation plot saved to: {args.save_plot}")
    else:
        print("No evaluation records collected.")

if __name__ == "__main__":
    main()
