#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import random
import queue
import csv
import numpy as np
import pandas as pd
import torch
import cv2

# Add parent directory and CARLA PythonAPI to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CarlaSteeringNet, CarlaSteeringExpertNet
from weather_config import set_weather_profile, apply_vehicle_lights

carla_root = "/home/za/carla"
try:
    sys.path.append(glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'agents'))[0])
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
    print("Error: Could not import CARLA Python library.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA DAgger-Lite Interactive Recovery Data Collector")
    parser.add_argument("--host", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to")
    parser.add_argument("--map", default="Town04", help="Name of the CARLA map/town to load")
    parser.add_argument("--weather", default="clear", choices=["clear", "rain", "fog", "night"], help="Weather profile")
    parser.add_argument("--model-path", default="models/carla_expert_clear.pth", help="Path to current steering model")
    parser.add_argument("--model-type", default="CarlaSteeringExpertNet", choices=["CarlaSteeringNet", "CarlaSteeringExpertNet"], help="Model architecture type")
    parser.add_argument("--dataset-root", default="datasets/carla_steering_e2e", help="Dataset directory to append data to")
    parser.add_argument("--max-steps", default=1500, type=int, help="Number of simulation steps to run dagger collection")
    parser.add_argument("--width", default=640, type=int, help="Image width")
    parser.add_argument("--height", default=480, type=int, help="Image height")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--drift-threshold", default=0.6, type=float, help="CTE threshold in meters to trigger autopilot takeover")
    parser.add_argument("--recovery-threshold", default=0.15, type=float, help="CTE threshold in meters to return control to AI model")
    return parser.parse_args()

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
    print(f"Loading steering model from {args.model_path} onto device: {device}...")
    if args.model_type == "CarlaSteeringNet":
        model = CarlaSteeringNet().to(device)
    else:
        model = CarlaSteeringExpertNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Open dataset index for appending new recovery frames
    save_dir = os.path.join(args.dataset_root, args.weather)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "index.csv")
    existing_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if existing_df is None:
        csv_writer.writerow(["frame", "image_path", "steering", "throttle", "brake", "speed_mph", "x", "y"])
        collected_frames = 0
    else:
        collected_frames = len(existing_df)
    
    print(f"Existing dataset has {collected_frames} frames. Appending new recovery data to {csv_path}...")
    
    actor_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    
    # Load reference coordinates to spawn vehicle at the nominal start position
    ref_x_list = existing_df["x"].tolist() if existing_df is not None else [-357.1]
    ref_y_list = existing_df["y"].tolist() if existing_df is not None else [30.0]
    
    try:
        print(f"Loading map {args.map}...")
        world = client.load_world(args.map)
        set_weather_profile(world, args.weather)
        
        # Synchronous mode at 5 Hz to match training frequency
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2
        world.apply_settings(settings)
        print("Synchronous mode enabled at 5 Hz.")
        
        blueprint_library = world.get_blueprint_library()
        
        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = carla.Transform(
            carla.Location(x=ref_x_list[0], y=ref_y_list[0], z=0.5),
            carla.Rotation(yaw=0.0)
        )
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned vehicle at: {spawn_point.location}")
        
        # Set up Traffic Manager for autopilot takeover
        traffic_manager = client.get_trafficmanager(8005)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.auto_lane_change(ego_vehicle, False)
        traffic_manager.set_desired_speed(ego_vehicle, 32.0) # 20 mph
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(ego_vehicle, 100.0)
        
        # Spawn camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(camera)
        
        image_queue = queue.Queue()
        camera.listen(lambda img: sensor_callback(img, image_queue))
        
        spectator = world.get_spectator()
        
        # Stabilize vehicle
        ego_vehicle.set_target_velocity(ego_vehicle.get_transform().get_forward_vector() * 8.94)
        for _ in range(30):
            world.tick()
            try:
                image_queue.get(timeout=0.05)
            except queue.Empty:
                pass
        
        print("Warmup complete. Starting DAgger-Lite interactive loop...")
        
        autopilot_override = False
        new_collected = 0
        
        for step in range(args.max_steps):
            world.tick()
            
            # Spectator camera follow
            try:
                transform = ego_vehicle.get_transform()
                forward_vec = transform.get_forward_vector()
                spectator_loc = transform.location - 6.0 * forward_vec + carla.Location(z=3.5)
                spectator_rot = carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0)
                spectator.set_transform(carla.Transform(spectator_loc, spectator_rot))
            except Exception:
                pass
            
            try:
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: Timed out waiting for camera frame.")
                continue
                
            loc = ego_vehicle.get_location()
            vel = ego_vehicle.get_velocity()
            speed_mph = 2.23694 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # Calculate actual Cross-Track Error (CTE) to the center of the current lane
            cte = 0.0
            wp = world.get_map().get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                wp_loc = wp.transform.location
                cte = float(np.sqrt((wp_loc.x - loc.x)**2 + (wp_loc.y - loc.y)**2))
            
            # Trigger conditions for DAgger takeover/release
            if not autopilot_override and cte > args.drift_threshold:
                autopilot_override = True
                ego_vehicle.set_autopilot(True, traffic_manager.get_port())
                print(f"[TAKEOVER] CTE={cte:.2f}m exceeded {args.drift_threshold}m. Autopilot enabled for recovery.")
                
            elif autopilot_override and cte < args.recovery_threshold:
                autopilot_override = False
                ego_vehicle.set_autopilot(False)
                print(f"[RELEASE] CTE={cte:.2f}m recovered below {args.recovery_threshold}m. AI model resumed control.")
                
            if not autopilot_override:
                # 1. Run AI model control
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                bgr_image = array[:, :, :3]
                
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                cropped = rgb_image[180:400, :]
                resized = cv2.resize(cropped, (80, 60))
                normalized = resized.astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_steer = float(model(img_tensor).squeeze().cpu().item())
                    pred_steer = max(-1.0, min(1.0, pred_steer))
                    
                target_speed = 20.0 # mph
                speed_error = target_speed - speed_mph
                throttle = min(0.8, 0.3 + speed_error * 0.15) if speed_error > 0 else 0.0
                brake = min(0.5, -speed_error * 0.1) if speed_error < 0 else 0.0
                
                ego_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=pred_steer, brake=brake))
            else:
                # 2. Autopilot control is active (recovery phase)
                # Read the control from the autopilot (expert) and save it to the dataset!
                ap_control = ego_vehicle.get_control()
                
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                bgr_image = array[:, :, :3]
                
                # Discard frames in intersections/junctions
                is_junction = wp.is_junction if wp else False
                if not is_junction:
                    img_name = f"frame_dagger_{collected_frames:06d}.png"
                    img_path = os.path.join(img_dir, img_name)
                    
                    cv2_saved = cv2.imwrite(img_path, bgr_image)
                    if cv2_saved:
                        relative_path = os.path.join("images", img_name)
                        csv_writer.writerow([
                            collected_frames,
                            relative_path,
                            ap_control.steer,
                            ap_control.throttle,
                            ap_control.brake,
                            speed_mph,
                            loc.x,
                            loc.y
                        ])
                        collected_frames += 1
                        new_collected += 1
                
            if (step + 1) % 100 == 0:
                print(f"Step {step+1:04d}/{args.max_steps:04d} | Autopilot Override: {autopilot_override} | CTE: {cte:.2f}m | New Recovery Frames: {new_collected}")
                
        print(f"DAgger interactive session complete. Collected {new_collected} new recovery frames.")
        
    except Exception as e:
        print(f"Error during DAgger: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        csv_file.close()
        print("Cleaning up simulation actors...")
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                ego_vehicle.set_autopilot(False)
            except Exception:
                pass
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        print("Cleanup done.")

if __name__ == "__main__":
    main()
