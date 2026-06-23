#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import csv
import queue
import numpy as np

# Add CARLA PythonAPI to path if not present
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    print("Error: Could not import CARLA Python library. Ensure CARLA is installed and python package is available.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: OpenCV (cv2) is required to run the data collector script.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Continuous Highway Segment Dataset Collector")
    parser.add_argument("--host", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to")
    parser.add_argument("--map", default="Town04", help="Name of the CARLA map/town to load")
    parser.add_argument("--weather", default="clear", choices=["clear", "rain", "fog", "night"], help="Weather profile for collection")
    parser.add_argument("--num-frames", default=5000, type=int, help="Maximum number of frames to collect (ignored/cap)")
    parser.add_argument("--width", default=640, type=int, help="Image width")
    parser.add_argument("--height", default=480, type=int, help="Image height")
    parser.add_argument("--output-dir", default="datasets/carla_steering_e2e", help="Root directory for saving collected datasets")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--spawn-point-idx", default=12, type=int, help="Ignored (backward compatibility)")
    return parser.parse_args()

def set_weather(world, profile):
    set_weather_profile(world, profile)

def sensor_callback(image, image_queue):
    image_queue.put(image)

def teleport_vehicle(world, vehicle, spawn_point, image_queue):
    """
    Teleports the vehicle safely to a new location, neutralizing velocity
    and waiting a few frames for physics/exposure to stabilize.
    """
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    
    transform = carla.Transform(
        carla.Location(x=spawn_point.location.x, y=spawn_point.location.y, z=spawn_point.location.z + 0.5),
        spawn_point.rotation
    )
    vehicle.set_transform(transform)
    
    # Warmup ticks to flush frame buffers and stabilize camera exposure
    for _ in range(25):
        world.tick()
        try:
            image_queue.get(timeout=0.05)
        except queue.Empty:
            pass
            
    # Set initial target velocity to 20 mph (8.94 m/s) to start moving smoothly
    forward_vec = vehicle.get_transform().get_forward_vector()
    vehicle.set_target_velocity(forward_vec * 8.9408)

def main():
    args = parse_args()
    
    # Establish subdirectories
    save_dir = os.path.join(args.output_dir, args.weather)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "index.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "image_path", "steering", "throttle", "brake", "speed_mph", "x", "y"])

    actor_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    
    try:
        # Load map (always reload to ensure a clean slate and delete previous actors)
        print(f"Loading map {args.map}...")
        world = client.load_world(args.map)
        
        # Set weather
        set_weather(world, args.weather)
        
        # Set synchronous mode at 5 Hz (0.2s step size) to reduce straight-line redundancy
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.2  # 5 Hz
        world.apply_settings(settings)
        print("Synchronous mode enabled at 5 Hz.")
        
        blueprint_library = world.get_blueprint_library()
        
        # Spawn ego vehicle deterministically
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # Define target highway segments using user GPS coordinate thresholds:
        # Stage 1 (Eastbound): start at (-357.1, 30.0), facing East (yaw=0.0)
        sp_cw = carla.Transform(
            carla.Location(x=-357.1, y=30.0, z=0.5),
            carla.Rotation(yaw=0.0)
        )
        # Stage 2 (Westbound): start at (-396.8, 12.8), facing West (yaw=180.0)
        sp_ccw = carla.Transform(
            carla.Location(x=-396.8, y=12.8, z=0.5),
            carla.Rotation(yaw=180.0)
        )
        
        # Start at Eastbound segment
        spawn_point = sp_cw
        
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned ego vehicle: {ego_vehicle.type_id}")
        
        # Configure vehicle headlights based on weather (shared config)
        apply_vehicle_lights(ego_vehicle, args.weather)
        
        # Configure Traffic Manager
        traffic_manager = client.get_trafficmanager(8005)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        # Disable automatic lane changing! This keeps the vehicle locked in the inner lane
        traffic_manager.auto_lane_change(ego_vehicle, False)
        
        # Set desired longitudinal speed to 32 km/h (~20 mph / 8.94 m/s) for highway driving
        traffic_manager.set_desired_speed(ego_vehicle, 32.0)
        
        # Set autopilot
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(ego_vehicle, 100.0)
        print(f"Autopilot registered on port 8005. Lane-change disabled. Speed locked at 32 km/h (~20 mph).")
        
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
        
        # Setup queue for image collection
        image_queue = queue.Queue()
        camera.listen(lambda img: sensor_callback(img, image_queue))
        
        # Get spectator actor to track vehicle position
        spectator = world.get_spectator()
        
        # Stabilize and align spectator at the start
        teleport_vehicle(world, ego_vehicle, spawn_point, image_queue)
                
        print(f"Starting continuous data collection...")
        collected_frames = 0
        stage_idx = 1
        
        # Tracking flags for loop phase progression
        has_left_start = False
        reached_return_segment = False
        
        while True:
            world.tick()
            
            # Spectator camera follows vehicle smoothly
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

            # Skip saving frames if the vehicle is in a junction (e.g. exit/entrance merges)
            # to guarantee absolutely zero intersection/junction corruption in the dataset.
            waypoint = world.get_map().get_waypoint(loc)
            is_junction_frame = waypoint.is_junction

            # Handle Stage Transitions and End Conditions using GPS coordinate phases:
            if stage_idx == 1:
                # Stage 1: Eastbound (90 deg E). Start: x=-357.1. Drive Eastbound.
                # Left start check
                if not has_left_start:
                    if loc.x > 0.0:
                        has_left_start = True
                        print("Stage 1: Passed midpoint, left start segment.")
                
                # Check return leg (after looping, X goes negative again and reaches West curve)
                if has_left_start and not reached_return_segment:
                    if loc.x < -430.0:
                        reached_return_segment = True
                        print("Stage 1: Reached return segment on the West side.")
                
                # End condition: Reaches -424.0 on the return segment
                if reached_return_segment:
                    if loc.x > -424.0:
                        print(f"Stage 1 Complete (Reached x={loc.x:.1f}). Teleporting to Stage 2...")
                        teleport_vehicle(world, ego_vehicle, sp_ccw, image_queue)
                        stage_idx = 2
                        has_left_start = False
                        reached_return_segment = False
                        continue
            
            elif stage_idx == 2:
                # Stage 2: Westbound (270 deg W). Start: x=-396.8. Drive Westbound.
                # Left start check
                if not has_left_start:
                    if loc.x > 0.0:
                        has_left_start = True
                        print("Stage 2: Passed midpoint, left start segment.")
                
                # Check return leg (after looping, X goes positive on the East side)
                if has_left_start and not reached_return_segment:
                    if loc.x > 300.0:
                        reached_return_segment = True
                        print("Stage 2: Reached return segment on the East side.")
                
                # End condition: Reaches -321.5 on the return segment going Westbound
                if reached_return_segment:
                    if loc.x < -321.5:
                        print(f"Stage 2 Complete (Reached x={loc.x:.1f}). Stopping data collection.")
                        break

            # Retrieve control state
            ap_control = ego_vehicle.get_control()
            steer = ap_control.steer
            velocity = ego_vehicle.get_velocity()
            speed_mph = 2.23694 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Format and save BGR pixels
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]
            
            img_name = f"frame_{collected_frames:06d}.png"
            img_path = os.path.join(img_dir, img_name)
            
            # Save frame only if it's not in a junction
            if not is_junction_frame:
                cv2_saved = cv2.imwrite(img_path, bgr_image)
                if cv2_saved:
                    relative_img_path = os.path.join("images", img_name)
                    csv_writer.writerow([
                        collected_frames,
                        relative_img_path,
                        steer,
                        ap_control.throttle,
                        ap_control.brake,
                        speed_mph,
                        loc.x,
                        loc.y
                    ])
                    collected_frames += 1
                    if collected_frames % 100 == 0:
                        print(f"Collected {collected_frames} frames (Stage {stage_idx})...")
                else:
                    print("Error saving image to disk.")
            else:
                # Frame discarded due to junction
                pass
                
        print(f"Successfully collected continuous dataset of {collected_frames} frames.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        print("Cleaning up simulation actors...")
        csv_file.close()
        
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                if 'traffic_manager' in locals() and traffic_manager is not None:
                    ego_vehicle.set_autopilot(False, traffic_manager.get_port())
            except Exception:
                pass

        if 'camera' in locals() and camera is not None:
            try:
                camera.stop()
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
            
        print("Done.")

if __name__ == "__main__":
    main()
