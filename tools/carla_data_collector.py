#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import random
import csv
import queue
import numpy as np

# Add CARLA PythonAPI to path if not present
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

def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Dataset Collector for Steering Behavioral Cloning")
    parser.add_argument("--host", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--port", default=2000, type=int, help="TCP port to listen to")
    parser.add_argument("--map", default="Town01", help="Name of the CARLA map/town to load")
    parser.add_argument("--weather", default="clear", choices=["clear", "rain", "fog", "night"], help="Weather profile for collection")
    parser.add_argument("--num-frames", default=5000, type=int, help="Number of frames to collect")
    parser.add_argument("--width", default=640, type=int, help="Image width")
    parser.add_argument("--height", default=480, type=int, help="Image height")
    parser.add_argument("--output-dir", default="datasets/carla_steering_e2e", help="Root directory for saving collected datasets")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--spawn-point-idx", default=12, type=int, help="Index of spawn point for the ego vehicle")
    return parser.parse_args()

def set_weather(world, profile):
    weather = world.get_weather()
    
    if profile == "clear":
        # Clear Noon
        weather.cloudiness = 0.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 0.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 75.0  # High noon
        weather.fog_density = 0.0
        weather.wetness = 0.0
    elif profile == "rain":
        # Wet road and rain reflections
        weather.cloudiness = 80.0
        weather.precipitation = 80.0
        weather.precipitation_deposits = 80.0
        weather.wind_intensity = 50.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 45.0
        weather.fog_density = 10.0
        weather.wetness = 80.0
    elif profile == "fog":
        # Dense fog visibility loss
        weather.cloudiness = 90.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 10.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 45.0
        weather.fog_density = 75.0  # Very dense fog
        weather.fog_distance = 5.0
        weather.wetness = 0.0
    elif profile == "night":
        # Low ambient light (night)
        weather.cloudiness = 10.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 0.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = -75.0  # Nighttime
        weather.fog_density = 0.0
        weather.wetness = 0.0

    world.set_weather(weather)
    print(f"Weather set to profile: {profile.upper()}")

def sensor_callback(image, image_queue):
    image_queue.put(image)

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
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
    client.set_timeout(20.0)
    
    try:
        # Load map (always reload to ensure a clean slate and delete previous actors)
        print(f"Loading map {args.map}...")
        world = client.load_world(args.map)
        
        # Set weather
        set_weather(world, args.weather)
        
        # Set synchronous mode
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # 10 Hz
        world.apply_settings(settings)
        print("Synchronous mode enabled at 10 Hz.")
        
        blueprint_library = world.get_blueprint_library()
        
        # Spawn ego vehicle deterministically
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            spawn_point = carla.Transform()
            print("Warning: No spawn points found on map. Using default Transform.")
        else:
            spawn_point_idx = args.spawn_point_idx % len(spawn_points)
            spawn_point = spawn_points[spawn_point_idx]
            # Add safety z-offset to prevent clipping into the road collider
            spawn_point.location.z += 0.5
            print(f"Selected spawn point index {spawn_point_idx} (out of {len(spawn_points)}) at location {spawn_point.location}")
        
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned ego vehicle: {ego_vehicle.type_id}")
        
        # Configure Traffic Manager
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        # Target speed: constant 10 mph. 
        # Map speed limit is 30 km/h. To get 10 mph (16 km/h), set percentage difference to 46.6% below.
        traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 46.6)
        
        # Set autopilot
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(ego_vehicle, 100.0)
        print(f"Autopilot registered on port {traffic_manager.get_port()}. Speed limited to 10 mph (16 km/h), ignoring lights.")
        
        # Spawn front camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(args.width))
        camera_bp.set_attribute('image_size_y', str(args.height))
        camera_bp.set_attribute('fov', '90')
        
        # Hood camera placement (hood height ~1.2m, x=1.5m forward)
        camera_transform = carla.Transform(carla.Location(x=1.6, y=0.0, z=1.2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(camera)
        print("Spawned front RGB camera.")
        
        # Setup queue for image collection
        image_queue = queue.Queue()
        camera.listen(lambda img: sensor_callback(img, image_queue))
        
        # Get spectator actor to track vehicle position
        spectator = world.get_spectator()
        
        # Warmup ticks to stabilize vehicle and auto-exposure
        print("Warming up auto-exposure and vehicle stabilization...")
        for _ in range(30):
            try:
                # Direct velocity override (10 mph = 4.4704 m/s)
                transform = ego_vehicle.get_transform()
                forward_vec = transform.get_forward_vector()
                ego_vehicle.set_target_velocity(forward_vec * 4.4704)
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
                
        # Main collection loop
        print(f"Starting collection of {args.num_frames} frames under {args.weather} weather...")
        collected_frames = 0
        
        while collected_frames < args.num_frames:
            try:
                # Direct velocity override (10 mph = 4.4704 m/s)
                transform = ego_vehicle.get_transform()
                forward_vec = transform.get_forward_vector()
                ego_vehicle.set_target_velocity(forward_vec * 4.4704)
            except Exception:
                pass
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
                # Synchronously retrieve image from queue
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: Timed out waiting for camera frame.")
                continue
                
            # 1. Retrieve the steering and speed from autopilot
            ap_control = ego_vehicle.get_control()
            velocity = ego_vehicle.get_velocity()
            speed_mph = 2.23694 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Format raw pixels to numpy array.
            # CARLA camera outputs BGRA format, so the first 3 channels are BGR.
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            bgr_image = array[:, :, :3]
            
            # Save frame to disk
            img_name = f"frame_{collected_frames:06d}.png"
            img_path = os.path.join(img_dir, img_name)
            
            # cv2.imwrite expects BGR, so save directly without channel swapping
            cv2_saved = cv2.imwrite(img_path, bgr_image)
            
            if cv2_saved:
                # Log metadata
                loc = ego_vehicle.get_location()
                relative_img_path = os.path.join("images", img_name)
                csv_writer.writerow([
                    collected_frames,
                    relative_img_path,
                    ap_control.steer,
                    ap_control.throttle,
                    ap_control.brake,
                    speed_mph,
                    loc.x,
                    loc.y
                ])
                
                collected_frames += 1
                if collected_frames % 500 == 0:
                    print(f"Collected {collected_frames}/{args.num_frames} frames...")
            else:
                print("Error saving image to disk.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Revert settings and destroy actors
        print("Cleaning up simulation actors...")
        csv_file.close()
        
        # Disable autopilot on vehicle first to unregister it from Traffic Manager
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                ego_vehicle.set_autopilot(False)
            except Exception:
                pass

        # Stop and destroy camera sensor first to prevent background thread callbacks
        if 'camera' in locals() and camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.destroy()
            except Exception:
                pass
                
        # Destroy ego vehicle second
        if 'ego_vehicle' in locals() and ego_vehicle is not None:
            try:
                ego_vehicle.destroy()
            except Exception:
                pass
        
        # Revert sync settings
        try:
            world.apply_settings(original_settings)
            print("Reverted world settings to original.")
        except Exception:
            pass
            
        print("Done.")

if __name__ == "__main__":
    # Ensure cv2 is imported inside main context
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV (cv2) is required to run the data collector script.")
        sys.exit(1)
        
    main()
