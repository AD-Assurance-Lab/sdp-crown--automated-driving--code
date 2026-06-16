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
    parser.add_argument("--output-dir", default="datasets/carla_testing", help="Root directory for saving collected datasets")
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
    
    # Establish subdirectories
    save_dir = os.path.join(args.output_dir, args.weather)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "index.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "image_path", "steering", "throttle", "brake", "speed"])

    actor_list = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    
    try:
        # Load map
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
        
        # Spawn ego vehicle
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned ego vehicle: {ego_vehicle.type_id}")
        
        # Set autopilot
        ego_vehicle.set_autopilot(True)
        # Configure autopilot speed limit
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(30.0)  # Moderate speed
        traffic_manager.set_synchronous_mode(True)
        
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
        
        # Warmup ticks to stabilize vehicle and auto-exposure
        print("Warming up auto-exposure and vehicle stabilization...")
        for _ in range(30):
            world.tick()
            try:
                image_queue.get(timeout=1.0)
            except queue.Empty:
                pass
                
        # Main collection loop
        print(f"Starting collection of {args.num_frames} frames under {args.weather} weather...")
        collected_frames = 0
        
        while collected_frames < args.num_frames:
            world.tick()
            
            try:
                # Synchronously retrieve image from queue
                image = image_queue.get(timeout=2.0)
            except queue.Empty:
                print("Warning: Timed out waiting for camera frame.")
                continue
                
            # Retrieve vehicle physics states
            control = ego_vehicle.get_control()
            velocity = ego_vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            
            # Format raw pixels to numpy RGB array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            rgb_image = array[:, :, :3]
            
            # Save frame to disk
            img_name = f"frame_{collected_frames:06d}.png"
            img_path = os.path.join(img_dir, img_name)
            
            # Convert RGB to BGR for OpenCV saving
            bgr_image = rgb_image[:, :, ::-1]
            cv2_saved = cv2.imwrite(img_path, bgr_image)
            
            if cv2_saved:
                # Log metadata
                relative_img_path = os.path.join("images", img_name)
                csv_writer.writerow([
                    collected_frames,
                    relative_img_path,
                    control.steer,
                    control.throttle,
                    control.brake,
                    speed
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
        
        # Destroy all spawned actors
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        
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
