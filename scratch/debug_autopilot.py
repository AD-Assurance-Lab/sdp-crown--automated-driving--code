#!/usr/bin/env python3
import os
import sys
import glob
import time
import random

# Add CARLA PythonAPI to path
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

import carla

def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    # Load map Town01
    print("Loading map Town01...")
    world = client.load_world("Town01")
    
    # Enable synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    
    # Configure Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(42)
    
    # Spawn vehicle at spawn point index 0
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    # Try different spawn point index if index 0 is blocked
    spawn_idx = 0
    spawn_point = spawn_points[spawn_idx]
    
    # Add a slight z-offset to prevent road clipping
    spawn_point.location.z += 0.5
    
    print(f"Spawning vehicle at index {spawn_idx}: {spawn_point.location}")
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    try:
        # Register with TM
        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(vehicle, 100.0)
        traffic_manager.ignore_signs_percentage(vehicle, 100.0)
        
        print("Starting simulation ticks...")
        for tick in range(100):
            world.tick()
            
            # Fetch state
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            speed = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            control = vehicle.get_control()
            
            # Check if vehicle is affected by any traffic light
            traffic_light = vehicle.get_traffic_light()
            light_state = "None"
            if traffic_light is not None:
                light_state = str(traffic_light.get_state())
                
            print(f"Tick {tick:02d} | Location: ({loc.x:.2f}, {loc.y:.2f}) | Speed: {speed:.2f} km/h | Steer: {control.steer:.2f} | Throttle: {control.throttle:.2f} | Brake: {control.brake:.2f} | Light: {light_state}")
            time.sleep(0.05)
            
    finally:
        print("Cleaning up...")
        vehicle.destroy()
        world.apply_settings(original_settings)
        print("Done.")

if __name__ == "__main__":
    main()
