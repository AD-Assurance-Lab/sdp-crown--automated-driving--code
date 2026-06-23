# tools/weather_config.py
import carla

def set_weather_profile(world, profile):
    """
    Applies calibrated ACDC-aligned weather parameters to the CARLA world.
    """
    weather = world.get_weather()
    
    if profile == "clear":
        # OvercastNoon: Diffuse sunlight, no harsh shadows, no sun glare
        weather.cloudiness = 80.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 10.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 90.0  # High sun, but covered by heavy clouds (diffuse)
        weather.fog_density = 0.0
        weather.wetness = 0.0
    elif profile == "rain":
        # Calibrated Rain (ACDC match): wet surfaces and reflections without torrential wash-out
        weather.cloudiness = 80.0
        weather.precipitation = 40.0
        weather.precipitation_deposits = 30.0
        weather.wind_intensity = 20.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 45.0
        weather.fog_density = 10.0
        weather.wetness = 50.0
    elif profile == "fog":
        # Calibrated Fog (ACDC match): realistic visual attenuation, not default zero-visibility fog
        weather.cloudiness = 60.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 10.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = 45.0
        weather.fog_density = 25.0
        weather.fog_distance = 30.0  # Safe visibility range for driving
        weather.wetness = 0.0
    elif profile == "night":
        # Nighttime with low ambient lighting
        weather.cloudiness = 20.0
        weather.precipitation = 0.0
        weather.precipitation_deposits = 0.0
        weather.wind_intensity = 0.0
        weather.sun_azimuth_angle = 0.0
        weather.sun_altitude_angle = -45.0  # Below horizon
        weather.fog_density = 0.0
        weather.wetness = 0.0

    world.set_weather(weather)
    print(f"Weather set to profile: {profile.upper()}")

def apply_vehicle_lights(vehicle, profile):
    """
    Configures vehicle headlights based on the weather profile.
    """
    if profile == "night":
        light_state = carla.VehicleLightState(
            carla.VehicleLightState.Position | 
            carla.VehicleLightState.LowBeam | 
            carla.VehicleLightState.HighBeam
        )
        vehicle.set_light_state(light_state)
        print("Nightlights turned ON (LowBeam, HighBeam, Position).")
    else:
        vehicle.set_light_state(carla.VehicleLightState.NONE)
