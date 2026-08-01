"""
Clean CARLA interface: connection, world/sync setup, spawning, a PHYSICS-HONEST
constant-speed controller (throttle/brake, not a velocity override), and image
helpers. The velocity-override approach in the legacy code corrupted lateral
dynamics and could stall the vehicle; a speed controller keeps physics intact so
the CTE we measure is real.
"""
import math
import queue

import carla

from config import (
    HOST, PORT, CLIENT_TIMEOUT_S, MAP_NAME, VEHICLE_BLUEPRINT,
    CAM_WIDTH, CAM_HEIGHT, CAM_FOV, CAM_X, CAM_Y, CAM_Z,
    TARGET_SPEED_MS, MPH_PER_MS, FIXED_DT,
)
# Re-export the shared image helpers so existing callers (env.raw_to_bgr,
# env.preprocess_for_model) keep working while the definition lives in imaging.
from imaging import raw_to_bgr, preprocess_for_model  # noqa: F401


# ── Connection / world ───────────────────────────────────────────────────────

def connect():
    client = carla.Client(HOST, PORT)
    client.set_timeout(CLIENT_TIMEOUT_S)
    return client


def load_town04(client, fresh=True):
    """Return a Town04 world. With fresh=True (default) the world is reloaded on
    every connect, clearing any accumulated actors/state from prior runs on a
    long-lived CARLA server (which can silently corrupt closed-loop results)."""
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != MAP_NAME:
        return client.load_world(MAP_NAME)      # loads a fresh map
    return client.reload_world() if fresh else world  # already Town04 -> reload fresh


def enable_sync_mode(world):
    """
    Enable fixed-step synchronous mode. Returns original settings to restore.

    CARLA requires  fixed_delta_seconds <= max_substep_delta_time * max_substeps,
    or physics silently advances less than the full step (the car covers half the
    distance its velocity implies). We size the substeps to cover the full dt.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = math.ceil(FIXED_DT / 0.01)  # 20 for dt=0.2 -> 0.2s physics
    world.apply_settings(settings)
    return original


def set_clear_weather(world):
    w = world.get_weather()
    w.cloudiness = 80.0
    w.precipitation = 0.0
    w.precipitation_deposits = 0.0
    w.sun_azimuth_angle = 0.0
    w.sun_altitude_angle = 90.0
    w.fog_density = 0.0
    w.wetness = 0.0
    world.set_weather(w)


# ── Spawning ─────────────────────────────────────────────────────────────────

def make_transform(spawn):
    return carla.Transform(
        carla.Location(x=spawn["x"], y=spawn["y"], z=spawn["z"]),
        carla.Rotation(yaw=spawn["yaw"]),
    )


def spawn_vehicle(world, spawn):
    bp = world.get_blueprint_library().filter(VEHICLE_BLUEPRINT)[0]
    tf = make_transform(spawn)
    vehicle = world.try_spawn_actor(bp, tf)
    if vehicle is None:
        tf.location.z += 0.5
        vehicle = world.spawn_actor(bp, tf)
    return vehicle


def spawn_camera(world, vehicle):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(CAM_WIDTH))
    bp.set_attribute("image_size_y", str(CAM_HEIGHT))
    bp.set_attribute("fov", str(CAM_FOV))
    tf = carla.Transform(carla.Location(x=CAM_X, y=CAM_Y, z=CAM_Z))
    camera = world.spawn_actor(bp, tf, attach_to=vehicle)
    img_queue = queue.Queue()
    camera.listen(img_queue.put)
    return camera, img_queue


# ── Speed control (physics-honest) ───────────────────────────────────────────

def speed_ms(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def speed_mph(vehicle):
    return speed_ms(vehicle) * MPH_PER_MS


class SpeedController:
    """
    PI controller on speed error -> (throttle, brake). The integral term removes
    the steady-state offset a pure-P controller leaves (so we hold the target
    speed exactly, satisfying the fixed-speed requirement). Anti-windup clamps
    the integral. Call reset() at the start of each drive.
    """

    def __init__(self, target_ms=TARGET_SPEED_MS, kp=0.5, ki=0.4, dt=FIXED_DT):
        self.target = target_ms
        self.kp, self.ki, self.dt = kp, ki, dt
        self.integ = 0.0

    def reset(self):
        self.integ = 0.0

    def control(self, vehicle):
        err = self.target - speed_ms(vehicle)
        # Conditional integration: only accumulate near the setpoint so the
        # integral can't wind up during the large-error warmup acceleration
        # (which otherwise overshoots to ~27 mph before settling).
        if abs(err) < 1.5:
            self.integ = max(-3.0, min(3.0, self.integ + err * self.dt))
        else:
            self.integ = 0.0
        u = self.kp * err + self.ki * self.integ
        return (min(1.0, u), 0.0) if u >= 0 else (0.0, min(1.0, -u))


def teleport(vehicle, spawn):
    """Reposition and zero out motion (for direction switches)."""
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
    tf = make_transform(spawn)
    tf.location.z += 0.3
    vehicle.set_transform(tf)


def warmup_to_speed(world, vehicle, img_queue, speed_ctrl, steer_fn=None,
                    settle_ticks=15, max_accel_ticks=80):
    """
    Let physics settle (held by brake), then accelerate to target speed while
    STEERING along the lane via steer_fn (default straight). Steering during
    warmup keeps the car centered on curved spawn lanes so recording starts
    on-center instead of recovering from a warmup-induced drift.
    """
    speed_ctrl.reset()
    for _ in range(settle_ticks):
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        world.tick()
        _drain(img_queue)
    for _ in range(max_accel_ticks):
        steer = steer_fn(vehicle) if steer_fn else 0.0
        thr, brk = speed_ctrl.control(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))
        world.tick()
        _drain(img_queue)
        if speed_ms(vehicle) >= 0.98 * TARGET_SPEED_MS:
            break


def _drain(img_queue):
    try:
        img_queue.get(timeout=1.0)
    except queue.Empty:
        pass


# ── Spectator / images / cleanup ─────────────────────────────────────────────

def update_spectator(world, vehicle):
    try:
        tf = vehicle.get_transform()
        fwd = tf.get_forward_vector()
        loc = tf.location - 6.0 * fwd + carla.Location(z=3.5)
        rot = carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw)
        world.get_spectator().set_transform(carla.Transform(loc, rot))
    except Exception:
        pass


def cleanup(actors, world=None, original_settings=None):
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass
    if world is not None and original_settings is not None:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
