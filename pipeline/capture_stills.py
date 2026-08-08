#!/usr/bin/env python3
"""Capture paper-quality CARLA stills: a cinematic chase view and the front-camera
view, under clear / fog / night, from the VALIDATED on-road eastbound spawn. Settles
physics and verifies the vehicle is resting on the road before shooting (an arbitrary
low-z route spawn made the car fall through the map). Night is a legible twilight
(representative of the ACDC-calibrated severity), not CARLA's near-black default.
Outputs to pipeline/results/stills/."""
import os
import sys
import queue
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import carla  # noqa: E402
import config as C  # noqa: E402
import carla_env as env  # noqa: E402

OUT = os.path.join(C.RESULTS_DIR, "stills")
os.makedirs(OUT, exist_ok=True)


def set_night_legible(world):
    """Twilight night: dark but road/lanes still visible (ACDC-severity, not -25 deg)."""
    w = world.get_weather()
    w.cloudiness, w.precipitation, w.precipitation_deposits = 20.0, 0.0, 0.0
    w.wetness, w.fog_density = 0.0, 0.0
    w.sun_altitude_angle = -6.0
    world.set_weather(w)


def add_camera(world, vehicle, w, h, tf, fov=90):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    cam = world.spawn_actor(bp, tf, attach_to=vehicle)
    q = queue.Queue()
    cam.listen(q.put)
    return cam, q


def grab(q, world, n_settle=20):
    img = None
    for _ in range(n_settle):
        world.tick()
        try:
            img = q.get(timeout=2.0)
        except queue.Empty:
            pass
    return img


def main():
    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)

    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)  # validated on-road spawn
    # settle firmly on the road (held by brake) before doing anything
    for _ in range(40):
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
        world.tick()
    z = vehicle.get_location().z
    print(f"  vehicle settled at z={z:.2f} m", flush=True)
    if z < -1.0:
        print("  ERROR: vehicle fell through the map; aborting", flush=True)
        env.cleanup([vehicle], world, original)
        return

    chase_tf = carla.Transform(carla.Location(x=-8.0, y=0.0, z=4.0),
                               carla.Rotation(pitch=-12.0))
    chase, chase_q = add_camera(world, vehicle, 1280, 720, chase_tf, fov=75)
    front_tf = carla.Transform(carla.Location(x=C.CAM_X, y=C.CAM_Y, z=C.CAM_Z))
    front, front_q = add_camera(world, vehicle, 960, 720, front_tf, fov=C.CAM_FOV)

    import cv2

    def drain(qq):
        while not qq.empty():
            try:
                qq.get_nowait()
            except queue.Empty:
                break

    weathers = [("clear", lambda: env.set_weather(world, "clear")),
                ("fog",   lambda: env.set_weather(world, "fog")),
                ("night", lambda: set_night_legible(world))]
    try:
        for name, setter in weathers:
            setter()
            env.update_spectator(world, vehicle)
            drain(chase_q); drain(front_q)     # discard buffered pre-change frames
            ci = fi = None
            for _ in range(30):                # render + settle the NEW weather, then grab fresh
                world.tick()
                try:
                    ci = chase_q.get(timeout=2.0)
                except queue.Empty:
                    pass
                try:
                    fi = front_q.get(timeout=2.0)
                except queue.Empty:
                    pass
            if ci is not None:
                cv2.imwrite(os.path.join(OUT, f"chase_{name}.png"), env.raw_to_bgr(ci))
            if fi is not None:
                cv2.imwrite(os.path.join(OUT, f"front_{name}.png"), env.raw_to_bgr(fi))
            print(f"  captured {name} (chase + front)", flush=True)
    finally:
        env.cleanup([chase, front, vehicle], world, original)
    print(f"stills -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
