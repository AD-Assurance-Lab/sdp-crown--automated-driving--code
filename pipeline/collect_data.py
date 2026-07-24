#!/usr/bin/env python3
"""
Milestone-1, step 3: collect behavior-cloning data by driving the full Town04
loop (both directions) with the pure-pursuit EXPERT and recording, per frame,
the raw camera image paired with the expert steering label.

Image[t] is paired with pose[t]/label[t] by ticking FIRST, then reading pose and
computing the label from the same frame — exact image/label alignment.

Saves raw 640x480 RGB PNGs (preprocessing deferred to train time) plus a single
manifest CSV. Usage:
    python collect_data.py --dataset clear --laps 2 --direction both
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import carla  # noqa: E402

import config as C  # noqa: E402
import carla_env as env  # noqa: E402
from route import load_route, signed_cte_route, pure_pursuit_route  # noqa: E402

SPAWNS = {"eastbound": C.SPAWN_EASTBOUND, "westbound": C.SPAWN_WESTBOUND}
FIELDS = ["image", "direction", "lap", "step", "steer", "steer_rad",
          "cte_m", "speed_mph", "x", "y", "yaw"]


def collect_lap(world, world_map, vehicle, img_queue, direction, lap, out_dir, max_steps):
    spawn = SPAWNS[direction]
    route = load_route(direction)
    hint = None
    speed_ctrl = env.SpeedController()
    env.teleport(vehicle, spawn)
    env.warmup_to_speed(
        world, vehicle, img_queue, speed_ctrl,
        steer_fn=lambda veh: pure_pursuit_route(route, veh.get_transform())[0],
    )

    frames_dir = os.path.join(out_dir, f"{direction}_lap{lap:02d}", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    start = carla.Location(x=spawn["x"], y=spawn["y"], z=spawn["z"])
    print(f"  [{direction} lap{lap:02d}] start speed={env.speed_mph(vehicle):.1f} mph")

    rows, left_start = [], False
    for step in range(max_steps):
        env.update_spectator(world, vehicle)
        world.tick()                              # advance -> frame t
        try:
            image = img_queue.get(timeout=2.0)    # image[t]
        except Exception:
            continue
        tf = vehicle.get_transform()              # pose[t]
        loc = tf.location
        cte, hint = signed_cte_route(route, loc.x, loc.y, hint)
        steer, steer_rad, _ = pure_pursuit_route(route, tf, hint)  # label[t]

        rel = os.path.join(f"{direction}_lap{lap:02d}", "frames", f"{step:05d}.png")
        cv2.imwrite(os.path.join(out_dir, rel), env.raw_to_bgr(image))
        rows.append(dict(
            image=rel, direction=direction, lap=lap, step=step,
            steer=steer, steer_rad=steer_rad, cte_m=cte,
            speed_mph=env.speed_mph(vehicle), x=loc.x, y=loc.y, yaw=tf.rotation.yaw,
        ))

        vehicle.apply_control(carla.VehicleControl(*_ctrl(speed_ctrl, vehicle, steer)))

        d0 = loc.distance(start)
        if d0 > 50.0:
            left_start = True
        if left_start and d0 < 12.0:
            print(f"    loop closed at step {step} ({len(rows)} frames)")
            break
    return rows


def _ctrl(speed_ctrl, vehicle, steer):
    thr, brk = speed_ctrl.control(vehicle)
    return thr, steer, brk  # VehicleControl(throttle, steer, brake)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clear")
    ap.add_argument("--laps", type=int, default=2)
    ap.add_argument("--direction", default="both", choices=["eastbound", "westbound", "both"])
    ap.add_argument("--max-steps", type=int, default=2500)
    args = ap.parse_args()

    out_dir = os.path.join(C.DATASET_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)
    env.set_clear_weather(world)
    world_map = world.get_map()

    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)
    camera, img_queue = env.spawn_camera(world, vehicle)

    dirs = ["eastbound", "westbound"] if args.direction == "both" else [args.direction]
    all_rows = []
    try:
        for lap in range(args.laps):
            for d in dirs:
                all_rows += collect_lap(world, world_map, vehicle, img_queue,
                                        d, lap, out_dir, args.max_steps)
    finally:
        env.cleanup([camera, vehicle], world, original)

    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    steers = [r["steer"] for r in all_rows]
    n_straight = sum(1 for s in steers if abs(s) <= 0.01)
    print(f"\nCollected {len(all_rows)} frames -> {manifest}")
    print(f"  straight (|steer|<=0.01): {n_straight} ({100*n_straight/len(all_rows):.0f}%) | "
          f"left: {sum(1 for s in steers if s>0.01)} | right: {sum(1 for s in steers if s<-0.01)}")
    print(f"  steer range: [{min(steers):.3f}, {max(steers):.3f}]")


if __name__ == "__main__":
    main()
