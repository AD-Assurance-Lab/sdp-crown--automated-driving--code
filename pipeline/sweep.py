#!/usr/bin/env python3
"""
M2 resolution sweep: distill the teacher into a StudentNet at several input
resolutions and closed-loop test each, to find the SMALLEST input that still
drives within budget (smallest = cheapest to verify). Reports over-budget % and
ReLU-neuron count per resolution. Connects to an already-running CARLA.

    python sweep.py --resolutions 120x40,96x32,84x28,72x24
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import carla

import config as C
import carla_env as env
from route import load_route, signed_cte_route, pure_pursuit_route
from metrics import summarize_cte
from student import StudentNet, student_preprocess
from distill import distill_student

SPAWNS = {"eastbound": C.SPAWN_EASTBOUND, "westbound": C.SPAWN_WESTBOUND}


def student_steer_fn(model, device, w, h):
    def fn(bgr):
        x = torch.from_numpy(student_preprocess(bgr, w, h)).unsqueeze(0).to(device)
        with torch.no_grad():
            return max(-1.0, min(1.0, float(model(x).item())))
    return fn


def drive(world, vehicle, img_queue, steer_fn, direction, max_steps):
    route = load_route(direction)
    hint = None
    sc = env.SpeedController()
    env.teleport(vehicle, SPAWNS[direction])
    env.warmup_to_speed(world, vehicle, img_queue, sc,
                        steer_fn=lambda v: pure_pursuit_route(route, v.get_transform())[0])
    start = carla.Location(x=SPAWNS[direction]["x"], y=SPAWNS[direction]["y"], z=SPAWNS[direction]["z"])
    ctes, left, stalled, offroad = [], False, 0, 0
    for step in range(max_steps):
        env.update_spectator(world, vehicle)
        world.tick()
        try:
            image = img_queue.get(timeout=2.0)
        except Exception:
            continue
        tf = vehicle.get_transform()
        loc = tf.location
        steer = steer_fn(env.raw_to_bgr(image))
        cte, hint = signed_cte_route(route, loc.x, loc.y, hint)
        thr, brk = sc.control(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=steer))
        ctes.append(cte)
        d0 = loc.distance(start)
        if d0 > 50:
            left = True
        if left and d0 < 12:
            break
        stalled = stalled + 1 if env.speed_mph(vehicle) < 1 else 0
        offroad = offroad + 1 if abs(cte) > 6 else 0
        if stalled >= 20 or offroad >= 15:
            break
    return summarize_cte(ctes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", default="120x40,96x32,84x28,72x24")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()
    res = [tuple(int(v) for v in r.split("x")) for r in args.resolutions.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)
    env.set_clear_weather(world)
    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)
    camera, img_queue = env.spawn_camera(world, vehicle)

    results = []
    try:
        for (w, h) in res:
            name = f"student_{w}x{h}"
            print(f"\n{'='*60}\n# {w}x{h} — distilling {name}\n{'='*60}", flush=True)
            info = distill_student(w, h, name, epochs=args.epochs, quiet=True)
            model = StudentNet(h, w).to(device)
            model.load_state_dict(torch.load(info["ckpt"], map_location=device))
            model.eval()
            fn = student_steer_fn(model, device, w, h)
            row = {"res": f"{w}x{h}", "neurons": info["neurons"],
                   "kd_rmse": info["best_val"] ** 0.5}
            for d in ["eastbound", "westbound"]:
                st = drive(world, vehicle, img_queue, fn, d, args.max_steps)
                row[d] = st.get("frac_over_budget", 1) * 100
                row[d + "_max"] = st.get("max_abs_cte_m", 0) * C.M_TO_FT
                print(f"  [{d}] over-budget={row[d]:5.1f}%  max|CTE|={row[d+'_max']:5.2f}ft  "
                      f"-> {'PASS' if st.get('passed') else 'FAIL'}", flush=True)
            results.append(row)
    finally:
        env.cleanup([camera, vehicle], world, original)

    print("\n===== SWEEP SUMMARY (KD-only, no DAgger polish yet) =====")
    print(f"{'res':>8} {'neurons':>8} {'KD-rmse':>8} {'E-over%':>8} {'W-over%':>8}")
    for r in results:
        print(f"{r['res']:>8} {r['neurons']:>8} {r['kd_rmse']:>8.4f} "
              f"{r['eastbound']:>8.1f} {r['westbound']:>8.1f}")


if __name__ == "__main__":
    main()
