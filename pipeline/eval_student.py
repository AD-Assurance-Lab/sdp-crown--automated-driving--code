#!/usr/bin/env python3
"""
Closed-loop evaluation of a StudentNet under a CARLA weather preset. Used for the
clear-weather re-confirm and (M3) the fog/rain/night failure tests of the
clear-only student. CTE is measured against the fixed reference route.

    python eval_student.py --student student_84x28_dagger_r02 --w 84 --h 28 --weather clear
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import carla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import carla_env as env
from route import load_route, signed_cte_route, pure_pursuit_route
from metrics import summarize_cte
from student import StudentNet, student_preprocess
from weather_bounds import BOUNDS_SETS, worst_corner

SPAWNS = {"eastbound": C.SPAWN_EASTBOUND, "westbound": C.SPAWN_WESTBOUND}


def set_weather(world, name):
    """Apply a CARLA weather preset. 'clear' matches the training weather."""
    if name == "clear":
        env.set_clear_weather(world)
        return
    w = world.get_weather()
    if name == "fog":
        w.cloudiness, w.fog_density, w.fog_distance = 90.0, 70.0, 10.0
        w.precipitation, w.wetness = 0.0, 0.0
        w.sun_altitude_angle = 45.0
    elif name == "rain":
        w.cloudiness, w.precipitation, w.precipitation_deposits = 90.0, 85.0, 70.0
        w.wetness, w.fog_density = 80.0, 5.0
        w.sun_altitude_angle = 40.0
    elif name == "night":
        w.cloudiness, w.precipitation, w.fog_density = 30.0, 0.0, 0.0
        w.sun_altitude_angle = -25.0   # sun below horizon
    else:
        raise ValueError(name)
    world.set_weather(w)


def drive(world, vehicle, img_queue, model, device, w, h, direction, max_steps, perturb=None):
    route = load_route(direction)
    hint = None
    sc = env.SpeedController()
    env.teleport(vehicle, SPAWNS[direction])
    env.warmup_to_speed(world, vehicle, img_queue, sc,
                        steer_fn=lambda v: pure_pursuit_route(route, v.get_transform())[0])
    start = carla.Location(x=SPAWNS[direction]["x"], y=SPAWNS[direction]["y"], z=SPAWNS[direction]["z"])
    records, left, stalled, offroad = [], False, 0, 0
    for step in range(max_steps):
        env.update_spectator(world, vehicle)
        world.tick()
        try:
            image = img_queue.get(timeout=2.0)
        except Exception:
            continue
        tf = vehicle.get_transform()
        loc = tf.location
        xin = torch.from_numpy(student_preprocess(env.raw_to_bgr(image), w, h)).unsqueeze(0).to(device)
        if perturb is not None:  # affine weather perturbation, same model as the verifier
            ec, eb = perturb
            xin = torch.clamp(xin * (1.0 + ec) + eb, 0.0, 1.0)
        with torch.no_grad():
            nn_steer = max(-1.0, min(1.0, float(model(xin).item())))
        cte, hint = signed_cte_route(route, loc.x, loc.y, hint)
        thr, brk = sc.control(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=nn_steer))
        records.append(dict(step=step, cte_m=cte, cte_ft=cte * C.M_TO_FT,
                            nn_steer=nn_steer, x=loc.x, y=loc.y))
        d0 = loc.distance(start)
        if d0 > 50:
            left = True
        if left and d0 < 12:
            break
        stalled = stalled + 1 if env.speed_mph(vehicle) < 1 else 0
        offroad = offroad + 1 if abs(cte) > 6 else 0
        if stalled >= 20 or offroad >= 15:
            print(f"    {direction}: aborted at step {step} (departed)")
            break
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--weather", default="clear", choices=["clear", "fog", "rain", "night"])
    ap.add_argument("--affine", default="none", choices=["none", "acdc", "carla"],
                    help="apply affine weather eps (worst corner) in-loop on CLEAR weather")
    ap.add_argument("--acond", default="night", choices=["fog", "rain", "night", "snow"],
                    help="condition for --affine")
    ap.add_argument("--direction", default="both", choices=["eastbound", "westbound", "both"])
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentNet(args.h, args.w).to(device)
    model.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{args.student}.pth"),
                                     map_location=device))
    model.eval()

    # Affine mode: force CLEAR weather and apply the affine eps in-loop (isolates
    # the affine perturbation the verifier reasons about, no rendered weather).
    perturb, label = None, args.weather
    if args.affine != "none":
        perturb = worst_corner(BOUNDS_SETS[args.affine][args.acond])
        label = f"affine-{args.affine}-{args.acond}"
        print(f"AFFINE mode: {label}  eps=(c={perturb[0]:+.4f}, b={perturb[1]:+.4f}) on CLEAR weather")

    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)
    set_weather(world, "clear" if args.affine != "none" else args.weather)
    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)
    camera, img_queue = env.spawn_camera(world, vehicle)

    dirs = ["eastbound", "westbound"] if args.direction == "both" else [args.direction]
    results = {}
    try:
        for d in dirs:
            recs = drive(world, vehicle, img_queue, model, device, args.w, args.h, d,
                         args.max_steps, perturb=perturb)
            st = summarize_cte([r["cte_m"] for r in recs])
            results[d] = st
            print(f"  [{label}/{d}] over-budget={st.get('frac_over_budget',1)*100:5.1f}%  "
                  f"max|CTE|={st.get('max_abs_cte_m',0)*C.M_TO_FT:5.2f}ft  "
                  f"-> {'PASS' if st.get('passed') else 'FAIL'}")
            os.makedirs(C.RESULTS_DIR, exist_ok=True)
            tag = f"_{args.tag}" if args.tag else ""
            with open(os.path.join(C.RESULTS_DIR, f"evalstu_{args.student}_{label}_{d}{tag}.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(recs[0].keys())); wr.writeheader(); wr.writerows(recs)
    finally:
        env.cleanup([camera, vehicle], world, original)

    print(f"\n===== {args.student} @ {label} =====")
    for d, st in results.items():
        print(f"  {d:10s}: {'PASS' if st.get('passed') else 'FAIL'} "
              f"(over {st.get('frac_over_budget',1)*100:.1f}%, max {st.get('max_abs_cte_m',0)*C.M_TO_FT:.2f}ft)")


if __name__ == "__main__":
    main()
