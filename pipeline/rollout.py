#!/usr/bin/env python3
"""
Verifier-in-the-loop worst-case rollout (M3-C, understanding experiment).

Each step: (1) get the clean frame from the ACTUAL (drifted) state; (2) bound the
student's steering under the weather eps-box via an oracle; (3) apply the worst-
case endpoint (the one that UNDER-corrects relative to pure-pursuit, i.e. drifts
outward); (4) tick the simulator, which gives the next real frame. This uses CARLA
as the high-fidelity dynamics+perception oracle (no vehicle model), and turns the
per-frame certificate into a single closed-loop worst-case trajectory.

Oracles (steering interval [lb, ub] over the eps-box):
  concrete   : grid the eps-box, student's TRUE min/max output (no relaxation loose-
               ness, fast) -- the clean baseline for understanding compounding.
  CROWN/SDP  : the certified relaxation bound (formal; slower; may be loose).

    python rollout.py --student student_84x28_dagger_r02 --w 84 --h 28 \
        --bounds acdc --acond night --oracle concrete --direction eastbound
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # auto_LiRPA

import numpy as np
import torch
import carla

import config as C
import carla_env as env
from route import load_route, signed_cte_route, pure_pursuit_route
from student import StudentNet, student_preprocess
from weather_bounds import BOUNDS_SETS

SPAWNS = {"eastbound": C.SPAWN_EASTBOUND, "westbound": C.SPAWN_WESTBOUND}


def concrete_bound(model, x0, crange, brange, device, grid=7):
    """True [min,max] student steering over the eps-box (unclamped affine, matching
    the semantic layer the verifier reasons about). Fast: grid^2 forward passes."""
    ecs = np.linspace(crange[0], crange[1], grid)
    ebs = np.linspace(brange[0], brange[1], grid)
    outs = []
    with torch.no_grad():
        for ec in ecs:
            for eb in ebs:
                outs.append(float(model(x0 * (1.0 + ec) + eb).item()))
    return min(outs), max(outs)


def steer_bound(model, x0, b, oracle, device):
    if oracle == "concrete":
        return concrete_bound(model, x0, b["c"], b["b"], device)
    from verify import verify_frame  # lazy (pulls auto_LiRPA)
    r = verify_frame(model, x0, b, C.STEER_CORRIDOR_NORM, device,
                     method={"crown": "CROWN", "sdp": "SDP-CROWN"}[oracle])
    return r["lb"], r["ub"]


def rollout(world, vehicle, img_queue, model, device, w, h, direction, b, oracle, max_steps):
    route = load_route(direction)
    hint = None
    sc = env.SpeedController()
    env.teleport(vehicle, SPAWNS[direction])
    env.warmup_to_speed(world, vehicle, img_queue, sc,
                        steer_fn=lambda v: pure_pursuit_route(route, v.get_transform())[0])
    start = carla.Location(x=SPAWNS[direction]["x"], y=SPAWNS[direction]["y"], z=SPAWNS[direction]["z"])

    records, left, offroad = [], False, 0
    for step in range(max_steps):
        env.update_spectator(world, vehicle)
        world.tick()
        try:
            image = img_queue.get(timeout=2.0)
        except Exception:
            continue
        tf = vehicle.get_transform()
        loc = tf.location
        x0 = torch.from_numpy(student_preprocess(env.raw_to_bgr(image), w, h)).unsqueeze(0).to(device)

        cte, hint = signed_cte_route(route, loc.x, loc.y, hint)
        pp_steer, _, _ = pure_pursuit_route(route, tf, hint)   # the correcting/route-following steer
        lb, ub = steer_bound(model, x0, b, oracle, device)
        # worst case = UNDER-correct relative to pure pursuit (drift outward / under-steer curve)
        worst = lb if pp_steer >= 0 else ub
        worst = max(-1.0, min(1.0, worst))

        thr, brk = sc.control(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=worst))
        records.append(dict(step=step, cte_ft=cte * C.M_TO_FT, pp_steer=pp_steer,
                            lb=lb, ub=ub, worst=worst, x=loc.x, y=loc.y))

        d0 = loc.distance(start)
        if d0 > 50:
            left = True
        if left and d0 < 12:
            print(f"  loop closed at step {step} (survived worst-case!)")
            break
        offroad = offroad + 1 if abs(cte) > 6 else 0
        if offroad >= 10:
            print(f"  DEPARTED at step {step}: x={loc.x:.0f} y={loc.y:.0f} CTE={cte*C.M_TO_FT:.1f}ft")
            break
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--bounds", default="acdc", choices=["acdc", "carla"])
    ap.add_argument("--acond", default="night", choices=["clear", "fog", "rain", "night", "snow"])
    ap.add_argument("--oracle", default="concrete", choices=["concrete", "crown", "sdp"])
    ap.add_argument("--direction", default="eastbound", choices=["eastbound", "westbound"])
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentNet(args.h, args.w).to(device)
    model.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{args.student}.pth"),
                                     map_location=device))
    model.eval()
    # 'clear' = zero perturbation control: rollout should reproduce the clear
    # closed-loop result (survive), confirming the machinery isn't spuriously departing.
    b = ({"c": (0.0, 0.0), "b": (0.0, 0.0), "masked": False}
         if args.acond == "clear" else BOUNDS_SETS[args.bounds][args.acond])

    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)
    env.set_clear_weather(world)
    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)
    camera, img_queue = env.spawn_camera(world, vehicle)

    print(f"WORST-CASE ROLLOUT | {args.direction} | {args.bounds}-{args.acond} "
          f"eps_c{b['c']} eps_b{b['b']} | oracle={args.oracle}")
    try:
        recs = rollout(world, vehicle, img_queue, model, device, args.w, args.h,
                       args.direction, b, args.oracle, args.max_steps)
    finally:
        env.cleanup([camera, vehicle], world, original)

    ctes = [abs(r["cte_ft"]) for r in recs]
    widths = [r["ub"] - r["lb"] for r in recs]
    print(f"\n== steps={len(recs)} max|CTE|={max(ctes):.2f}ft "
          f"mean steer-interval width={np.mean(widths):.4f} "
          f"({'DEPARTED' if max(ctes) > 6*C.M_TO_FT else 'survived'}) ==")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    out = os.path.join(C.RESULTS_DIR, f"rollout_{args.direction}_{args.bounds}_{args.acond}_{args.oracle}.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(recs[0].keys())); wr.writeheader(); wr.writerows(recs)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
