#!/usr/bin/env python3
"""
DAgger (Ross et al., 2011) for the steering policy.

Each round: the CURRENT policy drives the full loop both directions; EVERY frame
it visits is labeled with the route pure-pursuit recovery action (steer back to
the intended centerline) and saved. That aggregated data is added to the training
set and the model is retrained from scratch. The off-center states the policy
wanders into ARE the recovery data — no autopilot hand-over, no PID oscillations.

The same drive both (a) evaluates the current policy (route-based CTE) and
(b) collects the next round's data. Stops when the driven policy meets budget.

    python dagger.py --init steering_bc_baseline --rounds 6
"""
import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import torch  # noqa: E402
import carla  # noqa: E402

import config as C  # noqa: E402
import carla_env as env  # noqa: E402
from imaging import preprocess_for_model  # noqa: E402
from route import load_route, signed_cte_route, pure_pursuit_route  # noqa: E402
from metrics import summarize_cte  # noqa: E402
from model import CarlaSteeringNet  # noqa: E402
from train import train_model  # noqa: E402

SPAWNS = {"eastbound": C.SPAWN_EASTBOUND, "westbound": C.SPAWN_WESTBOUND}
FIELDS = ["image", "weather", "direction", "step", "steer", "steer_rad", "nn_steer",
          "cte_m", "speed_mph", "x", "y", "yaw"]


def load_model(name, device):
    m = CarlaSteeringNet().to(device)
    m.load_state_dict(torch.load(os.path.join(C.CHECKPOINT_DIR, f"{name}.pth"),
                                 map_location=device))
    m.eval()
    return m


def drive_collect(world, vehicle, img_queue, model, device, weather, direction, round_dir, max_steps):
    """Policy drives; every frame is labeled with the route recovery expert and saved.
    Returns (rows, cte_stats). CTE is against the fixed reference route."""
    route = load_route(direction)
    hint = None
    speed_ctrl = env.SpeedController()
    env.teleport(vehicle, SPAWNS[direction])
    # warm up on-center with the expert, then hand control to the policy
    env.warmup_to_speed(
        world, vehicle, img_queue, speed_ctrl,
        steer_fn=lambda veh: pure_pursuit_route(route, veh.get_transform())[0],
    )
    seg = os.path.join(weather, direction)
    frames_dir = os.path.join(round_dir, seg, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    start = carla.Location(x=SPAWNS[direction]["x"], y=SPAWNS[direction]["y"],
                           z=SPAWNS[direction]["z"])

    rows, left, stalled, offroad = [], False, 0, 0
    for step in range(max_steps):
        env.update_spectator(world, vehicle)
        world.tick()
        try:
            image = img_queue.get(timeout=2.0)
        except Exception:
            continue
        tf = vehicle.get_transform()
        loc = tf.location

        bgr = env.raw_to_bgr(image)
        xin = torch.from_numpy(preprocess_for_model(bgr)).unsqueeze(0).to(device)
        with torch.no_grad():
            nn_steer = max(-1.0, min(1.0, float(model(xin).item())))

        cte, hint = signed_cte_route(route, loc.x, loc.y, hint)
        exp_steer, exp_rad, _ = pure_pursuit_route(route, tf, hint)   # LABEL

        rel = os.path.join(seg, "frames", f"{step:05d}.png")
        cv2.imwrite(os.path.join(round_dir, rel), bgr)
        rows.append(dict(
            image=rel, weather=weather, direction=direction, step=step,
            steer=exp_steer, steer_rad=exp_rad, nn_steer=nn_steer,
            cte_m=cte, speed_mph=env.speed_mph(vehicle), x=loc.x, y=loc.y, yaw=tf.rotation.yaw,
        ))

        thr, brk = speed_ctrl.control(vehicle)
        vehicle.apply_control(carla.VehicleControl(throttle=thr, brake=brk, steer=nn_steer))

        d0 = loc.distance(start)
        if d0 > 50.0:
            left = True
        if left and d0 < 12.0:
            break
        stalled = stalled + 1 if env.speed_mph(vehicle) < 1.0 else 0
        offroad = offroad + 1 if abs(cte) > 6.0 else 0
        if stalled >= 20 or offroad >= 15:
            print(f"    {direction}: aborted at step {step} (stall/offroad)")
            break
    return rows, summarize_cte([r["cte_m"] for r in rows])


def write_manifest(round_dir, rows):
    os.makedirs(round_dir, exist_ok=True)
    path = os.path.join(round_dir, "manifest.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="clear", help="base BC dataset name")
    ap.add_argument("--init", default="steering_bc_baseline", help="initial policy checkpoint")
    ap.add_argument("--rounds", type=int, default=6, help="max DAgger retrains")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-4,
                    help="LR for warm-start retrains (gentle fine-tune from prior round)")
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--out-prefix", default="steering_dagger")
    ap.add_argument("--weathers", default="clear",
                    help="comma-separated weather presets to drive/collect each round")
    ap.add_argument("--dagger-dir", default="dagger",
                    help="subdir under data/ for this run's rounds (use a distinct name "
                         "per model, e.g. dagger_mixed, so rounds don't collide)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weathers = args.weathers.split(",")
    dagger_dir = os.path.join(C.DATASET_DIR, args.dagger_dir)
    manifests = [os.path.join(C.DATASET_DIR, args.base, "manifest.csv")]
    model = load_model(args.init, device)
    current = args.init

    client = env.connect()
    world = env.load_town04(client)
    original = env.enable_sync_mode(world)
    vehicle = env.spawn_vehicle(world, C.SPAWN_EASTBOUND)
    camera, img_queue = env.spawn_camera(world, vehicle)

    history = []
    try:
        for r in range(args.rounds + 1):
            round_dir = os.path.join(dagger_dir, f"round{r:02d}")
            print(f"\n{'#'*64}\n# DAgger round {r} — evaluating policy '{current}'\n{'#'*64}")
            rows, passed = [], True
            for weather in weathers:
                env.set_weather(world, weather)
                for d in ["eastbound", "westbound"]:
                    drows, st = drive_collect(world, vehicle, img_queue, model, device,
                                              weather, d, round_dir, args.max_steps)
                    rows += drows
                    ob = st.get("frac_over_budget", 1) * 100
                    mx = st.get("max_abs_cte_m", 0) * C.M_TO_FT
                    print(f"  [{weather}/{d}] over-budget={ob:5.1f}%  max|CTE|={mx:5.2f}ft  "
                          f"-> {'PASS' if st.get('passed') else 'FAIL'}")
                    if not st.get("passed"):
                        passed = False
            mpath = write_manifest(round_dir, rows)
            history.append((r, current, passed))

            if passed:
                print(f"\n*** PASSED at round {r} with policy '{current}' ***")
                break
            if r == args.rounds:
                print(f"\nExhausted {args.rounds} rounds without passing "
                      f"(last policy '{current}').")
                break

            manifests.append(mpath)
            new = f"{args.out_prefix}_r{r:02d}"
            print(f"  aggregating {len(manifests)} manifests, warm-start from '{current}' -> {new}")
            train_model(manifests, new, epochs=args.epochs, balance=True, quiet=True,
                        weathers=weathers, init_from=current, lr=args.lr)
            model = load_model(new, device)
            current = new
    finally:
        env.cleanup([camera, vehicle], world, original)

    print("\n===== DAgger summary =====")
    for r, name, passed in history:
        print(f"  round {r}: {name:24s} {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
