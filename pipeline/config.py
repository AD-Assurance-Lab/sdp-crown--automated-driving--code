"""
Single source of truth for the E2E steering pipeline.

Design rule: measured PRIMITIVES are declared explicitly; every SAFETY number
(CTE budget, steering corridor) is DERIVED from them below, so the two can never
silently disagree. All primitives marked [MEASURED] were verified in CARLA
(Town04, Tesla Model 3) on 2026-07-22 via scripts/probe_geometry.
"""
import math
import os

# ── CARLA connection ─────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 2000
CLIENT_TIMEOUT_S = 120.0
TRAFFIC_MANAGER_PORT = 8005
CARLA_ROOT = "/home/za/carla"

# ── Map ──────────────────────────────────────────────────────────────────────
MAP_NAME = "Town04"

# ── Vehicle (Tesla Model 3, as instantiated in CARLA) ────────────────────────
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
WHEELBASE_M = 3.005          # [MEASURED] from CARLA wheel positions (spec is 2.87)
VEHICLE_WIDTH_M = 2.164      # [MEASURED] CARLA bounding box (includes mirrors)
MAX_STEER_RAD = 1.2217       # [MEASURED] 70.0 deg front-wheel max steer

# ── Camera ───────────────────────────────────────────────────────────────────
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FOV = 90
CAM_X, CAM_Y, CAM_Z = 1.6, 0.0, 1.2   # hood mount (m, vehicle frame)
CROP_TOP, CROP_BOT = 180, 400          # remove sky + hood before resize
INPUT_W, INPUT_H = 200, 66             # model input (PilotNet-style)

# ── Speed (fixed longitudinal, to remove velocity as a variable) ─────────────
TARGET_SPEED_MPH = 20.0
TARGET_SPEED_MS = 8.9408
MPH_PER_MS = 2.23694

# ── Simulation timing ────────────────────────────────────────────────────────
FIXED_DT = 0.2               # s per tick
SIM_HZ = 1.0 / FIXED_DT      # 5 Hz

# ── Pure-pursuit expert ──────────────────────────────────────────────────────
LOOKAHEAD_M = 5.0

# ── Road geometry ────────────────────────────────────────────────────────────
LANE_WIDTH_M = 3.500         # [MEASURED] constant on Town04 highway, both dirs

# ── Spawn points (start just after the western intersection) ─────────────────
SPAWN_EASTBOUND = {"x": -357.1, "y": 30.0, "z": 0.5, "yaw": 0.0}
SPAWN_WESTBOUND = {"x": -396.8, "y": 12.8, "z": 0.5, "yaw": 180.0}

# ── Unit conversions ─────────────────────────────────────────────────────────
M_TO_FT = 3.28084

# ═══════════════════════════════════════════════════════════════════════════
# DERIVED SAFETY CRITERIA  (do not hardcode — computed from primitives above)
# ═══════════════════════════════════════════════════════════════════════════

# Success criterion: no part of the vehicle body leaves its lane. Expressed as a
# limit on the vehicle-center-to-lane-center CTE. Using the CARLA bounding box
# (2.164 m, includes mirrors) is the defensible in-simulator choice.
#   (spec body 1.849 m would give 2.71 ft — kept for paper discussion only.)
CTE_BUDGET_M = (LANE_WIDTH_M - VEHICLE_WIDTH_M) / 2.0     # 0.668 m
CTE_BUDGET_FT = CTE_BUDGET_M * M_TO_FT                    # 2.19 ft

# Verification corridor: max per-frame steering deviation that keeps CTE within
# budget if a systematic bias persists for T_HORIZON_S. Bicycle model:
#   y(t) = v^2 * dtheta / (2L) * t^2  ->  dtheta_max = 2 L y / (v^2 T^2)
T_HORIZON_S = 1.0
STEER_CORRIDOR_RAD = (2.0 * WHEELBASE_M * CTE_BUDGET_M) / (TARGET_SPEED_MS ** 2 * T_HORIZON_S ** 2)
STEER_CORRIDOR_DEG = math.degrees(STEER_CORRIDOR_RAD)    # 2.88 deg
STEER_CORRIDOR_NORM = STEER_CORRIDOR_RAD / MAX_STEER_RAD  # 0.041 (network output units)

# ── Output paths ─────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_BASE)
DATASET_DIR = os.path.join(_BASE, "data")
CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
RESULTS_DIR = os.path.join(_BASE, "results")


def summary():
    """Human-readable dump of the derived safety criteria."""
    return (
        f"CTE budget      : {CTE_BUDGET_M:.4f} m ({CTE_BUDGET_FT:.3f} ft)\n"
        f"Steer corridor  : {STEER_CORRIDOR_DEG:.3f} deg "
        f"({STEER_CORRIDOR_RAD:.4f} rad, {STEER_CORRIDOR_NORM:.4f} norm) "
        f"@ T={T_HORIZON_S}s\n"
        f"Speed / dt      : {TARGET_SPEED_MPH} mph, {FIXED_DT}s ({SIM_HZ:.0f} Hz)\n"
        f"Wheelbase       : {WHEELBASE_M} m | max steer {math.degrees(MAX_STEER_RAD):.1f} deg"
    )


if __name__ == "__main__":
    print(summary())
