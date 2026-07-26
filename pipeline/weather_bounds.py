"""
Shared calibrated weather epsilon-bounds (eps_c contrast, eps_b brightness),
used by both the verifier (verify.py) and the closed-loop affine test
(eval_student.py) so the two can never drift apart.

ACDC  = real-world clear/adverse image pairs.
CARLA = the simulator's own rendered weather (much more severe).
"""

ACDC_BOUNDS = {
    "rain":  {"c": (-0.4337, 0.0), "b": (0.0, 0.1013), "masked": True},
    "fog":   {"c": (-0.1504, 0.0), "b": (0.0, 0.1145), "masked": False},
    "night": {"c": (-0.5865, 0.0), "b": (-0.1557, 0.0), "masked": False},
    "snow":  {"c": (-0.3989, 0.0), "b": (0.0, 0.1809), "masked": True},
}
CARLA_BOUNDS = {
    "rain":  {"c": (-0.5875, 0.0211), "b": (-0.1635, 0.3059), "masked": True},
    "fog":   {"c": (-0.7830, -0.3979), "b": (0.2671, 0.5946), "masked": False},
    "night": {"c": (-0.8879, 0.0601), "b": (-0.6082, -0.0454), "masked": False},
    "snow":  {"c": (-0.3989, 0.0), "b": (0.0, 0.1809), "masked": True},
}
BOUNDS_SETS = {"acdc": ACDC_BOUNDS, "carla": CARLA_BOUNDS}


def worst_corner(bounds):
    """Most-degraded corner of the eps-box: max contrast drop + the larger-magnitude
    brightness shift. A single representative worst-case for the closed-loop test."""
    ec = bounds["c"][0]  # most negative contrast (biggest drop)
    blo, bhi = bounds["b"]
    eb = blo if abs(blo) >= abs(bhi) else bhi
    return ec, eb
