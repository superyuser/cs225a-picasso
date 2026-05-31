import math

import numpy as np

from config import DAMP_POWER, DAMP_START


def smoothstep(s):
    """
    Smooth scalar interpolation.
    """
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def make_landing_s_values(n, damp_start=DAMP_START, power=DAMP_POWER):
    """
    Generate progress values from 0 to 1 with smaller increments near the end.

    This makes the robot land gently at the target point.
    """
    values = []

    for i in range(1, n + 1):
        raw_s = i / n

        if raw_s <= damp_start:
            local = raw_s / damp_start
            s = damp_start * smoothstep(local)
        else:
            local = (raw_s - damp_start) / (1.0 - damp_start)
            eased = 1.0 - (1.0 - local) ** power
            s = damp_start + (1.0 - damp_start) * eased

        values.append(s)

    return values


def make_position_segment(p0, p1, step_m, min_steps=2):
    """
    Build an interpolated Cartesian position segment from p0 to p1.

    This is still position-only.
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)

    dist = np.linalg.norm(p1 - p0)
    n = max(min_steps, int(math.ceil(dist / step_m)))

    s_values = make_landing_s_values(n)

    segment = []

    for s in s_values:
        p = (1.0 - s) * p0 + s * p1
        segment.append(p)

    return segment
