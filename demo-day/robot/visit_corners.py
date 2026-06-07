"""Visit each calibrated canvas corner in succession.

Sequence:
    INIT_POS -> TL -> TR -> BR -> BL -> INIT_POS

The orientation at the time the script starts (assumed to match the
orientation used during calibration) is held throughout. Corner
positions are read from the JSON written by
demo-day/robot/calibrate_canvas.py (default: demo-day/canvas_corners.json).

This script is fully self-contained and does not import from the
sibling robot/ package, mirroring the style of
demo-day/robot/calibrate_canvas.py and demo-day/robot/orient_camera.py.

Usage:
    python visit_corners.py
    python visit_corners.py --dwell-s 2.0
    python visit_corners.py --corners-json path/to/canvas_corners.json
    python visit_corners.py --skip-final-init   # stop at BL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import redis
except ImportError:
    redis = None


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_CORNERS_JSON = DEMO_DAY_DIR / "canvas_corners.json"
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"

ROBOT_NAME = "Titania"
CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"
CARTESIAN_CONTROLLER = "cartesian_controller"

DT = 0.01                            # 100 Hz main loop
POS_TOL_M = 1.0e-2                   # 10 mm position tolerance
DWELL_AT_WAYPOINT_S = 1.0
INTER_GOAL_REFRESH_S = 0.05          # how often to re-send the cartesian goal


def _load_init_pos() -> np.ndarray:
    """Load INIT_POS from demo-day/config.json so all scripts share one source."""
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return np.array(cfg["init_pos_m"], dtype=float)


# Robot "home" pose -- shared with calibrate_canvas.py / orient_camera.py.
INIT_POS = _load_init_pos()

CORNER_ORDER = ("TL", "TR", "BR", "BL")


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::current_orientation"
    )
    active_controller: str = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


redis_keys = RedisKeys()


# ============================================================
# REDIS HELPERS (same style as the other scripts in this folder)
# ============================================================

def decode_redis_value(val):
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def read_np(redis_client, key, expected_shape):
    while True:
        val = redis_client.get(key)
        if val is not None:
            try:
                arr = np.array(json.loads(decode_redis_value(val)), dtype=float)
                if arr.shape == expected_shape:
                    return arr
            except Exception:
                pass
        time.sleep(0.01)


def set_cartesian_goal(redis_client, position, orientation):
    redis_client.set(
        redis_keys.cartesian_task_goal_position,
        json.dumps(np.array(position, dtype=float).tolist()),
    )
    redis_client.set(
        redis_keys.cartesian_task_goal_orientation,
        json.dumps(np.array(orientation, dtype=float).tolist()),
    )


def set_active_controller(redis_client, name):
    while True:
        active_raw = redis_client.get(redis_keys.active_controller)
        active = decode_redis_value(active_raw) if active_raw is not None else None
        if active == name:
            break
        redis_client.set(redis_keys.active_controller, name)
        time.sleep(0.05)


def ensure_robot_ready(redis_client) -> bool:
    config_raw = redis_client.get(redis_keys.config_file_name)
    if config_raw is None:
        print("Could not read config file name from Redis.")
        print("Missing key:", redis_keys.config_file_name)
        return False
    config_file_name = decode_redis_value(config_raw)
    if config_file_name != CONFIG_FILE_FOR_THIS_SCRIPT:
        print(f"This script expects config file: {CONFIG_FILE_FOR_THIS_SCRIPT}")
        print(f"Current config file: {config_file_name}")
        return False
    return True


def position_error(current_pos, goal_pos) -> float:
    return float(
        np.linalg.norm(
            np.array(goal_pos, dtype=float) - np.array(current_pos, dtype=float)
        )
    )


# ============================================================
# CORNER LOADING
# ============================================================

def load_corner_world_positions(corners_json: Path) -> dict[str, np.ndarray]:
    """Read the calibration JSON and return TL/TR/BR/BL in world frame.

    Preferred source: ``payload["corners_in_world_frame_m"]`` (added by
    calibrate_canvas.py).

    Fallback (for older JSONs): reconstruct world coords by simple addition
    ``INIT_POS + corner_in_tip_frame_m``. This matches the convention used
    by the updated calibrate_canvas.py, which treats the tip-frame axes as
    aligned to the world frame at INIT_POS.
    """
    with corners_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    world = payload.get("corners_in_world_frame_m")
    if world is not None:
        return {name: np.array(world[name], dtype=float) for name in CORNER_ORDER}

    tip = payload.get("corners_in_tip_frame_m")
    if tip is None:
        raise RuntimeError(
            f"{corners_json} is missing both 'corners_in_world_frame_m' and "
            "'corners_in_tip_frame_m'. Re-run demo-day/robot/calibrate_canvas.py."
        )

    print(
        "Reconstructing world-frame corners as INIT_POS + tip-frame offset "
        "(older JSON format)."
    )
    return {
        name: INIT_POS + np.array(tip[name], dtype=float) for name in CORNER_ORDER
    }


# ============================================================
# STATE MACHINE
# ============================================================

def go_to_waypoint(
    redis_client,
    target_pos: np.ndarray,
    hold_orientation: np.ndarray,
    label: str,
    dwell_s: float,
) -> None:
    print(f"\n-> {label}: target = {np.round(target_pos, 5).tolist()}")
    set_cartesian_goal(redis_client, target_pos, hold_orientation)

    loop_time = 0.0
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9
    last_refresh = 0.0

    while True:
        loop_time += DT
        time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

        # Continuously re-assert the goal so the controller doesn't drift if
        # another process touches the Redis key.
        if loop_time - last_refresh >= INTER_GOAL_REFRESH_S:
            set_cartesian_goal(redis_client, target_pos, hold_orientation)
            last_refresh = loop_time

        current = read_np(redis_client, redis_keys.cartesian_task_current_position, (3,))
        err = position_error(current, target_pos)
        print(f"{label} | pos_error: {err:.5f}")

        if err < POS_TOL_M:
            break

    print(f"Reached {label}. Dwelling {dwell_s:.2f}s.")
    time.sleep(dwell_s)


def run_visit_sequence(
    *,
    corners_json: Path,
    dwell_s: float,
    skip_final_init: bool,
) -> int:
    if not corners_json.is_file():
        print(f"Corners JSON not found: {corners_json}")
        print("Run demo-day/robot/calibrate_canvas.py first.")
        return 1

    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print(f"Using controller: {CARTESIAN_CONTROLLER}")

    hold_orientation = read_np(
        redis_client,
        redis_keys.cartesian_task_current_orientation,
        (3, 3),
    )
    print("Holding orientation throughout (read from current EE pose):")
    print(hold_orientation)

    current = read_np(
        redis_client, redis_keys.cartesian_task_current_position, (3,)
    )
    print(f"Starting position: {current}")

    corner_world = load_corner_world_positions(corners_json)
    print("\nCanvas corners (world frame, meters):")
    print(f"  INIT_POS: {INIT_POS.tolist()}")
    for name in CORNER_ORDER:
        print(f"  {name}: {np.round(corner_world[name], 5).tolist()}")

    sequence: list[tuple[str, np.ndarray]] = [("INIT_POS", INIT_POS)]
    sequence.extend((name, corner_world[name]) for name in CORNER_ORDER)
    if not skip_final_init:
        sequence.append(("INIT_POS", INIT_POS))

    print("\nWaypoint sequence:")
    for label, target in sequence:
        print(f"  {label}: {np.round(target, 5).tolist()}")
    print()

    for label, target in sequence:
        go_to_waypoint(
            redis_client,
            target,
            hold_orientation,
            label=label,
            dwell_s=dwell_s,
        )

    print("\nFinished visiting all waypoints.")
    return 0


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corners-json",
        default=str(DEFAULT_CORNERS_JSON),
        help=f"Path to the canvas corners JSON (default: {DEFAULT_CORNERS_JSON}).",
    )
    parser.add_argument(
        "--dwell-s",
        type=float,
        default=DWELL_AT_WAYPOINT_S,
        help=f"Seconds to dwell at each waypoint after arrival (default: {DWELL_AT_WAYPOINT_S}).",
    )
    parser.add_argument(
        "--skip-final-init",
        action="store_true",
        help="Stop at BL instead of returning to INIT_POS.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_visit_sequence(
        corners_json=Path(args.corners_json),
        dwell_s=args.dwell_s,
        skip_final_init=args.skip_final_init,
    )


if __name__ == "__main__":
    sys.exit(main())
