"""Probe the Cartesian TOOL_INIT approach followed by a simple -Z move.

Flow:
    INIT_POS -> TOOL_WP1 -> TOOL_WP2 -> TOOL_INIT as Cartesian poses
    then command:
        current_position + [0, 0, -0.1] m

Usage:
    python robot/tool_init_z_probe.py
    python robot/tool_init_z_probe.py --z-distance-m 0.03
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

try:
    import redis
except ImportError:
    redis = None

from helpers.tool_station_coords import (
    CARTESIAN_MAX_STEP_M,
    CONFIG_FILE_FOR_THIS_SCRIPT,
    DT,
    POS_TOL_M,
    STATUS_PERIOD_S,
    TIMEOUT_PER_WAYPOINT_S,
    TOOL_ARC_MIN_RADIUS_M,
    TOOL_ARC_SAGITTA_M,
    TOOL_ARC_SIDE,
    approach_tool_init_cartesian,
    decode_redis_value,
    position_error,
    read_cartesian_pose,
    read_np,
    redis_keys,
    rotation_error_rad,
    set_cartesian_goal,
)


DEFAULT_Z_DISTANCE_M = 0.1
DEFAULT_CARTESIAN_DWELL_S = 0.5


def ensure_expected_config(redis_client, expected_config_file: str) -> bool:
    config_raw = redis_client.get(redis_keys.config_file_name)
    if config_raw is None:
        print("Could not read config file name from Redis.")
        print("Missing key:", redis_keys.config_file_name)
        return False

    config_file_name = decode_redis_value(config_raw)
    if config_file_name != expected_config_file:
        print("This script is meant to be used with config file:", expected_config_file)
        print("Current config file:", config_file_name)
        return False

    return True


def command_minus_z_translation(
    redis_client,
    *,
    start_position: np.ndarray,
    hold_orientation: np.ndarray,
    z_distance_m: float,
    max_step_m: float,
    pos_tol_m: float,
    dwell_s: float,
    status_period_s: float,
    timeout_s: float,
) -> bool:
    target_position = np.asarray(start_position, dtype=float).copy()
    target_position[2] -= abs(float(z_distance_m))

    print("\nCommanding Cartesian -Z translation.")
    print("  start position:", np.round(start_position, 5).tolist())
    print("  target position:", np.round(target_position, 5).tolist())
    print("  delta m:", np.round(target_position - start_position, 5).tolist())

    delta = target_position - np.asarray(start_position, dtype=float)
    distance = float(np.linalg.norm(delta))
    num_steps = max(1, int(math.ceil(distance / max(max_step_m, 1.0e-5))))
    print("  Cartesian samples:", num_steps)

    loop_time = 0.0
    last_status = 0.0
    start = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    for step_index in range(1, num_steps + 1):
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        u = step_index / num_steps
        s = 3.0 * u * u - 2.0 * u * u * u
        commanded_position = start_position + s * delta
        set_cartesian_goal(redis_client, commanded_position, hold_orientation)

        current_position, current_orientation = read_cartesian_pose(redis_client)
        err_to_final = position_error(current_position, target_position)
        err_to_command = position_error(current_position, commanded_position)
        orientation_err = rotation_error_rad(current_orientation, hold_orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "STREAMING_MINUS_Z",
                "|",
                f"sample {step_index}/{num_steps}",
                "| final_error:",
                round(err_to_final, 5),
                "| tracking_error:",
                round(err_to_command, 5),
                "| orientation_error_rad:",
                round(orientation_err, 5),
                "| current:",
                np.round(current_position, 5).tolist(),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print("Timed out while streaming -Z target.")
            print("Final position error:", round(err_to_final, 5))
            return False

    while True:
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        current_position, current_orientation = read_cartesian_pose(redis_client)
        err = position_error(current_position, target_position)
        orientation_err = rotation_error_rad(current_orientation, hold_orientation)
        set_cartesian_goal(redis_client, target_position, hold_orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "SETTLING_MINUS_Z",
                "| pos_error:",
                round(err, 5),
                "| orientation_error_rad:",
                round(orientation_err, 5),
                "| current:",
                np.round(current_position, 5).tolist(),
            )
            last_status = loop_time

        if err < pos_tol_m:
            print("Reached -Z target.")
            if dwell_s > 0.0:
                time.sleep(dwell_s)
            return True

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print("Timed out before reaching -Z target.")
            print("Final position error:", round(err, 5))
            return False


def run_probe(
    *,
    config_file_name_expected: str,
    dwell_at_tool_init_s: float,
    z_distance_m: float,
    cartesian_max_step_m: float,
    tool_arc_sagitta_m: float,
    tool_arc_min_radius_m: float,
    tool_arc_side: str,
    pos_tol_m: float,
    cartesian_dwell_s: float,
    status_period_s: float,
    timeout_s: float,
) -> int:
    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_expected_config(redis_client, config_file_name_expected):
        return 1

    print("Running Cartesian TOOL_INIT -Z probe.")
    tool_init_pose = approach_tool_init_cartesian(
        redis_client,
        dwell_at_tool_init_s=dwell_at_tool_init_s,
        max_cartesian_step_m=cartesian_max_step_m,
        arc_sagitta_m=tool_arc_sagitta_m,
        arc_min_radius_m=tool_arc_min_radius_m,
        arc_side=tool_arc_side,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    )
    if tool_init_pose is None:
        return 1

    _tool_init_position, tool_init_orientation = tool_init_pose
    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )

    if not command_minus_z_translation(
        redis_client,
        start_position=current_position,
        hold_orientation=tool_init_orientation,
        z_distance_m=z_distance_m,
        max_step_m=cartesian_max_step_m,
        pos_tol_m=pos_tol_m,
        dwell_s=cartesian_dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    ):
        return 1

    print("Finished Cartesian TOOL_INIT -Z probe.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default=CONFIG_FILE_FOR_THIS_SCRIPT,
        help=f"Expected Sai config file (default: {CONFIG_FILE_FOR_THIS_SCRIPT}).",
    )
    parser.add_argument(
        "--dwell-at-tool-init-s",
        type=float,
        default=0.0,
        help="Seconds to dwell after TOOL_INIT arrival before the -Z move (default: 0.0).",
    )
    parser.add_argument(
        "--z-distance-m",
        type=float,
        default=DEFAULT_Z_DISTANCE_M,
        help=(
            "Positive distance to translate in world -Z after TOOL_INIT "
            f"(default: {DEFAULT_Z_DISTANCE_M})."
        ),
    )
    parser.add_argument(
        "--cartesian-max-step-m",
        type=float,
        default=CARTESIAN_MAX_STEP_M,
        help=(
            "Maximum commanded Cartesian step per 10 ms tick "
            f"(default: {CARTESIAN_MAX_STEP_M})."
        ),
    )
    parser.add_argument(
        "--tool-arc-sagitta-m",
        type=float,
        default=TOOL_ARC_SAGITTA_M,
        help=(
            "XY arc sagitta for the Cartesian INIT_POS -> TOOL_INIT path "
            f"(default: {TOOL_ARC_SAGITTA_M})."
        ),
    )
    parser.add_argument(
        "--tool-arc-min-radius-m",
        type=float,
        default=TOOL_ARC_MIN_RADIUS_M,
        help=(
            "Minimum XY arc curvature radius for the Cartesian tool path "
            f"(default: {TOOL_ARC_MIN_RADIUS_M})."
        ),
    )
    parser.add_argument(
        "--tool-arc-side",
        choices=("auto", "left", "right"),
        default=TOOL_ARC_SIDE,
        help=(
            "Which side to bulge the XY arcs. auto picks the side with the "
            f"larger minimum distance from the robot base (default: {TOOL_ARC_SIDE})."
        ),
    )
    parser.add_argument(
        "--pos-tol-m",
        type=float,
        default=POS_TOL_M,
        help=f"Cartesian arrival tolerance in meters (default: {POS_TOL_M}).",
    )
    parser.add_argument(
        "--cartesian-dwell-s",
        type=float,
        default=DEFAULT_CARTESIAN_DWELL_S,
        help=f"Seconds to dwell at the -Z target (default: {DEFAULT_CARTESIAN_DWELL_S}).",
    )
    parser.add_argument(
        "--status-period-s",
        type=float,
        default=STATUS_PERIOD_S,
        help=(
            "Seconds between progress prints. Use 0 to print every control loop "
            f"(default: {STATUS_PERIOD_S})."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=TIMEOUT_PER_WAYPOINT_S,
        help=(
            "Maximum seconds to wait for the Cartesian approach and -Z move. "
            f"Use 0 to disable (default: {TIMEOUT_PER_WAYPOINT_S})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_probe(
            config_file_name_expected=args.config_file,
            dwell_at_tool_init_s=args.dwell_at_tool_init_s,
            z_distance_m=args.z_distance_m,
            cartesian_max_step_m=args.cartesian_max_step_m,
            tool_arc_sagitta_m=args.tool_arc_sagitta_m,
            tool_arc_min_radius_m=args.tool_arc_min_radius_m,
            tool_arc_side=args.tool_arc_side,
            pos_tol_m=args.pos_tol_m,
            cartesian_dwell_s=args.cartesian_dwell_s,
            status_period_s=args.status_period_s,
            timeout_s=args.timeout_s,
        )
    except RuntimeError as exc:
        print("Could not start Cartesian TOOL_INIT -Z probe:")
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
