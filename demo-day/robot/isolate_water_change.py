"""Isolate the TOOL_INIT cached transition plus water/napkin paint-change segment.

Sequence:
    INIT_POS -> TOOL_INIT using tool_init_cartesian_path.json
    -> WATER_INIT_POS -> swirl water
    -> WATER_TO_NAPKIN_INIT_POS -> wipe napkin
    -> TOOL_INIT with fixed TOOL_INIT orientation
    -> INIT_POS using the reversed cached Cartesian path

Usage:
    python robot/isolate_water_change.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import redis
except ImportError:
    redis = None

from helpers.primitives import (
    DEFAULT_DIP_DWELL_S,
    DEFAULT_HOVER_DWELL_S,
    load_paint_init_pos_m,
    swirl_water,
    wipe_napkin,
)
from helpers.tool_station_coords import (
    DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
    STATUS_PERIOD_S,
    TIMEOUT_PER_WAYPOINT_S,
    approach_tool_init_cartesian,
    ensure_robot_ready,
    go_to_waypoint,
    load_demo_day_config,
    read_cartesian_pose,
    return_from_tool_init_cartesian,
    rotation_error_rad,
)


def log_pose(redis_client, label: str, reference_orientation: np.ndarray) -> None:
    position, orientation = read_cartesian_pose(redis_client)
    orientation_error = rotation_error_rad(orientation, reference_orientation)
    print(
        f"{label}: position={np.round(position, 5).tolist()} "
        f"| orientation_error_from_TOOL_INIT={orientation_error:.5f} rad"
    )


def run(
    *,
    tool_init_cartesian_path_json: Path,
    hover_dwell_s: float,
    dip_dwell_s: float,
    status_period_s: float,
    timeout_s: float,
) -> int:
    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    cfg = load_demo_day_config()

    print("Running isolated cached TOOL_INIT -> water change -> TOOL_INIT test.")
    tool_init_pose = approach_tool_init_cartesian(
        redis_client,
        dwell_at_tool_init_s=hover_dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        path_json=tool_init_cartesian_path_json,
        use_cached_path=True,
        rebuild_path=False,
        save_path=False,
        require_cached_path=True,
    )
    if tool_init_pose is None:
        return 1

    tool_init_position, tool_init_orientation = tool_init_pose
    log_pose(redis_client, "AT_TOOL_INIT_AFTER_CACHED_APPROACH", tool_init_orientation)

    water_init = load_paint_init_pos_m(cfg, "WATER_INIT_POS")
    if not go_to_waypoint(
        redis_client,
        target_pos=water_init,
        hold_orientation=tool_init_orientation,
        label="WATER_INIT_POS",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1
    log_pose(redis_client, "AT_WATER_INIT_POS", tool_init_orientation)

    if not swirl_water(
        redis_client,
        hold_orientation=tool_init_orientation,
        config=cfg,
        dwell_at_dip_s=dip_dwell_s,
        dwell_at_hover_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1
    log_pose(redis_client, "AFTER_WATER_SWIRL", tool_init_orientation)

    water_to_napkin = load_paint_init_pos_m(cfg, "WATER_TO_NAPKIN_INIT_POS")
    if not go_to_waypoint(
        redis_client,
        target_pos=water_to_napkin,
        hold_orientation=tool_init_orientation,
        label="WATER_TO_NAPKIN_INIT_POS",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1
    log_pose(redis_client, "AT_WATER_TO_NAPKIN_INIT_POS", tool_init_orientation)

    if not wipe_napkin(
        redis_client,
        hold_orientation=tool_init_orientation,
        config=cfg,
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1
    log_pose(redis_client, "AFTER_NAPKIN_WIPE", tool_init_orientation)

    if not go_to_waypoint(
        redis_client,
        target_pos=tool_init_position,
        hold_orientation=tool_init_orientation,
        label="TOOL_INIT",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1
    log_pose(redis_client, "AT_TOOL_INIT_BEFORE_CACHED_RETURN", tool_init_orientation)

    if not return_from_tool_init_cartesian(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        path_json=tool_init_cartesian_path_json,
    ):
        return 1

    print("Finished isolated water-change transition test.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool-init-cartesian-path-json",
        default=str(DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON),
        help=(
            "Cached Cartesian INIT_POS -> TOOL_INIT path JSON "
            f"(default: {DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON})."
        ),
    )
    parser.add_argument(
        "--hover-dwell-s",
        type=float,
        default=DEFAULT_HOVER_DWELL_S,
        help=f"Seconds to dwell at hover points (default: {DEFAULT_HOVER_DWELL_S}).",
    )
    parser.add_argument(
        "--dip-dwell-s",
        type=float,
        default=DEFAULT_DIP_DWELL_S,
        help=f"Seconds to dwell during water contact (default: {DEFAULT_DIP_DWELL_S}).",
    )
    parser.add_argument(
        "--status-period-s",
        type=float,
        default=STATUS_PERIOD_S,
        help=f"Seconds between status prints (default: {STATUS_PERIOD_S}).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=TIMEOUT_PER_WAYPOINT_S,
        help=f"Motion timeout in seconds (default: {TIMEOUT_PER_WAYPOINT_S}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        tool_init_cartesian_path_json=Path(args.tool_init_cartesian_path_json),
        hover_dwell_s=args.hover_dwell_s,
        dip_dwell_s=args.dip_dwell_s,
        status_period_s=args.status_period_s,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    sys.exit(main())
