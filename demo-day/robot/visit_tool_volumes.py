"""Visit each paint/water station container after visit_tool_station.py.

Pre-conditions:
    * Robot is at the TOOL_INIT joint pose, reached by running
      demo-day/robot/visit_tool_station.py first.
    * demo-day/tool_station_model.json and
      demo-day/tool_station_observation.json have just been written by
      visit_tool_station.py.

The script transforms the tool-station model (see
demo-day/robot/model_tool_station.py) into the robot world frame using the
Redis Cartesian pose recorded during the AprilTag scan (not the nominal
TOOL_INIT joint target), then issues a Cartesian sequence:

    SCAN_REFERENCE -> P1_INIT -> P2_INIT -> P3_INIT -> WATER_INIT
                                                    -> SCAN_REFERENCE

P*_INIT and WATER_INIT are "hover" poses (init_hover_z_offset_mm above each
container top). Orientation from the tag-scan snapshot is held fixed
throughout; only position changes between waypoints.

Usage:
    python robot/visit_tool_volumes.py
    python robot/visit_tool_volumes.py --dwell-s 0.75
    python robot/visit_tool_volumes.py --skip-final-tool-init
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import redis
except ImportError:
    redis = None

# Local import; the model lives next to this script.
from model_tool_station import (  # type: ignore[import-not-found]
    INIT_VISIT_ORDER,
    PaintToolStationModel,
    VOLUME_NAME_TO_INIT_LABEL,
    station_point_in_reference_tag_local_m,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent

DEFAULT_TOOL_STATION_MODEL_JSON = DEMO_DAY_DIR / "tool_station_model.json"
DEFAULT_TOOL_STATION_OBSERVATION_JSON = DEMO_DAY_DIR / "tool_station_observation.json"
DEFAULT_TOOL_STATION_WORLD_JSON = DEMO_DAY_DIR / "tool_station_world_positions.json"

ROBOT_NAME = "Titania"
CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"
CARTESIAN_CONTROLLER = "cartesian_controller"

DT = 0.01
POS_TOL_M = 1.0e-2
DWELL_AT_WAYPOINT_S = 0.75
INTER_GOAL_REFRESH_S = 0.05
TIMEOUT_PER_WAYPOINT_S = 30.0
STATUS_PERIOD_S = 0.25
CARTESIAN_SETTLE_S = 0.25


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
# REDIS HELPERS (same style as visit_corners.py / visit_tool_station.py)
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
# WORLD-FRAME MATH
# ============================================================

# Rotation mapping station-frame vectors into a tag's local PnP frame.
# See model_tool_station.py for the derivation.
R_STATION_TO_TAG_LOCAL = np.diag([-1.0, -1.0, 1.0])


def station_point_in_camera_frame_m(
    p_station_m: np.ndarray,
    *,
    ref_tvec_cam: np.ndarray,
    ref_rvec_cam: np.ndarray,
    tag_size_m: float,
    tag_tl_station_m: np.ndarray,
) -> np.ndarray:
    """Transform a point from station frame to camera frame (meters)."""
    import cv2  # local import keeps the module importable without OpenCV

    p_in_tag_local = station_point_in_reference_tag_local_m(
        p_station_m,
        tag_size_m=tag_size_m,
        tag_tl_station_m=tag_tl_station_m,
    )
    R_tag_to_cam, _ = cv2.Rodrigues(np.asarray(ref_rvec_cam, dtype=float))
    return R_tag_to_cam @ p_in_tag_local + np.asarray(ref_tvec_cam, dtype=float)


def camera_point_in_world_frame_m(
    p_cam_m: np.ndarray,
    *,
    ee_pos_world: np.ndarray,
    R_ee_world: np.ndarray,
    R_cam_to_tip: np.ndarray,
    camera_offset_in_tip_m: np.ndarray,
) -> np.ndarray:
    """Transform a camera-frame point through tip frame to world frame."""
    p_tip = R_cam_to_tip @ np.asarray(p_cam_m, dtype=float) + camera_offset_in_tip_m
    return ee_pos_world + R_ee_world @ p_tip


def load_scan_reference_pose(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the EE pose recorded during the AprilTag scan snapshot.

    This is the actual Redis Cartesian pose averaged over the same camera
    frames as the tag detections, not the nominal TOOL_INIT joint target.
    """
    if "ee_position_world_m" not in observation or "ee_orientation_world" not in observation:
        raise RuntimeError(
            "Observation JSON is missing ee_position_world_m / "
            "ee_orientation_world. Re-run visit_tool_station.py."
        )
    position = np.array(observation["ee_position_world_m"], dtype=float)
    orientation = np.array(observation["ee_orientation_world"], dtype=float)
    if position.shape != (3,):
        raise RuntimeError("ee_position_world_m must have 3 elements.")
    if orientation.shape != (3, 3):
        raise RuntimeError("ee_orientation_world must be a 3x3 matrix.")
    return position, orientation


def station_point_in_world_frame_m(
    p_station_m: np.ndarray,
    *,
    observation: dict[str, Any],
    model: PaintToolStationModel,
    tag_size_m: float,
    reference_tag_id: int,
) -> np.ndarray:
    poses = observation["tag_poses_in_camera_frame"]
    ref_key = str(reference_tag_id)
    if ref_key not in poses:
        raise RuntimeError(
            f"Observation JSON is missing tag pose for reference tag {reference_tag_id}."
        )
    ref_pose = poses[ref_key]
    tvec = np.array(ref_pose["tvec"], dtype=float)
    rvec = np.array(ref_pose["rvec"], dtype=float)

    ee_pos = np.array(observation["ee_position_world_m"], dtype=float)
    R_ee = np.array(observation["ee_orientation_world"], dtype=float)
    R_cam_to_tip = np.array(observation["R_camera_to_tip"], dtype=float)
    cam_offset = np.array(observation["camera_offset_in_tip_frame_m"], dtype=float)
    tag_tl_station_m = model.get_tag_tl_station_m(reference_tag_id)

    p_cam = station_point_in_camera_frame_m(
        p_station_m,
        ref_tvec_cam=tvec,
        ref_rvec_cam=rvec,
        tag_size_m=tag_size_m,
        tag_tl_station_m=tag_tl_station_m,
    )
    return camera_point_in_world_frame_m(
        p_cam,
        ee_pos_world=ee_pos,
        R_ee_world=R_ee,
        R_cam_to_tip=R_cam_to_tip,
        camera_offset_in_tip_m=cam_offset,
    )


# ============================================================
# WAYPOINT BUILDER
# ============================================================

@dataclass
class Waypoint:
    label: str
    world_position_m: np.ndarray
    station_position_mm: np.ndarray | None
    volume_name: str | None


def build_volume_waypoints(
    *,
    model: PaintToolStationModel,
    observation: dict[str, Any],
) -> list[Waypoint]:
    tag_size_m = float(observation.get("tag_size_m", model.tag_size_mm * 1.0e-3))
    reference_tag_id = int(
        observation.get("reference_tag_id", model.reference_tag_id)
    )

    waypoints: list[Waypoint] = []
    for volume_name in INIT_VISIT_ORDER:
        volume = model.get_volume(volume_name)
        station_point_mm = volume.init_hover_point_mm()
        p_world = station_point_in_world_frame_m(
            station_point_mm * 1.0e-3,
            observation=observation,
            model=model,
            tag_size_m=tag_size_m,
            reference_tag_id=reference_tag_id,
        )
        waypoints.append(
            Waypoint(
                label=VOLUME_NAME_TO_INIT_LABEL[volume_name],
                world_position_m=p_world,
                station_position_mm=station_point_mm,
                volume_name=volume_name,
            )
        )
    return waypoints


def save_world_positions_json(
    *,
    path: Path,
    waypoints: list[Waypoint],
    scan_reference_position_m: np.ndarray,
    hold_orientation: np.ndarray,
    observation_path: Path,
    model_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "frame": "world (robot base) frame",
        "units": "meters",
        "scan_reference_position_world_m": [float(v) for v in scan_reference_position_m],
        "hold_orientation_world": [
            [float(v) for v in row] for row in hold_orientation
        ],
        "observation_json_path": str(observation_path),
        "model_json_path": str(model_path),
        "waypoints": [
            {
                "label": wp.label,
                "volume_name": wp.volume_name,
                "world_position_m": [float(v) for v in wp.world_position_m],
                "station_position_mm": (
                    [float(v) for v in wp.station_position_mm]
                    if wp.station_position_mm is not None
                    else None
                ),
            }
            for wp in waypoints
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ============================================================
# CARTESIAN MOTION
# ============================================================

def go_to_waypoint(
    redis_client,
    *,
    target_pos: np.ndarray,
    hold_orientation: np.ndarray,
    label: str,
    dwell_s: float,
    timeout_s: float,
    status_period_s: float,
) -> bool:
    print(f"\n-> {label}: target = {np.round(target_pos, 5).tolist()}")
    set_cartesian_goal(redis_client, target_pos, hold_orientation)

    loop_time = 0.0
    last_status = 0.0
    last_refresh = 0.0
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9
    start = time.perf_counter()

    while True:
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        if loop_time - last_refresh >= INTER_GOAL_REFRESH_S:
            set_cartesian_goal(redis_client, target_pos, hold_orientation)
            last_refresh = loop_time

        current = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        err = position_error(current, target_pos)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(f"MOVING_TO_{label} | pos_error: {err:.5f}")
            last_status = loop_time

        if err < POS_TOL_M:
            break

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(f"Timed out before reaching {label}. Final error:", round(err, 5))
            return False

    print(f"Reached {label}. Dwelling {dwell_s:.2f}s.")
    if dwell_s > 0.0:
        time.sleep(dwell_s)
    return True


# ============================================================
# MAIN
# ============================================================

def run_visit_volumes(
    *,
    model_json_path: Path,
    observation_json_path: Path,
    world_json_path: Path,
    dwell_s: float,
    skip_final_tool_init: bool,
    timeout_per_waypoint_s: float,
    status_period_s: float,
) -> int:
    if not model_json_path.is_file():
        print(f"Tool-station model JSON not found: {model_json_path}")
        print("Run demo-day/robot/visit_tool_station.py first.")
        return 1
    if not observation_json_path.is_file():
        print(f"Tool-station observation JSON not found: {observation_json_path}")
        print("Run demo-day/robot/visit_tool_station.py first.")
        return 1

    with observation_json_path.open("r", encoding="utf-8") as f:
        observation = json.load(f)

    scan_reference_position_m, hold_orientation = load_scan_reference_pose(
        observation
    )

    # The model in memory shares the schema with the saved JSON; we use the
    # in-memory instance for `init_hover_point_mm()`. If you tune the model
    # parameters between runs, re-export via visit_tool_station.py.
    model = PaintToolStationModel()

    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )

    print("Using scan-reference pose from observation JSON (not nominal TOOL_INIT):")
    print("  scan_reference_position_world_m:", scan_reference_position_m.tolist())
    print("  hold_orientation (fixed throughout):")
    print(hold_orientation)
    print("Current Cartesian position:", current_position.tolist())

    waypoints = build_volume_waypoints(model=model, observation=observation)

    print("\nComputed world-frame hover waypoints (relative to scan reference):")
    for wp in waypoints:
        delta = wp.world_position_m - scan_reference_position_m
        print(
            f"  {wp.label} (from {wp.volume_name}):"
            f" station {np.round(wp.station_position_mm, 3).tolist()} mm"
            f" -> world {np.round(wp.world_position_m, 5).tolist()} m"
            f" (delta {np.round(delta, 5).tolist()} m)"
        )

    saved_world_path = save_world_positions_json(
        path=world_json_path,
        waypoints=waypoints,
        scan_reference_position_m=scan_reference_position_m,
        hold_orientation=hold_orientation,
        observation_path=observation_json_path,
        model_path=model_json_path,
    )
    print(f"Saved world-frame waypoints JSON: {saved_world_path}")

    # Hand control to the Cartesian controller. Hold the scan orientation and
    # start from the measured current position so the controller does not snap.
    set_cartesian_goal(redis_client, current_position, hold_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print(f"\nUsing controller: {CARTESIAN_CONTROLLER}")

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < CARTESIAN_SETTLE_S:
        set_cartesian_goal(redis_client, current_position, hold_orientation)
        time.sleep(DT)

    sequence: list[tuple[str, np.ndarray]] = [
        (wp.label, wp.world_position_m) for wp in waypoints
    ]
    if not skip_final_tool_init:
        sequence.append(("SCAN_REFERENCE", scan_reference_position_m))

    print("\nWaypoint sequence:")
    for label, target in sequence:
        print(f"  {label}: {np.round(target, 5).tolist()}")

    for label, target in sequence:
        ok = go_to_waypoint(
            redis_client,
            target_pos=target,
            hold_orientation=hold_orientation,
            label=label,
            dwell_s=dwell_s,
            timeout_s=timeout_per_waypoint_s,
            status_period_s=status_period_s,
        )
        if not ok:
            return 1

    print("\nFinished visiting all tool-station volumes.")
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
        "--model-json",
        default=str(DEFAULT_TOOL_STATION_MODEL_JSON),
        help=f"Tool-station model JSON (default: {DEFAULT_TOOL_STATION_MODEL_JSON}).",
    )
    parser.add_argument(
        "--observation-json",
        default=str(DEFAULT_TOOL_STATION_OBSERVATION_JSON),
        help=(
            "Snapshot from visit_tool_station.py with tag poses + EE pose "
            f"(default: {DEFAULT_TOOL_STATION_OBSERVATION_JSON})."
        ),
    )
    parser.add_argument(
        "--world-json",
        default=str(DEFAULT_TOOL_STATION_WORLD_JSON),
        help=(
            "Where to save the computed world-frame waypoints "
            f"(default: {DEFAULT_TOOL_STATION_WORLD_JSON})."
        ),
    )
    parser.add_argument(
        "--dwell-s",
        type=float,
        default=DWELL_AT_WAYPOINT_S,
        help=f"Seconds to dwell at each waypoint (default: {DWELL_AT_WAYPOINT_S}).",
    )
    parser.add_argument(
        "--skip-final-tool-init",
        action="store_true",
        help="Stop at WATER_INIT instead of returning to the scan-reference pose.",
    )
    parser.add_argument(
        "--timeout-per-waypoint-s",
        type=float,
        default=TIMEOUT_PER_WAYPOINT_S,
        help=(
            "Maximum seconds to wait at each waypoint. Use 0 to disable "
            f"(default: {TIMEOUT_PER_WAYPOINT_S})."
        ),
    )
    parser.add_argument(
        "--status-period-s",
        type=float,
        default=STATUS_PERIOD_S,
        help=(
            "Seconds between progress prints. Use 0 to print every loop "
            f"(default: {STATUS_PERIOD_S})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_visit_volumes(
        model_json_path=Path(args.model_json),
        observation_json_path=Path(args.observation_json),
        world_json_path=Path(args.world_json),
        dwell_s=args.dwell_s,
        skip_final_tool_init=args.skip_final_tool_init,
        timeout_per_waypoint_s=args.timeout_per_waypoint_s,
        status_period_s=args.status_period_s,
    )


if __name__ == "__main__":
    sys.exit(main())
