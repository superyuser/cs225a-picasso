"""Tool-station coordinate math, JSON I/O, and Cartesian volume visits.

Used by demo-day/robot/visit_tool_station.py for coordinate math, JSON I/O,
and Cartesian volume visits.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .model_tool_station import (
    INIT_VISIT_ORDER,
    PaintToolStationModel,
    VOLUME_NAME_TO_INIT_LABEL,
    station_point_in_reference_tag_local_m,
)

try:
    import redis
except ImportError:
    redis = None


HELPERS_DIR = Path(__file__).resolve().parent
ROBOT_DIR = HELPERS_DIR.parent
DEMO_DAY_DIR = ROBOT_DIR.parent
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"

DEFAULT_TOOL_STATION_MODEL_JSON = DEMO_DAY_DIR / "tool_station_model.json"
DEFAULT_TOOL_STATION_OBSERVATION_JSON = DEMO_DAY_DIR / "tool_station_observation.json"
DEFAULT_TOOL_STATION_WORLD_JSON = DEMO_DAY_DIR / "tool_station_world_positions.json"
DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON = DEMO_DAY_DIR / "tool_init_cartesian_path.json"

ROBOT_NAME = "Titania"
CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"
CARTESIAN_CONTROLLER = "cartesian_controller"
TOOL_INIT_CARTESIAN_PATH_VERSION = 1

DT = 0.01
MM_TO_M = 1.0e-3
DEG_TO_RAD = math.pi / 180.0
POS_TOL_M = 1.0e-2
ORI_TOL_RAD = 5.0e-2
DWELL_AT_WAYPOINT_S = 0.75
INTER_GOAL_REFRESH_S = 0.05
TIMEOUT_PER_WAYPOINT_S = 30.0
STATUS_PERIOD_S = 0.25
CARTESIAN_SETTLE_S = 0.25
CARTESIAN_MAX_STEP_M = 0.002
CARTESIAN_MAX_ORI_STEP_RAD = 0.0025
TOOL_ARC_SAGITTA_M = 0.25
TOOL_ARC_MIN_RADIUS_M = 1.6
TOOL_ARC_SIDE = "auto"
CACHED_TOOL_PATH_START_POS_TOL_M = 1.0e-2
CACHED_TOOL_PATH_START_ORI_TOL_RAD = 2.0e-2
CACHED_TOOL_PATH_CONFIG_POS_TOL_M = 1.0e-4
CACHED_TOOL_PATH_CONFIG_ORI_TOL_RAD = 1.0e-4
CACHED_TOOL_PATH_PARAM_TOL = 1.0e-9


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


@dataclass
class VolumeWaypoint:
    label: str
    world_position_m: np.ndarray
    station_position_mm: np.ndarray | None
    volume_name: str | None


@dataclass
class CartesianPoseWaypoint:
    label: str
    position_m: np.ndarray
    orientation: np.ndarray


@dataclass
class CartesianPosePath:
    source_waypoints: list[CartesianPoseWaypoint]
    samples: list[CartesianPoseWaypoint]
    segment_summaries: list[dict[str, Any]]
    source: str
    path_json: Path | None


@dataclass
class ToolStationCalibration:
    """World-frame targets derived from model + observation JSON."""

    model_json_path: Path
    observation_json_path: Path
    world_json_path: Path
    observation: dict[str, Any]
    scan_reference_position_m: np.ndarray
    hold_orientation: np.ndarray
    waypoints: list[VolumeWaypoint]


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

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


def load_demo_day_config() -> dict[str, Any]:
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def rotation_error_rad(current_orientation: np.ndarray, goal_orientation: np.ndarray) -> float:
    relative_rotation = np.asarray(goal_orientation, dtype=float).T @ np.asarray(
        current_orientation,
        dtype=float,
    )
    cos_angle = (float(np.trace(relative_rotation)) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def rpy_deg_to_matrix(rpy_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy_deg, dtype=float) * DEG_TO_RAD
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rz @ ry @ rx


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qw, qx, qy, qz], dtype=float)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-12)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat, dtype=float)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def slerp_orientation(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    q0 = matrix_to_quat(start)
    q1 = matrix_to_quat(target)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    fraction = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        quat = q0 + fraction * (q1 - q0)
        quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
        return quat_to_matrix(quat)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    scale_0 = math.sin(theta_0 - theta) / sin_theta_0
    scale_1 = math.sin(theta) / sin_theta_0
    return quat_to_matrix(scale_0 * q0 + scale_1 * q1)


def smootherstep(fraction: float) -> float:
    """Quintic interpolation with zero velocity and acceleration at endpoints."""
    u = float(np.clip(fraction, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def load_init_pos_m(config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or load_demo_day_config()
    init_pos = np.array(cfg["init_pos_m"], dtype=float)
    if init_pos.shape != (3,):
        raise RuntimeError(f"{DEMO_DAY_CONFIG_PATH} init_pos_m must contain 3 values.")
    return init_pos


def load_cartesian_tool_waypoints(
    config: dict[str, Any] | None = None,
) -> list[CartesianPoseWaypoint]:
    cfg = config or load_demo_day_config()
    waypoint_specs = [
        ("TOOL_WP1", "TOOL_WP1_POS", "TOOL_WP1_ORI"),
        ("TOOL_WP2", "TOOL_WP2_POS", "TOOL_WP2_ORI"),
        ("TOOL_INIT", "TOOL_INIT_POS", "TOOL_INIT_ORI"),
    ]

    waypoints: list[CartesianPoseWaypoint] = []
    for label, pos_key, ori_key in waypoint_specs:
        if pos_key not in cfg or ori_key not in cfg:
            raise RuntimeError(
                f"{DEMO_DAY_CONFIG_PATH} is missing {pos_key} or {ori_key}."
            )

        position_mm = np.array(cfg[pos_key], dtype=float)
        orientation_deg = np.array(cfg[ori_key], dtype=float)
        if position_mm.shape != (3,):
            raise RuntimeError(f"{DEMO_DAY_CONFIG_PATH} {pos_key} must contain 3 values.")
        if orientation_deg.shape != (3,):
            raise RuntimeError(f"{DEMO_DAY_CONFIG_PATH} {ori_key} must contain 3 values.")

        waypoints.append(
            CartesianPoseWaypoint(
                label=label,
                position_m=position_mm * MM_TO_M,
                orientation=rpy_deg_to_matrix(orientation_deg),
            )
        )

    return waypoints


def min_xy_radius(points: np.ndarray) -> float:
    if len(points) == 0:
        return math.inf
    return float(np.min(np.linalg.norm(points[:, :2], axis=1)))


def build_xy_arc_path(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    sagitta_m: float,
    step_m: float,
    side: str,
    min_radius_m: float = 0.0,
) -> tuple[np.ndarray, float]:
    start_xy = np.array(start_pos[:2], dtype=float)
    target_xy = np.array(target_pos[:2], dtype=float)
    chord = target_xy - start_xy
    chord_len = float(np.linalg.norm(chord))

    if chord_len < 1.0e-6:
        return np.array([target_pos], dtype=float), math.inf

    sagitta = max(abs(sagitta_m), 1.0e-4)
    sagitta = min(sagitta, chord_len * 0.49)
    step_m = max(abs(step_m), 1.0e-4)

    radius = chord_len**2 / (8.0 * sagitta) + sagitta / 2.0
    half_chord = chord_len / 2.0

    # Match orient_camera.py: if the requested sagitta would create a sharp
    # small-radius arc, shrink the sagitta until the arc is flatter.
    if min_radius_m > 0.0 and radius < min_radius_m:
        if min_radius_m >= half_chord:
            new_sagitta = min_radius_m - math.sqrt(
                max(min_radius_m**2 - half_chord**2, 0.0)
            )
            sagitta = min(max(new_sagitta, 1.0e-4), chord_len * 0.49)
        else:
            sagitta = max(min(sagitta, 1.0e-3), 1.0e-4)
        radius = chord_len**2 / (8.0 * sagitta) + sagitta / 2.0

    center_offset = math.sqrt(max(radius**2 - half_chord**2, 0.0))
    unit_chord = chord / chord_len
    left_normal = np.array([-unit_chord[1], unit_chord[0]], dtype=float)
    bulge_normal = left_normal if side == "left" else -left_normal
    center = (start_xy + target_xy) / 2.0 - bulge_normal * center_offset

    start_angle = math.atan2(start_xy[1] - center[1], start_xy[0] - center[0])
    target_angle = math.atan2(target_xy[1] - center[1], target_xy[0] - center[0])
    angle_delta = (target_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi

    arc_length = abs(angle_delta) * radius
    num_segments = max(1, int(math.ceil(arc_length / step_m)))

    samples = []
    for index in range(1, num_segments + 1):
        fraction = index / num_segments
        theta = start_angle + angle_delta * fraction
        xy = center + radius * np.array(
            [math.cos(theta), math.sin(theta)],
            dtype=float,
        )
        z = start_pos[2] + (target_pos[2] - start_pos[2]) * fraction
        samples.append(np.array([xy[0], xy[1], z], dtype=float))

    samples[-1] = np.array(target_pos, dtype=float)
    return np.array(samples, dtype=float), radius


def build_tool_translation_path(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    sagitta_m: float,
    step_m: float,
    arc_side: str,
    min_radius_m: float,
) -> tuple[np.ndarray, float, str]:
    if arc_side in ("left", "right"):
        path, radius = build_xy_arc_path(
            start_pos,
            target_pos,
            sagitta_m=sagitta_m,
            step_m=step_m,
            side=arc_side,
            min_radius_m=min_radius_m,
        )
        return path, radius, arc_side

    left_path, left_radius = build_xy_arc_path(
        start_pos,
        target_pos,
        sagitta_m=sagitta_m,
        step_m=step_m,
        side="left",
        min_radius_m=min_radius_m,
    )
    right_path, right_radius = build_xy_arc_path(
        start_pos,
        target_pos,
        sagitta_m=sagitta_m,
        step_m=step_m,
        side="right",
        min_radius_m=min_radius_m,
    )

    if min_xy_radius(left_path) >= min_xy_radius(right_path):
        return left_path, left_radius, "left"
    return right_path, right_radius, "right"


def cartesian_pose_waypoint_to_json(
    waypoint: CartesianPoseWaypoint,
) -> dict[str, Any]:
    return {
        "label": waypoint.label,
        "position_m": [float(v) for v in np.asarray(waypoint.position_m, dtype=float)],
        "orientation": [
            [float(v) for v in row]
            for row in np.asarray(waypoint.orientation, dtype=float)
        ],
    }


def cartesian_pose_waypoint_from_json(data: dict[str, Any]) -> CartesianPoseWaypoint:
    position = np.array(data["position_m"], dtype=float)
    orientation = np.array(data["orientation"], dtype=float)
    if position.shape != (3,):
        raise RuntimeError("Cached Cartesian waypoint position_m must have 3 values.")
    if orientation.shape != (3, 3):
        raise RuntimeError("Cached Cartesian waypoint orientation must be 3x3.")
    return CartesianPoseWaypoint(
        label=str(data["label"]),
        position_m=position,
        orientation=orientation,
    )


def tool_init_cartesian_path_parameters(
    *,
    max_cartesian_step_m: float,
    max_orientation_step_rad: float,
    arc_sagitta_m: float,
    arc_min_radius_m: float,
    arc_side: str,
) -> dict[str, Any]:
    return {
        "max_cartesian_step_m": float(max_cartesian_step_m),
        "max_orientation_step_rad": float(max_orientation_step_rad),
        "arc_sagitta_m": float(arc_sagitta_m),
        "arc_min_radius_m": float(arc_min_radius_m),
        "arc_side": str(arc_side),
    }


def cached_tool_init_parameters_match(
    saved_parameters: dict[str, Any],
    current_parameters: dict[str, Any],
) -> bool:
    for key, current_value in current_parameters.items():
        if key not in saved_parameters:
            print(f"Cached Cartesian tool path is missing parameter {key}; rebuild required.")
            return False

        saved_value = saved_parameters[key]
        if isinstance(current_value, str):
            if str(saved_value) != current_value:
                print(
                    "Cached Cartesian tool path parameter mismatch for",
                    f"{key}: saved {saved_value!r}, current {current_value!r};",
                    "rebuild required.",
                )
                return False
            continue

        try:
            saved_float = float(saved_value)
            current_float = float(current_value)
        except (TypeError, ValueError):
            print(
                "Cached Cartesian tool path parameter is invalid for",
                f"{key}: {saved_value!r}; rebuild required.",
            )
            return False

        if abs(saved_float - current_float) > CACHED_TOOL_PATH_PARAM_TOL:
            print(
                "Cached Cartesian tool path parameter mismatch for",
                f"{key}: saved {saved_float:.9f}, current {current_float:.9f};",
                "rebuild required.",
            )
            return False
    return True


def cached_tool_init_waypoints_match(
    saved_waypoints: list[CartesianPoseWaypoint],
    current_waypoints: list[CartesianPoseWaypoint],
    *,
    require_matching_config_waypoints: bool = False,
) -> bool:
    if not saved_waypoints or not current_waypoints:
        print("Cached Cartesian tool path has no source/start waypoint; rebuild required.")
        return False

    if require_matching_config_waypoints and len(saved_waypoints) != len(current_waypoints):
        print("Cached Cartesian tool path waypoint count changed; rebuild required.")
        return False

    waypoint_pairs = (
        zip(saved_waypoints, current_waypoints)
        if require_matching_config_waypoints
        else [(saved_waypoints[0], current_waypoints[0])]
    )
    for index, (saved_wp, current_wp) in enumerate(waypoint_pairs):
        if saved_wp.label != current_wp.label:
            print(
                "Cached Cartesian tool path waypoint labels changed:",
                f"{saved_wp.label!r} != {current_wp.label!r}; rebuild required.",
            )
            return False

        position_tol = (
            CACHED_TOOL_PATH_START_POS_TOL_M
            if index == 0
            else CACHED_TOOL_PATH_CONFIG_POS_TOL_M
        )
        orientation_tol = (
            CACHED_TOOL_PATH_START_ORI_TOL_RAD
            if index == 0
            else CACHED_TOOL_PATH_CONFIG_ORI_TOL_RAD
        )
        position_delta = position_error(saved_wp.position_m, current_wp.position_m)
        orientation_delta = rotation_error_rad(saved_wp.orientation, current_wp.orientation)
        if position_delta > position_tol or orientation_delta > orientation_tol:
            print(
                "Cached Cartesian tool path waypoint changed:",
                saved_wp.label,
                f"| position_delta {position_delta:.6f} m",
                f"(tol {position_tol:.6f})",
                f"| orientation_delta {orientation_delta:.6f} rad",
                f"(tol {orientation_tol:.6f}); rebuild required.",
            )
            return False

    return True


def build_cartesian_pose_path_samples(
    waypoints: list[CartesianPoseWaypoint],
    *,
    max_cartesian_step_m: float = CARTESIAN_MAX_STEP_M,
    max_orientation_step_rad: float = CARTESIAN_MAX_ORI_STEP_RAD,
    arc_sagitta_m: float = TOOL_ARC_SAGITTA_M,
    arc_min_radius_m: float = TOOL_ARC_MIN_RADIUS_M,
    arc_side: str = TOOL_ARC_SIDE,
) -> tuple[list[CartesianPoseWaypoint], list[dict[str, Any]]]:
    samples: list[CartesianPoseWaypoint] = []
    segment_summaries: list[dict[str, Any]] = []

    for segment_index, (start_wp, end_wp) in enumerate(
        zip(waypoints[:-1], waypoints[1:]),
        start=1,
    ):
        angle = rotation_error_rad(start_wp.orientation, end_wp.orientation)
        orientation_steps = max(
            1,
            int(math.ceil(angle / max(max_orientation_step_rad, 1.0e-4))),
        )
        position_path, arc_radius, chosen_arc_side = build_tool_translation_path(
            start_wp.position_m,
            end_wp.position_m,
            sagitta_m=arc_sagitta_m,
            step_m=max_cartesian_step_m,
            arc_side=arc_side,
            min_radius_m=arc_min_radius_m,
        )
        if len(position_path) < orientation_steps:
            position_path, arc_radius, chosen_arc_side = build_tool_translation_path(
                start_wp.position_m,
                end_wp.position_m,
                sagitta_m=arc_sagitta_m,
                step_m=max_cartesian_step_m * len(position_path) / orientation_steps,
                arc_side=arc_side,
                min_radius_m=arc_min_radius_m,
            )

        num_steps = len(position_path)
        path_length = float(
            np.sum(
                np.linalg.norm(
                    np.diff(
                        np.vstack([start_wp.position_m, position_path]),
                        axis=0,
                    ),
                    axis=1,
                )
            )
        )
        segment_label = f"{start_wp.label}->{end_wp.label}"

        segment_summaries.append(
            {
                "segment_index": int(segment_index),
                "start_label": start_wp.label,
                "end_label": end_wp.label,
                "num_steps": int(num_steps),
                "path_length_m": path_length,
                "rotation_rad": float(angle),
                "max_orientation_step_rad": float(max_orientation_step_rad),
                "xy_arc_radius_m": float(arc_radius),
                "min_xy_radius_m": min_xy_radius(position_path),
                "arc_side": chosen_arc_side,
            }
        )

        for step_index, commanded_position in enumerate(position_path, start=1):
            u = step_index / num_steps
            s = smootherstep(u)
            samples.append(
                CartesianPoseWaypoint(
                    label=segment_label,
                    position_m=np.array(commanded_position, dtype=float),
                    orientation=slerp_orientation(
                        start_wp.orientation,
                        end_wp.orientation,
                        s,
                    ),
                )
            )

    return samples, segment_summaries


def save_tool_init_cartesian_path(
    cartesian_path: CartesianPosePath,
    *,
    parameters: dict[str, Any],
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
) -> Path:
    payload = {
        "version": TOOL_INIT_CARTESIAN_PATH_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "units": "meters",
        "parameters": parameters,
        "source_waypoints": [
            cartesian_pose_waypoint_to_json(waypoint)
            for waypoint in cartesian_path.source_waypoints
        ],
        "samples": [
            cartesian_pose_waypoint_to_json(sample)
            for sample in cartesian_path.samples
        ],
        "segment_summaries": cartesian_path.segment_summaries,
    }

    path_json.parent.mkdir(parents=True, exist_ok=True)
    with path_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved tool-init Cartesian path JSON: {path_json}")
    print("Cartesian path samples:", len(cartesian_path.samples))
    return path_json


def load_tool_init_cartesian_path(
    *,
    current_waypoints: list[CartesianPoseWaypoint],
    parameters: dict[str, Any],
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
    require_matching_config_waypoints: bool = False,
) -> CartesianPosePath | None:
    if not path_json.is_file():
        return None

    try:
        with path_json.open("r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read cached Cartesian tool path {path_json}: {exc}")
        print("Rebuild required.")
        return None

    if saved.get("version") != TOOL_INIT_CARTESIAN_PATH_VERSION:
        print(f"Unsupported Cartesian tool path JSON version in {path_json}.")
        return None

    saved_parameters = saved.get("parameters")
    if not isinstance(saved_parameters, dict) or not cached_tool_init_parameters_match(
        saved_parameters,
        parameters,
    ):
        return None

    saved_source = saved.get("source_waypoints")
    if not isinstance(saved_source, list):
        print("Cached Cartesian tool path is missing source_waypoints; rebuild required.")
        return None
    try:
        source_waypoints = [
            cartesian_pose_waypoint_from_json(waypoint)
            for waypoint in saved_source
        ]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Cached Cartesian tool path source_waypoints are invalid: {exc}")
        print("Rebuild required.")
        return None
    if not cached_tool_init_waypoints_match(
        source_waypoints,
        current_waypoints,
        require_matching_config_waypoints=require_matching_config_waypoints,
    ):
        return None

    saved_samples = saved.get("samples")
    if not isinstance(saved_samples, list) or not saved_samples:
        print("Cached Cartesian tool path is missing samples; rebuild required.")
        return None
    try:
        samples = [
            cartesian_pose_waypoint_from_json(sample)
            for sample in saved_samples
        ]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Cached Cartesian tool path samples are invalid: {exc}")
        print("Rebuild required.")
        return None

    segment_summaries = saved.get("segment_summaries", [])
    if not isinstance(segment_summaries, list):
        print("Cached Cartesian tool path segment_summaries are invalid; rebuild required.")
        return None
    try:
        expected_samples = sum(
            int(summary.get("num_steps", 0))
            for summary in segment_summaries
            if isinstance(summary, dict)
        )
    except (TypeError, ValueError):
        print("Cached Cartesian tool path segment_summaries are invalid; rebuild required.")
        return None
    if expected_samples and expected_samples != len(samples):
        print(
            "Cached Cartesian tool path sample count does not match segment summaries; "
            "rebuild required."
        )
        return None

    print(f"Loaded tool-init Cartesian path JSON: {path_json}")
    print("Cartesian path samples:", len(samples))
    return CartesianPosePath(
        source_waypoints=source_waypoints,
        samples=samples,
        segment_summaries=segment_summaries,
        source="cached",
        path_json=path_json,
    )


def load_saved_tool_init_cartesian_path(
    *,
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
) -> CartesianPosePath | None:
    if not path_json.is_file():
        return None

    try:
        with path_json.open("r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read cached Cartesian tool path {path_json}: {exc}")
        return None

    saved_parameters = saved.get("parameters")
    saved_source = saved.get("source_waypoints")
    if not isinstance(saved_parameters, dict) or not isinstance(saved_source, list):
        print("Cached Cartesian tool path is missing parameters/source_waypoints.")
        return None

    try:
        source_waypoints = [
            cartesian_pose_waypoint_from_json(waypoint)
            for waypoint in saved_source
        ]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Cached Cartesian tool path source_waypoints are invalid: {exc}")
        return None

    return load_tool_init_cartesian_path(
        current_waypoints=source_waypoints,
        parameters=saved_parameters,
        path_json=path_json,
        require_matching_config_waypoints=True,
    )


def resolve_tool_init_cartesian_path(
    waypoints: list[CartesianPoseWaypoint],
    *,
    max_cartesian_step_m: float = CARTESIAN_MAX_STEP_M,
    max_orientation_step_rad: float = CARTESIAN_MAX_ORI_STEP_RAD,
    arc_sagitta_m: float = TOOL_ARC_SAGITTA_M,
    arc_min_radius_m: float = TOOL_ARC_MIN_RADIUS_M,
    arc_side: str = TOOL_ARC_SIDE,
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
    use_cached_path: bool = True,
    rebuild_path: bool = False,
    save_path: bool = False,
    require_cached_path: bool = False,
) -> CartesianPosePath:
    parameters = tool_init_cartesian_path_parameters(
        max_cartesian_step_m=max_cartesian_step_m,
        max_orientation_step_rad=max_orientation_step_rad,
        arc_sagitta_m=arc_sagitta_m,
        arc_min_radius_m=arc_min_radius_m,
        arc_side=arc_side,
    )

    if use_cached_path and not rebuild_path:
        cached = load_tool_init_cartesian_path(
            current_waypoints=waypoints,
            parameters=parameters,
            path_json=path_json,
        )
        if cached is not None:
            return cached
        if require_cached_path:
            raise RuntimeError(
                "Cached Cartesian INIT_POS -> TOOL_INIT path could not be loaded "
                f"from {path_json}. Run with --rebuild-tool-init-cartesian-path "
                "only when you intentionally want to regenerate it."
            )

    samples, segment_summaries = build_cartesian_pose_path_samples(
        waypoints,
        max_cartesian_step_m=max_cartesian_step_m,
        max_orientation_step_rad=max_orientation_step_rad,
        arc_sagitta_m=arc_sagitta_m,
        arc_min_radius_m=arc_min_radius_m,
        arc_side=arc_side,
    )
    cartesian_path = CartesianPosePath(
        source_waypoints=waypoints,
        samples=samples,
        segment_summaries=segment_summaries,
        source="generated",
        path_json=path_json,
    )
    if save_path:
        save_tool_init_cartesian_path(
            cartesian_path,
            parameters=parameters,
            path_json=path_json,
        )
    return cartesian_path


def read_cartesian_pose(redis_client) -> tuple[np.ndarray, np.ndarray]:
    position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )
    orientation = read_np(
        redis_client,
        redis_keys.cartesian_task_current_orientation,
        (3, 3),
    )
    return position, orientation


def seed_cartesian_goal_at_current(
    redis_client,
    *,
    hold_orientation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Set the Cartesian goal to the live measured pose before a new target."""
    current_position, current_orientation = read_cartesian_pose(redis_client)
    orientation = (
        hold_orientation
        if hold_orientation is not None
        else current_orientation
    )
    set_cartesian_goal(redis_client, current_position, orientation)
    time.sleep(DT)
    return current_position, orientation


def switch_to_cartesian_hold_current(
    redis_client,
    *,
    settle_s: float = CARTESIAN_SETTLE_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Activate Cartesian control while holding the measured EE orientation."""
    current_position, current_orientation = read_cartesian_pose(redis_client)
    locked_orientation = current_orientation.copy()

    print("Switching to Cartesian controller at measured pose.")
    print("Current Cartesian position:", np.round(current_position, 5))
    print("Locking current Cartesian orientation.")

    set_cartesian_goal(redis_client, current_position, locked_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < settle_s:
        current_position = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        set_cartesian_goal(redis_client, current_position, locked_orientation)
        time.sleep(DT)

    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )
    return current_position, locked_orientation


def go_to_pose_waypoint(
    redis_client,
    *,
    target_pos: np.ndarray,
    target_orientation: np.ndarray,
    label: str,
    dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    pos_tol_m: float = POS_TOL_M,
    ori_tol_rad: float = ORI_TOL_RAD,
) -> bool:
    print(f"\n-> {label}: target = {np.round(target_pos, 5).tolist()}")
    print(f"Aligning position and orientation for {label}.")

    seed_cartesian_goal_at_current(
        redis_client,
        hold_orientation=target_orientation,
    )
    set_cartesian_goal(redis_client, target_pos, target_orientation)

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
            set_cartesian_goal(redis_client, target_pos, target_orientation)
            last_refresh = loop_time

        current_position, current_orientation = read_cartesian_pose(redis_client)
        pos_err = position_error(current_position, target_pos)
        ori_err = rotation_error_rad(current_orientation, target_orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                f"MOVING_TO_{label}",
                "| pos_error:",
                f"{pos_err:.5f}",
                "| ori_error_rad:",
                f"{ori_err:.5f}",
            )
            last_status = loop_time

        if pos_err < pos_tol_m and ori_err < ori_tol_rad:
            break

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(
                f"Timed out before reaching {label}.",
                "Final position error:",
                round(pos_err, 5),
                "Final orientation error rad:",
                round(ori_err, 5),
            )
            return False

    print(f"Reached {label}. Dwelling {dwell_s:.2f}s.")
    if dwell_s > 0.0:
        time.sleep(dwell_s)
    return True


def stream_cartesian_pose_path(
    redis_client,
    waypoints: list[CartesianPoseWaypoint],
    *,
    pos_tol_m: float = POS_TOL_M,
    ori_tol_rad: float = ORI_TOL_RAD,
    max_cartesian_step_m: float = CARTESIAN_MAX_STEP_M,
    max_orientation_step_rad: float = CARTESIAN_MAX_ORI_STEP_RAD,
    arc_sagitta_m: float = TOOL_ARC_SAGITTA_M,
    arc_min_radius_m: float = TOOL_ARC_MIN_RADIUS_M,
    arc_side: str = TOOL_ARC_SIDE,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_PER_WAYPOINT_S,
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
    use_cached_path: bool = True,
    rebuild_path: bool = False,
    save_path: bool = False,
    require_cached_path: bool = True,
) -> bool:
    if len(waypoints) < 2:
        return True

    if use_cached_path and not rebuild_path:
        cartesian_path = load_saved_tool_init_cartesian_path(path_json=path_json)
        if cartesian_path is None:
            if require_cached_path:
                raise RuntimeError(
                    "Cached Cartesian INIT_POS -> TOOL_INIT path could not be "
                    f"loaded from {path_json}. Run with rebuild only when you "
                    "intentionally want to regenerate it."
                )
            cartesian_path = resolve_tool_init_cartesian_path(
                waypoints,
                max_cartesian_step_m=max_cartesian_step_m,
                max_orientation_step_rad=max_orientation_step_rad,
                arc_sagitta_m=arc_sagitta_m,
                arc_min_radius_m=arc_min_radius_m,
                arc_side=arc_side,
                path_json=path_json,
                use_cached_path=False,
                rebuild_path=False,
                save_path=save_path,
                require_cached_path=False,
            )
    else:
        cartesian_path = resolve_tool_init_cartesian_path(
            waypoints,
            max_cartesian_step_m=max_cartesian_step_m,
            max_orientation_step_rad=max_orientation_step_rad,
            arc_sagitta_m=arc_sagitta_m,
            arc_min_radius_m=arc_min_radius_m,
            arc_side=arc_side,
            path_json=path_json,
            use_cached_path=use_cached_path,
            rebuild_path=rebuild_path,
            save_path=save_path,
            require_cached_path=require_cached_path,
        )
    if not cartesian_path.samples:
        return True

    active_waypoints = cartesian_path.source_waypoints or waypoints
    print("\nCartesian TOOL_INIT pose path:")
    for waypoint in active_waypoints:
        print(f"  {waypoint.label}: {np.round(waypoint.position_m, 5).tolist()} m")
    print(f"Cartesian path source: {cartesian_path.source}")
    if cartesian_path.path_json is not None:
        print(f"Cartesian path JSON: {cartesian_path.path_json}")

    max_ori_step_deg = max_orientation_step_rad / DEG_TO_RAD
    for summary in cartesian_path.segment_summaries:
        print(
            f"Streaming {summary.get('start_label')}->{summary.get('end_label')}:",
            f"{int(summary.get('num_steps', 0))} samples",
            f"| path {float(summary.get('path_length_m', 0.0)):.4f} m",
            f"| rotation {float(summary.get('rotation_rad', 0.0)):.4f} rad",
            f"| max_ori_step {max_orientation_step_rad:.5f} rad ({max_ori_step_deg:.3f} deg)",
            f"| xy_arc_radius {float(summary.get('xy_arc_radius_m', math.inf)):.4f} m",
            f"| min_xy_radius {float(summary.get('min_xy_radius_m', math.inf)):.4f} m",
            f"| arc_side {summary.get('arc_side')}",
        )

    waypoint_by_label = {waypoint.label: waypoint for waypoint in active_waypoints}
    segment_ranges: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 1
    for summary in cartesian_path.segment_summaries:
        num_steps = int(summary.get("num_steps", 0))
        if num_steps <= 0:
            continue
        segment_ranges.append((cursor, cursor + num_steps - 1, summary))
        cursor += num_steps
    segment_count = max(1, len(segment_ranges))

    loop_time = 0.0
    last_status = 0.0
    start_time = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    for sample_index, sample in enumerate(cartesian_path.samples, start=1):
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        active_segment_index = 1
        segment_step_index = sample_index
        segment_num_steps = len(cartesian_path.samples)
        target_wp = active_waypoints[-1]
        for candidate_index, (start_index, end_index, summary) in enumerate(
            segment_ranges,
            start=1,
        ):
            if start_index <= sample_index <= end_index:
                active_segment_index = candidate_index
                segment_step_index = sample_index - start_index + 1
                segment_num_steps = end_index - start_index + 1
                target_wp = waypoint_by_label.get(
                    str(summary.get("end_label")),
                    active_waypoints[-1],
                )
                break

        set_cartesian_goal(redis_client, sample.position_m, sample.orientation)

        current_position, current_orientation = read_cartesian_pose(redis_client)
        pos_err = position_error(current_position, target_wp.position_m)
        ori_err = rotation_error_rad(current_orientation, target_wp.orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "STREAMING_TOOL_CARTESIAN",
                "| segment:",
                f"{active_segment_index}/{segment_count}",
                "| target:",
                target_wp.label,
                "| sample:",
                f"{segment_step_index}/{segment_num_steps}",
                "| pos_error:",
                round(pos_err, 5),
                "| ori_error_rad:",
                round(ori_err, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start_time > timeout_s:
            print(f"Timed out while streaming to {target_wp.label}.")
            print("Final position error:", round(pos_err, 5))
            print("Final orientation error rad:", round(ori_err, 5))
            return False

    final_wp = active_waypoints[-1]
    set_cartesian_goal(redis_client, final_wp.position_m, final_wp.orientation)
    while True:
        current_position, current_orientation = read_cartesian_pose(redis_client)
        pos_err = position_error(current_position, final_wp.position_m)
        ori_err = rotation_error_rad(current_orientation, final_wp.orientation)
        if pos_err < pos_tol_m and ori_err < ori_tol_rad:
            print(f"Reached {final_wp.label}.")
            return True

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "SETTLING_TOOL_CARTESIAN",
                "| pos_error:",
                round(pos_err, 5),
                "| ori_error_rad:",
                round(ori_err, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start_time > timeout_s:
            print(f"Timed out settling at {final_wp.label}.")
            print("Final position error:", round(pos_err, 5))
            print("Final orientation error rad:", round(ori_err, 5))
            return False

        set_cartesian_goal(redis_client, final_wp.position_m, final_wp.orientation)
        time.sleep(DT)
        loop_time += DT


def approach_tool_init_cartesian(
    redis_client,
    *,
    dwell_at_init_s: float = 0.25,
    dwell_at_tool_init_s: float = 0.5,
    pos_tol_m: float = POS_TOL_M,
    ori_tol_rad: float = ORI_TOL_RAD,
    max_cartesian_step_m: float = CARTESIAN_MAX_STEP_M,
    max_orientation_step_rad: float = CARTESIAN_MAX_ORI_STEP_RAD,
    arc_sagitta_m: float = TOOL_ARC_SAGITTA_M,
    arc_min_radius_m: float = TOOL_ARC_MIN_RADIUS_M,
    arc_side: str = TOOL_ARC_SIDE,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_PER_WAYPOINT_S,
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
    use_cached_path: bool = True,
    rebuild_path: bool = False,
    save_path: bool = False,
    require_cached_path: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """INIT_POS -> TOOL_WP1 -> TOOL_WP2 -> TOOL_INIT as Cartesian poses."""
    cfg = load_demo_day_config()
    current_position, current_orientation = read_cartesian_pose(redis_client)
    cached_path = (
        load_saved_tool_init_cartesian_path(path_json=path_json)
        if use_cached_path and not rebuild_path
        else None
    )
    cached_start_wp = (
        cached_path.source_waypoints[0]
        if cached_path is not None and cached_path.source_waypoints
        else None
    )
    init_pos = (
        cached_start_wp.position_m
        if cached_start_wp is not None
        else load_init_pos_m(cfg)
    )
    init_orientation = (
        cached_start_wp.orientation
        if cached_start_wp is not None
        else current_orientation
    )

    print("Moving to INIT_POS before Cartesian tool path.")
    print("Current Cartesian position:", np.round(current_position, 5))
    print("INIT_POS target:", np.round(init_pos, 5))
    if cached_start_wp is not None:
        print("Using cached path INIT_POS pose from:", path_json)

    set_cartesian_goal(redis_client, current_position, current_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    if not go_to_pose_waypoint(
        redis_client,
        target_pos=init_pos,
        target_orientation=init_orientation,
        label="INIT_POS",
        dwell_s=dwell_at_init_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        pos_tol_m=pos_tol_m,
        ori_tol_rad=min(ori_tol_rad, CACHED_TOOL_PATH_START_ORI_TOL_RAD),
    ):
        return None

    start_position, start_orientation = read_cartesian_pose(redis_client)
    pose_waypoints = [
        CartesianPoseWaypoint("INIT_POS", start_position, start_orientation),
        *load_cartesian_tool_waypoints(cfg),
    ]

    if not stream_cartesian_pose_path(
        redis_client,
        pose_waypoints,
        pos_tol_m=pos_tol_m,
        ori_tol_rad=ori_tol_rad,
        max_cartesian_step_m=max_cartesian_step_m,
        max_orientation_step_rad=max_orientation_step_rad,
        arc_sagitta_m=arc_sagitta_m,
        arc_min_radius_m=arc_min_radius_m,
        arc_side=arc_side,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        path_json=path_json,
        use_cached_path=use_cached_path,
        rebuild_path=rebuild_path,
        save_path=save_path,
        require_cached_path=require_cached_path,
    ):
        return None

    if dwell_at_tool_init_s > 0.0:
        time.sleep(dwell_at_tool_init_s)
    return read_cartesian_pose(redis_client)


def return_from_tool_init_cartesian(
    redis_client,
    *,
    pos_tol_m: float = POS_TOL_M,
    ori_tol_rad: float = ORI_TOL_RAD,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_PER_WAYPOINT_S,
    path_json: Path = DEFAULT_TOOL_INIT_CARTESIAN_PATH_JSON,
) -> bool:
    """TOOL_INIT -> INIT_POS by replaying the cached Cartesian path in reverse."""
    cartesian_path = load_saved_tool_init_cartesian_path(path_json=path_json)
    if cartesian_path is None:
        print(f"Could not load cached Cartesian TOOL_INIT return path: {path_json}")
        return False
    if not cartesian_path.source_waypoints:
        print("Cached Cartesian TOOL_INIT path is missing source waypoints.")
        return False

    final_wp = cartesian_path.source_waypoints[0]
    tool_init_wp = cartesian_path.source_waypoints[-1]
    reverse_samples = list(reversed(cartesian_path.samples[:-1]))

    current_position, current_orientation = read_cartesian_pose(redis_client)
    start_pos_err = position_error(current_position, tool_init_wp.position_m)
    start_ori_err = rotation_error_rad(current_orientation, tool_init_wp.orientation)
    print("\nCartesian TOOL_INIT return path:")
    print("  source JSON:", path_json)
    print("  TOOL_INIT start error m:", round(start_pos_err, 5))
    print("  TOOL_INIT start orientation error rad:", round(start_ori_err, 5))
    print("  reverse samples:", len(reverse_samples))

    set_cartesian_goal(redis_client, current_position, current_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    loop_time = 0.0
    last_status = 0.0
    start_time = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    for sample_index, sample in enumerate(reverse_samples, start=1):
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )
        set_cartesian_goal(redis_client, sample.position_m, sample.orientation)

        current_position, current_orientation = read_cartesian_pose(redis_client)
        pos_err = position_error(current_position, final_wp.position_m)
        ori_err = rotation_error_rad(current_orientation, final_wp.orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "STREAMING_TOOL_CARTESIAN_RETURN",
                "| sample:",
                f"{sample_index}/{len(reverse_samples)}",
                "| pos_error:",
                round(pos_err, 5),
                "| ori_error_rad:",
                round(ori_err, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start_time > timeout_s:
            print("Timed out while streaming Cartesian TOOL_INIT return path.")
            print("Final position error:", round(pos_err, 5))
            print("Final orientation error rad:", round(ori_err, 5))
            return False

    set_cartesian_goal(redis_client, final_wp.position_m, final_wp.orientation)
    while True:
        current_position, current_orientation = read_cartesian_pose(redis_client)
        pos_err = position_error(current_position, final_wp.position_m)
        ori_err = rotation_error_rad(current_orientation, final_wp.orientation)
        if pos_err < pos_tol_m and ori_err < ori_tol_rad:
            print(f"Reached {final_wp.label} via cached Cartesian return path.")
            return True

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                "SETTLING_TOOL_CARTESIAN_RETURN",
                "| pos_error:",
                round(pos_err, 5),
                "| ori_error_rad:",
                round(ori_err, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start_time > timeout_s:
            print("Timed out settling after Cartesian TOOL_INIT return path.")
            print("Final position error:", round(pos_err, 5))
            print("Final orientation error rad:", round(ori_err, 5))
            return False

        set_cartesian_goal(redis_client, final_wp.position_m, final_wp.orientation)
        time.sleep(DT)
        loop_time += DT


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def station_point_in_camera_frame_m(
    p_station_m: np.ndarray,
    *,
    ref_tvec_cam: np.ndarray,
    ref_rvec_cam: np.ndarray,
    tag_size_m: float,
    tag_tl_station_m: np.ndarray,
) -> np.ndarray:
    import cv2

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
    p_tip = R_cam_to_tip @ np.asarray(p_cam_m, dtype=float) + camera_offset_in_tip_m
    return ee_pos_world + R_ee_world @ p_tip


def load_scan_reference_pose(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if "ee_position_world_m" not in observation or "ee_orientation_world" not in observation:
        raise RuntimeError(
            "Observation JSON is missing ee_position_world_m / ee_orientation_world."
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


def build_volume_waypoints(
    *,
    model: PaintToolStationModel,
    observation: dict[str, Any],
) -> list[VolumeWaypoint]:
    tag_size_m = float(observation.get("tag_size_m", model.tag_size_mm * 1.0e-3))
    reference_tag_id = int(
        observation.get("reference_tag_id", model.reference_tag_id)
    )

    waypoints: list[VolumeWaypoint] = []
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
            VolumeWaypoint(
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
    waypoints: list[VolumeWaypoint],
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


def load_observation_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_tool_station_calibration(
    *,
    model_json_path: Path,
    observation_json_path: Path,
    world_json_path: Path,
) -> ToolStationCalibration:
    """Load observation JSON and compute/export world-frame volume waypoints."""
    if not model_json_path.is_file():
        raise FileNotFoundError(f"Tool-station model JSON not found: {model_json_path}")
    if not observation_json_path.is_file():
        raise FileNotFoundError(
            f"Tool-station observation JSON not found: {observation_json_path}"
        )

    observation = load_observation_json(observation_json_path)
    scan_reference_position_m, hold_orientation = load_scan_reference_pose(observation)
    model = PaintToolStationModel()
    waypoints = build_volume_waypoints(model=model, observation=observation)

    saved_world_path = save_world_positions_json(
        path=world_json_path,
        waypoints=waypoints,
        scan_reference_position_m=scan_reference_position_m,
        hold_orientation=hold_orientation,
        observation_path=observation_json_path,
        model_path=model_json_path,
    )
    print(f"Saved world-frame waypoints JSON: {saved_world_path}")

    print("\nComputed world-frame hover waypoints (relative to scan reference):")
    for wp in waypoints:
        delta = wp.world_position_m - scan_reference_position_m
        print(
            f"  {wp.label} (from {wp.volume_name}):"
            f" station {np.round(wp.station_position_mm, 3).tolist()} mm"
            f" -> world {np.round(wp.world_position_m, 5).tolist()} m"
            f" (delta {np.round(delta, 5).tolist()} m)"
        )

    return ToolStationCalibration(
        model_json_path=model_json_path,
        observation_json_path=observation_json_path,
        world_json_path=saved_world_path,
        observation=observation,
        scan_reference_position_m=scan_reference_position_m,
        hold_orientation=hold_orientation,
        waypoints=waypoints,
    )


# ---------------------------------------------------------------------------
# Cartesian motion
# ---------------------------------------------------------------------------

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
    seed_cartesian_goal_at_current(
        redis_client,
        hold_orientation=hold_orientation,
    )
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


def visit_tool_volume_waypoints(
    redis_client,
    calibration: ToolStationCalibration,
    *,
    dwell_s: float = DWELL_AT_WAYPOINT_S,
    skip_final_scan_reference: bool = False,
    timeout_per_waypoint_s: float = TIMEOUT_PER_WAYPOINT_S,
    status_period_s: float = STATUS_PERIOD_S,
) -> bool:
    """Move P1_INIT -> P2_INIT -> P3_INIT -> WATER_INIT (orientation fixed)."""
    scan_reference_position_m = calibration.scan_reference_position_m
    hold_orientation = calibration.hold_orientation

    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )

    print("Using scan-reference pose from calibration JSON:")
    print("  scan_reference_position_world_m:", scan_reference_position_m.tolist())
    print("  hold_orientation (fixed throughout):")
    print(hold_orientation)
    print("Current Cartesian position:", current_position.tolist())

    set_cartesian_goal(redis_client, current_position, hold_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print(f"\nUsing controller: {CARTESIAN_CONTROLLER}")

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < CARTESIAN_SETTLE_S:
        set_cartesian_goal(redis_client, current_position, hold_orientation)
        time.sleep(DT)

    sequence: list[tuple[str, np.ndarray]] = [
        (wp.label, wp.world_position_m) for wp in calibration.waypoints
    ]
    if not skip_final_scan_reference:
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
            return False

    print("\nFinished visiting all tool-station volumes.")
    return True


def run_volume_visit_from_json(
    *,
    model_json_path: Path = DEFAULT_TOOL_STATION_MODEL_JSON,
    observation_json_path: Path = DEFAULT_TOOL_STATION_OBSERVATION_JSON,
    world_json_path: Path = DEFAULT_TOOL_STATION_WORLD_JSON,
    dwell_s: float = DWELL_AT_WAYPOINT_S,
    skip_final_scan_reference: bool = False,
    timeout_per_waypoint_s: float = TIMEOUT_PER_WAYPOINT_S,
    status_period_s: float = STATUS_PERIOD_S,
    redis_client=None,
) -> int:
    """Load calibration JSON and run the volume visit sequence."""
    if redis is None and redis_client is None:
        print("`redis` package is not installed.")
        return 1

    try:
        calibration = compute_tool_station_calibration(
            model_json_path=model_json_path,
            observation_json_path=observation_json_path,
            world_json_path=world_json_path,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1

    client = redis_client if redis_client is not None else redis.Redis()
    if not ensure_robot_ready(client):
        return 1

    ok = visit_tool_volume_waypoints(
        client,
        calibration,
        dwell_s=dwell_s,
        skip_final_scan_reference=skip_final_scan_reference,
        timeout_per_waypoint_s=timeout_per_waypoint_s,
        status_period_s=status_period_s,
    )
    return 0 if ok else 1
