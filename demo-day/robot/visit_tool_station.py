"""Move the robot to the tool-station TOOL_INIT joint pose.

Flow:
    INIT_POS (Cartesian) -> TOOL_SAFE_APPROACH -> TOOL_SAFE_APPROACH_2 -> TOOL_INIT
    then Cartesian paint dips:
        P1_INIT_POS -> dip -> P2_INIT_POS -> dip -> P3_INIT_POS -> dip
        -> WATER_INIT_POS -> swirl water -> WATER_TO_NAPKIN_INIT_POS -> wipe napkin

Each dip is hover -> hover + z_dive_in_offset -> hover (see helpers/primitives.py).

Usage:
    python robot/visit_tool_station.py
    python robot/visit_tool_station.py --joint-max-step-deg 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from typing import Callable

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import redis
except ImportError:
    redis = None

from helpers.joint_motion import approach_tool_init
from helpers.primitives import (
    dip_paint_1,
    dip_paint_2,
    dip_paint_3,
    load_paint_init_pos_m,
    load_z_dive_in_offset_m,
    swirl_water,
    wipe_napkin,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"

ROBOT_NAME = "Titania"
CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"
CARTESIAN_CONTROLLER = "cartesian_controller"
JOINT_CONTROLLER = "joint_controller"

DT = 0.01
DEG_TO_RAD = math.pi / 180.0
POS_TOL_M = 1.0e-2
DWELL_AT_INIT_POS_S = 0.25
JOINT_ARRIVAL_THRESHOLD = 0.25
JOINT_MAX_STEP_DEG = 0.5
JOINT_CONTROLLER_SETTLE_S = 0.25
CARTESIAN_SETTLE_S = 0.25
DWELL_AT_TOOL_INIT_S = 0.5
DWELL_AT_PAINT_HOVER_S = 0.25
DWELL_AT_PAINT_DIP_S = 0.5
STATUS_PERIOD_S = 0.25
TIMEOUT_S = 60.0

DEFAULT_CAMERA_INDEX = 0
PREVIEW_WINDOW = "Tool Station AprilTag Scan"
TAG_FAMILY = "tag36h11"
TOOL_TAG_IDS = (10, 11, 12, 13)
SCAN_STABLE_FRAMES_REQUIRED = 10
SCAN_STATUS_PERIOD_S = 0.25
SCAN_TIMEOUT_S = 30.0
DEFAULT_PATH_LOG_DIR = DEMO_DAY_DIR / "path-logs"

# Outputs written for calibration / replay.
DEFAULT_TOOL_STATION_MODEL_JSON = DEMO_DAY_DIR / "tool_station_model.json"
DEFAULT_TOOL_STATION_OBSERVATION_JSON = DEMO_DAY_DIR / "tool_station_observation.json"
DEFAULT_TOOL_STATION_WORLD_JSON = DEMO_DAY_DIR / "tool_station_world_positions.json"

# Camera intrinsics file (same one used by demo-day/robot/calibrate_canvas.py).
DEFAULT_CAMERA_INTRINSICS_JSON = SCRIPT_DIR / "camera_intrinsics.json"
TAG_SIZE_M = 0.035
TOOL_OBSERVATION_FRAMES = 12
DEFAULT_HFOV_DEG = 65.0


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
    joint_task_goal_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::joint_controller::joint_task::goal_position"
    )
    joint_task_goal_velocity: str = (
        f"opensai::controllers::{ROBOT_NAME}::joint_controller::joint_task::goal_velocity"
    )
    joint_task_goal_acceleration: str = (
        f"opensai::controllers::{ROBOT_NAME}::joint_controller::joint_task::goal_acceleration"
    )
    sensor_joint_positions: str = f"opensai::sensors::{ROBOT_NAME}::joint_positions"
    joint_names: str = f"opensai::controllers::{ROBOT_NAME}::joint_names"
    active_controller: str = f"opensai::controllers::{ROBOT_NAME}::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


redis_keys = RedisKeys()


@dataclass
class PathTraceSample:
    t: float
    phase: str
    position: np.ndarray


def load_demo_day_config() -> dict:
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_init_pos() -> np.ndarray:
    cfg = load_demo_day_config()
    init_pos = np.array(cfg["init_pos_m"], dtype=float)
    if init_pos.shape != (3,):
        raise RuntimeError(
            f"{DEMO_DAY_CONFIG_PATH} init_pos_m must contain 3 coordinates."
        )
    return init_pos


def load_joint_waypoint_deg(config: dict, name: str) -> np.ndarray:
    if name not in config:
        raise RuntimeError(f"{DEMO_DAY_CONFIG_PATH} is missing {name}.")

    waypoint = np.array(config[name], dtype=float)
    if waypoint.shape != (7,):
        raise RuntimeError(
            f"{DEMO_DAY_CONFIG_PATH} {name} must contain 7 joint angles."
        )
    return waypoint


def load_tool_waypoints_deg() -> list[tuple[str, np.ndarray]]:
    cfg = load_demo_day_config()
    waypoint_names = [
        "TOOL_SAFE_APPROACH",
        "TOOL_SAFE_APPROACH_2",
        "TOOL_INIT",
    ]
    return [
        (name, load_joint_waypoint_deg(cfg, name))
        for name in waypoint_names
    ]


def decode_redis_value(val):
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def read_np(redis_client, key: str, expected_shape: tuple[int, ...]) -> np.ndarray:
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


def read_optional_np(
    redis_client,
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    val = redis_client.get(key)
    if val is None:
        return None
    try:
        arr = np.array(json.loads(decode_redis_value(val)), dtype=float)
        if arr.shape == expected_shape:
            return arr
    except Exception:
        pass
    return None


def read_optional_json(redis_client, key: str):
    val = redis_client.get(key)
    if val is None:
        return None
    try:
        return json.loads(decode_redis_value(val))
    except Exception:
        return decode_redis_value(val)


def ensure_robot_ready(redis_client, config_file_name_expected: str) -> bool:
    config_raw = redis_client.get(redis_keys.config_file_name)
    if config_raw is None:
        print("Could not read config file name from Redis.")
        print("Missing key:", redis_keys.config_file_name)
        return False

    config_file_name = decode_redis_value(config_raw)
    if config_file_name != config_file_name_expected:
        print("This script is meant to be used with config file:", config_file_name_expected)
        print("Current config file:", config_file_name)
        return False

    return True


def set_active_controller(redis_client, controller_name: str) -> None:
    while True:
        active_raw = redis_client.get(redis_keys.active_controller)
        active_controller = decode_redis_value(active_raw) if active_raw is not None else None
        if active_controller == controller_name:
            break
        redis_client.set(redis_keys.active_controller, controller_name)
        time.sleep(0.001)


def set_joint_goal(redis_client, position: np.ndarray) -> None:
    redis_client.set(redis_keys.joint_task_goal_position, json.dumps(position.tolist()))
    redis_client.set(redis_keys.joint_task_goal_velocity, json.dumps(np.zeros_like(position).tolist()))
    redis_client.set(
        redis_keys.joint_task_goal_acceleration,
        json.dumps(np.zeros_like(position).tolist()),
    )


def set_cartesian_goal(
    redis_client,
    position: np.ndarray,
    orientation: np.ndarray,
) -> None:
    redis_client.set(
        redis_keys.cartesian_task_goal_position,
        json.dumps(np.asarray(position, dtype=float).tolist()),
    )
    redis_client.set(
        redis_keys.cartesian_task_goal_orientation,
        json.dumps(np.asarray(orientation, dtype=float).tolist()),
    )


def position_error(current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(goal_pos) - np.asarray(current_pos)))


def append_path_trace(
    samples: list[PathTraceSample],
    *,
    trace_start: float,
    phase: str,
    position: np.ndarray | None,
) -> None:
    if position is None:
        return
    samples.append(
        PathTraceSample(
            t=time.perf_counter() - trace_start,
            phase=phase,
            position=np.asarray(position, dtype=float).copy(),
        )
    )


def save_path_trace_csv(samples: list[PathTraceSample], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "phase", "x", "y", "z"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "t": f"{sample.t:.6f}",
                    "phase": sample.phase,
                    "x": f"{sample.position[0]:.9f}",
                    "y": f"{sample.position[1]:.9f}",
                    "z": f"{sample.position[2]:.9f}",
                }
            )


def set_axes_equal_3d(ax, positions: np.ndarray) -> None:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0e-3)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def save_path_visualization(
    samples: list[PathTraceSample],
    output_dir: Path,
) -> tuple[Path, Path] | None:
    if not samples:
        print("No Cartesian path samples recorded; skipping path visualization.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    png_path = output_dir / f"tool_station_path_{stamp}.png"
    csv_path = output_dir / f"tool_station_path_{stamp}.csv"
    save_path_trace_csv(samples, csv_path)

    try:
        matplotlib_config_dir = Path("/tmp/matplotlib")
        matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping path visualization.")
        print(f"Saved path samples CSV: {csv_path}")
        return None

    positions = np.vstack([sample.position for sample in samples])
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)

    start = 0
    while start < len(samples):
        phase = samples[start].phase
        end = start + 1
        while end < len(samples) and samples[end].phase == phase:
            end += 1
        segment = positions[start:end]
        ax3d.plot(segment[:, 0], segment[:, 1], segment[:, 2], label=phase)
        axxy.plot(segment[:, 0], segment[:, 1], label=phase)
        start = end

    ax3d.scatter(positions[0, 0], positions[0, 1], positions[0, 2], marker="o", label="start")
    ax3d.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], marker="x", label="end")
    axxy.scatter(positions[0, 0], positions[0, 1], marker="o", label="start")
    axxy.scatter(positions[-1, 0], positions[-1, 1], marker="x", label="end")

    ax3d.set_title("Measured Cartesian Path")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    set_axes_equal_3d(ax3d, positions)
    ax3d.legend(loc="best")

    axxy.set_title("XY Projection")
    axxy.set_xlabel("X (m)")
    axxy.set_ylabel("Y (m)")
    axxy.axis("equal")
    axxy.grid(True)
    axxy.legend(loc="best")

    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved path visualization: {png_path}")
    print(f"Saved path samples CSV: {csv_path}")
    return png_path, csv_path


def move_to_init_pos(
    redis_client,
    *,
    pos_tol_m: float = POS_TOL_M,
    dwell_s: float = DWELL_AT_INIT_POS_S,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    path_trace: list[PathTraceSample] | None = None,
    trace_start: float | None = None,
) -> int:
    init_pos = load_init_pos()
    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )
    hold_orientation = read_np(
        redis_client,
        redis_keys.cartesian_task_current_orientation,
        (3, 3),
    )

    print("Current Cartesian position:", np.round(current_position, 5))
    print("INIT_POS target:", np.round(init_pos, 5))
    print("Holding current Cartesian orientation during INIT_POS move.")
    if path_trace is not None and trace_start is not None:
        append_path_trace(
            path_trace,
            trace_start=trace_start,
            phase="INIT_POS",
            position=current_position,
        )

    set_cartesian_goal(redis_client, current_position, hold_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    start = time.perf_counter()
    if not move_cartesian_to_target(
        redis_client,
        target_position=init_pos,
        hold_orientation=hold_orientation,
        label="INIT_POS",
        pos_tol_m=pos_tol_m,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        start_time=start,
        path_trace=path_trace,
        trace_start=trace_start,
    ):
        return 1

    print("Reached INIT_POS.")
    if dwell_s > 0.0:
        time.sleep(dwell_s)
    return 0


def move_cartesian_to_target(
    redis_client,
    *,
    target_position: np.ndarray,
    hold_orientation: np.ndarray,
    label: str,
    pos_tol_m: float,
    status_period_s: float,
    timeout_s: float,
    start_time: float,
    path_trace: list[PathTraceSample] | None = None,
    trace_start: float | None = None,
) -> bool:
    set_cartesian_goal(redis_client, target_position, hold_orientation)
    loop_time = 0.0
    last_status = 0.0
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    while True:
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        current_position = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        if path_trace is not None and trace_start is not None:
            append_path_trace(
                path_trace,
                trace_start=trace_start,
                phase=label,
                position=current_position,
            )
        err = position_error(current_position, target_position)
        set_cartesian_goal(redis_client, target_position, hold_orientation)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(f"MOVING_TO_{label}", "| pos_error:", round(err, 5))
            last_status = loop_time

        if err < pos_tol_m:
            return True

        if timeout_s > 0.0 and time.perf_counter() - start_time > timeout_s:
            print(f"Timed out before reaching {label}.")
            print("Final position error:", round(err, 5))
            return False


def cubic_hermite(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    duration_steps: int,
    step_index: int,
) -> np.ndarray:
    u = step_index / duration_steps
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return (
        h00 * p0
        + h10 * duration_steps * v0
        + h01 * p1
        + h11 * duration_steps * v1
    )


def build_smooth_joint_path(
    waypoints: list[np.ndarray],
    *,
    max_step: float,
) -> tuple[np.ndarray, list[int]]:
    """Build one C1-continuous joint path through ordered waypoints.

    The returned path includes each waypoint exactly. Interior waypoints are
    passed with shared nonzero tangents where possible; the controller does
    not stop or dwell there.
    """
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")

    distances = [
        float(np.linalg.norm(waypoints[index + 1] - waypoints[index]))
        for index in range(len(waypoints) - 1)
    ]

    if max_step <= 0.0:
        raise ValueError("max_step must be positive")

    scale = 1.0
    for _attempt in range(8):
        durations = [
            max(1, int(math.ceil(scale * distance / max_step)))
            for distance in distances
        ]
        waypoint_indices = [0]
        for duration in durations:
            waypoint_indices.append(waypoint_indices[-1] + duration)
        knot_times = np.array(waypoint_indices, dtype=float)

        velocities: list[np.ndarray] = []
        for index in range(len(waypoints)):
            if index == 0 or index == len(waypoints) - 1:
                velocities.append(np.zeros(7, dtype=float))
            else:
                previous_delta = waypoints[index] - waypoints[index - 1]
                next_delta = waypoints[index + 1] - waypoints[index]
                previous_distance = float(np.linalg.norm(previous_delta))
                next_distance = float(np.linalg.norm(next_delta))
                if previous_distance < 1.0e-12 or next_distance < 1.0e-12:
                    velocities.append(np.zeros(7, dtype=float))
                    continue

                previous_direction = previous_delta / previous_distance
                next_direction = next_delta / next_distance
                tangent_direction = previous_direction + next_direction
                tangent_norm = float(np.linalg.norm(tangent_direction))
                if tangent_norm < 1.0e-6:
                    velocities.append(np.zeros(7, dtype=float))
                    continue

                tangent_direction = tangent_direction / tangent_norm
                previous_speed = previous_distance / (
                    knot_times[index] - knot_times[index - 1]
                )
                next_speed = next_distance / (
                    knot_times[index + 1] - knot_times[index]
                )
                velocities.append(
                    tangent_direction * min(previous_speed, next_speed)
                )

        samples = [waypoints[0].copy()]
        for segment_index, duration in enumerate(durations):
            for step_index in range(1, duration + 1):
                samples.append(
                    cubic_hermite(
                        waypoints[segment_index],
                        waypoints[segment_index + 1],
                        velocities[segment_index],
                        velocities[segment_index + 1],
                        duration,
                        step_index,
                    )
                )

        path = np.vstack(samples)
        max_observed_step = float(
            np.max(np.linalg.norm(np.diff(path, axis=0), axis=1))
        )
        if max_observed_step <= max_step * 1.001:
            return path, waypoint_indices

        scale *= max(1.25, 1.1 * max_observed_step / max_step)

    return path, waypoint_indices


def open_camera(camera_index: int) -> cv2.VideoCapture:
    if cv2 is None:
        raise RuntimeError("`cv2` package is not installed.")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):
        cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    ok, _frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Opened camera index {camera_index}, but could not read a frame.")
    print(f"Opened camera index {camera_index}.")
    return cap


def make_detector():
    """Build an ArUco/AprilTag detector compatible with old and new OpenCV."""
    if cv2 is None:
        raise RuntimeError("`cv2` package is not installed.")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV is installed without cv2.aruco support.")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    try:
        params = cv2.aruco.DetectorParameters()
        return ("new", cv2.aruco.ArucoDetector(dictionary, params), dictionary, params)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        return ("old", None, dictionary, params)


def detect_tags(detector_bundle, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return detected tag IDs and corners. Corners shape is (N, 4, 2)."""
    kind, detector, dictionary, params = detector_bundle
    if kind == "new":
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=params,
        )
    if ids is None:
        return np.array([], dtype=int), np.zeros((0, 4, 2), dtype=np.float32)
    ids = ids.flatten().astype(int)
    corners_2d = np.array([c.reshape(4, 2) for c in corners], dtype=np.float32)
    return ids, corners_2d


def draw_scan_overlay(
    frame: np.ndarray,
    ids: np.ndarray,
    corners_2d: np.ndarray,
    status_text: str,
    stable_count: int,
    stable_frames_required: int,
) -> None:
    if len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(
            frame,
            [c.reshape(1, 4, 2) for c in corners_2d],
            ids.reshape(-1, 1),
        )
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"stable {stable_count}/{stable_frames_required}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 255),
        2,
    )


def scan_tool_tags(
    *,
    cap,
    detector_bundle,
    preview: bool = True,
    stable_frames_required: int = SCAN_STABLE_FRAMES_REQUIRED,
    status_period_s: float = SCAN_STATUS_PERIOD_S,
    timeout_s: float = SCAN_TIMEOUT_S,
) -> int:
    stable_frames_required = max(1, stable_frames_required)
    required_ids = set(TOOL_TAG_IDS)
    stable_count = 0
    last_status = 0.0
    start = time.perf_counter()

    print(f"Scanning for {TAG_FAMILY} AprilTags:", sorted(required_ids))
    if preview:
        print("Press q or Esc in the preview window to stop.")

    for _ in range(5):
        cap.read()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read camera frame.")
            return 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids, corners_2d = detect_tags(detector_bundle, gray)
        visible_target_ids = sorted(set(ids.tolist()) & required_ids)
        missing_ids = sorted(required_ids - set(visible_target_ids))

        if not missing_ids:
            stable_count += 1
            status = f"ALL TOOL TAGS visible: {visible_target_ids}"
        elif visible_target_ids:
            stable_count = 0
            status = f"missing: {missing_ids}  visible: {visible_target_ids}"
        else:
            stable_count = 0
            status = "no tool tags detected"

        now = time.perf_counter()
        if preview:
            draw_scan_overlay(
                frame,
                ids,
                corners_2d,
                status,
                stable_count,
                stable_frames_required,
            )
            cv2.imshow(PREVIEW_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("Tool tag scan stopped by user.")
                return 1
        elif status_period_s <= 0.0 or now - last_status >= status_period_s:
            print(
                "TOOL_TAG_SCAN",
                "|",
                status,
                "| stable",
                f"{stable_count}/{stable_frames_required}",
            )
            last_status = now

        if stable_count >= stable_frames_required:
            print("Detected all tool tags:", visible_target_ids)
            return 0

        if timeout_s > 0.0 and now - start > timeout_s:
            print("Timed out before all tool tags were visible.")
            print("Last visible target IDs:", visible_target_ids)
            print("Missing target IDs:", missing_ids)
            return 1


def load_camera_intrinsics(path: Path, frame_w: int, frame_h: int):
    """Load (K, dist, calibrated) from a JSON file, or fall back to an HFOV
    approximation if the file is missing. Mirrors calibrate_canvas.py."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        K = np.array(payload["camera_matrix"], dtype=float)
        dist = np.array(
            payload.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        print(f"Loaded camera intrinsics from {path}")
        return K, dist, True

    fx = fy = 0.5 * frame_w / math.tan(math.radians(DEFAULT_HFOV_DEG / 2.0))
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float
    )
    dist = np.zeros((5,), dtype=float)
    print(
        f"WARNING: no intrinsics file at {path}. Using {DEFAULT_HFOV_DEG} deg"
        " HFOV approximation. Tool-station Z depth will be inaccurate --"
        " calibrate the camera and provide it for accurate results."
    )
    return K, dist, False


def average_rotation_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """Average rotation matrices via SVD re-orthogonalization."""
    if not matrices:
        raise ValueError("matrices must be non-empty")
    if len(matrices) == 1:
        return np.asarray(matrices[0], dtype=float)

    mean_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
    u, _, vt = np.linalg.svd(mean_matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def estimate_tool_tag_pose(
    image_corners_2d: np.ndarray,
    tag_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run cv2.solvePnP for one tool-station tag (returns tvec, rvec)."""
    s = tag_size_m / 2.0
    object_points = np.array(
        [
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = np.ascontiguousarray(
        image_corners_2d.reshape(4, 2)
    ).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    return tvec.flatten().astype(float), rvec.flatten().astype(float)


@dataclass
class ToolTagObservationSnapshot:
    """Tag poses and EE pose captured on the same camera frames."""

    reference_tag_id: int
    tag_poses_in_camera_frame: dict[int, dict[str, list[float]]]
    ee_position_world_m: np.ndarray
    ee_orientation_world: np.ndarray
    num_frames: int
    visible_tag_ids: list[int]


def capture_tool_tag_observation(
    *,
    cap,
    detector_bundle,
    redis_client,
    intrinsics_json: Path,
    num_frames: int,
) -> ToolTagObservationSnapshot | None:
    """Capture tag PnP poses and EE Cartesian pose on synchronized frames.

    Only the reference tag must be visible in every collected frame. The
    reference tag is the first detected tag in preference order
    (10, 11, 12, 13). Other visible tool tags are recorded when present.
    """
    if cv2 is None:
        raise RuntimeError("`cv2` package is not installed.")

    try:
        from helpers.model_tool_station import pick_reference_tag_id
    except ImportError as exc:
        raise RuntimeError("Could not import model_tool_station") from exc

    reference_tag_id: int | None = None
    tvec_history: dict[int, list[np.ndarray]] = {tid: [] for tid in TOOL_TAG_IDS}
    rvec_history: dict[int, list[np.ndarray]] = {tid: [] for tid in TOOL_TAG_IDS}
    ee_pos_history: list[np.ndarray] = []
    ee_ori_history: list[np.ndarray] = []
    K = None
    dist = None

    frames_collected = 0
    attempts = 0
    max_attempts = max(num_frames * 8, 40)

    while frames_collected < num_frames and attempts < max_attempts:
        attempts += 1
        ok, frame = cap.read()
        if not ok:
            print("Could not read camera frame during tool-station snapshot.")
            return None

        if K is None:
            h, w = frame.shape[:2]
            K, dist, _ = load_camera_intrinsics(intrinsics_json, w, h)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ids, corners_2d = detect_tags(detector_bundle, gray)
        ids_list = ids.tolist() if ids.size > 0 else []

        per_frame_tvecs: dict[int, np.ndarray] = {}
        per_frame_rvecs: dict[int, np.ndarray] = {}
        for tag_id, corner_pts in zip(ids_list, corners_2d):
            if tag_id not in TOOL_TAG_IDS:
                continue
            pose = estimate_tool_tag_pose(corner_pts, TAG_SIZE_M, K, dist)
            if pose is None:
                continue
            tvec, rvec = pose
            per_frame_tvecs[tag_id] = tvec
            per_frame_rvecs[tag_id] = rvec

        visible_tool_ids = set(per_frame_tvecs.keys())
        if not visible_tool_ids:
            continue

        if reference_tag_id is None:
            reference_tag_id = pick_reference_tag_id(visible_tool_ids)
            if reference_tag_id is None:
                continue
            print(
                "Selected reference tag for tool-station calibration:",
                reference_tag_id,
                "| visible:",
                sorted(visible_tool_ids),
            )

        if reference_tag_id not in per_frame_tvecs:
            continue

        ee_pos = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        ee_ori = read_np(
            redis_client,
            redis_keys.cartesian_task_current_orientation,
            (3, 3),
        )

        for tag_id in visible_tool_ids:
            tvec_history[tag_id].append(per_frame_tvecs[tag_id])
            rvec_history[tag_id].append(per_frame_rvecs[tag_id])
        ee_pos_history.append(ee_pos)
        ee_ori_history.append(ee_ori)
        frames_collected += 1

    if frames_collected == 0 or reference_tag_id is None:
        print(
            "Tool-station snapshot failed: no frame with a usable tool tag"
            f" after {attempts} attempts."
        )
        return None

    poses: dict[int, dict[str, list[float]]] = {}
    visible_tag_ids = sorted(
        tag_id for tag_id in TOOL_TAG_IDS if tvec_history[tag_id]
    )
    for tag_id in visible_tag_ids:
        tvecs = np.array(tvec_history[tag_id], dtype=float)
        rvecs = np.array(rvec_history[tag_id], dtype=float)
        poses[tag_id] = {
            "tvec": [round(float(v), 6) for v in tvecs.mean(axis=0)],
            "rvec": [round(float(v), 6) for v in rvecs.mean(axis=0)],
            "num_frames": int(tvecs.shape[0]),
        }

    ee_pos_mean = np.mean(np.stack(ee_pos_history, axis=0), axis=0)
    ee_ori_mean = average_rotation_matrices(ee_ori_history)

    print(
        f"Captured tool-station snapshot: averaged {frames_collected} frames"
        f" with reference tag {reference_tag_id} visible."
    )
    print("Visible tool tags in snapshot:", visible_tag_ids)
    print(
        "Synchronized EE Cartesian pose at tag scan (world frame):",
        np.round(ee_pos_mean, 5).tolist(),
    )
    return ToolTagObservationSnapshot(
        reference_tag_id=reference_tag_id,
        tag_poses_in_camera_frame=poses,
        ee_position_world_m=ee_pos_mean,
        ee_orientation_world=ee_ori_mean,
        num_frames=frames_collected,
        visible_tag_ids=visible_tag_ids,
    )


def save_tool_station_artifacts(
    *,
    cap,
    detector_bundle,
    redis_client,
    intrinsics_json: Path,
    model_json_path: Path,
    observation_json_path: Path,
    snapshot_frames: int,
) -> int:
    """Save tool-station JSON files as soon as TOOL_INIT is reached.

    Always writes the static model JSON first. Then snapshots whatever tool
    tags are visible (reference tag required) together with the Redis
    Cartesian pose and writes the observation JSON.
    """
    try:
        from helpers.model_tool_station import PaintToolStationModel
    except ImportError as exc:  # pragma: no cover - defensive
        print("Could not import model_tool_station:", exc)
        return 1

    model = PaintToolStationModel()
    saved_model_path = model.save_json(model_json_path)
    print(f"Saved tool-station model JSON: {saved_model_path}")

    snapshot = capture_tool_tag_observation(
        cap=cap,
        detector_bundle=detector_bundle,
        redis_client=redis_client,
        intrinsics_json=intrinsics_json,
        num_frames=snapshot_frames,
    )
    if snapshot is None:
        print(
            "Observation JSON was NOT written: no usable tool tag was visible"
            " at TOOL_INIT. Model JSON was still saved."
        )
        return 1

    observation_payload = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "robot_name": ROBOT_NAME,
        "tag_family": TAG_FAMILY,
        "tag_size_m": TAG_SIZE_M,
        "tool_tag_ids": list(TOOL_TAG_IDS),
        "reference_tag_id": snapshot.reference_tag_id,
        "visible_tag_ids": snapshot.visible_tag_ids,
        "snapshot_num_frames": snapshot.num_frames,
        "note": (
            "ee_position_world_m and ee_orientation_world are the Redis "
            "Cartesian pose averaged over the same frames as the tag scan. "
            "reference_tag_id is the detected tag used to anchor the station "
            "frame (prefer 10, else first visible in 10/11/12/13 order)."
        ),
        "ee_position_world_m": [float(v) for v in snapshot.ee_position_world_m],
        "ee_orientation_world": [
            [float(v) for v in row] for row in snapshot.ee_orientation_world
        ],
        "camera_offset_in_tip_frame_m": [-0.07560, 0.00000, 0.04211],
        "R_camera_to_tip": [
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        "tag_poses_in_camera_frame": {
            str(tag_id): pose
            for tag_id, pose in snapshot.tag_poses_in_camera_frame.items()
        },
        "model_json_path": str(saved_model_path),
    }

    observation_json_path.parent.mkdir(parents=True, exist_ok=True)
    with observation_json_path.open("w", encoding="utf-8") as f:
        json.dump(observation_payload, f, indent=2)
    print(f"Saved tool-station observation JSON: {observation_json_path}")
    print(f"Reference tag id: {snapshot.reference_tag_id}")
    return 0


def follow_joint_path_to_goal(
    redis_client,
    *,
    start_joint_position: np.ndarray,
    goal_joint_position: np.ndarray,
    label: str,
    max_joint_step: float,
    joint_arrival_threshold: float,
    dwell_s: float = 0.0,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    path_trace: list[PathTraceSample] | None = None,
    trace_start: float | None = None,
    trace_phase: str = "TOOL_JOINT_PATH",
    settle_phase: str = "TOOL_SETTLE",
) -> bool:
    path, _waypoint_indices = build_smooth_joint_path(
        [start_joint_position, goal_joint_position],
        max_step=max_joint_step,
    )

    print(f"Moving to {label}...")
    print(f"{label} target (deg):", np.round(goal_joint_position / DEG_TO_RAD, 3))
    print("Smooth joint path samples:", len(path))

    loop_time = 0.0
    last_status = 0.0
    start = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9
    path_index = 1

    while path_index < len(path):
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        commanded_joint_position = path[path_index]
        current_joint_position = read_np(
            redis_client,
            redis_keys.sensor_joint_positions,
            (7,),
        )
        if path_trace is not None and trace_start is not None:
            append_path_trace(
                path_trace,
                trace_start=trace_start,
                phase=trace_phase,
                position=read_optional_np(
                    redis_client,
                    redis_keys.cartesian_task_current_position,
                    (3,),
                ),
            )

        joint_error = float(np.linalg.norm(goal_joint_position - current_joint_position))
        commanded_error = float(
            np.linalg.norm(goal_joint_position - commanded_joint_position)
        )
        set_joint_goal(redis_client, commanded_joint_position)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                f"FOLLOWING_{label}",
                "|",
                f"path_sample {path_index + 1}/{len(path)}",
                "| joint_error:",
                round(joint_error, 5),
                "| commanded_remaining:",
                round(commanded_error, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(f"Timed out while streaming joint path to {label}.")
            print("Final joint error:", round(joint_error, 5))
            return False

        path_index += 1

    set_joint_goal(redis_client, goal_joint_position)
    while True:
        current_joint_position = read_np(
            redis_client,
            redis_keys.sensor_joint_positions,
            (7,),
        )
        if path_trace is not None and trace_start is not None:
            append_path_trace(
                path_trace,
                trace_start=trace_start,
                phase=settle_phase,
                position=read_optional_np(
                    redis_client,
                    redis_keys.cartesian_task_current_position,
                    (3,),
                ),
            )
        joint_error = float(np.linalg.norm(goal_joint_position - current_joint_position))
        if joint_error < joint_arrival_threshold:
            print(f"Reached {label}.")
            if dwell_s > 0.0:
                time.sleep(dwell_s)
            return True

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(f"Timed out before settling at {label}.")
            print("Final joint error:", round(joint_error, 5))
            return False

        set_joint_goal(redis_client, goal_joint_position)
        time.sleep(DT)


def make_cartesian_move_fn(
    *,
    pos_tol_m: float,
    path_trace: list[PathTraceSample] | None = None,
    trace_start: float | None = None,
):
    def move_fn(
        redis_client,
        *,
        target_pos: np.ndarray,
        hold_orientation: np.ndarray,
        label: str,
        dwell_s: float,
        timeout_s: float,
        status_period_s: float,
    ) -> bool:
        start = time.perf_counter()
        ok = move_cartesian_to_target(
            redis_client,
            target_position=target_pos,
            hold_orientation=hold_orientation,
            label=label,
            pos_tol_m=pos_tol_m,
            status_period_s=status_period_s,
            timeout_s=timeout_s,
            start_time=start,
            path_trace=path_trace,
            trace_start=trace_start,
        )
        if ok and dwell_s > 0.0:
            time.sleep(dwell_s)
        return ok

    return move_fn


def run_paint_dip_sequence(
    redis_client,
    *,
    pos_tol_m: float = POS_TOL_M,
    hover_dwell_s: float = DWELL_AT_PAINT_HOVER_S,
    dip_dwell_s: float = DWELL_AT_PAINT_DIP_S,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    path_trace: list[PathTraceSample] | None = None,
    trace_start: float | None = None,
) -> bool:
    cfg = load_demo_day_config()
    hold_orientation = read_np(
        redis_client,
        redis_keys.cartesian_task_current_orientation,
        (3, 3),
    )
    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )

    print("Switching to Cartesian controller for paint-station visits.")
    print("Holding current Cartesian orientation throughout.")
    print("Current Cartesian position:", np.round(current_position, 5))
    print(
        "z_dive_in_offset (mm):",
        round(load_z_dive_in_offset_m(cfg) * 1000.0, 3),
    )

    set_cartesian_goal(redis_client, current_position, hold_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < CARTESIAN_SETTLE_S:
        set_cartesian_goal(redis_client, current_position, hold_orientation)
        time.sleep(DT)

    move_fn = make_cartesian_move_fn(
        pos_tol_m=pos_tol_m,
        path_trace=path_trace,
        trace_start=trace_start,
    )
    sequence_start = time.perf_counter()

    paint_legs: list[tuple[str, np.ndarray, Callable[..., bool]]] = [
        ("P1_INIT_POS", load_paint_init_pos_m(cfg, "P1_INIT_POS"), dip_paint_1),
        ("P2_INIT_POS", load_paint_init_pos_m(cfg, "P2_INIT_POS"), dip_paint_2),
        ("P3_INIT_POS", load_paint_init_pos_m(cfg, "P3_INIT_POS"), dip_paint_3),
    ]

    print("\nPaint-station Cartesian sequence:")
    for label, target_pos, _dip_fn in paint_legs:
        print(f"  {label}: {np.round(target_pos, 5).tolist()} m")

    for label, target_pos, dip_fn in paint_legs:
        if not move_cartesian_to_target(
            redis_client,
            target_position=target_pos,
            hold_orientation=hold_orientation,
            label=label,
            pos_tol_m=pos_tol_m,
            status_period_s=status_period_s,
            timeout_s=timeout_s,
            start_time=sequence_start,
            path_trace=path_trace,
            trace_start=trace_start,
        ):
            return False

        print(f"Reached {label}.")
        if hover_dwell_s > 0.0:
            time.sleep(hover_dwell_s)

        if not dip_fn(
            redis_client,
            hold_orientation=hold_orientation,
            config=cfg,
            dwell_at_dip_s=dip_dwell_s,
            dwell_at_hover_s=hover_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
            move_fn=move_fn,
        ):
            return False

    water_pos = load_paint_init_pos_m(cfg, "WATER_INIT_POS")
    if not move_cartesian_to_target(
        redis_client,
        target_position=water_pos,
        hold_orientation=hold_orientation,
        label="WATER_INIT_POS",
        pos_tol_m=pos_tol_m,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        start_time=sequence_start,
        path_trace=path_trace,
        trace_start=trace_start,
    ):
        return False

    print("Reached WATER_INIT_POS.")
    if hover_dwell_s > 0.0:
        time.sleep(hover_dwell_s)

    if not swirl_water(
        redis_client,
        hold_orientation=hold_orientation,
        config=cfg,
        dwell_at_dip_s=dip_dwell_s,
        dwell_at_hover_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    ):
        return False

    napkin_approach_pos = load_paint_init_pos_m(cfg, "WATER_TO_NAPKIN_INIT_POS")
    if not move_cartesian_to_target(
        redis_client,
        target_position=napkin_approach_pos,
        hold_orientation=hold_orientation,
        label="WATER_TO_NAPKIN_INIT_POS",
        pos_tol_m=pos_tol_m,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        start_time=sequence_start,
        path_trace=path_trace,
        trace_start=trace_start,
    ):
        return False

    print("Reached WATER_TO_NAPKIN_INIT_POS.")
    if hover_dwell_s > 0.0:
        time.sleep(hover_dwell_s)

    if not wipe_napkin(
        redis_client,
        hold_orientation=hold_orientation,
        config=cfg,
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    ):
        return False

    print("\nFinished paint-station dip sequence.")
    return True


def move_to_tool_init(
    *,
    config_file_name_expected: str = CONFIG_FILE_FOR_THIS_SCRIPT,
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    joint_controller_settle_s: float = JOINT_CONTROLLER_SETTLE_S,
    dwell_s: float = DWELL_AT_TOOL_INIT_S,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    save_path_plot: bool = True,
    path_log_dir: Path = DEFAULT_PATH_LOG_DIR,
    use_cached_tool_init_path: bool = True,
    rebuild_tool_init_path: bool = False,
) -> int:
    path_trace: list[PathTraceSample] = []
    trace_start = time.perf_counter()

    def finish(result: int) -> int:
        if save_path_plot:
            save_path_visualization(path_trace, path_log_dir)
        return result

    if redis is None:
        print("`redis` package is not installed.")
        return finish(1)

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client, config_file_name_expected):
        return finish(1)

    approach = approach_tool_init(
        redis_client,
        joint_arrival_threshold=joint_arrival_threshold,
        joint_max_step_deg=joint_max_step_deg,
        joint_controller_settle_s=joint_controller_settle_s,
        dwell_at_tool_init_s=dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        use_cached_path=use_cached_tool_init_path,
        rebuild_path=rebuild_tool_init_path,
    )
    if approach is None:
        return finish(1)

    if not run_paint_dip_sequence(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        path_trace=path_trace,
        trace_start=trace_start,
    ):
        return finish(1)
    return finish(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-log-dir",
        default=str(DEFAULT_PATH_LOG_DIR),
        help=f"Directory for path PNG/CSV logs (default: {DEFAULT_PATH_LOG_DIR}).",
    )
    parser.add_argument(
        "--no-path-plot",
        action="store_true",
        help="Do not write the measured Cartesian path matplotlib PNG/CSV.",
    )
    parser.add_argument(
        "--config-file",
        default=CONFIG_FILE_FOR_THIS_SCRIPT,
        help=f"Expected Sai config file (default: {CONFIG_FILE_FOR_THIS_SCRIPT}).",
    )
    parser.add_argument(
        "--joint-arrival-threshold",
        type=float,
        default=JOINT_ARRIVAL_THRESHOLD,
        help=f"Joint-space arrival threshold in radians (default: {JOINT_ARRIVAL_THRESHOLD}).",
    )
    parser.add_argument(
        "--joint-max-step-deg",
        type=float,
        default=JOINT_MAX_STEP_DEG,
        help=f"Maximum joint-command step per loop in degrees (default: {JOINT_MAX_STEP_DEG}).",
    )
    parser.add_argument(
        "--joint-controller-settle-s",
        type=float,
        default=JOINT_CONTROLLER_SETTLE_S,
        help=(
            "Seconds to hold measured joints after switching to joint_controller "
            f"(default: {JOINT_CONTROLLER_SETTLE_S})."
        ),
    )
    parser.add_argument(
        "--dwell-s",
        type=float,
        default=DWELL_AT_TOOL_INIT_S,
        help=f"Seconds to hold after arrival (default: {DWELL_AT_TOOL_INIT_S}).",
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
        default=TIMEOUT_S,
        help=f"Maximum seconds to wait for arrival. Use 0 to disable (default: {TIMEOUT_S}).",
    )
    parser.add_argument(
        "--rebuild-tool-init-path",
        action="store_true",
        help=(
            "Rebuild and save demo-day/tool_init_joint_path.json instead of "
            "using the cached joint path."
        ),
    )
    parser.add_argument(
        "--no-cached-tool-init-path",
        action="store_true",
        help="Do not load the cached INIT_POS -> TOOL_INIT joint path JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return move_to_tool_init(
            config_file_name_expected=args.config_file,
            joint_arrival_threshold=args.joint_arrival_threshold,
            joint_max_step_deg=args.joint_max_step_deg,
            joint_controller_settle_s=args.joint_controller_settle_s,
            dwell_s=args.dwell_s,
            status_period_s=args.status_period_s,
            timeout_s=args.timeout_s,
            save_path_plot=not args.no_path_plot,
            path_log_dir=Path(args.path_log_dir),
            use_cached_tool_init_path=not args.no_cached_tool_init_path,
            rebuild_tool_init_path=args.rebuild_tool_init_path,
        )
    except RuntimeError as exc:
        print("Could not start tool-station run:")
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
