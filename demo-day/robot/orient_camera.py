"""Open the camera stream, translate to CAMERA_INIT, then rotate the last joint."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np


DEFAULT_CAMERA_INDEX = 0
PREVIEW_WINDOW = "Camera Preview"
DEG_TO_RAD = math.pi / 180.0

ROBOT_NAME = "Titania"
CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"
CARTESIAN_CONTROLLER = "cartesian_controller"
JOINT_CONTROLLER = "joint_controller"

DT = 0.01
POS_TOL_M = 1.0e-2
DWELL_AFTER_TRANSLATION_S = 0.25
ARC_STEP_M = 0.02
ARC_SAGITTA_M = 0.12
LAST_JOINT_TARGET_DEG = 90.0
JOINT_ARRIVAL_THRESHOLD = 8.0e-2
JOINT_MAX_STEP_DEG = 0.5
JOINT_CONTROLLER_SETTLE_S = 0.25

CAMERA_INIT_POS_MM = np.array([-310.22, 592.11, 340.43], dtype=float)
CAMERA_INIT_POS_M = CAMERA_INIT_POS_MM / 1000.0


class CameraState(Enum):
    TRANSLATING = auto()
    ROTATING_LAST_JOINT = auto()
    DONE = auto()


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


def set_cartesian_goal(redis_client, position: np.ndarray, orientation: np.ndarray) -> None:
    redis_client.set(redis_keys.cartesian_task_goal_position, json.dumps(position.tolist()))
    redis_client.set(redis_keys.cartesian_task_goal_orientation, json.dumps(orientation.tolist()))


def set_joint_goal(redis_client, position: np.ndarray) -> None:
    redis_client.set(redis_keys.joint_task_goal_position, json.dumps(position.tolist()))
    redis_client.set(redis_keys.joint_task_goal_velocity, json.dumps(np.zeros_like(position).tolist()))
    redis_client.set(
        redis_keys.joint_task_goal_acceleration,
        json.dumps(np.zeros_like(position).tolist()),
    )


def position_error(current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
    return float(np.linalg.norm(goal_pos - current_pos))


def orientation_error(current_ori: np.ndarray, goal_ori: np.ndarray) -> float:
    return float(np.linalg.norm(goal_ori - current_ori))


def min_xy_radius(points: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points[:, :2], axis=1)))


def build_xy_arc_path(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    sagitta_m: float,
    step_m: float,
    side: str,
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
    center_offset = math.sqrt(max(radius**2 - half_chord**2, 0.0))

    unit_chord = chord / chord_len
    left_normal = np.array([-unit_chord[1], unit_chord[0]], dtype=float)
    bulge_sign = 1.0 if side == "left" else -1.0
    bulge_normal = bulge_sign * left_normal

    # Put the circle center opposite the intended bulge so the minor arc bows
    # toward ``bulge_normal``.
    center = (start_xy + target_xy) / 2.0 - bulge_normal * center_offset

    start_angle = math.atan2(start_xy[1] - center[1], start_xy[0] - center[0])
    target_angle = math.atan2(target_xy[1] - center[1], target_xy[0] - center[0])
    angle_delta = (target_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi

    arc_length = abs(angle_delta) * radius
    num_segments = max(1, int(math.ceil(arc_length / step_m)))

    waypoints = []
    for index in range(1, num_segments + 1):
        fraction = index / num_segments
        theta = start_angle + angle_delta * fraction
        xy = center + radius * np.array([math.cos(theta), math.sin(theta)], dtype=float)
        z = start_pos[2] + (target_pos[2] - start_pos[2]) * fraction
        waypoints.append(np.array([xy[0], xy[1], z], dtype=float))

    waypoints[-1] = np.array(target_pos, dtype=float)
    return np.array(waypoints, dtype=float), radius


def build_translation_path(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    sagitta_m: float,
    step_m: float,
    arc_side: str,
) -> tuple[np.ndarray, float, str]:
    if arc_side in ("left", "right"):
        path, radius = build_xy_arc_path(
            start_pos,
            target_pos,
            sagitta_m=sagitta_m,
            step_m=step_m,
            side=arc_side,
        )
        return path, radius, arc_side

    left_path, left_radius = build_xy_arc_path(
        start_pos,
        target_pos,
        sagitta_m=sagitta_m,
        step_m=step_m,
        side="left",
    )
    right_path, right_radius = build_xy_arc_path(
        start_pos,
        target_pos,
        sagitta_m=sagitta_m,
        step_m=step_m,
        side="right",
    )

    if min_xy_radius(left_path) >= min_xy_radius(right_path):
        return left_path, left_radius, "left"
    return right_path, right_radius, "right"


def open_camera(camera_index: int) -> cv2.VideoCapture:
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


def update_camera_stream(cap: cv2.VideoCapture, *, preview: bool, label: str) -> bool:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from camera stream.")
        return False

    if not preview:
        return True

    cv2.putText(
        frame,
        label,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imshow(PREVIEW_WINDOW, frame)

    key = cv2.waitKey(1) & 0xFF
    return key not in (27, ord("q"))


def preview_camera_stream(camera_index: int = DEFAULT_CAMERA_INDEX) -> bool:
    cap: cv2.VideoCapture | None = None

    try:
        cap = open_camera(camera_index)
        print("Press q or Esc in the preview window to stop.")

        while update_camera_stream(cap, preview=True, label="Camera preview"):
            time.sleep(0.001)

        return True

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        return False

    except Exception as exc:
        print("Exception occurred:")
        print(exc)
        return False

    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def move_to_camera_init(
    *,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    preview: bool = True,
    config_file_name_expected: str = CONFIG_FILE_FOR_THIS_SCRIPT,
    arc_sagitta_m: float = ARC_SAGITTA_M,
    arc_step_m: float = ARC_STEP_M,
    arc_side: str = "auto",
    last_joint_target_deg: float = LAST_JOINT_TARGET_DEG,
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    joint_controller_settle_s: float = JOINT_CONTROLLER_SETTLE_S,
) -> bool:
    cap: cv2.VideoCapture | None = None

    try:
        cap = open_camera(camera_index)

        import redis

        redis_client = redis.Redis()
        if not ensure_robot_ready(redis_client, config_file_name_expected):
            return False

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

        print("Current position:", current_position)
        print("Hold orientation:", hold_orientation)
        print("CAMERA_INIT position (m):", CAMERA_INIT_POS_M)
        print("Last joint target (deg):", last_joint_target_deg)
        print("Joint max step (deg):", joint_max_step_deg)

        translation_path, arc_radius, chosen_arc_side = build_translation_path(
            current_position,
            CAMERA_INIT_POS_M,
            sagitta_m=arc_sagitta_m,
            step_m=arc_step_m,
            arc_side=arc_side,
        )
        path_index = 0
        translation_target = translation_path[path_index]

        print("Translation waypoints:", len(translation_path))
        print("XY arc radius (m):", arc_radius)
        print("XY arc side:", chosen_arc_side)
        print("Minimum waypoint XY radius (m):", min_xy_radius(translation_path))

        set_cartesian_goal(redis_client, current_position, hold_orientation)
        set_active_controller(redis_client, CARTESIAN_CONTROLLER)
        print("Using controller:", CARTESIAN_CONTROLLER)

        state = CameraState.TRANSLATING
        set_cartesian_goal(redis_client, translation_target, hold_orientation)
        print("Phase 1: following XY arc to CAMERA_INIT while holding current orientation.")

        joint_goal = None
        commanded_joint_position = None
        max_joint_step = max(abs(joint_max_step_deg) * DEG_TO_RAD, 1.0e-5)

        loop_time = 0.0
        time.sleep(0.01)
        init_time = time.perf_counter_ns() * 1e-9

        while True:
            loop_time += DT
            time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

            if not update_camera_stream(cap, preview=preview, label=state.name):
                print("Camera stream stopped. Exiting.")
                return False

            if state == CameraState.TRANSLATING:
                current_position = read_np(
                    redis_client,
                    redis_keys.cartesian_task_current_position,
                    (3,),
                )
                current_orientation = read_np(
                    redis_client,
                    redis_keys.cartesian_task_current_orientation,
                    (3, 3),
                )
                translation_target = translation_path[path_index]
                pos_err = position_error(current_position, translation_target)
                hold_ori_err = orientation_error(current_orientation, hold_orientation)
                set_cartesian_goal(redis_client, translation_target, hold_orientation)
                print(
                    "TRANSLATING",
                    path_index + 1,
                    "/",
                    len(translation_path),
                    "| pos_error:",
                    round(pos_err, 5),
                    "| hold_ori_error:",
                    round(hold_ori_err, 5),
                )

                if pos_err < POS_TOL_M:
                    path_index += 1
                    if path_index >= len(translation_path):
                        time.sleep(DWELL_AFTER_TRANSLATION_S)
                        current_joint_position = read_np(
                            redis_client,
                            redis_keys.sensor_joint_positions,
                            (7,),
                        )
                        joint_goal = current_joint_position.copy()
                        joint_goal[-1] = last_joint_target_deg * DEG_TO_RAD
                        commanded_joint_position = current_joint_position.copy()

                        joint_names = read_optional_json(redis_client, redis_keys.joint_names)
                        if joint_names is not None:
                            print("Joint names:", joint_names)

                        set_joint_goal(redis_client, current_joint_position)
                        set_active_controller(redis_client, JOINT_CONTROLLER)
                        print("Using controller:", JOINT_CONTROLLER)
                        settle_start = time.perf_counter()
                        while time.perf_counter() - settle_start < joint_controller_settle_s:
                            set_joint_goal(redis_client, current_joint_position)
                            time.sleep(DT)

                        state = CameraState.ROTATING_LAST_JOINT
                        set_joint_goal(redis_client, commanded_joint_position)
                        print("Phase 2: position reached. Moving only the last joint.")
                        print("Current joint position:", current_joint_position)
                        print("Target joint position:", joint_goal)
                        print(
                            "Commanded joint delta (deg):",
                            ((joint_goal - current_joint_position) / DEG_TO_RAD).round(3),
                        )
                    else:
                        translation_target = translation_path[path_index]
                        set_cartesian_goal(redis_client, translation_target, hold_orientation)

            elif state == CameraState.ROTATING_LAST_JOINT:
                current_joint_position = read_np(
                    redis_client,
                    redis_keys.sensor_joint_positions,
                    (7,),
                )
                delta = joint_goal - commanded_joint_position
                distance = float(np.linalg.norm(delta))
                if distance > max_joint_step:
                    delta *= max_joint_step / distance
                commanded_joint_position = commanded_joint_position + delta

                joint_error = float(np.linalg.norm(joint_goal - current_joint_position))
                last_joint_error = abs(float(joint_goal[-1] - current_joint_position[-1]))
                commanded_error = float(np.linalg.norm(joint_goal - commanded_joint_position))
                set_joint_goal(redis_client, commanded_joint_position)
                print(
                    "ROTATING_LAST_JOINT",
                    "| joint_error:",
                    round(joint_error, 5),
                    "| last_joint_error:",
                    round(last_joint_error, 5),
                    "| commanded_remaining:",
                    round(commanded_error, 5),
                )

                if joint_error < joint_arrival_threshold:
                    state = CameraState.DONE
                    print("Reached target last-joint position.")
                    return True

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        return False

    except Exception as exc:
        print("Exception occurred:")
        print(exc)
        return False

    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the camera stream and move the robot to CAMERA_INIT in two phases."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"OpenCV camera index (default: {DEFAULT_CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Read the camera stream without showing an OpenCV preview window.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only show the camera stream; do not connect to Redis or command the robot.",
    )
    parser.add_argument(
        "--config-file",
        default=CONFIG_FILE_FOR_THIS_SCRIPT,
        help=f"Expected Sai config file (default: {CONFIG_FILE_FOR_THIS_SCRIPT}).",
    )
    parser.add_argument(
        "--arc-sagitta-m",
        type=float,
        default=ARC_SAGITTA_M,
        help=f"XY arc sagitta in meters (default: {ARC_SAGITTA_M}).",
    )
    parser.add_argument(
        "--arc-step-m",
        type=float,
        default=ARC_STEP_M,
        help=f"Approximate translation waypoint spacing in meters (default: {ARC_STEP_M}).",
    )
    parser.add_argument(
        "--arc-side",
        choices=("auto", "left", "right"),
        default="auto",
        help="XY arc side from current position to target (default: auto).",
    )
    parser.add_argument(
        "--last-joint-target-deg",
        type=float,
        default=LAST_JOINT_TARGET_DEG,
        help=f"Absolute target angle for the last joint in degrees (default: {LAST_JOINT_TARGET_DEG}).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview_only:
        ok = preview_camera_stream(camera_index=args.camera_index)
        if ok is False:
            sys.exit(1)
        return

    ok = move_to_camera_init(
        camera_index=args.camera_index,
        preview=not args.no_preview,
        config_file_name_expected=args.config_file,
        arc_sagitta_m=args.arc_sagitta_m,
        arc_step_m=args.arc_step_m,
        arc_side=args.arc_side,
        last_joint_target_deg=args.last_joint_target_deg,
        joint_arrival_threshold=args.joint_arrival_threshold,
        joint_max_step_deg=args.joint_max_step_deg,
        joint_controller_settle_s=args.joint_controller_settle_s,
    )
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
