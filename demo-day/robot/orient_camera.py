"""Open the camera stream, move to INIT_POS, move in joint space to the camera
pose, then hold still until a face is detected and stationary, snap a picture,
and return to the INIT_POS joint configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_CAPTURES_DIR = DEMO_DAY_DIR / "raw-captures"

# Wait-for-stationary-face capture parameters.
# The face's bbox center must stay within STATIONARY_RADIUS_PX of an anchor
# point for STATIONARY_REQUIRED_S seconds before a frame is captured.
STATIONARY_REQUIRED_S = 2.0
STATIONARY_RADIUS_PX = 80
POST_CAPTURE_DISPLAY_S = 1.5
FLASH_PERIOD_S = 0.25

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
JOINT_ARRIVAL_THRESHOLD = 0.20
JOINT_MAX_STEP_DEG = 0.5
JOINT_CONTROLLER_SETTLE_S = 0.25

# Face detection parameters.
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (60, 60)

CAMERA_INIT_POS_MM = np.array([-310.22, 592.11, 340.43], dtype=float)
CAMERA_INIT_POS_M = CAMERA_INIT_POS_MM / 1000.0
CAMERA_INIT_JOINT_CONFIG_DEG = np.array(
    [-70.63, -32.43, -27.25, 99.52, 17.86, 38.24, 74.29],
    dtype=float,
)
CAMERA_INIT_JOINT_CONFIG_RAD = CAMERA_INIT_JOINT_CONFIG_DEG * DEG_TO_RAD
INIT_POS_JOINT_CONFIG_DEG = np.array(
    [0.42, -27.42, -0.50, 116.48, 0.41, 53.89, -0.34],
    dtype=float,
)
INIT_POS_JOINT_CONFIG_RAD = INIT_POS_JOINT_CONFIG_DEG * DEG_TO_RAD

# Robot "home" pose to return to after the centered capture is taken. Sourced
# from demo-day/config.json so all demo-day scripts share one source of truth.
_DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"


def _load_init_pos() -> np.ndarray:
    with _DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return np.array(cfg["init_pos_m"], dtype=float)


INIT_POS = _load_init_pos()


class CameraState(Enum):
    TRANSLATING = auto()
    MOVING_TO_CAMERA_JOINTS = auto()
    WAITING_FOR_FACE = auto()
    RETURNING_TO_INIT_JOINTS = auto()
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


def make_raw_capture_name() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S") + ".jpg"


def save_raw_capture(frame: np.ndarray, captures_dir: Path) -> Path:
    captures_dir.mkdir(parents=True, exist_ok=True)
    out_path = captures_dir / make_raw_capture_name()
    suffix = 1
    while out_path.exists():
        out_path = captures_dir / (out_path.stem + f"_{suffix}.jpg")
        suffix += 1
    cv2.imwrite(str(out_path), frame)
    return out_path


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

    return detector


def detect_closest_face(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=FACE_DETECTION_SCALE_FACTOR,
        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
        minSize=FACE_DETECTION_MIN_SIZE,
    )

    if len(faces) == 0:
        return None

    # Monocular closest-face approximation:
    # larger detected face bounding box area usually means closer to camera.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


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


def update_camera_stream(
    cap: cv2.VideoCapture,
    *,
    preview: bool,
    label: str,
) -> tuple[bool, np.ndarray | None]:
    ok, frame = cap.read()
    if not ok:
        print("Could not read from camera stream.")
        return False, None

    if not preview:
        return True, frame

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
    return key not in (27, ord("q")), frame


def draw_face_bbox(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int] | None,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> None:
    if face_box is None:
        return
    x, y, w, h = face_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def draw_status_text(
    frame: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    scale: float = 0.9,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
    )
    cv2.putText(
        frame,
        text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    *,
    color: tuple[int, int, int] = (0, 220, 0),
    scale: float = 6.0,
    thickness: int = 14,
) -> None:
    height, width = frame.shape[:2]
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    org = ((width - text_w) // 2, (height + text_h) // 2)
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 8,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
    )


def preview_camera_stream(camera_index: int = DEFAULT_CAMERA_INDEX) -> bool:
    cap: cv2.VideoCapture | None = None

    try:
        cap = open_camera(camera_index)
        print("Press q or Esc in the preview window to stop.")

        while True:
            keep_running, _frame = update_camera_stream(
                cap,
                preview=True,
                label="Camera preview",
            )
            if not keep_running:
                break
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
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    joint_controller_settle_s: float = JOINT_CONTROLLER_SETTLE_S,
    capture_when_centered: bool = True,
    raw_captures_dir: str | os.PathLike[str] = DEFAULT_RAW_CAPTURES_DIR,
) -> bool:
    cap: cv2.VideoCapture | None = None
    raw_captures_dir = Path(raw_captures_dir)

    try:
        cap = open_camera(camera_index)
        face_detector = create_face_detector()

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
        print("INIT_POS position (m):", INIT_POS)
        print("CAMERA_INIT Cartesian reference (m):", CAMERA_INIT_POS_M)
        print("Camera joint target (deg):", CAMERA_INIT_JOINT_CONFIG_DEG)
        print("Return INIT_POS joint target (deg):", INIT_POS_JOINT_CONFIG_DEG)
        print("Joint max step (deg):", joint_max_step_deg)

        translation_path = np.array([INIT_POS], dtype=float)
        path_index = 0
        translation_target = translation_path[path_index]

        print("Translation waypoints:", len(translation_path))
        print("Translation phase target: INIT_POS only")

        set_cartesian_goal(redis_client, current_position, hold_orientation)
        set_active_controller(redis_client, CARTESIAN_CONTROLLER)
        print("Using controller:", CARTESIAN_CONTROLLER)

        state = CameraState.TRANSLATING
        set_cartesian_goal(redis_client, translation_target, hold_orientation)
        print("Phase 1: translating to INIT_POS while holding current orientation.")

        joint_goal = None
        commanded_joint_position = None

        # Wait-for-stationary-face capture state.
        face_anchor_center: tuple[float, float] | None = None
        stationary_since: float | None = None
        captured_at: float | None = None
        saved_raw_capture_path: Path | None = None

        max_joint_step = max(abs(joint_max_step_deg) * DEG_TO_RAD, 1.0e-5)

        loop_time = 0.0
        time.sleep(0.01)
        init_time = time.perf_counter_ns() * 1e-9

        while True:
            loop_time += DT
            time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

            if state != CameraState.WAITING_FOR_FACE:
                keep_running, _frame = update_camera_stream(
                    cap,
                    preview=preview,
                    label=state.name,
                )
                if not keep_running:
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
                        joint_goal = CAMERA_INIT_JOINT_CONFIG_RAD.copy()
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

                        state = CameraState.MOVING_TO_CAMERA_JOINTS
                        set_joint_goal(redis_client, commanded_joint_position)
                        print("Phase 2: INIT_POS reached. Moving to camera joint configuration.")
                        print("Current joint position:", current_joint_position)
                        print("Target joint position:", joint_goal)
                        print(
                            "Commanded joint delta (deg):",
                            ((joint_goal - current_joint_position) / DEG_TO_RAD).round(3),
                        )
                    else:
                        translation_target = translation_path[path_index]
                        set_cartesian_goal(redis_client, translation_target, hold_orientation)

            elif state == CameraState.MOVING_TO_CAMERA_JOINTS:
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
                commanded_error = float(np.linalg.norm(joint_goal - commanded_joint_position))
                set_joint_goal(redis_client, commanded_joint_position)
                print(
                    "MOVING_TO_CAMERA_JOINTS",
                    "| joint_error:",
                    round(joint_error, 5),
                    "| commanded_remaining:",
                    round(commanded_error, 5),
                )

                if joint_error < joint_arrival_threshold:
                    state = CameraState.WAITING_FOR_FACE
                    # Hold the camera joint target until we begin the return sequence.
                    set_joint_goal(redis_client, joint_goal)
                    face_anchor_center = None
                    stationary_since = None
                    captured_at = None
                    print("Reached camera joint configuration.")
                    print(
                        "Phase 3: waiting for stationary face."
                        " Hold still for"
                        f" {STATIONARY_REQUIRED_S:.1f}s to trigger capture."
                        " Press q or Esc in preview window to stop."
                    )

            elif state == CameraState.WAITING_FOR_FACE:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read from camera stream.")
                    return False

                clean_frame = frame.copy()
                face_box = detect_closest_face(frame, face_detector)
                now = time.perf_counter()

                # Hold the robot stationary: no joint commanding while we wait.
                # Post-capture: display "Photo taken!" for a moment, then begin
                # the reverse motion sequence.
                if captured_at is not None:
                    if preview:
                        draw_status_text(
                            frame,
                            "WAITING_FOR_FACE",
                            color=(0, 255, 0),
                        )
                        if face_box is not None:
                            draw_face_bbox(
                                frame, face_box, color=(0, 220, 0), thickness=4
                            )
                        draw_centered_text(
                            frame,
                            "Photo taken!",
                            color=(0, 220, 0),
                            scale=2.5,
                            thickness=6,
                        )
                        cv2.imshow(PREVIEW_WINDOW, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            return False

                    if now - captured_at < POST_CAPTURE_DISPLAY_S:
                        continue

                    # Begin return sequence: move directly to the INIT_POS joint
                    # configuration.
                    current_joint_position = read_np(
                        redis_client,
                        redis_keys.sensor_joint_positions,
                        (7,),
                    )
                    joint_goal = INIT_POS_JOINT_CONFIG_RAD.copy()
                    commanded_joint_position = current_joint_position.copy()
                    set_joint_goal(redis_client, commanded_joint_position)
                    state = CameraState.RETURNING_TO_INIT_JOINTS
                    print("Phase 4: returning to INIT_POS joint configuration.")
                    print("Return joint target:", joint_goal)
                    print(
                        "Commanded joint delta (deg):",
                        ((joint_goal - current_joint_position) / DEG_TO_RAD).round(3),
                    )
                    continue

                # No face: reset stationary tracking and show a "looking" hint.
                if face_box is None:
                    face_anchor_center = None
                    stationary_since = None
                    if preview:
                        draw_status_text(
                            frame,
                            "Looking for face...",
                            color=(0, 200, 255),
                        )
                        cv2.imshow(PREVIEW_WINDOW, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            print("Capture stopped by user.")
                            return False
                    else:
                        print("WAITING_FOR_FACE | no face detected")
                    continue

                # Face detected: track stationarity via a fixed anchor center.
                x, y, w, h = face_box
                face_center = (x + w / 2.0, y + h / 2.0)

                if face_anchor_center is None or stationary_since is None:
                    face_anchor_center = face_center
                    stationary_since = now
                else:
                    dx = face_center[0] - face_anchor_center[0]
                    dy = face_center[1] - face_anchor_center[1]
                    if math.hypot(dx, dy) > STATIONARY_RADIUS_PX:
                        face_anchor_center = face_center
                        stationary_since = now

                stationary_elapsed = now - stationary_since
                remaining = STATIONARY_REQUIRED_S - stationary_elapsed

                # Stationary long enough: capture the un-annotated frame and
                # transition into the post-capture display phase.
                if remaining <= 0.0 and capture_when_centered and saved_raw_capture_path is None:
                    saved_raw_capture_path = save_raw_capture(
                        clean_frame, raw_captures_dir
                    )
                    captured_at = now
                    print(
                        f"Saved centered raw capture: {saved_raw_capture_path}"
                    )
                    continue

                print(
                    "WAITING_FOR_FACE",
                    "| face_center:",
                    (round(face_center[0], 1), round(face_center[1], 1)),
                    "| stationary_elapsed:",
                    round(stationary_elapsed, 2),
                    "/",
                    STATIONARY_REQUIRED_S,
                )

                if preview:
                    # Flashing green bbox while a face is held in view.
                    flash_on = int(now / FLASH_PERIOD_S) % 2 == 0
                    bbox_color = (0, 255, 0) if flash_on else (0, 160, 0)
                    draw_face_bbox(frame, face_box, color=bbox_color, thickness=4)
                    draw_status_text(
                        frame,
                        "Hold still...",
                        color=(0, 220, 0),
                    )

                    if remaining > 0.0:
                        countdown_num = max(1, int(math.ceil(remaining)))
                        draw_centered_text(
                            frame,
                            str(countdown_num),
                            color=(0, 220, 0),
                            scale=6.0,
                            thickness=14,
                        )

                    cv2.imshow(PREVIEW_WINDOW, frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (27, ord("q")):
                        print("Capture stopped by user.")
                        state = CameraState.DONE
                        return True

            elif state == CameraState.RETURNING_TO_INIT_JOINTS:
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
                commanded_error = float(np.linalg.norm(joint_goal - commanded_joint_position))
                set_joint_goal(redis_client, commanded_joint_position)
                print(
                    "RETURNING_TO_INIT_JOINTS",
                    "| joint_error:",
                    round(joint_error, 5),
                    "| commanded_remaining:",
                    round(commanded_error, 5),
                )

                if joint_error < joint_arrival_threshold:
                    set_joint_goal(redis_client, joint_goal)
                    print("Reached INIT_POS joint configuration. Return sequence complete.")
                    state = CameraState.DONE
                    return True

            elif state == CameraState.DONE:
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
        description=(
            "Open the camera stream, translate to INIT_POS, move to the camera "
            "joint configuration, then track the closest detected face."
        )
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
        "--no-capture",
        action="store_true",
        help=(
            "Do not save a raw capture when the face becomes stationary. "
            "By default, once a face's bbox center stays within "
            f"{STATIONARY_RADIUS_PX}px for {STATIONARY_REQUIRED_S:.1f}s a JPEG "
            f"is saved to {DEFAULT_RAW_CAPTURES_DIR} and the robot returns "
            "to INIT_POS."
        ),
    )
    parser.add_argument(
        "--raw-captures-dir",
        default=str(DEFAULT_RAW_CAPTURES_DIR),
        help=f"Directory for the centered raw JPEG capture (default: {DEFAULT_RAW_CAPTURES_DIR}).",
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
        joint_arrival_threshold=args.joint_arrival_threshold,
        joint_max_step_deg=args.joint_max_step_deg,
        joint_controller_settle_s=args.joint_controller_settle_s,
        capture_when_centered=not args.no_capture,
        raw_captures_dir=args.raw_captures_dir,
    )
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
