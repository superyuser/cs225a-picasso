"""Calibrate canvas corners using AprilTags seen from the wrist camera.

Flow:
    1. Send the robot to INIT_POS so the camera is at a known pose.
    2. Open the USB camera (same logic as demo-day/robot/orient_camera.py).
    3. Detect 4 AprilTags (tag36h11 family) with IDs 0, 1, 2, 3 -- the
       canvas corners TL, TR, BR, BL. Each tag is a 35 mm black square.
    4. Estimate each tag's 3D position in the camera frame using
       cv2.solvePnP with known tag geometry + camera intrinsics.
    5. Transform from the camera frame into the brush-tip frame using the
       fixed camera-on-tip offset (-75.60, 0.0, +42.11 mm) and an assumed
       camera-mount rotation. Output XYZ distances from the tip to each
       corner.
    6. Save the calibration JSON for downstream robot use.

This script lives in demo-day/robot/ and is fully self-contained: it does
not import from the sibling robot/ package, mirroring the style of
demo-day/robot/orient_camera.py.

Usage:
    python calibrate-canvas.py                  # full pipeline
    python calibrate-canvas.py --no-move        # skip robot motion
    python calibrate-canvas.py --no-preview     # headless
    python calibrate-canvas.py --intrinsics-json camera_intrinsics.json
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

import cv2
import numpy as np

try:
    import redis
except ImportError:
    redis = None


# ============================================================
# ROBOT CONFIG (mirrors robot/config.py + robot/redis_io.py so this
# script is self-contained within demo-day/)
# ============================================================

ROBOT_NAME = "Titania"
config_file_for_this_example = "basket.xml"
controller_to_use = "cartesian_controller"

DT = 0.01                            # 100 Hz main loop
POS_TOL_M = 1.0e-2                   # 10 mm position tolerance
DWELL_AT_INIT_S = 0.5

# Robot "home" pose is sourced from demo-day/config.json so all demo-day
# scripts share a single source of truth.
_SCRIPT_DIR_FOR_INIT = Path(__file__).resolve().parent
_DEMO_DAY_CONFIG_PATH = _SCRIPT_DIR_FOR_INIT.parent / "config.json"


def _load_init_pos() -> np.ndarray:
    with _DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return np.array(cfg["init_pos_m"], dtype=float)


INIT_POS = _load_init_pos()


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = (
        f"opensai::controllers::{ROBOT_NAME}::cartesian_controller::cartesian_task::goal_position"
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


def send_position(redis_client, pos):
    redis_client.set(
        redis_keys.cartesian_task_goal_position,
        json.dumps(np.array(pos, dtype=float).tolist()),
    )


def position_error(current_pos, goal_pos):
    return float(np.linalg.norm(np.array(goal_pos, dtype=float) - np.array(current_pos, dtype=float)))


# ============================================================
# CALIBRATION CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_JSON = DEMO_DAY_DIR / "canvas_corners.json"
DEFAULT_INTRINSICS_JSON = DEMO_DAY_DIR / "robot" / "camera_intrinsics.json"

DEFAULT_CAMERA_INDEX = 0
PREVIEW_WINDOW = "Canvas Calibration"

# AprilTag geometry.
TAG_SIZE_M = 0.035                    # 35 mm black square
TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11
TAG_ID_TO_CORNER = {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
CORNER_ORDER = ("TL", "TR", "BR", "BL")

# Which specific corner of each tag corresponds to the physical canvas
# corner we want. Each entry is the (sign_x, sign_y) of the corner in the
# tag's own local frame (x: right, y: up, z: out of the tag face) -- the
# point will be (sx * s, sy * s, 0) with s = TAG_SIZE_M / 2.
#
# Layout (looking at the canvas, tags right-side-up):
#   TL canvas corner  <- bottom-right corner of TL tag  (+x, -y)
#   TR canvas corner  <- bottom-left  corner of TR tag  (-x, -y)
#   BR canvas corner  <- bottom-left  corner of BR tag  (-x, -y)
#   BL canvas corner  <- bottom-right corner of BL tag  (+x, -y)
TAG_LOCAL_CORNER_SIGNS = {
    "TL": (+1.0, -1.0),
    "TR": (-1.0, -1.0),
    "BR": (-1.0, -1.0),
    "BL": (+1.0, -1.0),
}

# Camera mounted behind+above the brush tip.
# Position of the camera origin expressed in the brush-tip frame, in meters.
CAMERA_OFFSET_IN_TIP_FRAME_M = np.array([-0.07560, 0.00000, 0.04211], dtype=float)

# Rotation: camera frame -> brush-tip frame.
# Camera frame convention: +X right, +Y down, +Z forward (out of lens).
# Brush-tip frame convention: +X forward (direction tip points), +Y left, +Z up.
# Camera is assumed to be mounted upright above the tip looking the same
# direction as the brush. If your physical mount differs, edit this matrix.
R_CAM_TO_TIP = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
], dtype=float)

# Stability before capturing the calibration.
STABLE_FRAMES_REQUIRED = 20           # ~0.7 s at 30 fps with all 4 tags visible
POSE_HISTORY_FRAMES = 30              # poses averaged over this many recent frames

# Default HFOV in degrees when no intrinsics file is supplied. Reasonable
# guess for a generic webcam at 60-70 deg. Z (depth) accuracy will be off
# without proper calibration -- supply --intrinsics-json for production use.
DEFAULT_HFOV_DEG = 65.0


# ============================================================
# CAMERA INTRINSICS
# ============================================================

def load_intrinsics(path: Path, frame_w: int, frame_h: int):
    """Load camera matrix and distortion coefficients.

    JSON format (units: pixels for fx/fy/cx/cy):
        {
          "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
          "dist_coeffs": [k1, k2, p1, p2, k3]
        }

    Falls back to a default approximation if the file is missing.
    """
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        K = np.array(payload["camera_matrix"], dtype=float)
        dist = np.array(payload.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=float)
        print(f"Loaded camera intrinsics from {path}")
        return K, dist, True

    fx = fy = 0.5 * frame_w / math.tan(math.radians(DEFAULT_HFOV_DEG / 2.0))
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    dist = np.zeros((5,), dtype=float)
    print(
        f"WARNING: no intrinsics file at {path}. Using {DEFAULT_HFOV_DEG} deg HFOV "
        f"approximation. Z depth will be inaccurate -- calibrate the camera and "
        f"provide --intrinsics-json for accurate results."
    )
    return K, dist, False


# ============================================================
# CAMERA
# ============================================================

def open_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):
        cap.release()
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Opened camera index {camera_index}, but could not read a frame.")
    print(f"Opened camera index {camera_index}.")
    return cap


# ============================================================
# APRILTAG DETECTION + POSE
# ============================================================

def make_detector():
    """Build an ArUco/AprilTag detector compatible with both old + new OpenCV."""
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    try:
        params = cv2.aruco.DetectorParameters()
        return ("new", cv2.aruco.ArucoDetector(dictionary, params), dictionary, params)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        return ("old", None, dictionary, params)


def detect_tags(detector_bundle, gray):
    """Return (ids, corners) for detected tags. corners shape: (N, 4, 2)."""
    kind, detector, dictionary, params = detector_bundle
    if kind == "new":
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return np.array([], dtype=int), np.zeros((0, 4, 2), dtype=np.float32)
    ids = ids.flatten().astype(int)
    pts = np.array([c.reshape(4, 2) for c in corners], dtype=np.float32)
    return ids, pts


def estimate_tag_pose(image_corners_2d, tag_size_m, K, dist):
    """Solve PnP for a single tag.

    Returns (tvec, rvec) in the camera frame. ``tvec`` is the tag center
    in meters; ``rvec`` is the Rodrigues rotation vector from the tag's
    local frame to the camera frame.
    image_corners_2d: shape (4, 2), order TL, TR, BR, BL (OpenCV aruco order).
    """
    s = tag_size_m / 2.0
    object_points = np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0],
    ], dtype=np.float32)
    image_points = np.ascontiguousarray(image_corners_2d.reshape(4, 2)).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None, None
    return tvec.flatten().astype(float), rvec.flatten().astype(float)


def tag_corner_in_camera_frame(
    tvec: np.ndarray,
    rvec: np.ndarray,
    tag_size_m: float,
    sign_x: float,
    sign_y: float,
) -> np.ndarray:
    """Project a specific tag-local corner into the camera frame.

    The desired corner in the tag's local frame is
    ``(sign_x * s, sign_y * s, 0)`` with ``s = tag_size_m / 2``.
    """
    s = tag_size_m / 2.0
    corner_local = np.array(
        [sign_x * s, sign_y * s, 0.0], dtype=float
    )
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return R @ corner_local + np.asarray(tvec, dtype=float)


# ============================================================
# COORDINATE TRANSFORMS
# ============================================================

def camera_to_tip(p_camera: np.ndarray) -> np.ndarray:
    """Convert a point in the camera frame to the brush-tip frame."""
    return R_CAM_TO_TIP @ np.asarray(p_camera, dtype=float) + CAMERA_OFFSET_IN_TIP_FRAME_M


# ============================================================
# OVERLAY
# ============================================================

def draw_overlay(frame, ids, corners_2d, status_text, stable_count):
    if len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, [c.reshape(1, 4, 2) for c in corners_2d], ids)
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"stable {stable_count}/{STABLE_FRAMES_REQUIRED}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)


# ============================================================
# ROBOT MOTION
# ============================================================

def ensure_robot_ready(redis_client) -> bool:
    config_raw = redis_client.get(redis_keys.config_file_name)
    if config_raw is None:
        print("Could not read config file name from Redis.")
        print("Missing key:", redis_keys.config_file_name)
        return False
    config_file_name = decode_redis_value(config_raw)
    if config_file_name != config_file_for_this_example:
        print("This script is meant to be used with config file:", config_file_for_this_example)
        print("Current config file:", config_file_name)
        return False
    while True:
        active_raw = redis_client.get(redis_keys.active_controller)
        active = decode_redis_value(active_raw) if active_raw is not None else None
        if active == controller_to_use:
            break
        redis_client.set(redis_keys.active_controller, controller_to_use)
        time.sleep(0.05)
    print("Using controller:", controller_to_use)
    return True


def move_to_init(redis_client, dwell_s: float = DWELL_AT_INIT_S) -> bool:
    current = read_np(redis_client, redis_keys.cartesian_task_current_position, (3,))
    print("Current position:", current)
    print("Target INIT:    ", INIT_POS)

    send_position(redis_client, INIT_POS)
    loop_time = 0.0
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    while True:
        loop_time += DT
        time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))
        send_position(redis_client, INIT_POS)
        current = read_np(redis_client, redis_keys.cartesian_task_current_position, (3,))
        err = position_error(current, INIT_POS)
        print(f"MOVE_TO_INIT | pos_error: {err:.5f}")
        if err < POS_TOL_M:
            break
    print(f"Reached INIT. Dwelling {dwell_s:.2f}s.")
    time.sleep(dwell_s)
    return True


# ============================================================
# MAIN CAPTURE LOOP
# ============================================================

def capture_corners(
    *,
    camera_index: int,
    intrinsics_json: Path,
    preview: bool,
):
    cap = open_camera(camera_index)
    detector_bundle = make_detector()

    K = None
    dist = None
    intrinsics_calibrated = False

    pose_history: dict[str, list[np.ndarray]] = {name: [] for name in CORNER_ORDER}
    stable_count = 0
    captured_camera_xyz: dict[str, np.ndarray] | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read camera frame.")
                return None
            h, w = frame.shape[:2]

            if K is None:
                K, dist, intrinsics_calibrated = load_intrinsics(intrinsics_json, w, h)
                print(f"Camera frame: {w}x{h}")
                print(f"K = \n{K}")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ids, corners_2d = detect_tags(detector_bundle, gray)

            this_frame_positions: dict[str, np.ndarray] = {}
            for tag_id, corner_pts in zip(ids.tolist(), corners_2d):
                if tag_id not in TAG_ID_TO_CORNER:
                    continue
                name = TAG_ID_TO_CORNER[tag_id]
                tvec, rvec = estimate_tag_pose(corner_pts, TAG_SIZE_M, K, dist)
                if tvec is None or rvec is None:
                    continue
                # Track the precise canvas-corner of this tag rather than its
                # center (e.g. for the TL tag we want its bottom-right corner).
                sx, sy = TAG_LOCAL_CORNER_SIGNS[name]
                corner_in_cam = tag_corner_in_camera_frame(
                    tvec, rvec, TAG_SIZE_M, sx, sy
                )
                this_frame_positions[name] = corner_in_cam
                pose_history[name].append(corner_in_cam)
                if len(pose_history[name]) > POSE_HISTORY_FRAMES:
                    pose_history[name].pop(0)

            all_four_visible = set(this_frame_positions.keys()) >= set(CORNER_ORDER)
            if all_four_visible:
                stable_count += 1
            else:
                stable_count = 0
                for name in CORNER_ORDER:
                    if name not in this_frame_positions and pose_history[name]:
                        pose_history[name].pop(0)

            visible_names = sorted(this_frame_positions.keys())
            missing_names = [n for n in CORNER_ORDER if n not in this_frame_positions]
            if all_four_visible:
                status = f"ALL 4 visible: {visible_names}"
            elif visible_names:
                status = f"missing: {missing_names}  visible: {visible_names}"
            else:
                status = "no tags detected"

            if preview:
                draw_overlay(frame, ids, corners_2d, status, stable_count)
                cv2.imshow(PREVIEW_WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("User aborted.")
                    return None
            else:
                print(f"CALIBRATE | {status} | stable {stable_count}/{STABLE_FRAMES_REQUIRED}")

            if stable_count >= STABLE_FRAMES_REQUIRED:
                captured_camera_xyz = {}
                for name in CORNER_ORDER:
                    history = np.array(pose_history[name], dtype=float)
                    captured_camera_xyz[name] = history.mean(axis=0)
                print("Captured stable corner positions.")
                break

        return {
            "frame_size": (w, h),
            "K": K,
            "dist": dist,
            "intrinsics_calibrated": intrinsics_calibrated,
            "corners_in_camera_frame_m": captured_camera_xyz,
        }

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# ============================================================
# OUTPUT
# ============================================================

def build_output_payload(capture_result, robot_pos_at_capture, robot_ori_at_capture):
    cam_corners = capture_result["corners_in_camera_frame_m"]
    tip_corners = {name: camera_to_tip(p) for name, p in cam_corners.items()}

    # World-frame corner positions: the tip is assumed to sit at INIT_POS when
    # calibration runs, with tip-frame axes aligned to the world frame. Then
    # the world position of each canvas corner is simply:
    #   corner_world = INIT_POS + corner_in_tip_frame
    world_corners = {
        name: INIT_POS + tip_corners[name] for name in CORNER_ORDER
    }

    payload = {
        "units": "meters",
        "tag_family": "tag36h11",
        "tag_ids": TAG_ID_TO_CORNER,
        "tag_local_corner_signs": {
            name: list(TAG_LOCAL_CORNER_SIGNS[name]) for name in CORNER_ORDER
        },
        "tag_size_m": TAG_SIZE_M,
        "camera_offset_in_tip_frame_m": CAMERA_OFFSET_IN_TIP_FRAME_M.tolist(),
        "R_camera_to_tip": R_CAM_TO_TIP.tolist(),
        "intrinsics_calibrated": bool(capture_result["intrinsics_calibrated"]),
        "camera_matrix": capture_result["K"].tolist(),
        "dist_coeffs": capture_result["dist"].tolist(),
        "frame_size_px": list(capture_result["frame_size"]),
        "robot_init_position_m": INIT_POS.tolist(),
        "robot_position_at_capture_m": (
            robot_pos_at_capture.tolist() if robot_pos_at_capture is not None else None
        ),
        "robot_orientation_at_capture": (
            robot_ori_at_capture.tolist() if robot_ori_at_capture is not None else None
        ),
        "corners_in_camera_frame_m": {
            name: [round(float(v), 5) for v in cam_corners[name]]
            for name in CORNER_ORDER
        },
        "corners_in_tip_frame_m": {
            name: [round(float(v), 5) for v in tip_corners[name]]
            for name in CORNER_ORDER
        },
        "corners_in_world_frame_m": {
            name: [round(float(v), 5) for v in world_corners[name]]
            for name in CORNER_ORDER
        },
        "tip_to_corner_distance_m": {
            name: round(float(np.linalg.norm(tip_corners[name])), 5)
            for name in CORNER_ORDER
        },
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    return payload


def print_summary(payload):
    print()
    print("=" * 60)
    print("CANVAS CORNER CALIBRATION RESULT")
    print("=" * 60)
    if not payload["intrinsics_calibrated"]:
        print("(!) Using HFOV-approximated intrinsics -- Z depth is rough.")
    print()
    print(f"{'corner':<6} {'X (mm)':>10} {'Y (mm)':>10} {'Z (mm)':>10}   {'dist (mm)':>10}")
    print("-" * 60)
    print("Tip-frame offsets (corner relative to brush tip):")
    for name in CORNER_ORDER:
        tip = payload["corners_in_tip_frame_m"][name]
        dist = payload["tip_to_corner_distance_m"][name]
        print(f"{name:<6} {tip[0]*1000:>10.2f} {tip[1]*1000:>10.2f} {tip[2]*1000:>10.2f}   {dist*1000:>10.2f}")
    world = payload.get("corners_in_world_frame_m")
    if world is not None:
        print()
        print("World-frame positions (robot base frame, meters):")
        print(f"{'corner':<6} {'X (m)':>10} {'Y (m)':>10} {'Z (m)':>10}")
        print("-" * 60)
        for name in CORNER_ORDER:
            w = world[name]
            print(f"{name:<6} {w[0]:>10.4f} {w[1]:>10.4f} {w[2]:>10.4f}")
    print()


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"OpenCV camera index (default: {DEFAULT_CAMERA_INDEX}).")
    parser.add_argument("--no-move", action="store_true",
                        help="Skip moving the robot to INIT_POS. Useful for tuning the camera.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Run headless (no OpenCV preview window).")
    parser.add_argument("--intrinsics-json", default=str(DEFAULT_INTRINSICS_JSON),
                        help=f"Camera intrinsics JSON (default: {DEFAULT_INTRINSICS_JSON}).")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON),
                        help=f"Where to write the calibration JSON (default: {DEFAULT_OUTPUT_JSON}).")
    parser.add_argument("--tag-size-mm", type=float, default=TAG_SIZE_M * 1000.0,
                        help=f"AprilTag black-square side length in mm (default: {TAG_SIZE_M * 1000:.1f}).")
    parser.add_argument("--dwell-s", type=float, default=DWELL_AT_INIT_S,
                        help=f"Seconds to dwell at INIT before capturing (default: {DWELL_AT_INIT_S}).")
    return parser.parse_args()


def main() -> int:
    global TAG_SIZE_M
    args = parse_args()
    TAG_SIZE_M = float(args.tag_size_mm) / 1000.0
    print(f"Tag size: {TAG_SIZE_M * 1000:.2f} mm")
    print(f"Camera offset in tip frame (mm): "
          f"{(CAMERA_OFFSET_IN_TIP_FRAME_M * 1000).tolist()}")

    robot_pos = None
    robot_ori = None

    if not args.no_move:
        if redis is None:
            print("`redis` package is not installed; run with --no-move or pip install redis.")
            return 1
        redis_client = redis.Redis()
        if not ensure_robot_ready(redis_client):
            return 1
        if not move_to_init(redis_client, dwell_s=args.dwell_s):
            return 1
        robot_pos = read_np(redis_client, redis_keys.cartesian_task_current_position, (3,))
        robot_ori = read_np(redis_client, redis_keys.cartesian_task_current_orientation, (3, 3))
        print(f"Robot orientation at capture:\n{robot_ori}")
    else:
        print("--no-move: skipping robot motion.")

    result = capture_corners(
        camera_index=args.camera_index,
        intrinsics_json=Path(args.intrinsics_json),
        preview=not args.no_preview,
    )
    if result is None:
        return 1

    payload = build_output_payload(result, robot_pos, robot_ori)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration to {out_path}")

    print_summary(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
