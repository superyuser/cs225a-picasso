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
CORNER_ORDER = ("TL", "TR", "BR", "BL")


def _load_demo_day_config() -> dict:
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


_DEMO_DAY_CONFIG = _load_demo_day_config()


def _load_init_pos() -> np.ndarray:
    """Load INIT_POS from demo-day/config.json so all scripts share one source."""
    return np.array(_DEMO_DAY_CONFIG["init_pos_m"], dtype=float)


def _load_canvas_corner_x_offset_m() -> float:
    """Load the X offset from calibrated/tag plane to the actual canvas plane.

    Applied additively to the X component of each TL/TR/BR/BL waypoint to
    account for canvas thickness. A negative value means the robot travels a
    shorter distance in +X before touching the drawable surface.
    """
    return float(_DEMO_DAY_CONFIG.get("canvas_corner_x_offset_m", 0.0))


def _load_canvas_padding_m() -> float:
    padding_m = float(_DEMO_DAY_CONFIG.get("CANVAS_PADDING", 0.0))
    if padding_m < 0.0:
        raise ValueError(f"CANVAS_PADDING must be non-negative, got {padding_m}")
    return padding_m


def _load_canvas_border_horizontal_offset_m() -> float:
    return float(_DEMO_DAY_CONFIG.get("canvas_border_horizontal_offset_m", 0.0))


def _load_canvas_border_vertical_offset_m() -> float:
    return float(_DEMO_DAY_CONFIG.get("canvas_border_vertical_offset_m", 0.0))


def _parse_corner_offsets(raw, *, key_name: str) -> dict[str, np.ndarray]:
    offsets = {name: np.zeros(3, dtype=float) for name in CORNER_ORDER}
    if raw is None:
        return offsets
    if not isinstance(raw, dict):
        raise ValueError(f"{key_name} must be an object keyed by TL/TR/BR/BL.")

    for name, value in raw.items():
        if name not in offsets:
            raise ValueError(f"{key_name} contains unknown corner {name!r}.")
        vec = np.array(value, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"{key_name}.{name} must be an [x, y, z] vector in meters.")
        offsets[name] = vec
    return offsets


def _load_canvas_corner_correction_offsets_m() -> dict[str, np.ndarray]:
    return _parse_corner_offsets(
        _DEMO_DAY_CONFIG.get("canvas_corner_correction_offsets_m", {}),
        key_name="canvas_corner_correction_offsets_m",
    )


# Robot "home" pose -- shared with calibrate_canvas.py / orient_camera.py.
INIT_POS = _load_init_pos()
CANVAS_CORNER_X_OFFSET_M = _load_canvas_corner_x_offset_m()
CANVAS_PADDING_M = _load_canvas_padding_m()
CANVAS_BORDER_HORIZONTAL_OFFSET_M = _load_canvas_border_horizontal_offset_m()
CANVAS_BORDER_VERTICAL_OFFSET_M = _load_canvas_border_vertical_offset_m()
CANVAS_CORNER_CORRECTION_OFFSETS_M = _load_canvas_corner_correction_offsets_m()


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


def _corners_from_payload_section(section: dict) -> dict[str, np.ndarray]:
    return {name: np.array(section[name], dtype=float) for name in CORNER_ORDER}


def _payload_applied_corner_correction_offsets(payload: dict) -> dict[str, np.ndarray]:
    if not payload.get("actual_canvas_correction_offsets_applied", False):
        return {name: np.zeros(3, dtype=float) for name in CORNER_ORDER}
    return _parse_corner_offsets(
        payload.get("actual_canvas_correction_offsets_m", {}),
        key_name="actual_canvas_correction_offsets_m",
    )


def _payload_canvas_border_offsets_m(payload: dict) -> tuple[float, float]:
    if not payload.get("canvas_border_offset_applied", False):
        return 0.0, 0.0
    return (
        float(payload.get("canvas_border_horizontal_offset_m", 0.0)),
        float(payload.get("canvas_border_vertical_offset_m", 0.0)),
    )


def _payload_capture_origin(payload: dict) -> np.ndarray:
    for key in (
        "calibration_capture_origin_m",
        "calibration_camera_init_position_m",
        "calibration_init_target_m",
    ):
        if key in payload:
            origin = np.array(payload[key], dtype=float)
            if origin.shape != (3,):
                raise ValueError(f"{key} must be an [x, y, z] vector in meters.")
            return origin
    return INIT_POS


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-9:
        raise ValueError("Cannot normalize near-zero vector while insetting canvas corners.")
    return vec / norm


def _canvas_plane_basis(corners: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.array([corners[name] for name in CORNER_ORDER], dtype=float)
    origin = points[0]

    normal = np.zeros(3, dtype=float)
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        normal += np.cross(point, next_point)
    normal = _normalize(normal)

    x_axis = _normalize(corners["TR"] - corners["TL"])
    y_axis = _normalize(np.cross(normal, x_axis))
    return origin, x_axis, y_axis


def _line_intersection_2d(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray:
    system = np.column_stack((first_direction, -second_direction))
    rhs = second_point - first_point
    if abs(float(np.linalg.det(system))) < 1.0e-9:
        raise ValueError("Canvas corner inset failed because adjacent edges are parallel.")
    t, _u = np.linalg.solve(system, rhs)
    return first_point + t * first_direction


def inset_canvas_corners(
    corners: dict[str, np.ndarray],
    padding_m: float,
) -> dict[str, np.ndarray]:
    """Inset ordered TL/TR/BR/BL corners by ``padding_m`` within their plane."""
    if padding_m <= 0.0:
        return {name: np.array(corners[name], dtype=float) for name in CORNER_ORDER}

    origin, x_axis, y_axis = _canvas_plane_basis(corners)
    points_2d = []
    for name in CORNER_ORDER:
        rel = np.array(corners[name], dtype=float) - origin
        points_2d.append(np.array([np.dot(rel, x_axis), np.dot(rel, y_axis)], dtype=float))

    offset_points: list[np.ndarray] = []
    offset_directions: list[np.ndarray] = []
    for index, point in enumerate(points_2d):
        next_point = points_2d[(index + 1) % len(points_2d)]
        edge = next_point - point
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= 2.0 * padding_m:
            raise ValueError(
                f"CANVAS_PADDING={padding_m:.4f} m is too large for canvas edge "
                f"{CORNER_ORDER[index]}->{CORNER_ORDER[(index + 1) % len(CORNER_ORDER)]} "
                f"with length {edge_len:.4f} m."
            )
        direction = edge / edge_len
        inward_normal = np.array([-direction[1], direction[0]], dtype=float)
        offset_points.append(point + inward_normal * padding_m)
        offset_directions.append(direction)

    inset_points_2d: list[np.ndarray] = []
    for index in range(len(points_2d)):
        prev_index = (index - 1) % len(points_2d)
        inset_points_2d.append(
            _line_intersection_2d(
                offset_points[prev_index],
                offset_directions[prev_index],
                offset_points[index],
                offset_directions[index],
            )
        )

    return {
        name: origin + point_2d[0] * x_axis + point_2d[1] * y_axis
        for name, point_2d in zip(CORNER_ORDER, inset_points_2d)
    }


def apply_canvas_border_offset(
    corners: dict[str, np.ndarray],
    *,
    horizontal_offset_m: float,
    vertical_offset_m: float,
) -> dict[str, np.ndarray]:
    """Shift every corner in the canvas plane.

    Positive horizontal moves from TL toward TR. Positive vertical moves from
    TL toward BL.
    """
    if abs(horizontal_offset_m) < 1.0e-12 and abs(vertical_offset_m) < 1.0e-12:
        return {name: np.array(corners[name], dtype=float) for name in CORNER_ORDER}

    right_axis = _normalize(corners["TR"] - corners["TL"])
    top_mid = 0.5 * (corners["TL"] + corners["TR"])
    bottom_mid = 0.5 * (corners["BL"] + corners["BR"])
    down_axis = _normalize(bottom_mid - top_mid)
    shift = horizontal_offset_m * right_axis + vertical_offset_m * down_axis
    return {name: np.array(corners[name], dtype=float) + shift for name in CORNER_ORDER}


# ============================================================
# CORNER LOADING
# ============================================================

def load_drawable_corner_world_positions(
    corners_json: Path,
) -> tuple[dict[str, np.ndarray], str, bool, dict[str, np.ndarray]]:
    """Read the calibration JSON and return TL/TR/BR/BL in world frame.

    Preferred source: explicit ``payload["actual_canvas_corners_in_world_frame_m"]``.
    It is corrected to the current config, then inset by the current
    ``CANVAS_PADDING`` so visit targets track config changes without requiring
    a fresh calibration capture.

    Fallback (for older JSONs): reconstruct world coords by simple addition
    ``calibration_capture_origin_m + corner_in_tip_frame_m`` when the capture
    origin exists, otherwise ``INIT_POS + corner_in_tip_frame_m``.
    """
    with corners_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    applied_correction_offsets = _payload_applied_corner_correction_offsets(payload)
    source_border_horizontal_offset_m, source_border_vertical_offset_m = (
        _payload_canvas_border_offsets_m(payload)
    )
    border_horizontal_delta_m = (
        CANVAS_BORDER_HORIZONTAL_OFFSET_M - source_border_horizontal_offset_m
    )
    border_vertical_delta_m = (
        CANVAS_BORDER_VERTICAL_OFFSET_M - source_border_vertical_offset_m
    )

    actual_world = payload.get("actual_canvas_corners_in_world_frame_m")
    if actual_world is not None:
        actual_corners = _corners_from_payload_section(actual_world)
        source_surface_x_offset_m = float(
            payload.get("actual_canvas_surface_x_offset_m", CANVAS_CORNER_X_OFFSET_M)
        )
        surface_x_offset_delta_m = CANVAS_CORNER_X_OFFSET_M - source_surface_x_offset_m
        surface_x_offset_delta = np.array([surface_x_offset_delta_m, 0.0, 0.0], dtype=float)
        correction_delta = {
            name: (
                CANVAS_CORNER_CORRECTION_OFFSETS_M[name]
                - applied_correction_offsets[name]
            )
            for name in CORNER_ORDER
        }
        corrected_actual_corners = {
            name: actual_corners[name] + surface_x_offset_delta + correction_delta[name]
            for name in CORNER_ORDER
        }
        corrected_actual_corners = apply_canvas_border_offset(
            corrected_actual_corners,
            horizontal_offset_m=border_horizontal_delta_m,
            vertical_offset_m=border_vertical_delta_m,
        )
        return (
            inset_canvas_corners(corrected_actual_corners, CANVAS_PADDING_M),
            (
                "actual_canvas_corners_in_world_frame_m "
                f"+ x_offset_delta({surface_x_offset_delta_m:+.4f}m) "
                f"+ border_offset_delta(h={border_horizontal_delta_m:+.4f}m, "
                f"v={border_vertical_delta_m:+.4f}m) "
                f"+ CANVAS_PADDING({CANVAS_PADDING_M:.4f}m)"
            ),
            True,
            CANVAS_CORNER_CORRECTION_OFFSETS_M,
        )

    drawable_world = payload.get("drawable_corners_in_world_frame_m")
    if drawable_world is not None:
        corners = apply_canvas_border_offset(
            _corners_from_payload_section(drawable_world),
            horizontal_offset_m=border_horizontal_delta_m,
            vertical_offset_m=border_vertical_delta_m,
        )
        return (
            corners,
            (
                "drawable_corners_in_world_frame_m "
                f"+ border_offset_delta(h={border_horizontal_delta_m:+.4f}m, "
                f"v={border_vertical_delta_m:+.4f}m)"
            ),
            False,
            applied_correction_offsets,
        )

    world = payload.get("corners_in_world_frame_m")
    if world is not None:
        corners = apply_canvas_border_offset(
            _corners_from_payload_section(world),
            horizontal_offset_m=border_horizontal_delta_m,
            vertical_offset_m=border_vertical_delta_m,
        )
        if "canvas_padding_m" in payload:
            return (
                corners,
                (
                    "corners_in_world_frame_m "
                    f"+ border_offset_delta(h={border_horizontal_delta_m:+.4f}m, "
                    f"v={border_vertical_delta_m:+.4f}m)"
                ),
                False,
                applied_correction_offsets,
            )
        return (
            inset_canvas_corners(corners, CANVAS_PADDING_M),
            (
                "corners_in_world_frame_m "
                f"+ border_offset_delta(h={border_horizontal_delta_m:+.4f}m, "
                f"v={border_vertical_delta_m:+.4f}m) "
                f"+ CANVAS_PADDING({CANVAS_PADDING_M:.4f}m)"
            ),
            False,
            applied_correction_offsets,
        )

    tip = payload.get("corners_in_tip_frame_m")
    if tip is None:
        raise RuntimeError(
            f"{corners_json} is missing both 'corners_in_world_frame_m' and "
            "'corners_in_tip_frame_m'. Re-run demo-day/robot/calibrate_canvas.py."
        )

    capture_origin = _payload_capture_origin(payload)
    print(
        "Reconstructing world-frame corners as calibration capture origin "
        "+ tip-frame offset (older JSON format)."
    )
    print(f"  calibration capture origin: {np.round(capture_origin, 5).tolist()}")
    corners = {
        name: capture_origin + np.array(tip[name], dtype=float) for name in CORNER_ORDER
    }
    corners = apply_canvas_border_offset(
        corners,
        horizontal_offset_m=border_horizontal_delta_m,
        vertical_offset_m=border_vertical_delta_m,
    )
    return (
        inset_canvas_corners(corners, CANVAS_PADDING_M),
        (
            "corners_in_tip_frame_m "
            f"+ border_offset_delta(h={border_horizontal_delta_m:+.4f}m, "
            f"v={border_vertical_delta_m:+.4f}m) "
            f"+ CANVAS_PADDING({CANVAS_PADDING_M:.4f}m)"
        ),
        False,
        applied_correction_offsets,
    )


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

    (
        corner_world_drawable,
        corner_source,
        source_has_surface_offset,
        source_corner_correction_offsets,
    ) = (
        load_drawable_corner_world_positions(corners_json)
    )
    x_offset_m = 0.0 if source_has_surface_offset else CANVAS_CORNER_X_OFFSET_M
    x_offset_vec = np.array([x_offset_m, 0.0, 0.0], dtype=float)
    correction_delta = {
        name: (
            CANVAS_CORNER_CORRECTION_OFFSETS_M[name]
            - source_corner_correction_offsets[name]
        )
        for name in CORNER_ORDER
    }
    corner_world = {
        name: corner_world_drawable[name] + x_offset_vec + correction_delta[name]
        for name in CORNER_ORDER
    }

    print("\nCanvas corners (world frame, meters):")
    print(f"  INIT_POS: {INIT_POS.tolist()}")
    print(
        f"  drawable corner source: {corner_source}"
    )
    if source_has_surface_offset:
        print(f"  CANVAS_PADDING = {CANVAS_PADDING_M:.4f} m")
        print("  actual canvas surface offset is reconciled from JSON to current config")
    else:
        print(
            f"  CANVAS_PADDING = {CANVAS_PADDING_M:.4f} m"
            " (already included in drawable source)"
        )
        print(
            f"  applying canvas_corner_x_offset_m = {x_offset_m:+.4f} m "
            "for actual canvas plane thickness (X axis only)"
        )
    max_correction_delta_m = max(
        float(np.linalg.norm(correction_delta[name])) for name in CORNER_ORDER
    )
    if max_correction_delta_m > 0.0:
        print(
            "  applying canvas_corner_correction_offsets_m delta "
            f"(max {max_correction_delta_m * 1000.0:.1f} mm)"
        )
    for name in CORNER_ORDER:
        print(
            f"  {name}: drawable {np.round(corner_world_drawable[name], 5).tolist()}"
            f" + correction {np.round(correction_delta[name], 5).tolist()}"
            f" -> target {np.round(corner_world[name], 5).tolist()}"
        )

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
