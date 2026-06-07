"""Joint-space path building and TOOL_INIT approach / return primitives."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tool_station_coords import (
    DT,
    POS_TOL_M,
    go_to_waypoint,
    read_np,
    redis_keys,
    set_active_controller,
    set_cartesian_goal,
)

HELPERS_DIR = Path(__file__).resolve().parent
ROBOT_DIR = HELPERS_DIR.parent
DEMO_DAY_DIR = ROBOT_DIR.parent
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"
DEFAULT_TOOL_INIT_JOINT_PATH_JSON = DEMO_DAY_DIR / "tool_init_joint_path.json"

TOOL_INIT_JOINT_PATH_VERSION = 1
JOINT_PATH_CONFIG_TOLERANCE_RAD = 1.0e-3
START_JOINT_TOLERANCE_RAD = 0.05

ROBOT_NAME = "Titania"
CARTESIAN_CONTROLLER = "cartesian_controller"
JOINT_CONTROLLER = "joint_controller"

DEG_TO_RAD = math.pi / 180.0
JOINT_ARRIVAL_THRESHOLD = 0.25
JOINT_MAX_STEP_DEG = 0.5
JOINT_CONTROLLER_SETTLE_S = 0.25
CARTESIAN_SETTLE_S = 0.25
DWELL_AT_INIT_POS_S = 0.25
DWELL_AT_TOOL_INIT_S = 0.5
STATUS_PERIOD_S = 0.25
TIMEOUT_S = 60.0

TOOL_JOINT_WAYPOINT_NAMES = (
    "TOOL_SAFE_APPROACH",
    "TOOL_SAFE_APPROACH_2",
    "TOOL_INIT",
)


@dataclass
class JointRedisKeys:
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


joint_redis_keys = JointRedisKeys()


@dataclass
class ToolInitApproachResult:
    forward_joint_path: np.ndarray
    tool_init_rad: np.ndarray
    start_joint_rad: np.ndarray
    waypoint_names: tuple[str, ...]
    waypoint_indices: list[int] | None = None
    joint_path_json: Path | None = None


def _joint_waypoints_deg_from_config(config: dict | None = None) -> dict[str, list[float]]:
    cfg = config or load_demo_day_config()
    return {
        name: load_joint_waypoint_deg(cfg, name).tolist()
        for name in TOOL_JOINT_WAYPOINT_NAMES
    }


def _config_matches_saved_joint_path(saved: dict) -> bool:
    saved_waypoints = saved.get("joint_waypoints_deg", {})
    current = _joint_waypoints_deg_from_config()
    for name in TOOL_JOINT_WAYPOINT_NAMES:
        if name not in saved_waypoints:
            return False
        saved_arr = np.array(saved_waypoints[name], dtype=float)
        current_arr = np.array(current[name], dtype=float)
        if saved_arr.shape != (7,) or current_arr.shape != (7,):
            return False
        if float(np.max(np.abs((saved_arr - current_arr) * DEG_TO_RAD))) > JOINT_PATH_CONFIG_TOLERANCE_RAD:
            return False

    saved_init = saved.get("init_pos_m")
    if saved_init is not None:
        current_init = load_init_pos_m()
        if float(np.max(np.abs(np.array(saved_init, dtype=float) - current_init))) > 1.0e-4:
            return False

    return True


def save_tool_init_joint_path(
    approach: ToolInitApproachResult,
    *,
    joint_path_json: Path = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    waypoint_indices: list[int] | None = None,
) -> Path:
    cfg = load_demo_day_config()
    payload = {
        "version": TOOL_INIT_JOINT_PATH_VERSION,
        "units": "degrees",
        "init_pos_m": load_init_pos_m().tolist(),
        "joint_max_step_deg": float(joint_max_step_deg),
        "waypoint_names": list(TOOL_JOINT_WAYPOINT_NAMES),
        "joint_waypoints_deg": {
            **_joint_waypoints_deg_from_config(cfg),
            "start_at_init_pos": (approach.start_joint_rad / DEG_TO_RAD).tolist(),
        },
        "waypoint_indices": waypoint_indices or [],
        "forward_path_deg": (approach.forward_joint_path / DEG_TO_RAD).tolist(),
        "start_joint_deg": (approach.start_joint_rad / DEG_TO_RAD).tolist(),
        "tool_init_deg": (approach.tool_init_rad / DEG_TO_RAD).tolist(),
    }

    joint_path_json.parent.mkdir(parents=True, exist_ok=True)
    with joint_path_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved tool-init joint path JSON: {joint_path_json}")
    print("Forward path samples:", len(approach.forward_joint_path))
    return joint_path_json


def load_tool_init_joint_path(
    *,
    joint_path_json: Path = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    require_valid_config: bool = True,
) -> ToolInitApproachResult | None:
    if not joint_path_json.is_file():
        return None

    with joint_path_json.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("version") != TOOL_INIT_JOINT_PATH_VERSION:
        print(f"Unsupported joint path JSON version in {joint_path_json}.")
        return None

    if require_valid_config and not _config_matches_saved_joint_path(saved):
        print(
            "Cached tool-init joint path does not match current config.json; "
            "rebuild required."
        )
        return None

    forward_joint_path = np.array(saved["forward_path_deg"], dtype=float) * DEG_TO_RAD
    if forward_joint_path.ndim != 2 or forward_joint_path.shape[1] != 7:
        raise RuntimeError(
            f"{joint_path_json} forward_path_deg must be an N x 7 array."
        )

    start_joint_rad = np.array(saved["start_joint_deg"], dtype=float) * DEG_TO_RAD
    tool_init_rad = np.array(saved["tool_init_deg"], dtype=float) * DEG_TO_RAD

    print(f"Loaded tool-init joint path JSON: {joint_path_json}")
    print("Forward path samples:", len(forward_joint_path))

    return ToolInitApproachResult(
        forward_joint_path=forward_joint_path,
        tool_init_rad=tool_init_rad,
        start_joint_rad=start_joint_rad,
        waypoint_names=tuple(saved.get("waypoint_names", TOOL_JOINT_WAYPOINT_NAMES)),
        waypoint_indices=[int(v) for v in saved.get("waypoint_indices", [])],
        joint_path_json=joint_path_json,
    )


def build_tool_init_approach_result(
    start_joint_rad: np.ndarray,
    *,
    max_joint_step: float,
    joint_path_json: Path | None = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    save_path: bool = True,
) -> ToolInitApproachResult:
    path, waypoint_indices, tool_init_rad = build_tool_init_approach_path(
        start_joint_rad,
        max_joint_step=max_joint_step,
    )
    approach = ToolInitApproachResult(
        forward_joint_path=path,
        tool_init_rad=tool_init_rad,
        start_joint_rad=start_joint_rad.copy(),
        waypoint_names=TOOL_JOINT_WAYPOINT_NAMES,
        waypoint_indices=waypoint_indices,
        joint_path_json=joint_path_json,
    )
    if save_path and joint_path_json is not None:
        save_tool_init_joint_path(
            approach,
            joint_path_json=joint_path_json,
            joint_max_step_deg=max_joint_step / DEG_TO_RAD,
            waypoint_indices=waypoint_indices,
        )
    return approach


def resolve_tool_init_approach_path(
    start_joint_rad: np.ndarray,
    *,
    max_joint_step: float,
    joint_path_json: Path = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    use_cached_path: bool = True,
    rebuild_path: bool = False,
    save_path: bool = True,
) -> ToolInitApproachResult:
    if use_cached_path and not rebuild_path:
        cached = load_tool_init_joint_path(joint_path_json=joint_path_json)
        if cached is not None:
            start_error = float(
                np.linalg.norm(start_joint_rad - cached.start_joint_rad)
            )
            if start_error > START_JOINT_TOLERANCE_RAD:
                print(
                    "Warning: current INIT joint config differs from cached path "
                    f"start by {start_error:.4f} rad; using cached path anyway."
                )
            return cached

    return build_tool_init_approach_result(
        start_joint_rad,
        max_joint_step=max_joint_step,
        joint_path_json=joint_path_json,
        save_path=save_path,
    )


def load_demo_day_config() -> dict:
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_init_pos_m() -> np.ndarray:
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


def load_tool_joint_waypoints_rad() -> list[tuple[str, np.ndarray]]:
    cfg = load_demo_day_config()
    return [
        (name, load_joint_waypoint_deg(cfg, name) * DEG_TO_RAD)
        for name in TOOL_JOINT_WAYPOINT_NAMES
    ]


def load_tool_init_rad() -> np.ndarray:
    cfg = load_demo_day_config()
    return load_joint_waypoint_deg(cfg, "TOOL_INIT") * DEG_TO_RAD


def set_joint_goal(redis_client, position: np.ndarray) -> None:
    redis_client.set(
        joint_redis_keys.joint_task_goal_position,
        json.dumps(position.tolist()),
    )
    redis_client.set(
        joint_redis_keys.joint_task_goal_velocity,
        json.dumps(np.zeros_like(position).tolist()),
    )
    redis_client.set(
        joint_redis_keys.joint_task_goal_acceleration,
        json.dumps(np.zeros_like(position).tolist()),
    )


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
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")

    distances = [
        float(np.linalg.norm(waypoints[index + 1] - waypoints[index]))
        for index in range(len(waypoints) - 1)
    ]

    if max_step <= 0.0:
        raise ValueError("max_step must be positive")

    scale = 1.0
    waypoint_indices = [0]
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


def build_tool_init_approach_path(
    start_joint_rad: np.ndarray,
    *,
    max_joint_step: float,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    tool_waypoints = [waypoint for _name, waypoint in load_tool_joint_waypoints_rad()]
    path, waypoint_indices = build_smooth_joint_path(
        [start_joint_rad, *tool_waypoints],
        max_step=max_joint_step,
    )
    return path, waypoint_indices, tool_waypoints[-1]


def move_cartesian_to_init_pos(
    redis_client,
    *,
    pos_tol_m: float = POS_TOL_M,
    dwell_s: float = DWELL_AT_INIT_POS_S,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
) -> bool:
    init_pos = load_init_pos_m()
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

    set_cartesian_goal(redis_client, current_position, hold_orientation)
    set_active_controller(redis_client, CARTESIAN_CONTROLLER)
    print("Using controller:", CARTESIAN_CONTROLLER)

    return go_to_waypoint(
        redis_client,
        target_pos=init_pos,
        hold_orientation=hold_orientation,
        label="INIT_POS",
        dwell_s=dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    )


def stream_joint_path_samples(
    redis_client,
    path: np.ndarray,
    *,
    goal_joint_position: np.ndarray,
    label: str,
    joint_arrival_threshold: float,
    dwell_s: float = 0.0,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    sample_indices: range | None = None,
) -> bool:
    if sample_indices is None:
        sample_indices = range(1, len(path))

    loop_time = 0.0
    last_status = 0.0
    start = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    for path_index in sample_indices:
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        commanded_joint_position = path[path_index]
        current_joint_position = read_np(
            redis_client,
            joint_redis_keys.sensor_joint_positions,
            (7,),
        )
        joint_error = float(
            np.linalg.norm(goal_joint_position - current_joint_position)
        )
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
            print(f"Timed out while streaming joint path for {label}.")
            print("Final joint error:", round(joint_error, 5))
            return False

    set_joint_goal(redis_client, goal_joint_position)
    while True:
        current_joint_position = read_np(
            redis_client,
            joint_redis_keys.sensor_joint_positions,
            (7,),
        )
        joint_error = float(
            np.linalg.norm(goal_joint_position - current_joint_position)
        )
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
) -> bool:
    path, _waypoint_indices = build_smooth_joint_path(
        [start_joint_position, goal_joint_position],
        max_step=max_joint_step,
    )
    print(f"Moving to {label}...")
    print(f"{label} target (deg):", np.round(goal_joint_position / DEG_TO_RAD, 3))
    print("Smooth joint path samples:", len(path))

    set_joint_goal(redis_client, start_joint_position)
    set_active_controller(redis_client, JOINT_CONTROLLER)

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < JOINT_CONTROLLER_SETTLE_S:
        set_joint_goal(redis_client, start_joint_position)
        time.sleep(DT)

    return stream_joint_path_samples(
        redis_client,
        path,
        goal_joint_position=goal_joint_position,
        label=label,
        joint_arrival_threshold=joint_arrival_threshold,
        dwell_s=dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    )


def approach_tool_init(
    redis_client,
    *,
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    joint_controller_settle_s: float = JOINT_CONTROLLER_SETTLE_S,
    dwell_at_tool_init_s: float = DWELL_AT_TOOL_INIT_S,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    joint_path_json: Path = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    use_cached_path: bool = True,
    rebuild_path: bool = False,
    save_path: bool = True,
) -> ToolInitApproachResult | None:
    """INIT_POS (Cartesian) -> TOOL_SAFE_APPROACH -> TOOL_SAFE_APPROACH_2 -> TOOL_INIT."""
    if not move_cartesian_to_init_pos(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    ):
        return None

    current_joint_position = read_np(
        redis_client,
        joint_redis_keys.sensor_joint_positions,
        (7,),
    )
    max_joint_step = max(abs(joint_max_step_deg) * DEG_TO_RAD, 1.0e-5)
    approach = resolve_tool_init_approach_path(
        current_joint_position,
        max_joint_step=max_joint_step,
        joint_path_json=joint_path_json,
        use_cached_path=use_cached_path,
        rebuild_path=rebuild_path,
        save_path=save_path,
    )
    path = approach.forward_joint_path
    tool_init_rad = approach.tool_init_rad
    waypoint_indices = approach.waypoint_indices or []

    print(
        "Current joint position (deg):",
        np.round(current_joint_position / DEG_TO_RAD, 3),
    )
    for name, waypoint_deg in [
        (name, load_joint_waypoint_deg(load_demo_day_config(), name))
        for name in TOOL_JOINT_WAYPOINT_NAMES
    ]:
        print(f"{name} target (deg):", np.round(waypoint_deg, 3))
    print("Joint max step (deg):", joint_max_step_deg)
    print("Smooth joint path samples:", len(path))
    if waypoint_indices:
        for waypoint_index, waypoint_name in zip(
            waypoint_indices[1:],
            TOOL_JOINT_WAYPOINT_NAMES,
        ):
            if waypoint_index < len(path):
                print(f"{waypoint_name} command sample:", waypoint_index + 1)

    set_joint_goal(redis_client, current_joint_position)
    set_active_controller(redis_client, JOINT_CONTROLLER)
    print("Using controller:", JOINT_CONTROLLER)

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < joint_controller_settle_s:
        set_joint_goal(redis_client, current_joint_position)
        time.sleep(DT)

    if not stream_joint_path_samples(
        redis_client,
        path,
        goal_joint_position=tool_init_rad,
        label="TOOL_INIT",
        joint_arrival_threshold=joint_arrival_threshold,
        dwell_s=dwell_at_tool_init_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    ):
        return None

    return approach


def return_from_tool_init(
    redis_client,
    approach: ToolInitApproachResult | None = None,
    *,
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    dwell_at_init_joint_s: float = 0.0,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    joint_path_json: Path = DEFAULT_TOOL_INIT_JOINT_PATH_JSON,
    use_cached_path: bool = True,
) -> bool:
    """Follow the exact reverse of the recorded TOOL_INIT approach joint path."""
    if approach is None:
        if not use_cached_path:
            print("No approach result provided and cached path use is disabled.")
            return False
        approach = load_tool_init_joint_path(joint_path_json=joint_path_json)
        if approach is None:
            print("Could not load cached tool-init joint path for return.")
            return False

    path = approach.forward_joint_path
    if len(path) < 2:
        return move_cartesian_to_init_pos(
            redis_client,
            status_period_s=status_period_s,
            timeout_s=timeout_s,
        )

    reverse_indices = range(len(path) - 2, -1, -1)
    goal_joint_position = path[0]

    print(
        "Returning from TOOL_INIT along reversed approach path "
        f"({len(reverse_indices)} samples)."
    )

    set_joint_goal(redis_client, path[-1])
    set_active_controller(redis_client, JOINT_CONTROLLER)
    print("Using controller:", JOINT_CONTROLLER)

    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < JOINT_CONTROLLER_SETTLE_S:
        set_joint_goal(redis_client, path[-1])
        time.sleep(DT)

    if not stream_joint_path_samples(
        redis_client,
        path,
        goal_joint_position=goal_joint_position,
        label="INIT_JOINT_CONFIG",
        joint_arrival_threshold=joint_arrival_threshold,
        dwell_s=dwell_at_init_joint_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        sample_indices=reverse_indices,
    ):
        return False

    return move_cartesian_to_init_pos(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    )
