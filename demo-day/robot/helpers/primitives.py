"""Reusable Cartesian motion primitives for the tool station."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .tool_station_coords import (
    DT,
    POS_TOL_M,
    go_to_waypoint,
    position_error,
    read_np,
    redis_keys,
    set_cartesian_goal,
)

HELPERS_DIR = Path(__file__).resolve().parent
ROBOT_DIR = HELPERS_DIR.parent
DEMO_DAY_DIR = ROBOT_DIR.parent
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"

MM_TO_M = 1.0e-3
DEFAULT_Z_DIVE_IN_OFFSET_MM = -86.0
DEFAULT_DIP_DWELL_S = 0.5
DEFAULT_HOVER_DWELL_S = 0.25
DEFAULT_SWIRL_DIAMETER_MM = 50.0
DEFAULT_SWIRL_REVOLUTIONS = 5
DEFAULT_PAINT_STATION_Y_OFFSET_M = -0.03
DEFAULT_PAINT_SWIRL_DIAMETER_MM = 20.0
DEFAULT_PAINT_SWIRL_REVOLUTIONS = 3
DEFAULT_SWIRL_STEP_MM = 2.0
PAINT_STATION_Y_OFFSET_CONFIG_KEY = "paint_station_y_offset_m"


def load_demo_day_config() -> dict:
    with DEMO_DAY_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_paint_init_pos_m(config: dict, key: str) -> np.ndarray:
    if key not in config:
        raise RuntimeError(f"{DEMO_DAY_CONFIG_PATH} is missing {key}.")

    pos_mm = np.array(config[key], dtype=float)
    if pos_mm.shape != (3,):
        raise RuntimeError(
            f"{DEMO_DAY_CONFIG_PATH} {key} must contain 3 coordinates (mm)."
        )
    pos_m = pos_mm * MM_TO_M
    pos_m[1] += load_paint_station_y_offset_m(config)
    return pos_m


def load_paint_station_y_offset_m(config: dict | None = None) -> float:
    cfg = config or load_demo_day_config()
    return float(
        cfg.get(
            PAINT_STATION_Y_OFFSET_CONFIG_KEY,
            DEFAULT_PAINT_STATION_Y_OFFSET_M,
        )
    )


def load_z_dive_in_offset_m(config: dict | None = None) -> float:
    cfg = config or load_demo_day_config()
    offset_mm = float(cfg.get("z_dive_in_offset_mm", DEFAULT_Z_DIVE_IN_OFFSET_MM))
    return offset_mm * MM_TO_M


def load_swirl_diameter_m(config: dict | None = None) -> float:
    cfg = config or load_demo_day_config()
    diameter_mm = float(cfg.get("swirl_diameter_mm", DEFAULT_SWIRL_DIAMETER_MM))
    return diameter_mm * MM_TO_M


def load_paint_swirl_diameter_m(config: dict | None = None) -> float:
    cfg = config or load_demo_day_config()
    diameter_mm = float(
        cfg.get("paint_swirl_diameter_mm", DEFAULT_PAINT_SWIRL_DIAMETER_MM)
    )
    return diameter_mm * MM_TO_M


def load_paint_swirl_revolutions(config: dict | None = None) -> int:
    cfg = config or load_demo_day_config()
    return int(cfg.get("paint_swirl_revolutions", DEFAULT_PAINT_SWIRL_REVOLUTIONS))


MoveFn = Callable[..., bool]
StreamFn = Callable[..., bool]


def build_circle_xy_path(
    center_m: np.ndarray,
    *,
    radius_m: float,
    revolutions: int,
    step_m: float,
) -> np.ndarray:
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if revolutions <= 0:
        raise ValueError("revolutions must be positive")
    if step_m <= 0.0:
        raise ValueError("step_m must be positive")

    circumference = 2.0 * math.pi * radius_m
    steps_per_rev = max(8, int(math.ceil(circumference / step_m)))
    total_steps = steps_per_rev * revolutions
    points: list[np.ndarray] = []
    for step_index in range(1, total_steps + 1):
        theta = 2.0 * math.pi * step_index / steps_per_rev
        points.append(
            np.array(
                [
                    center_m[0] + radius_m * math.cos(theta),
                    center_m[1] + radius_m * math.sin(theta),
                    center_m[2],
                ],
                dtype=float,
            )
        )
    return np.array(points, dtype=float)


def stream_cartesian_path(
    redis_client,
    path: np.ndarray,
    *,
    hold_orientation: np.ndarray,
    label: str,
    pos_tol_m: float = POS_TOL_M,
    status_period_s: float,
    timeout_s: float,
) -> bool:
    if len(path) == 0:
        return True

    print(f"Streaming {label}: {len(path)} Cartesian samples.")

    loop_time = 0.0
    last_status = 0.0
    start = time.perf_counter()
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9
    path_index = 0

    while path_index < len(path):
        loop_time += DT
        time.sleep(
            max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time))
        )

        target_position = path[path_index]
        set_cartesian_goal(redis_client, target_position, hold_orientation)
        current_position = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        err = position_error(current_position, target_position)

        if status_period_s <= 0.0 or loop_time - last_status >= status_period_s:
            print(
                f"STREAMING_{label}",
                "|",
                f"sample {path_index + 1}/{len(path)}",
                "| pos_error:",
                round(err, 5),
            )
            last_status = loop_time

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(f"Timed out while streaming {label}.")
            print("Final position error:", round(err, 5))
            return False

        path_index += 1

    final_position = path[-1]
    set_cartesian_goal(redis_client, final_position, hold_orientation)
    while True:
        current_position = read_np(
            redis_client,
            redis_keys.cartesian_task_current_position,
            (3,),
        )
        err = position_error(current_position, final_position)
        if err < pos_tol_m:
            print(f"Finished streaming {label}.")
            return True

        if timeout_s > 0.0 and time.perf_counter() - start > timeout_s:
            print(f"Timed out settling after streaming {label}.")
            print("Final position error:", round(err, 5))
            return False

        set_cartesian_goal(redis_client, final_position, hold_orientation)
        time.sleep(DT)


def dip_at_hover(
    redis_client,
    hover_pos_m: np.ndarray,
    *,
    hold_orientation: np.ndarray,
    label: str,
    z_dive_offset_m: float,
    swirl_diameter_m: float | None = None,
    swirl_revolutions: int = DEFAULT_PAINT_SWIRL_REVOLUTIONS,
    swirl_step_m: float = DEFAULT_SWIRL_STEP_MM * MM_TO_M,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    """Dip cycle: hover -> dive -> small paint swirl -> dive center -> hover."""
    move = move_fn or go_to_waypoint
    stream = stream_fn or stream_cartesian_path
    dipped_pos = hover_pos_m.copy()
    dipped_pos[2] += z_dive_offset_m
    diameter_m = (
        swirl_diameter_m
        if swirl_diameter_m is not None
        else DEFAULT_PAINT_SWIRL_DIAMETER_MM * MM_TO_M
    )

    print(
        f"\n{label} dip: paint circle diameter={diameter_m * 1000.0:.1f} mm, "
        f"revolutions={swirl_revolutions}, dive_z_offset={z_dive_offset_m * 1000.0:.1f} mm"
    )

    if not move(
        redis_client,
        target_pos=dipped_pos,
        hold_orientation=hold_orientation,
        label=f"{label}_DIVE",
        dwell_s=dwell_at_dip_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    swirl_path = build_circle_xy_path(
        dipped_pos,
        radius_m=diameter_m / 2.0,
        revolutions=swirl_revolutions,
        step_m=swirl_step_m,
    )
    if not stream(
        redis_client,
        swirl_path,
        hold_orientation=hold_orientation,
        label=f"{label}_DIP_CIRCLE",
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not move(
        redis_client,
        target_pos=dipped_pos,
        hold_orientation=hold_orientation,
        label=f"{label}_DIVE_CENTER",
        dwell_s=0.0,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return move(
        redis_client,
        target_pos=hover_pos_m,
        hold_orientation=hold_orientation,
        label=label,
        dwell_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    )


def _dip_paint(
    redis_client,
    config_key: str,
    label: str,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    swirl_diameter_m: float | None = None,
    swirl_revolutions: int | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    cfg = config or load_demo_day_config()
    hover_pos_m = load_paint_init_pos_m(cfg, config_key)
    z_offset_m = (
        z_dive_offset_m
        if z_dive_offset_m is not None
        else load_z_dive_in_offset_m(cfg)
    )
    diameter_m = (
        swirl_diameter_m
        if swirl_diameter_m is not None
        else load_paint_swirl_diameter_m(cfg)
    )
    revolutions = (
        swirl_revolutions
        if swirl_revolutions is not None
        else load_paint_swirl_revolutions(cfg)
    )
    return dip_at_hover(
        redis_client,
        hover_pos_m,
        hold_orientation=hold_orientation,
        label=label,
        z_dive_offset_m=z_offset_m,
        swirl_diameter_m=diameter_m,
        swirl_revolutions=revolutions,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
        stream_fn=stream_fn,
    )


def dip_paint_1(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    swirl_diameter_m: float | None = None,
    swirl_revolutions: int | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    """P1_INIT_POS -> dive -> 3 paint circles -> dive center -> P1_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P1_INIT_POS",
        "P1_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        swirl_diameter_m=swirl_diameter_m,
        swirl_revolutions=swirl_revolutions,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
        stream_fn=stream_fn,
    )


def dip_paint_2(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    swirl_diameter_m: float | None = None,
    swirl_revolutions: int | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    """P2_INIT_POS -> dive -> 3 paint circles -> dive center -> P2_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P2_INIT_POS",
        "P2_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        swirl_diameter_m=swirl_diameter_m,
        swirl_revolutions=swirl_revolutions,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
        stream_fn=stream_fn,
    )


def dip_paint_3(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    swirl_diameter_m: float | None = None,
    swirl_revolutions: int | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    """P3_INIT_POS -> dive -> 3 paint circles -> dive center -> P3_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P3_INIT_POS",
        "P3_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        swirl_diameter_m=swirl_diameter_m,
        swirl_revolutions=swirl_revolutions,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
        stream_fn=stream_fn,
    )


def swirl_water(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    swirl_diameter_m: float | None = None,
    revolutions: int = DEFAULT_SWIRL_REVOLUTIONS,
    swirl_step_m: float = DEFAULT_SWIRL_STEP_MM * MM_TO_M,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
    stream_fn: StreamFn | None = None,
) -> bool:
    """WATER_INIT_POS -> dive -> swirl -> center -> WATER_INIT_POS."""
    cfg = config or load_demo_day_config()
    move = move_fn or go_to_waypoint
    stream = stream_fn or stream_cartesian_path

    hover_pos_m = load_paint_init_pos_m(cfg, "WATER_INIT_POS")
    z_offset_m = (
        z_dive_offset_m
        if z_dive_offset_m is not None
        else load_z_dive_in_offset_m(cfg)
    )
    diameter_m = (
        swirl_diameter_m
        if swirl_diameter_m is not None
        else load_swirl_diameter_m(cfg)
    )
    radius_m = diameter_m / 2.0
    dive_center_m = hover_pos_m.copy()
    dive_center_m[2] += z_offset_m

    print(
        f"\nSwirl water: diameter={diameter_m * 1000.0:.1f} mm, "
        f"revolutions={revolutions}, dive_z_offset={z_offset_m * 1000.0:.1f} mm"
    )

    if not move(
        redis_client,
        target_pos=dive_center_m,
        hold_orientation=hold_orientation,
        label="WATER_DIVE",
        dwell_s=dwell_at_dip_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    swirl_path = build_circle_xy_path(
        dive_center_m,
        radius_m=radius_m,
        revolutions=revolutions,
        step_m=swirl_step_m,
    )
    if not stream(
        redis_client,
        swirl_path,
        hold_orientation=hold_orientation,
        label="WATER_SWIRL",
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not move(
        redis_client,
        target_pos=dive_center_m,
        hold_orientation=hold_orientation,
        label="WATER_SWIRL_CENTER",
        dwell_s=0.0,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return move(
        redis_client,
        target_pos=hover_pos_m,
        hold_orientation=hold_orientation,
        label="WATER_INIT",
        dwell_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    )


DEFAULT_NAPKIN_BACK_AND_FORTHS = 3

NAPKIN_APPROACH_KEYS = (
    "NAPKIN_LIFT1_POS",
    "NAPKIN_S1_POS",
)


def build_napkin_wipe_sequence(
    *,
    back_and_forths: int = DEFAULT_NAPKIN_BACK_AND_FORTHS,
) -> list[tuple[str, str]]:
    """Build ordered (config_key, motion_label) pairs for a napkin wipe."""
    if back_and_forths <= 0:
        raise ValueError("back_and_forths must be positive")

    sequence: list[tuple[str, str]] = [(key, key) for key in NAPKIN_APPROACH_KEYS]

    for stroke_index in range(1, back_and_forths + 1):
        sequence.append(("NAPKIN_S2_POS", f"NAPKIN_S2_POS_STROKE{stroke_index}"))
        sequence.append(("NAPKIN_S1_POS", f"NAPKIN_S1_POS_STROKE{stroke_index}"))

    sequence.append(("NAPKIN_LIFT1_POS", "NAPKIN_LIFT1_POS_FINISH"))

    return sequence


def wipe_napkin(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    back_and_forths: int = DEFAULT_NAPKIN_BACK_AND_FORTHS,
    dwell_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    """Wipe the brush on the napkin after water swirl.

    Expects the robot at WATER_TO_NAPKIN_INIT_POS. Runs NAPKIN_LIFT1_POS ->
    NAPKIN_S1_POS, then S1/S2/S1 back-and-forth strokes, then lifts off at
    NAPKIN_LIFT1_POS.
    """
    cfg = config or load_demo_day_config()
    move = move_fn or go_to_waypoint
    key_sequence = build_napkin_wipe_sequence(
        back_and_forths=back_and_forths,
    )

    print(
        f"\nNapkin wipe: {len(key_sequence)} Cartesian waypoints "
        f"({back_and_forths} S1/S2/S1 back-and-forths)."
    )

    for key, label in key_sequence:
        target_pos = load_paint_init_pos_m(cfg, key)
        if not move(
            redis_client,
            target_pos=target_pos,
            hold_orientation=hold_orientation,
            label=label,
            dwell_s=dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
        ):
            return False

    print("Finished napkin wipe.")
    return True
