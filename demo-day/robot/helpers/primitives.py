"""Reusable Cartesian motion primitives for the tool station."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .tool_station_coords import go_to_waypoint

HELPERS_DIR = Path(__file__).resolve().parent
ROBOT_DIR = HELPERS_DIR.parent
DEMO_DAY_DIR = ROBOT_DIR.parent
DEMO_DAY_CONFIG_PATH = DEMO_DAY_DIR / "config.json"

MM_TO_M = 1.0e-3
DEFAULT_Z_DIVE_IN_OFFSET_MM = -86.0
DEFAULT_DIP_DWELL_S = 0.5
DEFAULT_HOVER_DWELL_S = 0.25


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
    return pos_mm * MM_TO_M


def load_z_dive_in_offset_m(config: dict | None = None) -> float:
    cfg = config or load_demo_day_config()
    offset_mm = float(cfg.get("z_dive_in_offset_mm", DEFAULT_Z_DIVE_IN_OFFSET_MM))
    return offset_mm * MM_TO_M


MoveFn = Callable[..., bool]


def dip_at_hover(
    redis_client,
    hover_pos_m: np.ndarray,
    *,
    hold_orientation: np.ndarray,
    label: str,
    z_dive_offset_m: float,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    """Dip cycle: hover -> hover + z offset -> hover."""
    move = move_fn or go_to_waypoint
    dipped_pos = hover_pos_m.copy()
    dipped_pos[2] += z_dive_offset_m

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
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    cfg = config or load_demo_day_config()
    hover_pos_m = load_paint_init_pos_m(cfg, config_key)
    z_offset_m = (
        z_dive_offset_m
        if z_dive_offset_m is not None
        else load_z_dive_in_offset_m(cfg)
    )
    return dip_at_hover(
        redis_client,
        hover_pos_m,
        hold_orientation=hold_orientation,
        label=label,
        z_dive_offset_m=z_offset_m,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    )


def dip_paint_1(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    """P1_INIT_POS -> P1_INIT_POS + z_dive_in_offset -> P1_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P1_INIT_POS",
        "P1_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    )


def dip_paint_2(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    """P2_INIT_POS -> P2_INIT_POS + z_dive_in_offset -> P2_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P2_INIT_POS",
        "P2_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    )


def dip_paint_3(
    redis_client,
    *,
    hold_orientation: np.ndarray,
    config: dict | None = None,
    z_dive_offset_m: float | None = None,
    dwell_at_dip_s: float = DEFAULT_DIP_DWELL_S,
    dwell_at_hover_s: float = DEFAULT_HOVER_DWELL_S,
    timeout_s: float,
    status_period_s: float,
    move_fn: MoveFn | None = None,
) -> bool:
    """P3_INIT_POS -> P3_INIT_POS + z_dive_in_offset -> P3_INIT_POS."""
    return _dip_paint(
        redis_client,
        "P3_INIT_POS",
        "P3_INIT",
        hold_orientation=hold_orientation,
        config=config,
        z_dive_offset_m=z_dive_offset_m,
        dwell_at_dip_s=dwell_at_dip_s,
        dwell_at_hover_s=dwell_at_hover_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        move_fn=move_fn,
    )
