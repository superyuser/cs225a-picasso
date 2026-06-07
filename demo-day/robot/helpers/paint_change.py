"""Paint-change sequence builder and runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .joint_motion import (
    JOINT_ARRIVAL_THRESHOLD,
    JOINT_MAX_STEP_DEG,
    TIMEOUT_S,
    ToolInitApproachResult,
    approach_tool_init,
    follow_joint_path_to_goal,
    joint_redis_keys,
    load_demo_day_config,
    load_tool_init_rad,
    return_from_tool_init,
)
from .primitives import (
    dip_paint_1,
    dip_paint_2,
    dip_paint_3,
    load_paint_init_pos_m,
    swirl_water,
    wipe_napkin,
)
from .tool_station_coords import (
    CARTESIAN_SETTLE_S,
    POS_TOL_M,
    go_to_waypoint,
    switch_to_cartesian_hold_current,
)

DEFAULT_HOVER_DWELL_S = 0.25
DEFAULT_DIP_DWELL_S = 0.5
STATUS_PERIOD_S = 0.25


DIP_PAINT_BY_NUMBER: dict[int, Callable[..., bool]] = {
    1: dip_paint_1,
    2: dip_paint_2,
    3: dip_paint_3,
}

PAINT_INIT_CONFIG_KEY: dict[int, str] = {
    1: "P1_INIT_POS",
    2: "P2_INIT_POS",
    3: "P3_INIT_POS",
}


class PaintChangeStepKind(str, Enum):
    APPROACH_TOOL_INIT = "approach_tool_init"
    CARTESIAN_GOTO = "cartesian_goto"
    SWIRL_WATER = "swirl_water"
    WIPE_NAPKIN = "wipe_napkin"
    DIP_PAINT = "dip_paint"
    GOTO_TOOL_INIT = "goto_tool_init"
    RETURN_HOME = "return_home"


@dataclass(frozen=True)
class PaintChangeStep:
    kind: PaintChangeStepKind
    config_key: str | None = None
    paint_number: int | None = None


def normalize_paint_number(paint_number: int | str) -> int:
    if isinstance(paint_number, str):
        normalized = paint_number.strip().lower()
        if normalized.startswith("p"):
            normalized = normalized[1:]
        paint_number = int(normalized)

    paint_number = int(paint_number)
    if paint_number not in DIP_PAINT_BY_NUMBER:
        raise ValueError("paint_number must be 1, 2, or 3")
    return paint_number


def build_paint_change_sequence(paint_number: int | str) -> list[PaintChangeStep]:
    """Build the ordered paint-change routine for one paint cup."""
    paint = normalize_paint_number(paint_number)
    paint_config_key = PAINT_INIT_CONFIG_KEY[paint]

    return [
        PaintChangeStep(PaintChangeStepKind.APPROACH_TOOL_INIT),
        PaintChangeStep(
            PaintChangeStepKind.CARTESIAN_GOTO,
            config_key="WATER_INIT_POS",
        ),
        PaintChangeStep(PaintChangeStepKind.SWIRL_WATER),
        PaintChangeStep(
            PaintChangeStepKind.CARTESIAN_GOTO,
            config_key="WATER_TO_NAPKIN_INIT_POS",
        ),
        PaintChangeStep(PaintChangeStepKind.WIPE_NAPKIN),
        PaintChangeStep(
            PaintChangeStepKind.CARTESIAN_GOTO,
            config_key=paint_config_key,
        ),
        PaintChangeStep(
            PaintChangeStepKind.DIP_PAINT,
            paint_number=paint,
        ),
        PaintChangeStep(PaintChangeStepKind.GOTO_TOOL_INIT),
        PaintChangeStep(PaintChangeStepKind.RETURN_HOME),
    ]


def _prepare_cartesian_motion(
    redis_client,
    *,
    status_period_s: float,
) -> np.ndarray:
    _position, hold_orientation = switch_to_cartesian_hold_current(
        redis_client,
        settle_s=CARTESIAN_SETTLE_S,
    )
    return hold_orientation


def run_paint_change_sequence(
    redis_client,
    paint_number: int | str,
    *,
    pos_tol_m: float = POS_TOL_M,
    hover_dwell_s: float = DEFAULT_HOVER_DWELL_S,
    dip_dwell_s: float = DEFAULT_DIP_DWELL_S,
    joint_arrival_threshold: float = JOINT_ARRIVAL_THRESHOLD,
    joint_max_step_deg: float = JOINT_MAX_STEP_DEG,
    status_period_s: float = STATUS_PERIOD_S,
    timeout_s: float = TIMEOUT_S,
    use_cached_tool_init_path: bool = True,
    rebuild_tool_init_path: bool = False,
) -> bool:
    """Execute a full paint change starting from the robot home position."""
    paint = normalize_paint_number(paint_number)
    steps = build_paint_change_sequence(paint)
    cfg = load_demo_day_config()
    approach: ToolInitApproachResult | None = None
    hold_orientation: np.ndarray | None = None
    cartesian_ready = False

    print(f"\nPaint change sequence for paint {paint}:")
    for index, step in enumerate(steps, start=1):
        print(f"  {index}. {step.kind.value}", end="")
        if step.config_key:
            print(f" ({step.config_key})", end="")
        if step.paint_number is not None:
            print(f" (paint {step.paint_number})", end="")
        print()

    for step in steps:
        if step.kind == PaintChangeStepKind.APPROACH_TOOL_INIT:
            approach = approach_tool_init(
                redis_client,
                joint_arrival_threshold=joint_arrival_threshold,
                joint_max_step_deg=joint_max_step_deg,
                status_period_s=status_period_s,
                timeout_s=timeout_s,
                use_cached_path=use_cached_tool_init_path,
                rebuild_path=rebuild_tool_init_path,
            )
            if approach is None:
                return False
            continue

        if not cartesian_ready:
            hold_orientation = _prepare_cartesian_motion(
                redis_client,
                status_period_s=status_period_s,
            )
            cartesian_ready = True

        assert hold_orientation is not None

        if step.kind == PaintChangeStepKind.CARTESIAN_GOTO:
            assert step.config_key is not None
            target_pos = load_paint_init_pos_m(cfg, step.config_key)
            if not go_to_waypoint(
                redis_client,
                target_pos=target_pos,
                hold_orientation=hold_orientation,
                label=step.config_key,
                dwell_s=hover_dwell_s,
                timeout_s=timeout_s,
                status_period_s=status_period_s,
            ):
                return False
            continue

        if step.kind == PaintChangeStepKind.SWIRL_WATER:
            if not swirl_water(
                redis_client,
                hold_orientation=hold_orientation,
                config=cfg,
                dwell_at_dip_s=dip_dwell_s,
                dwell_at_hover_s=hover_dwell_s,
                timeout_s=timeout_s,
                status_period_s=status_period_s,
            ):
                return False
            continue

        if step.kind == PaintChangeStepKind.WIPE_NAPKIN:
            if not wipe_napkin(
                redis_client,
                hold_orientation=hold_orientation,
                config=cfg,
                dwell_s=hover_dwell_s,
                timeout_s=timeout_s,
                status_period_s=status_period_s,
            ):
                return False
            continue

        if step.kind == PaintChangeStepKind.DIP_PAINT:
            assert step.paint_number is not None
            dip_fn = DIP_PAINT_BY_NUMBER[step.paint_number]
            if not dip_fn(
                redis_client,
                hold_orientation=hold_orientation,
                config=cfg,
                dwell_at_dip_s=dip_dwell_s,
                dwell_at_hover_s=hover_dwell_s,
                timeout_s=timeout_s,
                status_period_s=status_period_s,
            ):
                return False
            continue

        if step.kind == PaintChangeStepKind.GOTO_TOOL_INIT:
            tool_init_rad = (
                approach.tool_init_rad if approach is not None else load_tool_init_rad()
            )
            current_joint = read_np(
                redis_client,
                joint_redis_keys.sensor_joint_positions,
                (7,),
            )
            max_joint_step = max(abs(joint_max_step_deg) * (np.pi / 180.0), 1.0e-5)
            if not follow_joint_path_to_goal(
                redis_client,
                start_joint_position=current_joint,
                goal_joint_position=tool_init_rad,
                label="TOOL_INIT",
                max_joint_step=max_joint_step,
                joint_arrival_threshold=joint_arrival_threshold,
                status_period_s=status_period_s,
                timeout_s=timeout_s,
            ):
                return False
            cartesian_ready = False
            continue

        if step.kind == PaintChangeStepKind.RETURN_HOME:
            if approach is None:
                print("Missing approach path; cannot return home.")
                return False
            if not return_from_tool_init(
                redis_client,
                approach,
                joint_arrival_threshold=joint_arrival_threshold,
                status_period_s=status_period_s,
                timeout_s=timeout_s,
                use_cached_path=use_cached_tool_init_path,
            ):
                return False
            continue

    print(f"\nFinished paint change for paint {paint}.")
    return True
