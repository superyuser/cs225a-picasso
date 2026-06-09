"""Draw a stroke JSON on the calibrated drawable canvas.

Sequence:
    INIT_POS
    -> TOOL_INIT via cached path -> dip paint 1 -> TOOL_INIT -> INIT_POS via cached reverse
    -> draw shirt_color strokes
    -> paint change to paint 2 -> draw hair_color strokes
    -> paint change to paint 3 -> draw first quarter of outline strokes
    -> INIT_POS -> TOOL_INIT via cached path -> dip paint 3 -> TOOL_INIT -> INIT_POS
    -> draw remaining outline strokes
    -> INIT_POS

The stroke JSON is expected to use the format produced by cv/run_pipeline.py.
Each stroke's ``points_mm`` coordinates are mapped onto the calibrated drawable
canvas corners in canvas_corners.json. ``points_mm`` uses (0, 0) at the bottom
left of the image plane, so it maps directly to BL/BR/TL/TR.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import redis
except ImportError:
    redis = None

from helpers.paint_change import run_paint_change_sequence
from helpers.primitives import (
    dip_paint_1,
    dip_paint_3,
    load_paint_init_pos_m,
    stream_cartesian_path,
)
from helpers.tool_station_coords import (
    STATUS_PERIOD_S,
    approach_tool_init_cartesian,
    ensure_robot_ready,
    go_to_waypoint,
    load_demo_day_config,
    load_init_pos_m,
    return_from_tool_init_cartesian,
    switch_to_cartesian_hold_current,
)
from visit_corners import (
    CANVAS_CORNER_CORRECTION_OFFSETS_M,
    CANVAS_CORNER_X_OFFSET_M,
    CORNER_ORDER,
    load_drawable_corner_world_positions,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_STROKE_JSON = DEMO_DAY_DIR / "stroke-jsons" / "20260604T220031_cartoon.json"
DEFAULT_CORNERS_JSON = DEMO_DAY_DIR / "canvas_corners.json"

LAYER_SEQUENCE = (
    ("shirt_color", 1),
    ("hair_color", 2),
    ("outline", 3),
)

DEFAULT_STROKE_MAX_STEP_M = 0.001
DEFAULT_STROKE_HOVER_X_OFFSET_M = -0.015
DEFAULT_STROKE_DWELL_S = 0.0
DEFAULT_INIT_DWELL_S = 0.25
DEFAULT_HOVER_DWELL_S = 0.1
DEFAULT_DIP_DWELL_S = 0.5
DEFAULT_MOTION_TIMEOUT_S = 120.0
DEFAULT_FIRST_OUTLINE_STROKE_SETTLE_S = 0.75


@dataclass(frozen=True)
class Stroke:
    layer_name: str
    points_mm: np.ndarray
    closed: bool
    brush_width_mm: float | None


@dataclass(frozen=True)
class StrokePlan:
    canvas_width_mm: float
    canvas_height_mm: float
    strokes_by_group: dict[str, list[Stroke]]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "stroke"


def _layer_matches(layer_name: str, group_name: str) -> bool:
    return layer_name.lower().startswith(group_name.lower())


def load_stroke_plan(stroke_json: Path) -> StrokePlan:
    with stroke_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    canvas_width_mm = float(payload["canvas_width_mm"])
    canvas_height_mm = float(payload["canvas_height_mm"])
    if canvas_width_mm <= 0.0 or canvas_height_mm <= 0.0:
        raise ValueError("canvas_width_mm and canvas_height_mm must be positive.")

    strokes_by_group = {group_name: [] for group_name, _paint in LAYER_SEQUENCE}

    for layer in payload.get("layers", []):
        layer_name = str(layer.get("name", ""))
        matched_group = next(
            (
                group_name
                for group_name, _paint in LAYER_SEQUENCE
                if _layer_matches(layer_name, group_name)
            ),
            None,
        )
        if matched_group is None:
            continue

        layer_brush_width = layer.get("brush_width_mm")
        for stroke_payload in layer.get("strokes", []):
            raw_points = stroke_payload.get("points_mm")
            if raw_points is None:
                raise ValueError(
                    f"Stroke in layer {layer_name!r} is missing points_mm."
                )

            points_mm = np.array(raw_points, dtype=float)
            if points_mm.ndim != 2 or points_mm.shape[1] != 2:
                raise ValueError(
                    f"Stroke in layer {layer_name!r} must have Nx2 points_mm."
                )
            if len(points_mm) < 2:
                continue

            brush_width = stroke_payload.get("brush_width_mm", layer_brush_width)
            strokes_by_group[matched_group].append(
                Stroke(
                    layer_name=layer_name,
                    points_mm=points_mm,
                    closed=bool(stroke_payload.get("closed", False)),
                    brush_width_mm=(
                        None if brush_width is None else float(brush_width)
                    ),
                )
            )

    return StrokePlan(
        canvas_width_mm=canvas_width_mm,
        canvas_height_mm=canvas_height_mm,
        strokes_by_group=strokes_by_group,
    )


def load_current_drawable_corners(corners_json: Path) -> tuple[dict[str, np.ndarray], str]:
    corners, source, source_has_surface_offset, source_correction_offsets = (
        load_drawable_corner_world_positions(corners_json)
    )

    x_offset_m = 0.0 if source_has_surface_offset else CANVAS_CORNER_X_OFFSET_M
    x_offset_vec = np.array([x_offset_m, 0.0, 0.0], dtype=float)
    correction_delta = {
        name: (
            CANVAS_CORNER_CORRECTION_OFFSETS_M[name]
            - source_correction_offsets[name]
        )
        for name in CORNER_ORDER
    }
    resolved = {
        name: corners[name] + x_offset_vec + correction_delta[name]
        for name in CORNER_ORDER
    }
    return resolved, source


def map_canvas_mm_to_world(
    point_mm: np.ndarray,
    *,
    corners: dict[str, np.ndarray],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> np.ndarray:
    u = float(point_mm[0]) / canvas_width_mm
    v = float(point_mm[1]) / canvas_height_mm

    bottom = (1.0 - u) * corners["BL"] + u * corners["BR"]
    top = (1.0 - u) * corners["TL"] + u * corners["TR"]
    return (1.0 - v) * bottom + v * top


def stroke_points_to_world(
    stroke: Stroke,
    *,
    corners: dict[str, np.ndarray],
    canvas_width_mm: float,
    canvas_height_mm: float,
) -> np.ndarray:
    points_mm = stroke.points_mm
    if stroke.closed and not np.allclose(points_mm[0], points_mm[-1]):
        points_mm = np.vstack([points_mm, points_mm[0]])

    points_world = [
        map_canvas_mm_to_world(
            point,
            corners=corners,
            canvas_width_mm=canvas_width_mm,
            canvas_height_mm=canvas_height_mm,
        )
        for point in points_mm
    ]
    return np.array(points_world, dtype=float)


def densify_path(points: np.ndarray, *, max_step_m: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive.")

    samples = [np.array(points[0], dtype=float)]
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-9:
            continue
        steps = max(1, int(math.ceil(distance / max_step_m)))
        for index in range(1, steps + 1):
            samples.append(start + delta * (index / steps))
    return np.array(samples, dtype=float)


def plan_group_paths(
    strokes: list[Stroke],
    *,
    corners: dict[str, np.ndarray],
    canvas_width_mm: float,
    canvas_height_mm: float,
    max_step_m: float,
) -> list[np.ndarray]:
    paths = []
    for stroke in strokes:
        path = stroke_points_to_world(
            stroke,
            corners=corners,
            canvas_width_mm=canvas_width_mm,
            canvas_height_mm=canvas_height_mm,
        )
        dense = densify_path(path, max_step_m=max_step_m)
        if len(dense) >= 2:
            paths.append(dense)
    return paths


def print_plan_summary(
    plan: StrokePlan,
    *,
    corners: dict[str, np.ndarray],
    corner_source: str,
    max_step_m: float,
    max_strokes_per_layer: int | None,
) -> None:
    print("Stroke JSON canvas:", f"{plan.canvas_width_mm:.1f} x {plan.canvas_height_mm:.1f} mm")
    print("Drawable corner source:", corner_source)
    print("Drawable corners (world frame, meters):")
    for name in CORNER_ORDER:
        print(f"  {name}: {np.round(corners[name], 5).tolist()}")

    for group_name, paint_number in LAYER_SEQUENCE:
        selected_strokes = (
            plan.strokes_by_group[group_name]
            if max_strokes_per_layer is None
            else plan.strokes_by_group[group_name][:max_strokes_per_layer]
        )
        paths = plan_group_paths(
            selected_strokes,
            corners=corners,
            canvas_width_mm=plan.canvas_width_mm,
            canvas_height_mm=plan.canvas_height_mm,
            max_step_m=max_step_m,
        )
        sample_count = sum(len(path) for path in paths)
        limit_note = "" if max_strokes_per_layer is None else " selected"
        print(
            f"Group {group_name} -> paint {paint_number}: "
            f"{len(selected_strokes)}{limit_note} strokes, "
            f"{sample_count} streamed samples"
        )


def move_to_init(
    redis_client,
    *,
    init_pos: np.ndarray,
    hold_orientation: np.ndarray,
    dwell_s: float,
    timeout_s: float,
    status_period_s: float,
) -> bool:
    return go_to_waypoint(
        redis_client,
        target_pos=init_pos,
        hold_orientation=hold_orientation,
        label="INIT_POS",
        dwell_s=dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    )


def dip_initial_paint_1(
    redis_client,
    *,
    config: dict[str, Any],
    hover_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
) -> bool:
    tool_init_pose = approach_tool_init_cartesian(
        redis_client,
        dwell_at_tool_init_s=hover_dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        use_cached_path=use_cached_tool_init_path,
        rebuild_path=rebuild_tool_init_path,
        save_path=rebuild_tool_init_path,
        require_cached_path=use_cached_tool_init_path and not rebuild_tool_init_path,
    )
    if tool_init_pose is None:
        return False

    tool_init_position, hold_orientation = tool_init_pose

    p1_hover = load_paint_init_pos_m(config, "P1_INIT_POS")
    if not go_to_waypoint(
        redis_client,
        target_pos=p1_hover,
        hold_orientation=hold_orientation,
        label="P1_INIT_POS",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not dip_paint_1(
        redis_client,
        hold_orientation=hold_orientation,
        config=config,
        dwell_at_dip_s=dip_dwell_s,
        dwell_at_hover_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not go_to_waypoint(
        redis_client,
        target_pos=tool_init_position,
        hold_orientation=hold_orientation,
        label="TOOL_INIT",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return return_from_tool_init_cartesian(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    )


def dip_same_paint_3(
    redis_client,
    *,
    config: dict[str, Any],
    hover_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
) -> bool:
    tool_init_pose = approach_tool_init_cartesian(
        redis_client,
        dwell_at_tool_init_s=hover_dwell_s,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
        use_cached_path=use_cached_tool_init_path,
        rebuild_path=rebuild_tool_init_path,
        save_path=rebuild_tool_init_path,
        require_cached_path=use_cached_tool_init_path and not rebuild_tool_init_path,
    )
    if tool_init_pose is None:
        return False

    tool_init_position, hold_orientation = tool_init_pose

    p3_hover = load_paint_init_pos_m(config, "P3_INIT_POS")
    if not go_to_waypoint(
        redis_client,
        target_pos=p3_hover,
        hold_orientation=hold_orientation,
        label="P3_INIT_POS_REDIP",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not dip_paint_3(
        redis_client,
        hold_orientation=hold_orientation,
        config=config,
        dwell_at_dip_s=dip_dwell_s,
        dwell_at_hover_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not go_to_waypoint(
        redis_client,
        target_pos=tool_init_position,
        hold_orientation=hold_orientation,
        label="TOOL_INIT_AFTER_P3_REDIP",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return return_from_tool_init_cartesian(
        redis_client,
        status_period_s=status_period_s,
        timeout_s=timeout_s,
    )


def prepare_initial_paint_1(
    redis_client,
    *,
    config: dict[str, Any],
    init_pos: np.ndarray,
    init_dwell_s: float,
    hover_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
) -> np.ndarray | None:
    _current_position, hold_orientation = switch_to_cartesian_hold_current(redis_client)
    if not move_to_init(
        redis_client,
        init_pos=init_pos,
        hold_orientation=hold_orientation,
        dwell_s=init_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return None

    if not dip_initial_paint_1(
        redis_client,
        config=config,
        hover_dwell_s=hover_dwell_s,
        dip_dwell_s=dip_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        use_cached_tool_init_path=use_cached_tool_init_path,
        rebuild_tool_init_path=rebuild_tool_init_path,
    ):
        return None

    _current_position, hold_orientation = switch_to_cartesian_hold_current(redis_client)
    if not move_to_init(
        redis_client,
        init_pos=init_pos,
        hold_orientation=hold_orientation,
        dwell_s=init_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return None

    return hold_orientation


def draw_stroke_path(
    redis_client,
    path: np.ndarray,
    *,
    hold_orientation: np.ndarray,
    label: str,
    hover_x_offset_m: float,
    hover_dwell_s: float,
    stroke_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
) -> bool:
    hover_offset = np.array([hover_x_offset_m, 0.0, 0.0], dtype=float)
    hover_start = path[0] + hover_offset
    hover_end = path[-1] + hover_offset

    if not go_to_waypoint(
        redis_client,
        target_pos=hover_start,
        hold_orientation=hold_orientation,
        label=f"{label}_HOVER_START",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not go_to_waypoint(
        redis_client,
        target_pos=path[0],
        hold_orientation=hold_orientation,
        label=f"{label}_TOUCH",
        dwell_s=stroke_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    if not stream_cartesian_path(
        redis_client,
        path,
        hold_orientation=hold_orientation,
        label=label,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return go_to_waypoint(
        redis_client,
        target_pos=hover_end,
        hold_orientation=hold_orientation,
        label=f"{label}_LIFT",
        dwell_s=hover_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    )


def draw_group(
    redis_client,
    group_name: str,
    strokes: list[Stroke],
    *,
    corners: dict[str, np.ndarray],
    canvas_width_mm: float,
    canvas_height_mm: float,
    hold_orientation: np.ndarray,
    max_step_m: float,
    hover_x_offset_m: float,
    hover_dwell_s: float,
    stroke_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    max_strokes: int | None,
    first_stroke_settle_s: float,
    stroke_index_offset: int = 0,
) -> bool:
    selected_strokes = strokes if max_strokes is None else strokes[:max_strokes]
    print(f"\nDrawing {group_name}: {len(selected_strokes)} strokes.")

    for local_stroke_index, stroke in enumerate(selected_strokes, start=1):
        stroke_index = stroke_index_offset + local_stroke_index
        path = stroke_points_to_world(
            stroke,
            corners=corners,
            canvas_width_mm=canvas_width_mm,
            canvas_height_mm=canvas_height_mm,
        )
        path = densify_path(path, max_step_m=max_step_m)
        if len(path) < 2:
            continue

        label = f"{_slug(group_name).upper()}_{stroke_index:03d}"
        print(
            f"  {label}: layer={stroke.layer_name}, "
            f"samples={len(path)}, brush={stroke.brush_width_mm} mm"
        )
        this_hover_dwell_s = hover_dwell_s
        this_stroke_dwell_s = stroke_dwell_s
        if local_stroke_index == 1 and first_stroke_settle_s > 0.0:
            print(
                f"    first-stroke settle: {first_stroke_settle_s:.2f}s "
                "at hover and touch"
            )
            this_hover_dwell_s += first_stroke_settle_s
            this_stroke_dwell_s = max(this_stroke_dwell_s, first_stroke_settle_s)
        if not draw_stroke_path(
            redis_client,
            path,
            hold_orientation=hold_orientation,
            label=label,
            hover_x_offset_m=hover_x_offset_m,
            hover_dwell_s=this_hover_dwell_s,
            stroke_dwell_s=this_stroke_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
        ):
            return False

    return True


def draw_outline_with_mid_redip(
    redis_client,
    strokes: list[Stroke],
    *,
    config: dict[str, Any],
    init_pos: np.ndarray,
    corners: dict[str, np.ndarray],
    canvas_width_mm: float,
    canvas_height_mm: float,
    hold_orientation: np.ndarray,
    max_step_m: float,
    hover_x_offset_m: float,
    init_dwell_s: float,
    hover_dwell_s: float,
    stroke_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    max_strokes: int | None,
    first_stroke_settle_s: float,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
) -> bool:
    selected_strokes = strokes if max_strokes is None else strokes[:max_strokes]
    if len(selected_strokes) <= 1:
        return draw_group(
            redis_client,
            "outline",
            selected_strokes,
            corners=corners,
            canvas_width_mm=canvas_width_mm,
            canvas_height_mm=canvas_height_mm,
            hold_orientation=hold_orientation,
            max_step_m=max_step_m,
            hover_x_offset_m=hover_x_offset_m,
            hover_dwell_s=hover_dwell_s,
            stroke_dwell_s=stroke_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
            max_strokes=None,
            first_stroke_settle_s=first_stroke_settle_s,
        )

    split_index = max(1, int(math.ceil(len(selected_strokes) / 4.0)))
    first_batch = selected_strokes[:split_index]
    remaining_batch = selected_strokes[split_index:]
    print(
        "\nOutline mid-run paint refresh: "
        f"{len(first_batch)} strokes, P3 re-dip, then "
        f"{len(remaining_batch)} remaining strokes."
    )

    if not draw_group(
        redis_client,
        "outline",
        first_batch,
        corners=corners,
        canvas_width_mm=canvas_width_mm,
        canvas_height_mm=canvas_height_mm,
        hold_orientation=hold_orientation,
        max_step_m=max_step_m,
        hover_x_offset_m=hover_x_offset_m,
        hover_dwell_s=hover_dwell_s,
        stroke_dwell_s=stroke_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        max_strokes=None,
        first_stroke_settle_s=first_stroke_settle_s,
    ):
        return False

    if not move_to_init(
        redis_client,
        init_pos=init_pos,
        hold_orientation=hold_orientation,
        dwell_s=init_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    print("\nRe-dipping paint 3 before latter half of outline strokes.")
    if not dip_same_paint_3(
        redis_client,
        config=config,
        hover_dwell_s=hover_dwell_s,
        dip_dwell_s=dip_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        use_cached_tool_init_path=use_cached_tool_init_path,
        rebuild_tool_init_path=rebuild_tool_init_path,
    ):
        return False

    _current_position, _current_hold_orientation = switch_to_cartesian_hold_current(
        redis_client
    )
    if not move_to_init(
        redis_client,
        init_pos=init_pos,
        hold_orientation=hold_orientation,
        dwell_s=init_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return False

    return draw_group(
        redis_client,
        "outline",
        remaining_batch,
        corners=corners,
        canvas_width_mm=canvas_width_mm,
        canvas_height_mm=canvas_height_mm,
        hold_orientation=hold_orientation,
        max_step_m=max_step_m,
        hover_x_offset_m=hover_x_offset_m,
        hover_dwell_s=hover_dwell_s,
        stroke_dwell_s=stroke_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        max_strokes=None,
        first_stroke_settle_s=first_stroke_settle_s,
        stroke_index_offset=len(first_batch),
    )


def run(
    *,
    stroke_json: Path,
    corners_json: Path,
    dry_run: bool,
    max_step_m: float,
    hover_x_offset_m: float,
    init_dwell_s: float,
    hover_dwell_s: float,
    stroke_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    max_strokes_per_layer: int | None,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
    skip_initial_dip: bool,
    first_outline_stroke_settle_s: float,
) -> int:
    if not use_cached_tool_init_path and not rebuild_tool_init_path:
        print(
            "Drawing requires saved TOOL_INIT paths for INIT_POS <-> TOOL_INIT "
            "transitions. Use the cached path, or pass --rebuild-tool-init-path "
            "to regenerate and save it."
        )
        return 1

    if not stroke_json.is_file():
        print(f"Stroke JSON not found: {stroke_json}")
        return 1
    if not corners_json.is_file():
        print(f"Corners JSON not found: {corners_json}")
        print("Run robot/calibrate_canvas.py first.")
        return 1

    plan = load_stroke_plan(stroke_json)
    drawable_corners, corner_source = load_current_drawable_corners(corners_json)
    print_plan_summary(
        plan,
        corners=drawable_corners,
        corner_source=corner_source,
        max_step_m=max_step_m,
        max_strokes_per_layer=max_strokes_per_layer,
    )

    if dry_run:
        print("\nDry run only; no robot commands were sent.")
        return 0

    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    cfg = load_demo_day_config()
    init_pos = load_init_pos_m(cfg)

    if skip_initial_dip:
        print("\nSkipping initial paint-1 dip; assuming brush is already loaded.")
        _current_position, hold_orientation = switch_to_cartesian_hold_current(redis_client)
        if not move_to_init(
            redis_client,
            init_pos=init_pos,
            hold_orientation=hold_orientation,
            dwell_s=init_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
        ):
            return 1
    else:
        hold_orientation = prepare_initial_paint_1(
            redis_client,
            config=cfg,
            init_pos=init_pos,
            init_dwell_s=init_dwell_s,
            hover_dwell_s=hover_dwell_s,
            dip_dwell_s=dip_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
            use_cached_tool_init_path=use_cached_tool_init_path,
            rebuild_tool_init_path=rebuild_tool_init_path,
        )
        if hold_orientation is None:
            return 1

    drawing_hold_orientation = hold_orientation.copy()
    print("\nLocked drawing orientation for all paint layers:")
    print(drawing_hold_orientation)

    if not draw_group(
        redis_client,
        "shirt_color",
        plan.strokes_by_group["shirt_color"],
        corners=drawable_corners,
        canvas_width_mm=plan.canvas_width_mm,
        canvas_height_mm=plan.canvas_height_mm,
        hold_orientation=drawing_hold_orientation,
        max_step_m=max_step_m,
        hover_x_offset_m=hover_x_offset_m,
        hover_dwell_s=hover_dwell_s,
        stroke_dwell_s=stroke_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        max_strokes=max_strokes_per_layer,
        first_stroke_settle_s=0.0,
    ):
        return 1

    if not move_to_init(
        redis_client,
        init_pos=init_pos,
        hold_orientation=drawing_hold_orientation,
        dwell_s=init_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
    ):
        return 1

    for group_name, paint_number in (("hair_color", 2), ("outline", 3)):
        if not run_paint_change_sequence(
            redis_client,
            paint_number,
            status_period_s=status_period_s,
            timeout_s=timeout_s,
            use_cached_tool_init_path=use_cached_tool_init_path,
            rebuild_tool_init_path=rebuild_tool_init_path,
        ):
            return 1

        _current_position, _current_hold_orientation = switch_to_cartesian_hold_current(
            redis_client
        )
        if not move_to_init(
            redis_client,
            init_pos=init_pos,
            hold_orientation=drawing_hold_orientation,
            dwell_s=init_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
        ):
            return 1

        if group_name == "outline":
            if not draw_outline_with_mid_redip(
                redis_client,
                plan.strokes_by_group[group_name],
                config=cfg,
                init_pos=init_pos,
                corners=drawable_corners,
                canvas_width_mm=plan.canvas_width_mm,
                canvas_height_mm=plan.canvas_height_mm,
                hold_orientation=drawing_hold_orientation,
                max_step_m=max_step_m,
                hover_x_offset_m=hover_x_offset_m,
                init_dwell_s=init_dwell_s,
                hover_dwell_s=hover_dwell_s,
                stroke_dwell_s=stroke_dwell_s,
                dip_dwell_s=dip_dwell_s,
                timeout_s=timeout_s,
                status_period_s=status_period_s,
                max_strokes=max_strokes_per_layer,
                first_stroke_settle_s=first_outline_stroke_settle_s,
                use_cached_tool_init_path=use_cached_tool_init_path,
                rebuild_tool_init_path=rebuild_tool_init_path,
            ):
                return 1
        elif not draw_group(
            redis_client,
            group_name,
            plan.strokes_by_group[group_name],
            corners=drawable_corners,
            canvas_width_mm=plan.canvas_width_mm,
            canvas_height_mm=plan.canvas_height_mm,
            hold_orientation=drawing_hold_orientation,
            max_step_m=max_step_m,
            hover_x_offset_m=hover_x_offset_m,
            hover_dwell_s=hover_dwell_s,
            stroke_dwell_s=stroke_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
            max_strokes=max_strokes_per_layer,
            first_stroke_settle_s=0.0,
        ):
            return 1

        if not move_to_init(
            redis_client,
            init_pos=init_pos,
            hold_orientation=drawing_hold_orientation,
            dwell_s=init_dwell_s,
            timeout_s=timeout_s,
            status_period_s=status_period_s,
        ):
            return 1

    print("\nFinished drawing stroke JSON.")
    return 0


def run_initial_dip_only(
    *,
    dry_run: bool,
    init_dwell_s: float,
    hover_dwell_s: float,
    dip_dwell_s: float,
    timeout_s: float,
    status_period_s: float,
    use_cached_tool_init_path: bool,
    rebuild_tool_init_path: bool,
) -> int:
    if not use_cached_tool_init_path and not rebuild_tool_init_path:
        print(
            "Initial paint dip requires saved TOOL_INIT paths for INIT_POS <-> "
            "TOOL_INIT transitions. Use the cached path, or pass "
            "--rebuild-tool-init-path to regenerate and save it."
        )
        return 1

    if dry_run:
        print("Dry run only; initial paint dip was not executed.")
        return 0

    if redis is None:
        print("`redis` package is not installed.")
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    cfg = load_demo_day_config()
    init_pos = load_init_pos_m(cfg)
    hold_orientation = prepare_initial_paint_1(
        redis_client,
        config=cfg,
        init_pos=init_pos,
        init_dwell_s=init_dwell_s,
        hover_dwell_s=hover_dwell_s,
        dip_dwell_s=dip_dwell_s,
        timeout_s=timeout_s,
        status_period_s=status_period_s,
        use_cached_tool_init_path=use_cached_tool_init_path,
        rebuild_tool_init_path=rebuild_tool_init_path,
    )
    if hold_orientation is None:
        return 1

    print("\nFinished initial paint-1 dip and returned to INIT_POS.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stroke_json",
        nargs="?",
        default=str(DEFAULT_STROKE_JSON),
        help=f"Stroke JSON to draw (default: {DEFAULT_STROKE_JSON}).",
    )
    parser.add_argument(
        "--corners-json",
        default=str(DEFAULT_CORNERS_JSON),
        help=f"Canvas calibration JSON (default: {DEFAULT_CORNERS_JSON}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and map the strokes, but do not send robot commands.",
    )
    parser.add_argument(
        "--initial-dip-only",
        action="store_true",
        help="Only dip paint 1 and return to INIT_POS; do not require or draw a stroke JSON.",
    )
    parser.add_argument(
        "--skip-initial-dip",
        action="store_true",
        help="Start drawing assuming paint 1 was already dipped and the robot is at INIT_POS.",
    )
    parser.add_argument(
        "--max-step-m",
        type=float,
        default=DEFAULT_STROKE_MAX_STEP_M,
        help=f"Max streamed distance between stroke samples (default: {DEFAULT_STROKE_MAX_STEP_M}).",
    )
    parser.add_argument(
        "--hover-x-offset-m",
        type=float,
        default=DEFAULT_STROKE_HOVER_X_OFFSET_M,
        help=(
            "World X offset used for stroke approach/lift. Negative retracts "
            f"from the canvas (default: {DEFAULT_STROKE_HOVER_X_OFFSET_M})."
        ),
    )
    parser.add_argument(
        "--init-dwell-s",
        type=float,
        default=DEFAULT_INIT_DWELL_S,
        help=f"Seconds to dwell at INIT_POS (default: {DEFAULT_INIT_DWELL_S}).",
    )
    parser.add_argument(
        "--hover-dwell-s",
        type=float,
        default=DEFAULT_HOVER_DWELL_S,
        help=f"Seconds to dwell at stroke/paint hover points (default: {DEFAULT_HOVER_DWELL_S}).",
    )
    parser.add_argument(
        "--stroke-dwell-s",
        type=float,
        default=DEFAULT_STROKE_DWELL_S,
        help=f"Seconds to dwell at stroke contact start (default: {DEFAULT_STROKE_DWELL_S}).",
    )
    parser.add_argument(
        "--dip-dwell-s",
        type=float,
        default=DEFAULT_DIP_DWELL_S,
        help=f"Seconds to dwell at paint dip contact (default: {DEFAULT_DIP_DWELL_S}).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_MOTION_TIMEOUT_S,
        help=f"Motion timeout per segment (default: {DEFAULT_MOTION_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--status-period-s",
        type=float,
        default=STATUS_PERIOD_S,
        help=f"Status print period in seconds (default: {STATUS_PERIOD_S}).",
    )
    parser.add_argument(
        "--max-strokes-per-layer",
        type=int,
        default=None,
        help="Limit strokes per layer for testing.",
    )
    parser.add_argument(
        "--first-outline-stroke-settle-s",
        type=float,
        default=DEFAULT_FIRST_OUTLINE_STROKE_SETTLE_S,
        help=(
            "Extra hover/touch settle before the first outline stroke, useful "
            "after the paint-3 change (default: "
            f"{DEFAULT_FIRST_OUTLINE_STROKE_SETTLE_S})."
        ),
    )
    parser.add_argument(
        "--rebuild-tool-init-path",
        action="store_true",
        help="Rebuild cached tool-init paths during initial dip and paint changes.",
    )
    parser.add_argument(
        "--no-cached-tool-init-path",
        action="store_true",
        help="Do not use cached tool-init paths during initial dip and paint changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.initial_dip_only:
        return run_initial_dip_only(
            dry_run=args.dry_run,
            init_dwell_s=args.init_dwell_s,
            hover_dwell_s=args.hover_dwell_s,
            dip_dwell_s=args.dip_dwell_s,
            timeout_s=args.timeout_s,
            status_period_s=args.status_period_s,
            use_cached_tool_init_path=not args.no_cached_tool_init_path,
            rebuild_tool_init_path=args.rebuild_tool_init_path,
        )

    return run(
        stroke_json=Path(args.stroke_json),
        corners_json=Path(args.corners_json),
        dry_run=args.dry_run,
        max_step_m=args.max_step_m,
        hover_x_offset_m=args.hover_x_offset_m,
        init_dwell_s=args.init_dwell_s,
        hover_dwell_s=args.hover_dwell_s,
        stroke_dwell_s=args.stroke_dwell_s,
        dip_dwell_s=args.dip_dwell_s,
        timeout_s=args.timeout_s,
        status_period_s=args.status_period_s,
        max_strokes_per_layer=args.max_strokes_per_layer,
        use_cached_tool_init_path=not args.no_cached_tool_init_path,
        rebuild_tool_init_path=args.rebuild_tool_init_path,
        skip_initial_dip=args.skip_initial_dip,
        first_outline_stroke_settle_s=args.first_outline_stroke_settle_s,
    )


if __name__ == "__main__":
    sys.exit(main())
