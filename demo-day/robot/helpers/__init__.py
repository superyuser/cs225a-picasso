"""Shared helpers for demo-day robot entrypoint scripts."""

from .model_tool_station import PaintToolStationModel, pick_reference_tag_id
from .tool_station_coords import (
    ToolStationCalibration,
    approach_tool_init_cartesian,
    compute_tool_station_calibration,
    run_volume_visit_from_json,
    visit_tool_volume_waypoints,
)
from .joint_motion import (
    ToolInitApproachResult,
    approach_tool_init,
    load_tool_init_joint_path,
    return_from_tool_init,
    save_tool_init_joint_path,
)
from .paint_change import (
    PaintChangeStep,
    PaintChangeStepKind,
    build_paint_change_sequence,
    normalize_paint_number,
    run_paint_change_sequence,
)
from .primitives import dip_paint_1, dip_paint_2, dip_paint_3, swirl_water, wipe_napkin

__all__ = [
    "PaintToolStationModel",
    "ToolStationCalibration",
    "ToolInitApproachResult",
    "PaintChangeStep",
    "PaintChangeStepKind",
    "approach_tool_init",
    "approach_tool_init_cartesian",
    "build_paint_change_sequence",
    "compute_tool_station_calibration",
    "dip_paint_1",
    "dip_paint_2",
    "dip_paint_3",
    "load_tool_init_joint_path",
    "normalize_paint_number",
    "return_from_tool_init",
    "run_paint_change_sequence",
    "save_tool_init_joint_path",
    "swirl_water",
    "wipe_napkin",
    "pick_reference_tag_id",
    "run_volume_visit_from_json",
    "visit_tool_volume_waypoints",
]
