import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_STROKES_JSON = "strokes.json"
OUTPUT_STROKES_JSON = "strokes_canvas_plane.json"
CALIBRATION_JSON = Path("robot") / "canvas_calibration.json"


# ============================================================
# CALIBRATED CANVAS CORNERS
# Units: meters
# ============================================================

# Optional offset applied to all mapped stroke points after the canvas frame is
# computed from calibration points. Negative world x pulls contact points back
# from the page for this robot setup; positive x pushes them into the page.
DEFAULT_STROKE_PLANE_OFFSET_XYZ_M = np.array([0.0, 0.0, 0.0], dtype=float)


# ============================================================
# MAPPING CONFIG
# ============================================================

# Physical padding between outermost stroke point and drawable rectangle boundary.
# This is in meters. Example: 0.02 = 20 mm.
CANVAS_PADDING_M = 0.025

# If None, use measured canvas width/height from calibrated corners.
# If you want to force a standard A3-like drawing rectangle, set these.
# A3 portrait is 0.297 x 0.420 m.
FORCE_CANVAS_WIDTH_M = None
FORCE_CANVAS_HEIGHT_M = None

# If True, preserve drawing aspect ratio and center it inside the padded rectangle.
# If False, stretch drawing to fill padded rectangle exactly.
PRESERVE_ASPECT_RATIO = True

# Strokes JSON usually has image y increasing downward.
# Canvas v-axis is defined TL -> BL, so y-down maps naturally to positive v.
FLIP_Y = False

# If True, clean raw quadrilateral into a perfect planar rectangle.
# Recommended.
CLEAN_CANVAS_TO_RECTANGLE = True

# Optional global scale inside padded region.
# 1.0 means fill as much as possible while respecting padding/aspect.
DRAWING_SCALE = 1.0

# Include both 2D local meters and 3D XYZ in output.
INCLUDE_LOCAL_UV_METERS = True

# Include original pixel points in output.
INCLUDE_ORIGINAL_PIXEL_POINTS = True


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def norm(v):
    return float(np.linalg.norm(v))


def unit(v, eps=1e-12):
    n = norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def project_to_plane(p, plane_point, normal):
    return p - np.dot(p - plane_point, normal) * normal


def compute_clean_canvas_frame(TL, TR, BL, BR):
    """
    Converts imperfect hand-calibrated quadrilateral corners into a clean
    rectangular canvas frame.

    Returns:
        frame dict with:
            origin: clean TL
            u_axis: left-to-right unit vector
            v_axis: top-to-bottom unit vector
            normal: plane normal
            width_m
            height_m
            center
            clean corners
    """

    TL = np.asarray(TL, dtype=float)
    TR = np.asarray(TR, dtype=float)
    BL = np.asarray(BL, dtype=float)
    BR = np.asarray(BR, dtype=float)

    center = 0.25 * (TL + TR + BL + BR)

    # Average horizontal direction from top and bottom edges.
    u_raw = 0.5 * ((TR - TL) + (BR - BL))
    u_axis = unit(u_raw)

    # Average vertical/down direction from left and right edges.
    v_raw = 0.5 * ((BL - TL) + (BR - TR))

    # Plane normal from rough u/v.
    normal = unit(np.cross(u_axis, v_raw))

    # Orthogonalize v against u while staying in the plane.
    v_axis = v_raw - np.dot(v_raw, u_axis) * u_axis
    v_axis = unit(v_axis)

    # Recompute normal so the frame is exactly orthonormal.
    normal = unit(np.cross(u_axis, v_axis))

    top_w = norm(TR - TL)
    bottom_w = norm(BR - BL)
    left_h = norm(BL - TL)
    right_h = norm(BR - TR)

    measured_width = 0.5 * (top_w + bottom_w)
    measured_height = 0.5 * (left_h + right_h)

    width_m = measured_width if FORCE_CANVAS_WIDTH_M is None else float(FORCE_CANVAS_WIDTH_M)
    height_m = measured_height if FORCE_CANVAS_HEIGHT_M is None else float(FORCE_CANVAS_HEIGHT_M)

    if CLEAN_CANVAS_TO_RECTANGLE:
        clean_TL = center - 0.5 * width_m * u_axis - 0.5 * height_m * v_axis
        clean_TR = center + 0.5 * width_m * u_axis - 0.5 * height_m * v_axis
        clean_BL = center - 0.5 * width_m * u_axis + 0.5 * height_m * v_axis
        clean_BR = center + 0.5 * width_m * u_axis + 0.5 * height_m * v_axis
    else:
        # Still use an orthonormal basis, but anchor at projected raw TL.
        plane_point = center
        clean_TL = project_to_plane(TL, plane_point, normal)
        clean_TR = clean_TL + width_m * u_axis
        clean_BL = clean_TL + height_m * v_axis
        clean_BR = clean_TL + width_m * u_axis + height_m * v_axis

    return {
        "origin": clean_TL,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "normal": normal,
        "width_m": width_m,
        "height_m": height_m,
        "center": center,
        "corners_raw": {
            "TL": TL,
            "TR": TR,
            "BL": BL,
            "BR": BR,
        },
        "corners_clean": {
            "TL": clean_TL,
            "TR": clean_TR,
            "BL": clean_BL,
            "BR": clean_BR,
        },
        "measured": {
            "top_width_m": top_w,
            "bottom_width_m": bottom_w,
            "left_height_m": left_h,
            "right_height_m": right_h,
            "avg_width_m": measured_width,
            "avg_height_m": measured_height,
        },
    }


def collect_all_pixel_points(strokes_payload):
    pts = []

    for stroke in strokes_payload["strokes"]:
        for p in stroke["points"]:
            pts.append([float(p[0]), float(p[1])])

    if len(pts) == 0:
        raise ValueError("No points found in strokes JSON.")

    return np.asarray(pts, dtype=float)


def compute_pixel_bbox(points_px):
    x_min = float(np.min(points_px[:, 0]))
    y_min = float(np.min(points_px[:, 1]))
    x_max = float(np.max(points_px[:, 0]))
    y_max = float(np.max(points_px[:, 1]))

    if x_max - x_min <= 1e-9:
        raise ValueError("Stroke bbox has near-zero width.")
    if y_max - y_min <= 1e-9:
        raise ValueError("Stroke bbox has near-zero height.")

    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def compute_fit_transform(pixel_bbox, canvas_width_m, canvas_height_m, padding_m):
    """
    Computes affine transform from pixel bbox to local canvas meters.

    The outermost stroke bbox maps inside:
        u in [padding, width - padding]
        v in [padding, height - padding]
    """

    inner_w = canvas_width_m - 2.0 * padding_m
    inner_h = canvas_height_m - 2.0 * padding_m

    if inner_w <= 0 or inner_h <= 0:
        raise ValueError(
            f"Padding too large. width={canvas_width_m:.4f}, "
            f"height={canvas_height_m:.4f}, padding={padding_m:.4f}"
        )

    src_w = pixel_bbox["width"]
    src_h = pixel_bbox["height"]

    sx = inner_w / src_w
    sy = inner_h / src_h

    if PRESERVE_ASPECT_RATIO:
        scale = min(sx, sy) * DRAWING_SCALE

        fitted_w = src_w * scale
        fitted_h = src_h * scale

        u0 = padding_m + 0.5 * (inner_w - fitted_w)
        v0 = padding_m + 0.5 * (inner_h - fitted_h)

        scale_x = scale
        scale_y = scale
    else:
        scale_x = sx * DRAWING_SCALE
        scale_y = sy * DRAWING_SCALE

        fitted_w = src_w * scale_x
        fitted_h = src_h * scale_y

        u0 = padding_m + 0.5 * (inner_w - fitted_w)
        v0 = padding_m + 0.5 * (inner_h - fitted_h)

    return {
        "u0_m": float(u0),
        "v0_m": float(v0),
        "scale_x_m_per_px": float(scale_x),
        "scale_y_m_per_px": float(scale_y),
        "fitted_width_m": float(fitted_w),
        "fitted_height_m": float(fitted_h),
        "inner_width_m": float(inner_w),
        "inner_height_m": float(inner_h),
    }


def pixel_to_local_uv(px, py, pixel_bbox, fit):
    """
    Converts pixel coordinate to local canvas meters:
        u: horizontal across canvas from clean TL toward clean TR
        v: vertical/down canvas from clean TL toward clean BL
    """

    x_norm_px = float(px) - pixel_bbox["x_min"]

    if FLIP_Y:
        y_norm_px = pixel_bbox["y_max"] - float(py)
    else:
        y_norm_px = float(py) - pixel_bbox["y_min"]

    u = fit["u0_m"] + x_norm_px * fit["scale_x_m_per_px"]
    v = fit["v0_m"] + y_norm_px * fit["scale_y_m_per_px"]

    return float(u), float(v)


def local_uv_to_xyz(u, v, frame):
    origin = frame["origin"]
    u_axis = frame["u_axis"]
    v_axis = frame["v_axis"]

    p = origin + u * u_axis + v * v_axis
    return p.astype(float)


def arr_to_list(a):
    return [float(x) for x in np.asarray(a).reshape(-1)]


def load_canvas_calibration(calibration_json_path=CALIBRATION_JSON):
    with open(calibration_json_path, "r") as f:
        calibration = json.load(f)

    corners = calibration.get("raw_corners_xyz")
    if not isinstance(corners, dict):
        raise ValueError(f"{calibration_json_path} missing raw_corners_xyz object.")

    required = ("TL", "TR", "BL", "BR")
    missing = [name for name in required if name not in corners]
    if missing:
        raise ValueError(
            f"{calibration_json_path} missing raw corner(s): {', '.join(missing)}"
        )

    raw_corners = {}
    for name in required:
        point = np.asarray(corners[name], dtype=float)
        if point.shape != (3,):
            raise ValueError(
                f"{calibration_json_path} raw_corners_xyz.{name} must be [x, y, z]."
            )
        raw_corners[name] = point

    offset = np.asarray(
        calibration.get("stroke_plane_offset_xyz_m", DEFAULT_STROKE_PLANE_OFFSET_XYZ_M),
        dtype=float,
    )
    if offset.shape != (3,):
        raise ValueError(
            f"{calibration_json_path} stroke_plane_offset_xyz_m must be [x, y, z]."
        )

    return raw_corners, offset


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_strokes_to_canvas_plane(
    input_json_path=INPUT_STROKES_JSON,
    output_json_path=OUTPUT_STROKES_JSON,
    calibration_json_path=CALIBRATION_JSON,
):
    with open(input_json_path, "r") as f:
        payload = json.load(f)

    raw_corners, stroke_plane_offset = load_canvas_calibration(
        calibration_json_path
    )
    frame = compute_clean_canvas_frame(
        raw_corners["TL"],
        raw_corners["TR"],
        raw_corners["BL"],
        raw_corners["BR"],
    )

    all_px = collect_all_pixel_points(payload)
    pixel_bbox = compute_pixel_bbox(all_px)

    fit = compute_fit_transform(
        pixel_bbox=pixel_bbox,
        canvas_width_m=frame["width_m"],
        canvas_height_m=frame["height_m"],
        padding_m=CANVAS_PADDING_M,
    )

    output = deepcopy(payload)

    output["coordinate_system"] = {
        "type": "canvas_plane_xyz",
        "units": "meters",
        "description": (
            "Each stroke point is mapped from original image pixel coordinates "
            "onto a cleaned rectangular canvas plane. u_axis points from TL to TR; "
            "v_axis points from TL to BL; normal = u_axis x v_axis."
        ),
    }

    output["canvas_mapping"] = {
        "input_strokes_json": input_json_path,
        "padding_m": float(CANVAS_PADDING_M),
        "preserve_aspect_ratio": bool(PRESERVE_ASPECT_RATIO),
        "flip_y": bool(FLIP_Y),
        "drawing_scale": float(DRAWING_SCALE),
        "calibration_json": str(calibration_json_path),
        "stroke_plane_offset_xyz_m": arr_to_list(stroke_plane_offset),

        "raw_corners_xyz": {
            "TL": arr_to_list(raw_corners["TL"]),
            "TR": arr_to_list(raw_corners["TR"]),
            "BL": arr_to_list(raw_corners["BL"]),
            "BR": arr_to_list(raw_corners["BR"]),
        },

        "clean_corners_xyz": {
            "TL": arr_to_list(frame["corners_clean"]["TL"]),
            "TR": arr_to_list(frame["corners_clean"]["TR"]),
            "BL": arr_to_list(frame["corners_clean"]["BL"]),
            "BR": arr_to_list(frame["corners_clean"]["BR"]),
        },

        "canvas_frame": {
            "origin_TL_xyz": arr_to_list(frame["origin"]),
            "u_axis_TL_to_TR": arr_to_list(frame["u_axis"]),
            "v_axis_TL_to_BL": arr_to_list(frame["v_axis"]),
            "normal_u_cross_v": arr_to_list(frame["normal"]),
            "width_m": float(frame["width_m"]),
            "height_m": float(frame["height_m"]),
            "center_xyz": arr_to_list(frame["center"]),
        },

        "measured_raw_canvas": {
            k: float(v) for k, v in frame["measured"].items()
        },

        "stroke_pixel_bbox_used_for_fit": pixel_bbox,

        "fit_transform": fit,
    }

    converted_strokes = []

    for stroke in payload["strokes"]:
        new_stroke = deepcopy(stroke)

        original_points_px = stroke["points"]
        points_uv = []
        points_xyz = []

        for p in original_points_px:
            px, py = float(p[0]), float(p[1])

            u, v = pixel_to_local_uv(px, py, pixel_bbox, fit)
            xyz = local_uv_to_xyz(u, v, frame) + stroke_plane_offset

            if INCLUDE_LOCAL_UV_METERS:
                points_uv.append([float(u), float(v)])

            points_xyz.append(arr_to_list(xyz))

        if INCLUDE_ORIGINAL_PIXEL_POINTS:
            new_stroke["points_px"] = original_points_px

        if INCLUDE_LOCAL_UV_METERS:
            new_stroke["points_uv_m"] = points_uv

        # Main output path for robot execution.
        new_stroke["points_xyz_m"] = points_xyz

        # Replace generic points field with XYZ for convenience.
        # If you prefer to preserve old behavior, comment this out.
        new_stroke["points"] = points_xyz

        converted_strokes.append(new_stroke)

    output["strokes"] = converted_strokes

    out_dir = os.path.dirname(output_json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print("===================================================")
    print("CANVAS PLANE MAPPING COMPLETE")
    print("===================================================")
    print(f"Input:  {input_json_path}")
    print(f"Output: {output_json_path}")
    print(f"Calibration: {calibration_json_path}")
    print()
    print("Raw measured canvas:")
    print(f"  top width:     {frame['measured']['top_width_m']:.4f} m")
    print(f"  bottom width:  {frame['measured']['bottom_width_m']:.4f} m")
    print(f"  left height:   {frame['measured']['left_height_m']:.4f} m")
    print(f"  right height:  {frame['measured']['right_height_m']:.4f} m")
    print()
    print("Clean canvas rectangle:")
    print(f"  width:         {frame['width_m']:.4f} m")
    print(f"  height:        {frame['height_m']:.4f} m")
    print(f"  padding:       {CANVAS_PADDING_M:.4f} m")
    print()
    print("Frame:")
    print(f"  origin TL:     {arr_to_list(frame['origin'])}")
    print(f"  u_axis:        {arr_to_list(frame['u_axis'])}")
    print(f"  v_axis:        {arr_to_list(frame['v_axis'])}")
    print(f"  normal:        {arr_to_list(frame['normal'])}")
    print(f"  stroke offset: {arr_to_list(stroke_plane_offset)}")
    print()
    print("Stroke pixel bbox fitted:")
    print(f"  x:             {pixel_bbox['x_min']:.2f} -> {pixel_bbox['x_max']:.2f}")
    print(f"  y:             {pixel_bbox['y_min']:.2f} -> {pixel_bbox['y_max']:.2f}")
    print()
    print("Fit:")
    print(f"  fitted width:  {fit['fitted_width_m']:.4f} m")
    print(f"  fitted height: {fit['fitted_height_m']:.4f} m")
    print(f"  sx:            {fit['scale_x_m_per_px']:.8f} m/px")
    print(f"  sy:            {fit['scale_y_m_per_px']:.8f} m/px")
    print()
    print(f"Converted strokes: {len(converted_strokes)}")

    return output


if __name__ == "__main__":
    convert_strokes_to_canvas_plane()
