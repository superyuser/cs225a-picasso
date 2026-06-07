import os
import cv2
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_STROKES_DIR = DEMO_DAY_DIR / "strokes"
DEFAULT_STROKE_JSONS_DIR = DEMO_DAY_DIR / "stroke-jsons"

# 10 in x 10 in canvas
CANVAS_WIDTH_MM = 254.0
CANVAS_HEIGHT_MM = 254.0

# Brush widths
OUTLINE_BRUSH_WIDTH_MM = 3.5
FILL_BRUSH_WIDTH_MM = 8.5

# Outline extraction
LINE_THRESH = 180
MIN_LINE_COMPONENT_AREA = 10
OUTLINE_SIMPLIFY_EPS_FRAC = 0.006

# Barrier thickness to keep fill off outlines
BARRIER_DILATE_PX = 5

# Fill generation
PEEL_STEP_FACTOR = 0.72     # < 1.0 gives overlap between fill strokes
MIN_FILL_CONTOUR_AREA = 40
MIN_FILL_CONTOUR_LENGTH = 20
FILL_SIMPLIFY_EPS_FRAC = 0.004

# ------------------------------------------------------------
# Corner filleting / smoothing
#
# After simplification, each polyline can still have sharp inflection
# corners. We round these using Chaikin-style corner cutting: at every
# vertex sharper than the threshold, replace the point with two new
# points placed FILLET_RATIO of the way back along each adjacent edge.
# Multiple iterations progressively smooth remaining sharp corners.
#
# Interior angle convention:
#   180 deg = perfectly straight (no turn)
#    90 deg = right-angle corner
#     0 deg = full reversal
# Smaller angle = sharper corner.
#
# Set FILLET_ANGLE_THRESHOLD_DEG = 180 to smooth every corner (pure Chaikin).
# Use FILLET_ITERATIONS = 0 to disable smoothing entirely.
# ------------------------------------------------------------
FILLET_ANGLE_THRESHOLD_DEG = 150.0
FILLET_RATIO = 0.25
FILLET_ITERATIONS = 2

# Randomization
RANDOM_SEED = 42
SHUFFLE_FILL_STROKES = True

# ------------------------------------------------------------
# Heuristic priors for cute centered portrait style
# These are the older-style masks that were working better.
# ------------------------------------------------------------

# Face / neck priors (kept uncolored)
FACE_CX = 0.50
FACE_CY = 0.49
FACE_RX = 0.23
FACE_RY = 0.22

NECK_CX = 0.50
NECK_CY = 0.72
NECK_RX = 0.11
NECK_RY = 0.10

# Accessories / headphones priors (kept uncolored)
LEFT_EAR_CX = 0.20
LEFT_EAR_CY = 0.43
LEFT_EAR_RX = 0.10
LEFT_EAR_RY = 0.18

RIGHT_EAR_CX = 0.80
RIGHT_EAR_CY = 0.43
RIGHT_EAR_RX = 0.10
RIGHT_EAR_RY = 0.18

BAND_OUTER_CX = 0.50
BAND_OUTER_CY = 0.23
BAND_OUTER_RX = 0.34
BAND_OUTER_RY = 0.20

BAND_INNER_CX = 0.50
BAND_INNER_CY = 0.24
BAND_INNER_RX = 0.29
BAND_INNER_RY = 0.15

# Hair priors
TOP_HAIR_CX = 0.50
TOP_HAIR_CY = 0.25
TOP_HAIR_RX = 0.28
TOP_HAIR_RY = 0.18

LEFT_BRAID1_CX = 0.30
LEFT_BRAID1_CY = 0.70
LEFT_BRAID1_RX = 0.08
LEFT_BRAID1_RY = 0.12

LEFT_BRAID2_CX = 0.29
LEFT_BRAID2_CY = 0.86
LEFT_BRAID2_RX = 0.05
LEFT_BRAID2_RY = 0.09

RIGHT_BRAID1_CX = 0.70
RIGHT_BRAID1_CY = 0.70
RIGHT_BRAID1_RX = 0.08
RIGHT_BRAID1_RY = 0.12

RIGHT_BRAID2_CX = 0.71
RIGHT_BRAID2_CY = 0.86
RIGHT_BRAID2_RX = 0.05
RIGHT_BRAID2_RY = 0.09

# Shirt prior
SHIRT_CX = 0.50
SHIRT_CY = 0.94
SHIRT_RX = 0.43
SHIRT_RY = 0.18
SHIRT_RECT_Y0 = 0.74


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def mask_u8(mask):
    return (mask.astype(np.uint8) * 255)


def ellipse_mask(h, w, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:h, 0:w]
    cx_px = cx * w
    cy_px = cy * h
    rx_px = max(rx * w, 1.0)
    ry_px = max(ry * h, 1.0)
    return (((xx - cx_px) / rx_px) ** 2 + ((yy - cy_px) / ry_px) ** 2) <= 1.0


def rect_mask(h, w, x0_frac, y0_frac, x1_frac, y1_frac):
    out = np.zeros((h, w), dtype=bool)
    x0 = int(round(x0_frac * w))
    x1 = int(round(x1_frac * w))
    y0 = int(round(y0_frac * h))
    y1 = int(round(y1_frac * h))
    out[y0:y1, x0:x1] = True
    return out


def morph_open(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_u8(mask), cv2.MORPH_OPEN, kernel) > 0


def morph_close(mask, k=5):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_u8(mask), cv2.MORPH_CLOSE, kernel) > 0


def morph_dilate(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8(mask), kernel) > 0


def morph_erode(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask_u8(mask), kernel) > 0


def keep_components_larger_than(mask, min_area_px):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8(mask), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            out |= labels == label
    return out


def save_mask(path, mask):
    cv2.imwrite(path, mask_u8(mask))


def px_to_mm(x_px, y_px, w, h, canvas_w_mm, canvas_h_mm):
    x_mm = (x_px / max(w - 1, 1)) * canvas_w_mm
    y_mm = ((h - 1 - y_px) / max(h - 1, 1)) * canvas_h_mm
    return [round(float(x_mm), 3), round(float(y_mm), 3)]


def mm_to_px(mm, image_w, canvas_w_mm):
    return max(1, int(round(mm * image_w / canvas_w_mm)))


# ============================================================
# CORNER FILLETING / CHAIKIN SMOOTHING
# ============================================================

def fillet_polyline(
    points,
    *,
    angle_threshold_deg=None,
    fillet_ratio=None,
    iterations=None,
    closed: bool = True,
):
    """Round sharp corners in a polyline using selective corner-cutting.

    For each interior vertex, computes the interior angle between the
    incoming and outgoing edges. If the angle is sharper than
    ``angle_threshold_deg`` (i.e. the corner turns more than the threshold
    allows), the vertex is replaced with two new points placed at
    ``fillet_ratio`` along each adjacent edge. Vertices on gentler bends
    are left in place.

    ``None`` parameters fall back to the module-level ``FILLET_*`` globals
    so that CLI overrides in ``main()`` take effect.

    Accepts arrays of shape ``(N, 2)`` or OpenCV's ``(N, 1, 2)``.
    Returns a float array of shape ``(M, 2)``.
    """
    if angle_threshold_deg is None:
        angle_threshold_deg = FILLET_ANGLE_THRESHOLD_DEG
    if fillet_ratio is None:
        fillet_ratio = FILLET_RATIO
    if iterations is None:
        iterations = FILLET_ITERATIONS

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts[:, 0, :]
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or iterations <= 0:
        return pts

    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            break

        new_pts = []
        for i in range(n):
            if not closed and (i == 0 or i == n - 1):
                new_pts.append(pts[i])
                continue

            p_prev = pts[(i - 1) % n]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-9 or n2 < 1e-9:
                new_pts.append(p_curr)
                continue

            cos_a = float(np.dot(v1, v2) / (n1 * n2))
            cos_a = max(-1.0, min(1.0, cos_a))
            angle_deg = math.degrees(math.acos(cos_a))  # 0..180

            if angle_deg < angle_threshold_deg:
                q = p_curr + fillet_ratio * (p_prev - p_curr)
                r = p_curr + fillet_ratio * (p_next - p_curr)
                new_pts.append(q)
                new_pts.append(r)
            else:
                new_pts.append(p_curr)

        pts = np.asarray(new_pts, dtype=np.float64)

    return pts


# ============================================================
# LINE ART EXTRACTION
# ============================================================

def extract_line_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # black lines on white background
    line_mask = gray < LINE_THRESH
    line_mask = morph_close(line_mask, 3)
    line_mask = keep_components_larger_than(line_mask, MIN_LINE_COMPONENT_AREA)

    return line_mask


def contours_to_outline_strokes(line_mask, canvas_w_mm, canvas_h_mm):
    """
    Convert black line mask into outline contour strokes.
    """
    h, w = line_mask.shape

    contours, _ = cv2.findContours(
        mask_u8(line_mask),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    strokes = []

    for contour in contours:
        if len(contour) < 10:
            continue

        area = abs(cv2.contourArea(contour))
        if area < 3:
            continue

        perimeter = cv2.arcLength(contour, True)
        eps = OUTLINE_SIMPLIFY_EPS_FRAC * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)

        pts_px = approx[:, 0, :]
        if len(pts_px) < 3:
            continue

        filleted = fillet_polyline(pts_px, closed=True)
        pts_px_int = filleted.round().astype(int)
        if len(pts_px_int) < 3:
            continue

        pts_mm = [
            px_to_mm(int(x), int(y), w, h, canvas_w_mm, canvas_h_mm)
            for x, y in pts_px_int
        ]

        strokes.append({
            "type": "polyline",
            "layer": "outline_black",
            "closed": True,
            "brush_width_mm": OUTLINE_BRUSH_WIDTH_MM,
            "points_px": pts_px_int.tolist(),
            "points_mm": pts_mm,
        })

    random.shuffle(strokes)
    return strokes


# ============================================================
# FEATURE PRIORS + REGION GROWING
# (this is the older masking strategy that behaved better)
# ============================================================

def build_priors(h, w):
    # face + neck (leave uncolored)
    face = ellipse_mask(h, w, FACE_CX, FACE_CY, FACE_RX, FACE_RY)
    neck = ellipse_mask(h, w, NECK_CX, NECK_CY, NECK_RX, NECK_RY)

    # accessories/headphones (leave uncolored)
    left_ear = ellipse_mask(h, w, LEFT_EAR_CX, LEFT_EAR_CY, LEFT_EAR_RX, LEFT_EAR_RY)
    right_ear = ellipse_mask(h, w, RIGHT_EAR_CX, RIGHT_EAR_CY, RIGHT_EAR_RX, RIGHT_EAR_RY)

    band_outer = ellipse_mask(h, w, BAND_OUTER_CX, BAND_OUTER_CY, BAND_OUTER_RX, BAND_OUTER_RY)
    band_inner = ellipse_mask(h, w, BAND_INNER_CX, BAND_INNER_CY, BAND_INNER_RX, BAND_INNER_RY)
    headband = band_outer & (~band_inner)

    accessories = left_ear | right_ear | headband

    # hair prior
    top_hair = ellipse_mask(h, w, TOP_HAIR_CX, TOP_HAIR_CY, TOP_HAIR_RX, TOP_HAIR_RY)
    left_braid1 = ellipse_mask(h, w, LEFT_BRAID1_CX, LEFT_BRAID1_CY, LEFT_BRAID1_RX, LEFT_BRAID1_RY)
    left_braid2 = ellipse_mask(h, w, LEFT_BRAID2_CX, LEFT_BRAID2_CY, LEFT_BRAID2_RX, LEFT_BRAID2_RY)
    right_braid1 = ellipse_mask(h, w, RIGHT_BRAID1_CX, RIGHT_BRAID1_CY, RIGHT_BRAID1_RX, RIGHT_BRAID1_RY)
    right_braid2 = ellipse_mask(h, w, RIGHT_BRAID2_CX, RIGHT_BRAID2_CY, RIGHT_BRAID2_RX, RIGHT_BRAID2_RY)

    hair = top_hair | left_braid1 | left_braid2 | right_braid1 | right_braid2

    # shirt prior
    shirt_ellipse = ellipse_mask(h, w, SHIRT_CX, SHIRT_CY, SHIRT_RX, SHIRT_RY)
    shirt_rect = rect_mask(h, w, 0.10, SHIRT_RECT_Y0, 0.90, 1.0)
    shirt = shirt_ellipse | shirt_rect

    return {
        "face": face,
        "neck": neck,
        "accessories": accessories,
        "hair": hair,
        "shirt": shirt,
    }


def grow_from_seeds(barrier_mask, constraint_mask, seeds):
    """
    Flood-fill reachable free space from seeds while respecting a constraint mask.
    """
    h, w = barrier_mask.shape

    free = (~barrier_mask) & constraint_mask
    visited = np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=bool)

    queue = []

    for (x, y) in seeds:
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < w and 0 <= y < h and free[y, x]:
            queue.append((x, y))
            visited[y, x] = 1
            out[y, x] = True

    head = 0
    while head < len(queue):
        x, y = queue[head]
        head += 1

        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h:
                if visited[ny, nx]:
                    continue
                visited[ny, nx] = 1
                if free[ny, nx]:
                    out[ny, nx] = True
                    queue.append((nx, ny))

    return out


def build_fill_masks(line_mask, priors):
    """
    Build hair fill + shirt fill.
    Face and accessories remain uncolored.
    """
    h, w = line_mask.shape

    barrier = morph_dilate(line_mask, BARRIER_DILATE_PX)

    face = priors["face"]
    neck = priors["neck"]
    accessories = priors["accessories"]
    hair_prior = priors["hair"]
    shirt_prior = priors["shirt"]

    # ------------------------
    # Hair seeds
    # ------------------------
    hair_seeds = [
        (0.50 * w, 0.18 * h),
        (0.31 * w, 0.66 * h),
        (0.29 * w, 0.84 * h),
        (0.69 * w, 0.66 * h),
        (0.71 * w, 0.84 * h),
    ]

    hair_constraint = hair_prior & (~face) & (~neck) & (~accessories)
    hair_mask = grow_from_seeds(barrier, hair_constraint, hair_seeds)
    hair_mask = morph_close(hair_mask, 7)
    hair_mask = morph_open(hair_mask, 3)
    hair_mask = hair_mask & (~barrier)
    hair_mask = hair_mask & (~face) & (~neck) & (~accessories)

    # ------------------------
    # Shirt seeds
    # ------------------------
    shirt_seeds = [
        (0.50 * w, 0.88 * h),
        (0.35 * w, 0.92 * h),
        (0.65 * w, 0.92 * h),
    ]

    shirt_constraint = shirt_prior & (~face) & (~neck) & (~accessories) & (~hair_mask)
    shirt_mask = grow_from_seeds(barrier, shirt_constraint, shirt_seeds)
    shirt_mask = morph_close(shirt_mask, 9)
    shirt_mask = morph_open(shirt_mask, 3)
    shirt_mask = shirt_mask & (~barrier)
    shirt_mask = shirt_mask & (~face) & (~neck) & (~accessories) & (~hair_mask)

    return hair_mask, shirt_mask, barrier


# ============================================================
# IMPROVED FILL STROKE GENERATION
# Onion-peel contour strokes for smoother coverage
# ============================================================

def simplify_contour(contour, eps_frac):
    perimeter = cv2.arcLength(contour, True)
    eps = eps_frac * perimeter
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx


def contour_to_stroke(contour, layer_name, brush_width_mm, canvas_w_mm, canvas_h_mm, image_shape, closed=True):
    h, w = image_shape
    pts_px = contour[:, 0, :]
    if len(pts_px) < 3:
        return None

    pts_mm = [
        px_to_mm(int(x), int(y), w, h, canvas_w_mm, canvas_h_mm)
        for x, y in pts_px
    ]

    return {
        "type": "contour_fill",
        "layer": layer_name,
        "closed": closed,
        "brush_width_mm": brush_width_mm,
        "points_px": pts_px.astype(int).tolist(),
        "points_mm": pts_mm,
    }


def generate_onion_fill_strokes(mask, region_name, canvas_w_mm, canvas_h_mm, brush_width_mm):
    """
    Generate smooth fill strokes by tracing inward offset contours.

    Idea:
      1. shrink region by about half a brush width -> safe center path
      2. trace contours
      3. erode inward by ~0.72 * brush width and trace again
      4. repeat until region disappears

    This produces broad, overlapping, smooth strokes with much better coverage
    than random scribbles.
    """
    h, w = mask.shape

    brush_px = mm_to_px(brush_width_mm, w, canvas_w_mm)
    safe_margin_px = max(1, brush_px // 2)
    peel_step_px = max(1, int(round(PEEL_STEP_FACTOR * brush_px)))

    # Safe zone: if brush center stays here, thick stroke remains inside original mask
    safe_zone = morph_erode(mask, 2 * safe_margin_px + 1)

    # Fallback: if region is too thin, just use original mask
    if np.sum(safe_zone) == 0:
        safe_zone = mask.copy()

    current = safe_zone.copy()
    strokes = []
    layer_idx = 0

    while np.sum(current) > 0:
        contours, hierarchy = cv2.findContours(
            mask_u8(current),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        layer_strokes = []

        for contour in contours:
            if len(contour) < MIN_FILL_CONTOUR_LENGTH:
                continue

            area = abs(cv2.contourArea(contour))
            if area < MIN_FILL_CONTOUR_AREA:
                continue

            smooth = simplify_contour(contour, FILL_SIMPLIFY_EPS_FRAC)
            if len(smooth) < 3:
                continue

            filleted = fillet_polyline(smooth, closed=True)
            if len(filleted) < 3:
                continue
            smooth_filleted = filleted.round().astype(np.int32).reshape(-1, 1, 2)

            stroke = contour_to_stroke(
                smooth_filleted,
                region_name,
                brush_width_mm,
                canvas_w_mm,
                canvas_h_mm,
                (h, w),
                closed=True,
            )

            if stroke is not None:
                # alternate path direction between layers / contours if desired
                if (layer_idx % 2) == 1:
                    stroke["points_px"] = stroke["points_px"][::-1]
                    stroke["points_mm"] = stroke["points_mm"][::-1]
                layer_strokes.append(stroke)

        if SHUFFLE_FILL_STROKES:
            random.shuffle(layer_strokes)

        strokes.extend(layer_strokes)

        # Peel inward
        next_current = morph_erode(current, 2 * peel_step_px + 1)

        # Stop if no meaningful change
        if np.array_equal(next_current, current):
            break

        current = next_current
        layer_idx += 1

    if SHUFFLE_FILL_STROKES:
        random.shuffle(strokes)

    return strokes


# ============================================================
# PREVIEW RENDERING
# ============================================================

def render_region_preview(line_mask, hair_mask, shirt_mask, out_path):
    """
    White background:
      black outlines
      hair = medium brown
      shirt = light warm beige
    """
    h, w = line_mask.shape
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255

    # BGR preview colors
    hair_col = np.array([80, 120, 180], dtype=np.uint8)
    shirt_col = np.array([170, 210, 235], dtype=np.uint8)

    vis[shirt_mask] = shirt_col
    vis[hair_mask] = hair_col
    vis[line_mask] = np.array([0, 0, 0], dtype=np.uint8)

    cv2.imwrite(out_path, vis)


def render_stroke_preview(
    image_shape,
    outline_strokes,
    hair_strokes,
    shirt_strokes,
    canvas_w_mm,
    canvas_h_mm,
    out_path,
    black_on_top=False,
):
    """
    black_on_top=False  -> requested order preview
    black_on_top=True   -> cleaner visual preview
    """
    h, w = image_shape
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255

    outline_px = mm_to_px(OUTLINE_BRUSH_WIDTH_MM, w, canvas_w_mm)
    fill_px = mm_to_px(FILL_BRUSH_WIDTH_MM, w, canvas_w_mm)

    def draw_fill_stroke(stroke, color):
        pts = np.array(stroke["points_px"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            vis,
            [pts],
            isClosed=stroke.get("closed", True),
            color=color,
            thickness=fill_px,
            lineType=cv2.LINE_AA,
        )

    def draw_outline_stroke(stroke):
        pts = np.array(stroke["points_px"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            vis,
            [pts],
            isClosed=stroke.get("closed", True),
            color=(0, 0, 0),
            thickness=outline_px,
            lineType=cv2.LINE_AA,
        )

    # Requested order: black -> hair -> shirt
    if not black_on_top:
        for stroke in outline_strokes:
            draw_outline_stroke(stroke)
        for stroke in hair_strokes:
            draw_fill_stroke(stroke, (80, 120, 180))
        for stroke in shirt_strokes:
            draw_fill_stroke(stroke, (170, 210, 235))
    else:
        # Cleaner visual preview: hair -> shirt -> black
        for stroke in hair_strokes:
            draw_fill_stroke(stroke, (80, 120, 180))
        for stroke in shirt_strokes:
            draw_fill_stroke(stroke, (170, 210, 235))
        for stroke in outline_strokes:
            draw_outline_stroke(stroke)

    cv2.imwrite(out_path, vis)


# ============================================================
# MAIN
# ============================================================

def main():
    global FILLET_ANGLE_THRESHOLD_DEG, FILLET_RATIO, FILLET_ITERATIONS

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Input line-art image")
    parser.add_argument(
        "--strokes-dir",
        default=str(DEFAULT_STROKES_DIR),
        type=str,
        help=(
            "Root directory for visual outputs (masks, region preview, stroke previews). "
            f"Outputs land in <strokes-dir>/<input_stem>/. Default: {DEFAULT_STROKES_DIR}."
        ),
    )
    parser.add_argument(
        "--stroke-jsons-dir",
        default=str(DEFAULT_STROKE_JSONS_DIR),
        type=str,
        help=(
            "Directory for the painting-plan JSON. "
            f"The file is written as <stroke-jsons-dir>/<input_stem>.json. "
            f"Default: {DEFAULT_STROKE_JSONS_DIR}."
        ),
    )
    parser.add_argument("--canvas_w_mm", default=CANVAS_WIDTH_MM, type=float)
    parser.add_argument("--canvas_h_mm", default=CANVAS_HEIGHT_MM, type=float)
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    parser.add_argument(
        "--fillet-angle-deg",
        type=float,
        default=FILLET_ANGLE_THRESHOLD_DEG,
        help=(
            "Interior-angle threshold for corner filleting (deg). 180=smooth every "
            "corner (pure Chaikin); lower values only round sharper corners. "
            f"Default: {FILLET_ANGLE_THRESHOLD_DEG}."
        ),
    )
    parser.add_argument(
        "--fillet-ratio",
        type=float,
        default=FILLET_RATIO,
        help=(
            "How far back along each adjacent edge the corner cut is placed "
            "(0..0.5). Larger = more aggressive rounding. "
            f"Default: {FILLET_RATIO}."
        ),
    )
    parser.add_argument(
        "--fillet-iterations",
        type=int,
        default=FILLET_ITERATIONS,
        help=(
            "Number of smoothing passes. 0 disables filleting entirely. "
            f"Default: {FILLET_ITERATIONS}."
        ),
    )
    args = parser.parse_args()

    FILLET_ANGLE_THRESHOLD_DEG = float(args.fillet_angle_deg)
    FILLET_RATIO = float(args.fillet_ratio)
    FILLET_ITERATIONS = int(args.fillet_iterations)
    print(
        f"Corner filleting: angle_threshold={FILLET_ANGLE_THRESHOLD_DEG} deg, "
        f"ratio={FILLET_RATIO}, iterations={FILLET_ITERATIONS}"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not read image: {input_path}")
    stem = input_path.stem

    strokes_dir = Path(args.strokes_dir) / stem
    stroke_jsons_dir = Path(args.stroke_jsons_dir)
    ensure_dir(str(strokes_dir))
    ensure_dir(str(stroke_jsons_dir))

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    h, w = image_bgr.shape[:2]

    # --------------------------------------------------------
    # 1. Extract line mask
    # --------------------------------------------------------
    line_mask = extract_line_mask(image_bgr)

    # --------------------------------------------------------
    # 2. Build priors + fill masks
    #    (keep the older masking strategy)
    # --------------------------------------------------------
    priors = build_priors(h, w)
    hair_mask, shirt_mask, barrier = build_fill_masks(line_mask, priors)

    # Ensure clean exclusivity
    hair_mask = hair_mask & (~line_mask)
    shirt_mask = shirt_mask & (~line_mask) & (~hair_mask)

    # --------------------------------------------------------
    # 3. Generate strokes
    # --------------------------------------------------------
    outline_strokes = contours_to_outline_strokes(
        line_mask,
        canvas_w_mm=args.canvas_w_mm,
        canvas_h_mm=args.canvas_h_mm,
    )

    hair_strokes = generate_onion_fill_strokes(
        hair_mask,
        region_name="hair_color_2",
        canvas_w_mm=args.canvas_w_mm,
        canvas_h_mm=args.canvas_h_mm,
        brush_width_mm=FILL_BRUSH_WIDTH_MM,
    )

    shirt_strokes = generate_onion_fill_strokes(
        shirt_mask,
        region_name="shirt_color_3",
        canvas_w_mm=args.canvas_w_mm,
        canvas_h_mm=args.canvas_h_mm,
        brush_width_mm=FILL_BRUSH_WIDTH_MM,
    )

    # --------------------------------------------------------
    # 4. Save masks / previews
    # --------------------------------------------------------
    cv2.imwrite(str(strokes_dir / "00_input.png"), image_bgr)

    save_mask(str(strokes_dir / "01_line_mask.png"), line_mask)
    save_mask(str(strokes_dir / "02_hair_mask.png"), hair_mask)
    save_mask(str(strokes_dir / "03_shirt_mask.png"), shirt_mask)
    save_mask(str(strokes_dir / "04_barrier_mask.png"), barrier)

    render_region_preview(
        line_mask,
        hair_mask,
        shirt_mask,
        str(strokes_dir / "05_region_preview.png"),
    )

    render_stroke_preview(
        (h, w),
        outline_strokes,
        hair_strokes,
        shirt_strokes,
        args.canvas_w_mm,
        args.canvas_h_mm,
        str(strokes_dir / "06_stroke_preview_requested_order.png"),
        black_on_top=False,
    )

    render_stroke_preview(
        (h, w),
        outline_strokes,
        hair_strokes,
        shirt_strokes,
        args.canvas_w_mm,
        args.canvas_h_mm,
        str(strokes_dir / "07_stroke_preview_clean_visual.png"),
        black_on_top=True,
    )

    # --------------------------------------------------------
    # 5. Save painting plan JSON
    # --------------------------------------------------------
    painting_plan = {
        "input": str(input_path),
        "canvas_width_mm": args.canvas_w_mm,
        "canvas_height_mm": args.canvas_h_mm,
        "execution_order_requested": [
            "outline_black",
            "hair_color_2",
            "shirt_color_3"
        ],
        "recommended_physical_order": [
            "hair_color_2",
            "shirt_color_3",
            "outline_black"
        ],
        "notes": [
            "Face and accessories/headphones are intentionally left uncolored.",
            "Hair and shirt masks use the earlier prior+seed-grow logic that behaved better.",
            "Fill stroke generation is improved using inward contour peeling ('onion-peel') for smoother, more complete coverage.",
            "Contour fill strokes are spaced based on brush size, giving a flatter brush-like reconstruction instead of scribbly random chunks."
        ],
        "layers": [
            {
                "name": "outline_black",
                "brush_width_mm": OUTLINE_BRUSH_WIDTH_MM,
                "stroke_count": len(outline_strokes),
                "strokes": outline_strokes,
            },
            {
                "name": "hair_color_2",
                "brush_width_mm": FILL_BRUSH_WIDTH_MM,
                "stroke_count": len(hair_strokes),
                "strokes": hair_strokes,
            },
            {
                "name": "shirt_color_3",
                "brush_width_mm": FILL_BRUSH_WIDTH_MM,
                "stroke_count": len(shirt_strokes),
                "strokes": shirt_strokes,
            },
        ]
    }

    json_path = stroke_jsons_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(painting_plan, f, indent=2)

    print("Done.")
    print(f"Visuals dir:   {strokes_dir}")
    print(f"Painting plan: {json_path}")
    print(f"Outline strokes: {len(outline_strokes)}")
    print(f"Hair strokes:    {len(hair_strokes)}")
    print(f"Shirt strokes:   {len(shirt_strokes)}")
    print()
    print("Inspect these first:")
    print(f"  {strokes_dir / '05_region_preview.png'}")
    print(f"  {strokes_dir / '06_stroke_preview_requested_order.png'}")
    print(f"  {strokes_dir / '07_stroke_preview_clean_visual.png'}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()