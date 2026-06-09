import os
import cv2
import json
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# ============================================================
# ROBUST ROBOT PAINTING PIPELINE
# ============================================================
# Main architecture:
#   1. Extract black line art as a physical barrier.
#   2. Build a portrait support mask so region selection does not leak into
#      the white background.
#   3. Build soft semantic priors for face / neck / accessories / hair / shirt.
#   4. Classify free-space connected components by overlap with those priors.
#   5. Use conservative fallbacks when line art is open or disconnected.
#   6. Generate robot strokes:
#        - outline: skeleton centerlines, not thick-contour boundaries
#        - hair: flow-aligned long hatch strokes, component-local angles
#        - shirt: smooth hatch strokes
#   7. Save diagnostic masks, previews, animation, and JSON plan.
#
# This is still a heuristic vision pipeline, but it is much more robust than
# seed-only flood fill because every region decision is backed by component
# geometry + priors + fallbacks.
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
FILL_BRUSH_WIDTH_MM = 5.5

# Line extraction
LINE_THRESH = 180
MIN_LINE_COMPONENT_AREA = 10

# Barrier for fill. Smaller than older 5 px to avoid blocking narrow hair pockets.
BARRIER_DILATE_PX = 2

# Portrait support extraction
SUPPORT_CLOSE_FRAC = 0.038      # relative to image width, closed silhouette helper
SUPPORT_DILATE_FRAC = 0.018
SUPPORT_MIN_AREA_FRAC = 0.030

# Component classification
MIN_REGION_COMPONENT_AREA_FRAC = 0.00008
HAIR_MIN_POSITIVE_OVERLAP = 0.08
HAIR_MAX_NEGATIVE_OVERLAP = 0.32
SHIRT_MIN_POSITIVE_OVERLAP = 0.10
SHIRT_MAX_NEGATIVE_OVERLAP = 0.35

# Fallback if classified hair is suspiciously small compared to the prior.
HAIR_FALLBACK_MIN_PRIOR_COVERAGE = 0.45
SHIRT_FALLBACK_MIN_PRIOR_COVERAGE = 0.45

# Hatching
HATCH_STEP_FACTOR = 0.52          # smaller = denser; 0.50-0.68 usually good
HATCH_MIN_SEGMENT_LEN_MM = 14.0   # keep shorter fragments in hair pockets
HATCH_POINT_SPACING_PX = 16
HATCH_SMOOTH_ITERATIONS = 1
HATCH_ANGLE_JITTER_DEG = 2.0
HATCH_SAFE_MARGIN_FACTOR = 0.25   # lower covers closer to color-mask boundaries

# Hair flow angles in image coordinates, degrees. 0 = left-to-right, 90 = downward.
HAIR_TOP_ANGLE_DEG = 88.0
HAIR_LEFT_ANGLE_DEG = 78.0
HAIR_RIGHT_ANGLE_DEG = 102.0
HAIR_BANGS_ANGLE_DEG = 86.0
SHIRT_HATCH_ANGLE_DEG = 10.0

# Outline centerline tracing
OUTLINE_SKELETONIZE = True
MIN_OUTLINE_PATH_LEN_MM = 4.0
OUTLINE_POINT_SPACING_PX = 5
OUTLINE_OPEN_SIMPLIFY_EPS_FRAC = 0.003

# Smoothing / filleting
FILLET_ANGLE_THRESHOLD_DEG = 150.0
FILLET_RATIO = 0.25
FILLET_ITERATIONS = 2

# Randomization
RANDOM_SEED = 42


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def odd(k):
    k = int(round(k))
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1


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
    x0 = max(0, min(w, int(round(x0_frac * w))))
    x1 = max(0, min(w, int(round(x1_frac * w))))
    y0 = max(0, min(h, int(round(y0_frac * h))))
    y1 = max(0, min(h, int(round(y1_frac * h))))
    out[y0:y1, x0:x1] = True
    return out


def morph_open(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k), odd(k)))
    return cv2.morphologyEx(mask_u8(mask), cv2.MORPH_OPEN, kernel) > 0


def morph_close(mask, k=5):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k), odd(k)))
    return cv2.morphologyEx(mask_u8(mask), cv2.MORPH_CLOSE, kernel) > 0


def morph_dilate(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k), odd(k)))
    return cv2.dilate(mask_u8(mask), kernel) > 0


def morph_erode(mask, k=3):
    if k <= 1:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(k), odd(k)))
    return cv2.erode(mask_u8(mask), kernel) > 0


def keep_components_larger_than(mask, min_area_px):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8(mask), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for label in range(1, num):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            out |= labels == label
    return out


def fill_holes(mask):
    """Fill holes inside a binary mask using border flood fill."""
    h, w = mask.shape
    inv = ~mask
    flood = mask_u8(inv)
    ff = flood.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), dtype=np.uint8), (0, 0), 128)
    outside = ff == 128
    holes = inv & (~outside)
    return mask | holes


def largest_component(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8(mask), connectivity=8)
    if num <= 1:
        return mask.copy()
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def save_mask(path, mask):
    cv2.imwrite(str(path), mask_u8(mask))


def px_to_mm(x_px, y_px, w, h, canvas_w_mm, canvas_h_mm):
    x_mm = (x_px / max(w - 1, 1)) * canvas_w_mm
    y_mm = ((h - 1 - y_px) / max(h - 1, 1)) * canvas_h_mm
    return [round(float(x_mm), 3), round(float(y_mm), 3)]


def mm_to_px(mm, image_w, canvas_w_mm):
    return max(1, int(round(mm * image_w / canvas_w_mm)))


def polyline_length_px(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def polyline_length_mm(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def projected_length_px(points, direction):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    direction = np.asarray(direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    projections = pts @ direction
    return float(np.max(projections) - np.min(projections))


def resample_polyline_px(points, spacing_px):
    """Downsample a polyline by accumulated arc length."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 2 or spacing_px <= 1:
        return pts
    out = [pts[0]]
    acc = 0.0
    prev = pts[0]
    for p in pts[1:]:
        seg = float(np.linalg.norm(p - prev))
        acc += seg
        if acc >= spacing_px:
            out.append(p)
            acc = 0.0
        prev = p
    if np.linalg.norm(out[-1] - pts[-1]) > 1e-6:
        out.append(pts[-1])
    return np.asarray(out, dtype=np.float64)


def chaikin_open_polyline(points, iterations=1, ratio=0.25):
    """Smooth an open robot path while preserving endpoints."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or iterations <= 0:
        return pts
    for _ in range(iterations):
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            out.append((1.0 - ratio) * p + ratio * q)
            out.append(ratio * p + (1.0 - ratio) * q)
        out.append(pts[-1])
        pts = np.asarray(out, dtype=np.float64)
    return pts


def fillet_polyline(points, angle_threshold_deg=None, fillet_ratio=None, iterations=None, closed=True):
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
            angle_deg = math.degrees(math.acos(cos_a))
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
# LINE ART AND PORTRAIT SUPPORT
# ============================================================

def extract_line_mask(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu sometimes helps with anti-aliased generated art, but clamp against
    # LINE_THRESH so colored fills in old previews are not accidentally treated
    # as line art.
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fixed = gray < LINE_THRESH
    line_mask = fixed | (otsu > 0)
    line_mask = morph_close(line_mask, 3)
    line_mask = keep_components_larger_than(line_mask, MIN_LINE_COMPONENT_AREA)
    return line_mask


def compute_ink_bbox(line_mask):
    ys, xs = np.nonzero(line_mask)
    h, w = line_mask.shape
    if len(xs) == 0:
        return (0, 0, w - 1, h - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def build_portrait_support(line_mask):
    """Return a conservative support area containing portrait interior.

    The support prevents priors from shading the infinite white background.  It
    is built by closing/dilating the line art into a coarse silhouette, filling
    holes, and keeping the largest component. For open-bottom bust portraits,
    this still includes the shirt/hair mass without exploding to the canvas.
    """
    h, w = line_mask.shape
    k_close = odd(max(9, SUPPORT_CLOSE_FRAC * w))
    k_dilate = odd(max(5, SUPPORT_DILATE_FRAC * w))

    coarse = morph_dilate(line_mask, k_dilate)
    coarse = morph_close(coarse, k_close)
    coarse = fill_holes(coarse)
    coarse = largest_component(coarse)
    coarse = morph_dilate(coarse, k_dilate)

    # Avoid tiny failure masks: if support is too small, use a bbox-padded mask.
    if np.mean(coarse) < SUPPORT_MIN_AREA_FRAC:
        x0, y0, x1, y1 = compute_ink_bbox(line_mask)
        padx = int(0.04 * w)
        pady = int(0.04 * h)
        coarse = np.zeros_like(line_mask, dtype=bool)
        coarse[max(0, y0-pady):min(h, y1+pady+1), max(0, x0-padx):min(w, x1+padx+1)] = True
    return coarse


# ============================================================
# PRIORS
# ============================================================

@dataclass
class Priors:
    face: np.ndarray
    neck: np.ndarray
    accessories: np.ndarray
    hair: np.ndarray
    shirt: np.ndarray
    portrait_support: np.ndarray
    not_color: np.ndarray


def build_priors(h, w, line_mask):
    support = build_portrait_support(line_mask)

    # These are normalized portrait priors, intentionally broad. They cover both
    # the older braided-headphones avatar and the newer long-hair avatar.
    face = ellipse_mask(h, w, 0.50, 0.47, 0.235, 0.225)
    face |= ellipse_mask(h, w, 0.50, 0.50, 0.205, 0.205)

    neck = ellipse_mask(h, w, 0.50, 0.705, 0.135, 0.105)
    neck |= rect_mask(h, w, 0.40, 0.60, 0.60, 0.78)

    # Accessories: broad headphones from old avatar + chest stars/buttons zone.
    left_ear = ellipse_mask(h, w, 0.18, 0.43, 0.095, 0.18)
    right_ear = ellipse_mask(h, w, 0.82, 0.43, 0.095, 0.18)
    band_outer = ellipse_mask(h, w, 0.50, 0.22, 0.36, 0.20)
    band_inner = ellipse_mask(h, w, 0.50, 0.235, 0.30, 0.155)
    headband = band_outer & (~band_inner)
    chest_decor = rect_mask(h, w, 0.57, 0.76, 0.78, 0.90)
    accessories = left_ear | right_ear | headband | chest_decor

    # Robust hair prior: top cap + side curtains + lower long hair + old braids.
    top_cap = ellipse_mask(h, w, 0.50, 0.245, 0.345, 0.225)
    bangs = ellipse_mask(h, w, 0.50, 0.36, 0.30, 0.13)
    left_curtain = ellipse_mask(h, w, 0.255, 0.58, 0.175, 0.35)
    right_curtain = ellipse_mask(h, w, 0.745, 0.58, 0.175, 0.35)
    left_lower = ellipse_mask(h, w, 0.275, 0.815, 0.145, 0.21)
    right_lower = ellipse_mask(h, w, 0.725, 0.815, 0.145, 0.21)
    left_braid_old = ellipse_mask(h, w, 0.30, 0.73, 0.085, 0.18) | ellipse_mask(h, w, 0.29, 0.88, 0.055, 0.10)
    right_braid_old = ellipse_mask(h, w, 0.70, 0.73, 0.085, 0.18) | ellipse_mask(h, w, 0.71, 0.88, 0.055, 0.10)
    hair = top_cap | bangs | left_curtain | right_curtain | left_lower | right_lower | left_braid_old | right_braid_old

    # Shirt prior: lower bust region, but avoid being too high.
    shirt_ellipse = ellipse_mask(h, w, 0.50, 0.94, 0.46, 0.21)
    shirt_rect = rect_mask(h, w, 0.05, 0.735, 0.95, 1.0)
    shirt = shirt_ellipse | shirt_rect

    # Clip all semantic priors to coarse portrait support where possible.
    face &= support
    neck &= support
    accessories &= support
    hair &= support
    shirt &= support

    not_color = face | neck | accessories
    return Priors(face=face, neck=neck, accessories=accessories, hair=hair, shirt=shirt, portrait_support=support, not_color=not_color)


# ============================================================
# COMPONENT-BASED REGION CLASSIFICATION
# ============================================================

@dataclass
class ComponentInfo:
    label: int
    area: int
    cx: float
    cy: float
    bbox: tuple
    hair_overlap: float
    shirt_overlap: float
    face_overlap: float
    neck_overlap: float
    accessory_overlap: float
    support_overlap: float
    touches_border: bool


def analyze_components(free_mask, priors):
    h, w = free_mask.shape
    min_area = max(20, int(MIN_REGION_COMPONENT_AREA_FRAC * h * w))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8(free_mask), connectivity=8)
    infos = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == label
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        denom = max(area, 1)
        touches_border = x <= 1 or y <= 1 or x + ww >= w - 2 or y + hh >= h - 2
        infos.append(ComponentInfo(
            label=label,
            area=area,
            cx=float(cx / w),
            cy=float(cy / h),
            bbox=(x, y, ww, hh),
            hair_overlap=float(np.sum(comp & priors.hair) / denom),
            shirt_overlap=float(np.sum(comp & priors.shirt) / denom),
            face_overlap=float(np.sum(comp & priors.face) / denom),
            neck_overlap=float(np.sum(comp & priors.neck) / denom),
            accessory_overlap=float(np.sum(comp & priors.accessories) / denom),
            support_overlap=float(np.sum(comp & priors.portrait_support) / denom),
            touches_border=touches_border,
        ))
    return labels, infos


def component_mask(labels, selected_labels):
    out = np.zeros(labels.shape, dtype=bool)
    for label in selected_labels:
        out |= labels == label
    return out


def grow_from_seeds(barrier_mask, constraint_mask, seeds):
    h, w = barrier_mask.shape
    free = (~barrier_mask) & constraint_mask
    visited = np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=bool)
    queue = []
    for (x, y) in seeds:
        x = int(round(x)); y = int(round(y))
        if 0 <= x < w and 0 <= y < h and free[y, x]:
            queue.append((x, y)); visited[y, x] = 1; out[y, x] = True
    head = 0
    while head < len(queue):
        x, y = queue[head]; head += 1
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = 1
                if free[ny, nx]:
                    out[ny, nx] = True
                    queue.append((nx, ny))
    return out


def expand_color_masks_inside_support(hair_mask, shirt_mask, barrier, priors):
    colorable = (~barrier) & (~priors.not_color) & priors.portrait_support

    expanded_hair = morph_dilate(hair_mask, 5) & colorable & (~shirt_mask)
    expanded_hair &= priors.hair | morph_dilate(priors.hair, 9)
    hair_mask |= expanded_hair

    expanded_shirt = morph_dilate(shirt_mask, 7) & colorable & (~hair_mask)
    expanded_shirt &= priors.shirt | morph_dilate(priors.shirt, 11)
    shirt_mask |= expanded_shirt

    return hair_mask & colorable, shirt_mask & colorable & (~hair_mask)


def build_fill_masks(line_mask, priors):
    h, w = line_mask.shape
    barrier = morph_dilate(line_mask, BARRIER_DILATE_PX)
    free = (~barrier) & priors.portrait_support
    labels, infos = analyze_components(free, priors)

    shirt_labels = []
    hair_labels = []

    for info in infos:
        neg_for_shirt = max(info.face_overlap, info.neck_overlap, info.accessory_overlap, info.hair_overlap * 0.55)
        shirt_score = info.shirt_overlap + 0.16 * max(0.0, info.cy - 0.68)
        if shirt_score >= SHIRT_MIN_POSITIVE_OVERLAP and neg_for_shirt <= SHIRT_MAX_NEGATIVE_OVERLAP:
            shirt_labels.append(info.label)

    preliminary_shirt = component_mask(labels, shirt_labels)

    for info in infos:
        if info.label in shirt_labels:
            continue
        neg_for_hair = max(info.face_overlap, info.neck_overlap, info.accessory_overlap, info.shirt_overlap * 0.75)
        # More permissive for top/side components; less permissive in center face.
        side_bonus = 0.08 if (info.cx < 0.36 or info.cx > 0.64) else 0.0
        top_bonus = 0.07 if info.cy < 0.36 else 0.0
        hair_score = info.hair_overlap + side_bonus + top_bonus
        is_center_face_like = 0.34 < info.cx < 0.66 and 0.34 < info.cy < 0.64 and info.face_overlap > 0.10
        if hair_score >= HAIR_MIN_POSITIVE_OVERLAP and neg_for_hair <= HAIR_MAX_NEGATIVE_OVERLAP and not is_center_face_like:
            hair_labels.append(info.label)

    hair_mask = component_mask(labels, hair_labels)
    shirt_mask = preliminary_shirt & (~hair_mask)

    # Seed-based recovery catches line-art that is not separated into clean components.
    hair_constraint = priors.hair & (~priors.not_color) & (~shirt_mask) & priors.portrait_support
    hair_seeds = [
        (0.50*w, 0.18*h), (0.39*w, 0.24*h), (0.61*w, 0.24*h),
        (0.27*w, 0.48*h), (0.24*w, 0.65*h), (0.27*w, 0.83*h),
        (0.73*w, 0.48*h), (0.76*w, 0.65*h), (0.73*w, 0.83*h),
    ]
    hair_seed_mask = grow_from_seeds(barrier, hair_constraint, hair_seeds)
    hair_mask |= hair_seed_mask

    shirt_constraint = priors.shirt & (~priors.not_color) & (~hair_mask) & priors.portrait_support
    shirt_seeds = [(0.50*w, 0.88*h), (0.30*w, 0.91*h), (0.70*w, 0.91*h)]
    shirt_seed_mask = grow_from_seeds(barrier, shirt_constraint, shirt_seeds)
    shirt_mask |= shirt_seed_mask

    # Strong fallback: if classified hair/shirt is tiny, use safe prior directly.
    hair_prior_safe = priors.hair & (~priors.not_color) & (~shirt_mask) & (~barrier) & priors.portrait_support
    hair_prior_area = max(1, int(np.sum(hair_prior_safe)))
    if np.sum(hair_mask) < HAIR_FALLBACK_MIN_PRIOR_COVERAGE * hair_prior_area:
        hair_mask |= hair_prior_safe

    shirt_prior_safe = priors.shirt & (~priors.not_color) & (~hair_mask) & (~barrier) & priors.portrait_support
    shirt_prior_area = max(1, int(np.sum(shirt_prior_safe)))
    if np.sum(shirt_mask) < SHIRT_FALLBACK_MIN_PRIOR_COVERAGE * shirt_prior_area:
        shirt_mask |= shirt_prior_safe

    hair_mask, shirt_mask = expand_color_masks_inside_support(
        hair_mask,
        shirt_mask,
        barrier,
        priors,
    )

    # Cleanup and exclusivity.
    hair_mask = morph_close(hair_mask, 7)
    hair_mask = morph_open(hair_mask, 2)
    hair_mask &= (~barrier) & (~priors.not_color) & priors.portrait_support

    shirt_mask = morph_close(shirt_mask, 9)
    shirt_mask = morph_open(shirt_mask, 2)
    shirt_mask &= (~barrier) & (~priors.not_color) & (~hair_mask) & priors.portrait_support

    return hair_mask, shirt_mask, barrier, free, infos


# ============================================================
# OUTLINE CENTERLINE TRACING
# ============================================================

def skeletonize_mask(mask):
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask.astype(bool))
        skel = keep_components_larger_than(skel, 3)
        return skel
    except Exception:
        img = mask_u8(mask)
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = cv2.erode(img, element)
            if cv2.countNonZero(img) == 0:
                break
        skel = skel > 0
        skel = keep_components_larger_than(skel, 3)
        return skel


def trace_skeleton_paths(skel):
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    pixels = set(zip(xs.tolist(), ys.tolist()))
    if not pixels:
        return []

    nbrs = {}
    for x, y in pixels:
        ns = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in pixels:
                    ns.append(q)
        nbrs[(x, y)] = ns

    degree = {p: len(nbrs[p]) for p in pixels}
    nodes = {p for p, d in degree.items() if d != 2}
    visited_edges = set()
    paths = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def walk(start, nxt):
        path = [start, nxt]
        visited_edges.add(edge_key(start, nxt))
        prev, curr = start, nxt
        while curr not in nodes:
            candidates = [q for q in nbrs[curr] if q != prev]
            if not candidates:
                break
            q = candidates[0]
            ek = edge_key(curr, q)
            if ek in visited_edges:
                break
            visited_edges.add(ek)
            path.append(q)
            prev, curr = curr, q
        return path

    for start in list(nodes):
        for nxt in nbrs[start]:
            if edge_key(start, nxt) not in visited_edges:
                paths.append(walk(start, nxt))

    # Closed loops with no endpoints/junctions.
    for p in list(pixels):
        for q in nbrs[p]:
            if edge_key(p, q) in visited_edges:
                continue
            loop = [p, q]
            visited_edges.add(edge_key(p, q))
            prev, curr = p, q
            while True:
                candidates = [r for r in nbrs[curr] if r != prev]
                if not candidates:
                    break
                r = candidates[0]
                if r == p:
                    break
                ek = edge_key(curr, r)
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                loop.append(r)
                prev, curr = curr, r
            paths.append(loop)

    return [np.asarray(path, dtype=np.float64) for path in paths if len(path) >= 3]


def skeleton_to_outline_strokes(line_mask, canvas_w_mm, canvas_h_mm):
    h, w = line_mask.shape
    skel = skeletonize_mask(line_mask)
    paths = trace_skeleton_paths(skel)
    strokes = []
    min_len_px = (MIN_OUTLINE_PATH_LEN_MM / canvas_w_mm) * w

    for path in paths:
        if polyline_length_px(path) < min_len_px:
            continue
        path = resample_polyline_px(path, OUTLINE_POINT_SPACING_PX)
        if len(path) < 2:
            continue
        pts_px_int = path.round().astype(int)
        pts_mm = [px_to_mm(int(x), int(y), w, h, canvas_w_mm, canvas_h_mm) for x, y in pts_px_int]
        strokes.append({
            "type": "centerline_outline",
            "layer": "outline_black",
            "closed": False,
            "brush_width_mm": OUTLINE_BRUSH_WIDTH_MM,
            "length_mm": round(float(polyline_length_mm(pts_mm)), 3),
            "points_px": pts_px_int.tolist(),
            "points_mm": pts_mm,
        })
    strokes.sort(key=lambda st: -polyline_length_px(np.asarray(st["points_px"])))
    return strokes, skel


def contours_to_outline_strokes(line_mask, canvas_w_mm, canvas_h_mm):
    h, w = line_mask.shape
    contours, _ = cv2.findContours(mask_u8(line_mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for contour in contours:
        if len(contour) < 10:
            continue
        area = abs(cv2.contourArea(contour))
        if area < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        eps = 0.006 * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)
        pts_px = approx[:, 0, :]
        if len(pts_px) < 3:
            continue
        filleted = fillet_polyline(pts_px, closed=True)
        pts_px_int = filleted.round().astype(int)
        pts_mm = [px_to_mm(int(x), int(y), w, h, canvas_w_mm, canvas_h_mm) for x, y in pts_px_int]
        strokes.append({
            "type": "polyline_outline_boundary_fallback",
            "layer": "outline_black",
            "closed": True,
            "brush_width_mm": OUTLINE_BRUSH_WIDTH_MM,
            "points_px": pts_px_int.tolist(),
            "points_mm": pts_mm,
        })
    random.shuffle(strokes)
    return strokes


# ============================================================
# FLOW-ALIGNED HATCH FILL STROKES
# ============================================================

def generate_hatch_fill_strokes(mask, region_name, canvas_w_mm, canvas_h_mm, brush_width_mm, angle_deg,
                                min_segment_len_mm=HATCH_MIN_SEGMENT_LEN_MM):
    """Generate clipped hatch strokes with enforced boustrophedon ordering.

    This function treats the hatch pattern as a scan process in a local 2D
    basis:

        d = along-stroke direction
        n = across-stroke direction

    Ordering is deterministic and robot-efficient:

      * Vertical-ish strokes:
            start at the left-most column, then move column-by-column to the
            right.  Column 0 paints top->bottom, column 1 paints bottom->top,
            column 2 paints top->bottom, etc.

      * Horizontal-ish strokes:
            start at the top-most row, then move row-by-row downward. Row 0
            paints left->right, row 1 paints right->left, row 2 paints
            left->right, etc.

      * Diagonal strokes:
            use the same projection-space rule: scan from low n to high n,
            alternating the sign of the d traversal every stripe.

    The final stroke list is NOT globally resorted afterward, because that
    would destroy the boustrophedon execution order.
    """
    h, w = mask.shape
    brush_px = mm_to_px(brush_width_mm, w, canvas_w_mm)
    safe_margin_px = max(1, int(round(HATCH_SAFE_MARGIN_FACTOR * brush_px)))
    step_px = max(2, int(round(HATCH_STEP_FACTOR * brush_px)))

    safe = morph_erode(mask, 2 * safe_margin_px + 1)
    # If eroding kills narrow strands, accept the original mask but hatches are
    # still clipped to the color region.
    if np.sum(safe) < 0.10 * max(1, np.sum(mask)):
        safe = mask.copy()

    ys, xs = np.nonzero(safe)
    if len(xs) == 0:
        return []

    # --------------------------------------------------------
    # Canonical scan basis.
    # --------------------------------------------------------
    theta = math.radians(angle_deg + random.uniform(-HATCH_ANGLE_JITTER_DEG, HATCH_ANGLE_JITTER_DEG))
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    d /= max(float(np.linalg.norm(d)), 1e-9)

    # Image coordinates: x grows rightward, y grows downward.
    # For horizontal-ish hatching, forward should mean left->right and the
    # scan should progress top->bottom.
    # For vertical-ish hatching, forward should mean top->bottom and the scan
    # should progress left->right.
    if abs(d[0]) >= abs(d[1]):
        # Horizontal-ish: d = left->right.
        if d[0] < 0:
            d = -d
        n = np.array([-d[1], d[0]], dtype=np.float64)  # mostly downward
        if n[1] < 0:
            n = -n
        scan_major_axis = "horizontal_rows_top_to_bottom"
        forward_semantics = "left_to_right"
        backward_semantics = "right_to_left"
    else:
        # Vertical-ish: d = top->bottom.
        if d[1] < 0:
            d = -d
        n = np.array([d[1], -d[0]], dtype=np.float64)  # mostly rightward
        if n[0] < 0:
            n = -n
        scan_major_axis = "vertical_columns_left_to_right"
        forward_semantics = "top_to_bottom"
        backward_semantics = "bottom_to_top"

    n /= max(float(np.linalg.norm(n)), 1e-9)

    coords = np.column_stack([xs, ys]).astype(np.float64)
    across = coords @ n
    min_a = float(np.min(across)) - step_px
    max_a = float(np.max(across)) + step_px

    diag = int(math.ceil(math.hypot(w, h))) + 20
    offsets = list(np.arange(min_a, max_a + 1e-6, step_px))
    offsets.sort()  # enforced scan order: top->bottom or left->right

    center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    min_seg_len_px = (min_segment_len_mm / canvas_w_mm) * w

    rows = []
    for raw_row_idx, off in enumerate(offsets):
        p0 = center + (off - float(center @ n)) * n
        p1 = p0 - diag * d
        p2 = p0 + diag * d

        line_img = np.zeros((h, w), dtype=np.uint8)
        cv2.line(
            line_img,
            tuple(np.round(p1).astype(int)),
            tuple(np.round(p2).astype(int)),
            255,
            1,
            cv2.LINE_AA,
        )
        clipped = (line_img > 0) & safe
        clipped = keep_components_larger_than(clipped, 2)
        contours, _ = cv2.findContours(mask_u8(clipped), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        row_segments = []
        for contour in contours:
            pts = contour[:, 0, :].astype(np.float64)
            if len(pts) < 2:
                continue

            # Within a stripe, impose geometric point order along d.
            order = np.argsort(pts @ d)
            pts = pts[order]

            if (
                polyline_length_px(pts) < min_seg_len_px
                or projected_length_px(pts, d) < min_seg_len_px
            ):
                continue

            pts = resample_polyline_px(pts, HATCH_POINT_SPACING_PX)
            pts = chaikin_open_polyline(pts, iterations=HATCH_SMOOTH_ITERATIONS, ratio=0.18)
            pts = resample_polyline_px(pts, HATCH_POINT_SPACING_PX)
            if len(pts) < 2:
                continue
            if (
                polyline_length_px(pts) < min_seg_len_px
                or projected_length_px(pts, d) < min_seg_len_px
            ):
                continue

            row_segments.append({
                "pts": pts,
                "along_mean": float(np.mean(pts @ d)),
                "across_mean": float(np.mean(pts @ n)),
            })

        if row_segments:
            # For forward rows, visit disconnected segments from low->high along d.
            # For backward rows, this will be reversed later.
            row_segments.sort(key=lambda seg: seg["along_mean"])
            rows.append({
                "raw_row_idx": raw_row_idx,
                "offset": float(off),
                "segments": row_segments,
            })

    strokes = []
    for scan_row_idx, row in enumerate(rows):
        forward = (scan_row_idx % 2 == 0)
        segs = row["segments"] if forward else list(reversed(row["segments"]))

        for seg in segs:
            pts = seg["pts"]
            if not forward:
                pts = pts[::-1]

            pts_px_int = pts.round().astype(int)
            pts_mm = [px_to_mm(int(x), int(y), w, h, canvas_w_mm, canvas_h_mm) for x, y in pts_px_int]
            stroke_length_mm = polyline_length_mm(pts_mm)
            if stroke_length_mm < min_segment_len_mm:
                continue

            strokes.append({
                "type": "boustrophedon_hatch_fill",
                "layer": region_name,
                "closed": False,
                "brush_width_mm": brush_width_mm,
                "length_mm": round(float(stroke_length_mm), 3),
                "angle_deg": round(float(angle_deg), 3),
                "scan_major_axis": scan_major_axis,
                "scan_row_index": int(scan_row_idx),
                "scan_raw_row_index": int(row["raw_row_idx"]),
                "scan_offset_px": round(float(row["offset"]), 3),
                "scan_forward": bool(forward),
                "direction_semantics": forward_semantics if forward else backward_semantics,
                "points_px": pts_px_int.tolist(),
                "points_mm": pts_mm,
            })

    return strokes

def generate_flow_hair_strokes(hair_mask, canvas_w_mm, canvas_h_mm, brush_width_mm):
    """Generate hair strokes with local flow angles per component.

    Long-hair portraits read much better when side-hair components are shaded
    vertically/diagonally down the strands, while bangs/top hair use a slightly
    different flow. This component-local choice also keeps strokes long inside
    each hair pocket instead of chopping the whole head with one global angle.
    """
    h, w = hair_mask.shape
    min_area = max(25, int(0.00005 * h * w))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8(hair_mask), connectivity=8)
    components = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label]
        components.append((label, area, cx / w, cy / h))

    # Largest/upper/outer components first.
    components.sort(key=lambda t: (t[3], -t[1]))

    all_strokes = []
    for label, area, cx, cy in components:
        comp = labels == label
        if cy < 0.38:
            angle = HAIR_BANGS_ANGLE_DEG
        elif cx < 0.48:
            angle = HAIR_LEFT_ANGLE_DEG
        elif cx > 0.52:
            angle = HAIR_RIGHT_ANGLE_DEG
        else:
            angle = HAIR_TOP_ANGLE_DEG

        # Smaller pockets need a lower minimum length or they will vanish.
        min_len = 10.0 if area < 0.01 * h * w else HATCH_MIN_SEGMENT_LEN_MM
        strokes = generate_hatch_fill_strokes(
            comp,
            region_name="hair_color_2",
            canvas_w_mm=canvas_w_mm,
            canvas_h_mm=canvas_h_mm,
            brush_width_mm=brush_width_mm,
            angle_deg=angle,
            min_segment_len_mm=min_len,
        )
        all_strokes.extend(strokes)

    return all_strokes


# ============================================================
# PREVIEW / ANIMATION
# ============================================================

def render_region_preview(line_mask, hair_mask, shirt_mask, out_path):
    h, w = line_mask.shape
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    hair_col = np.array([80, 120, 180], dtype=np.uint8)    # BGR brown
    shirt_col = np.array([170, 210, 235], dtype=np.uint8)  # BGR beige
    vis[shirt_mask] = shirt_col
    vis[hair_mask] = hair_col
    vis[line_mask] = np.array([0, 0, 0], dtype=np.uint8)
    cv2.imwrite(str(out_path), vis)


def render_debug_overlay(line_mask, priors, hair_mask, shirt_mask, free_mask, out_path):
    h, w = line_mask.shape
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255
    vis[priors.portrait_support] = (245, 245, 245)
    vis[free_mask] = (255, 250, 245)
    vis[priors.face] = (240, 240, 255)
    vis[priors.hair] = (220, 235, 255)
    vis[priors.shirt] = (225, 255, 255)
    vis[shirt_mask] = (170, 210, 235)
    vis[hair_mask] = (80, 120, 180)
    vis[line_mask] = (0, 0, 0)
    cv2.imwrite(str(out_path), vis)


def render_stroke_preview(image_shape, outline_strokes, hair_strokes, shirt_strokes, canvas_w_mm, canvas_h_mm, out_path, black_on_top=False):
    h, w = image_shape
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255

    def draw_stroke(stroke, color, default_width_mm):
        pts = np.array(stroke["points_px"], dtype=np.int32).reshape(-1, 1, 2)
        thickness = mm_to_px(float(stroke.get("brush_width_mm", default_width_mm)), w, canvas_w_mm)
        cv2.polylines(vis, [pts], isClosed=stroke.get("closed", False), color=color, thickness=thickness, lineType=cv2.LINE_AA)

    if not black_on_top:
        for st in outline_strokes:
            draw_stroke(st, (0, 0, 0), OUTLINE_BRUSH_WIDTH_MM)
        for st in hair_strokes:
            draw_stroke(st, (80, 120, 180), FILL_BRUSH_WIDTH_MM)
        for st in shirt_strokes:
            draw_stroke(st, (170, 210, 235), FILL_BRUSH_WIDTH_MM)
    else:
        for st in shirt_strokes:
            draw_stroke(st, (170, 210, 235), FILL_BRUSH_WIDTH_MM)
        for st in hair_strokes:
            draw_stroke(st, (80, 120, 180), FILL_BRUSH_WIDTH_MM)
        for st in outline_strokes:
            draw_stroke(st, (0, 0, 0), OUTLINE_BRUSH_WIDTH_MM)
    cv2.imwrite(str(out_path), vis)


def render_paint_stroke_animation(image_shape, outline_strokes, hair_strokes, shirt_strokes, canvas_w_mm, out_path, fps=24, strokes_per_frame=1, hold_seconds=1.0):
    h, w = image_shape
    fps = max(1, int(fps))
    strokes_per_frame = max(1, int(strokes_per_frame))
    hold_frames = max(0, int(round(float(hold_seconds) * fps)))
    ensure_dir(str(Path(out_path).parent))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    def draw_stroke(stroke, color, default_width_mm):
        pts = np.array(stroke["points_px"], dtype=np.int32).reshape(-1, 1, 2)
        thickness = mm_to_px(float(stroke.get("brush_width_mm", default_width_mm)), w, canvas_w_mm)
        cv2.polylines(canvas, [pts], isClosed=stroke.get("closed", False), color=color, thickness=thickness, lineType=cv2.LINE_AA)

    ordered_layers = [
        (shirt_strokes, (170, 210, 235), FILL_BRUSH_WIDTH_MM),
        (hair_strokes, (80, 120, 180), FILL_BRUSH_WIDTH_MM),
        (outline_strokes, (0, 0, 0), OUTLINE_BRUSH_WIDTH_MM),
    ]
    for _ in range(max(1, hold_frames // 2)):
        writer.write(canvas)
    counter = 0
    for strokes, color, default_width in ordered_layers:
        for st in strokes:
            draw_stroke(st, color, default_width)
            counter += 1
            if counter >= strokes_per_frame:
                writer.write(canvas)
                counter = 0
        if counter:
            writer.write(canvas); counter = 0
        for _ in range(hold_frames):
            writer.write(canvas)
    writer.release()


def write_component_debug_json(path, infos):
    data = []
    for info in infos:
        data.append({
            "label": info.label,
            "area_px": info.area,
            "centroid_frac": [round(info.cx, 4), round(info.cy, 4)],
            "bbox_px": list(map(int, info.bbox)),
            "overlap": {
                "hair": round(info.hair_overlap, 4),
                "shirt": round(info.shirt_overlap, 4),
                "face": round(info.face_overlap, 4),
                "neck": round(info.neck_overlap, 4),
                "accessory": round(info.accessory_overlap, 4),
                "support": round(info.support_overlap, 4),
            },
            "touches_border": bool(info.touches_border),
        })
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================
# MAIN
# ============================================================

def main():
    global FILLET_ANGLE_THRESHOLD_DEG, FILLET_RATIO, FILLET_ITERATIONS
    global FILL_BRUSH_WIDTH_MM, OUTLINE_BRUSH_WIDTH_MM, BARRIER_DILATE_PX

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Input line-art image")
    parser.add_argument("--strokes-dir", default=str(DEFAULT_STROKES_DIR), type=str)
    parser.add_argument("--stroke-jsons-dir", default=str(DEFAULT_STROKE_JSONS_DIR), type=str)
    parser.add_argument("--animation-output", default=None, type=str)
    parser.add_argument("--animation-fps", default=24, type=int)
    parser.add_argument("--animation-strokes-per-frame", default=1, type=int)
    parser.add_argument("--animation-hold-seconds", default=1.0, type=float)
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--canvas_w_mm", default=CANVAS_WIDTH_MM, type=float)
    parser.add_argument("--canvas_h_mm", default=CANVAS_HEIGHT_MM, type=float)
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    parser.add_argument("--fill-brush-mm", default=FILL_BRUSH_WIDTH_MM, type=float)
    parser.add_argument("--outline-brush-mm", default=OUTLINE_BRUSH_WIDTH_MM, type=float)
    parser.add_argument("--barrier-dilate-px", default=BARRIER_DILATE_PX, type=int)
    parser.add_argument("--no-skeleton-outline", action="store_true")
    parser.add_argument("--fillet-angle-deg", type=float, default=FILLET_ANGLE_THRESHOLD_DEG)
    parser.add_argument("--fillet-ratio", type=float, default=FILLET_RATIO)
    parser.add_argument("--fillet-iterations", type=int, default=FILLET_ITERATIONS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    FILL_BRUSH_WIDTH_MM = float(args.fill_brush_mm)
    OUTLINE_BRUSH_WIDTH_MM = float(args.outline_brush_mm)
    BARRIER_DILATE_PX = int(args.barrier_dilate_px)
    FILLET_ANGLE_THRESHOLD_DEG = float(args.fillet_angle_deg)
    FILLET_RATIO = float(args.fillet_ratio)
    FILLET_ITERATIONS = int(args.fillet_iterations)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Could not read image: {input_path}")
    stem = input_path.stem
    strokes_dir = Path(args.strokes_dir) / stem
    stroke_jsons_dir = Path(args.stroke_jsons_dir)
    ensure_dir(str(strokes_dir)); ensure_dir(str(stroke_jsons_dir))

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    h, w = image_bgr.shape[:2]

    print("Robust pipeline settings:")
    print(f"  fill_brush_mm={FILL_BRUSH_WIDTH_MM}")
    print(f"  outline_brush_mm={OUTLINE_BRUSH_WIDTH_MM}")
    print(f"  barrier_dilate_px={BARRIER_DILATE_PX}")
    print(f"  skeleton_outline={not args.no_skeleton_outline}")

    # 1. Extract line art and priors.
    line_mask = extract_line_mask(image_bgr)
    priors = build_priors(h, w, line_mask)

    # 2. Robust region segmentation.
    hair_mask, shirt_mask, barrier, free_mask, comp_infos = build_fill_masks(line_mask, priors)
    hair_mask &= (~line_mask)
    shirt_mask &= (~line_mask) & (~hair_mask)

    # 3. Generate strokes.
    if not args.no_skeleton_outline:
        outline_strokes, skel = skeleton_to_outline_strokes(line_mask, args.canvas_w_mm, args.canvas_h_mm)
    else:
        outline_strokes = contours_to_outline_strokes(line_mask, args.canvas_w_mm, args.canvas_h_mm)
        skel = np.zeros_like(line_mask, dtype=bool)

    hair_strokes = generate_flow_hair_strokes(hair_mask, args.canvas_w_mm, args.canvas_h_mm, FILL_BRUSH_WIDTH_MM)
    shirt_strokes = generate_hatch_fill_strokes(shirt_mask, "shirt_color_3", args.canvas_w_mm, args.canvas_h_mm, FILL_BRUSH_WIDTH_MM, SHIRT_HATCH_ANGLE_DEG)

    # 4. Visual diagnostics.
    cv2.imwrite(str(strokes_dir / "00_input.png"), image_bgr)
    save_mask(strokes_dir / "01_line_mask.png", line_mask)
    save_mask(strokes_dir / "02_hair_mask.png", hair_mask)
    save_mask(strokes_dir / "03_shirt_mask.png", shirt_mask)
    save_mask(strokes_dir / "04_barrier_mask.png", barrier)
    save_mask(strokes_dir / "04b_portrait_support.png", priors.portrait_support)
    save_mask(strokes_dir / "04c_skeleton.png", skel)
    save_mask(strokes_dir / "04d_free_space.png", free_mask)

    render_region_preview(line_mask, hair_mask, shirt_mask, strokes_dir / "05_region_preview.png")
    render_debug_overlay(line_mask, priors, hair_mask, shirt_mask, free_mask, strokes_dir / "05b_debug_overlay.png")
    render_stroke_preview((h, w), outline_strokes, hair_strokes, shirt_strokes, args.canvas_w_mm, args.canvas_h_mm, strokes_dir / "06_stroke_preview_requested_order.png", black_on_top=False)
    render_stroke_preview((h, w), outline_strokes, hair_strokes, shirt_strokes, args.canvas_w_mm, args.canvas_h_mm, strokes_dir / "07_stroke_preview_clean_visual.png", black_on_top=True)
    write_component_debug_json(strokes_dir / "09_component_debug.json", comp_infos)

    animation_path = None
    if not args.no_animation:
        animation_path = Path(args.animation_output) if args.animation_output else strokes_dir / "08_paint_stroke_animation.mp4"
        render_paint_stroke_animation((h, w), outline_strokes, hair_strokes, shirt_strokes, args.canvas_w_mm, str(animation_path), fps=args.animation_fps, strokes_per_frame=args.animation_strokes_per_frame, hold_seconds=args.animation_hold_seconds)

    # 5. JSON painting plan.
    painting_plan = {
        "input": str(input_path),
        "canvas_width_mm": args.canvas_w_mm,
        "canvas_height_mm": args.canvas_h_mm,
        "execution_order_requested": ["shirt_color_3", "hair_color_2", "outline_black"],
        "recommended_physical_order": ["shirt_color_3", "hair_color_2", "outline_black"],
        "animation_output": str(animation_path) if animation_path else None,
        "diagnostics": {
            "visuals_dir": str(strokes_dir),
            "component_debug_json": str(strokes_dir / "09_component_debug.json"),
            "hair_mask_area_px": int(np.sum(hair_mask)),
            "shirt_mask_area_px": int(np.sum(shirt_mask)),
            "line_mask_area_px": int(np.sum(line_mask)),
            "portrait_support_area_px": int(np.sum(priors.portrait_support)),
            "hatch_step_factor": HATCH_STEP_FACTOR,
            "hatch_safe_margin_factor": HATCH_SAFE_MARGIN_FACTOR,
            "barrier_dilate_px": BARRIER_DILATE_PX,
        },
        "notes": [
            "Region segmentation uses portrait-support masking plus connected-component classification by overlap with broad semantic priors.",
            "Hair uses broad top-cap, bangs, side-curtain, lower-hair, and old-braid priors, then component classification plus seed/prior fallback.",
            "This avoids the old failure mode where only a tiny braid-like ellipse is colored for long-hair portraits.",
            "Hair strokes are flow-aligned per component: top/bangs mostly vertical, left side tilted down-right, right side tilted down-left.",
            "Fill strokes are enforced boustrophedon scanlines: vertical-ish regions scan left-to-right with alternating top-to-bottom/bottom-to-top traversal; horizontal-ish regions scan top-to-bottom with alternating left-to-right/right-to-left traversal.",
            "Outline generation uses skeleton centerline tracing so the robot follows intended line centers instead of jagged thick-line boundaries.",
            "Robot executor should still time-parameterize each polyline with velocity/acceleration limits and approach-contact-draw-retract phases.",
        ],
        "layers": [
            {"name": "outline_black", "brush_width_mm": OUTLINE_BRUSH_WIDTH_MM, "stroke_count": len(outline_strokes), "strokes": outline_strokes},
            {"name": "hair_color_2", "brush_width_mm": FILL_BRUSH_WIDTH_MM, "stroke_count": len(hair_strokes), "strokes": hair_strokes},
            {"name": "shirt_color_3", "brush_width_mm": FILL_BRUSH_WIDTH_MM, "stroke_count": len(shirt_strokes), "strokes": shirt_strokes},
        ],
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
    print(f"Hair mask area:  {int(np.sum(hair_mask))} px")
    print(f"Shirt mask area: {int(np.sum(shirt_mask))} px")
    if animation_path:
        print(f"Animation:       {animation_path}")
    print("Inspect these first:")
    print(f"  {strokes_dir / '05_region_preview.png'}")
    print(f"  {strokes_dir / '05b_debug_overlay.png'}")
    print(f"  {strokes_dir / '07_stroke_preview_clean_visual.png'}")
    print(f"  {strokes_dir / '09_component_debug.json'}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
