import argparse
import os
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.morphology import skeletonize

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else iter(())


# ============================================================
# DEBUG LOGGING
# ============================================================

_step_counter = {"n": 0, "total": 0}


def _log(msg):
    print(f"[pipeline] {msg}", flush=True)


def _set_total_steps(total):
    _step_counter["n"] = 0
    _step_counter["total"] = total


@contextmanager
def _step(label):
    _step_counter["n"] += 1
    idx = _step_counter["n"]
    total = _step_counter["total"]
    prefix = f"[{idx}/{total}]" if total else f"[{idx}]"
    _log(f"{prefix} {label} ...")
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _log(f"{prefix} {label} done in {elapsed:.2f}s")


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_JSONS_DIR = SCRIPT_DIR / "stroke-jsons"
DEFAULT_RENDERS_DIR = SCRIPT_DIR / "stroke-renders"

# ----------------------------
# Binary / raster cleanup
# ----------------------------
THRESHOLD = 210

MIN_COMPONENT_AREA = 100
OPEN_KERNEL_SIZE = 2
CLOSE_KERNEL_SIZE = 1
BBOX_PAD = 20

# ----------------------------
# Skeleton cleanup
# ----------------------------
PRUNE_SPUR_ITERS = 18
MIN_BRANCH_PIXELS = 8

# ----------------------------
# Pupil -> iris replacement
# ----------------------------
PUPIL_MIN_AREA = 120
PUPIL_MAX_AREA = 5000
PUPIL_CIRCULARITY_MIN = 0.70
PUPIL_Y_MIN_FRAC = 0.20
PUPIL_Y_MAX_FRAC = 0.70

IRIS_SCALE = 1.35
IRIS_EXTRA_PX = 5
IRIS_NUM_POINTS = 36

# ----------------------------
# Glasses fallback iris placement
# ----------------------------
USE_GLASSES_FALLBACK_IRIS = True
GLASSES_Y_MIN_FRAC = 0.25
GLASSES_Y_MAX_FRAC = 0.62
GLASSES_AREA_MIN_FRAC = 0.006
GLASSES_AREA_MAX_FRAC = 0.15
GLASSES_CIRCULARITY_MIN = 0.45
GLASSES_ASPECT_MIN = 0.60
GLASSES_ASPECT_MAX = 1.60
IRIS_FROM_GLASSES_RATIO = 1.0 / 3.0

# ----------------------------
# Stroke extraction / simplification
# ----------------------------
MIN_STROKE_POINTS = 4
MIN_STROKE_LENGTH = 28.0
MIN_STROKE_BBOX_AREA = 30.0
SIMPLIFY_EPS = 2.4
CLOSE_THRESH = 4.0

# ----------------------------
# Post-processing: smoothing + stroke merging
# ----------------------------
ENABLE_STROKE_SMOOTHING = True

SMOOTH_ITERATIONS = 2
POST_SMOOTH_SIMPLIFY_EPS = 2.0

# Robot interpolates between control points, so this can stay low.
MAX_POINTS_PER_STROKE = 60

ENABLE_STROKE_MERGING = True
MAX_MERGE_PASSES = 6

# Endpoint distance threshold in pixels.
MERGE_ENDPOINT_DIST = 16.0

MERGE_ALLOWED_FEATURES = {
    "outer_contour",
    "hair",
    "nose",
    "mouth",
    "ear",
    "neck_collar",
    "shoulders",
    "misc",
}

MERGE_BLOCKED_FEATURES = {
    "left_eye",
    "right_eye",
    "left_iris",
    "right_iris",
}

# If endpoints are extremely close, allow merge even if direction is imperfect.
FORCE_MERGE_CLOSE_DIST = 6.0

# Angle compatibility. Higher = stricter. -1 disables direction check.
MERGE_DIRECTION_DOT_MIN = -0.25

# ----------------------------
# Filleting only for merge connector segments
# ----------------------------
USE_FILLETED_MERGE_CONNECTORS = True

# Maximum connector fillet radius in pixels.
# This is only used between two merged stroke fragments.
MERGE_CONNECTOR_FILLET_RADIUS_PX = 14.0

# Number of points on the connector curve.
MERGE_CONNECTOR_SAMPLES = 8

# If the gap is tiny, use fewer samples.
MERGE_CONNECTOR_MIN_SAMPLES = 3

# If gap is bigger than this, still merge only if allowed by MERGE_ENDPOINT_DIST,
# but keep connector visually softer.
MERGE_CONNECTOR_MAX_CONTROL_FRAC = 0.45

# ----------------------------
# Geometric junk deletion
# ----------------------------
DELETE_TINY_STROKES_NEAR_EYES = True
TINY_NEAR_EYE_LENGTH = 45.0
TINY_NEAR_EYE_BBOX_AREA = 100.0

DELETE_TOP_BORDER_ARTIFACTS = True
TOP_BORDER_FRAC = 0.06
TOP_BORDER_MIN_LENGTH_KEEP = 80.0

# ----------------------------
# Rendering
# ----------------------------
CENTERLINE_PREVIEW_THICKNESS = 1
MARKER_PREVIEW_THICKNESS = 7
ANIM_LINE_THICKNESS = 7
ANIM_FRAME_DURATION = 0.045
ANIM_PROGRESS_STEPS_PER_STROKE = 10


# ============================================================
# BASIC UTILS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def contour_circularity(cnt):
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def path_length(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def is_closed_path(pts, close_thresh=CLOSE_THRESH):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return False
    return np.linalg.norm(pts[0] - pts[-1]) <= close_thresh


def stroke_bbox(pts):
    pts = np.asarray(pts, dtype=np.float32)
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    return x0, y0, x1, y1


def stroke_centroid(pts):
    pts = np.asarray(pts, dtype=np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def stroke_area_if_closed(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3 or not is_closed_path(pts):
        return 0.0
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def stroke_circularity_if_closed(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3 or not is_closed_path(pts):
        return 0.0

    area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
    perim = cv2.arcLength(pts.reshape(-1, 1, 2), True)

    if perim <= 1e-6:
        return 0.0

    return float(4.0 * math.pi * area / (perim * perim))


def draw_polyline(canvas, pts, thickness=2, color=0):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return

    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas,
        [pts_i],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def make_circle_polyline(cx, cy, r, n=36):
    ts = np.linspace(0, 2 * np.pi, n, endpoint=True)
    xs = cx + r * np.cos(ts)
    ys = cy + r * np.sin(ts)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    return pts


# ============================================================
# POINT REDUCTION / SMOOTHING
# ============================================================

def resample_polyline_by_count(pts, max_points):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) <= max_points:
        return pts

    closed = is_closed_path(pts)

    if closed:
        work = pts.copy()
        if np.linalg.norm(work[0] - work[-1]) > 1e-6:
            work = np.vstack([work, work[0]])
        sample_count = max(4, max_points - 1)
    else:
        work = pts.copy()
        sample_count = max_points

    seg = np.diff(work, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]

    if total <= 1e-6:
        return pts[:1]

    samples = np.linspace(0.0, total, sample_count)
    out = []

    for s in samples:
        idx = np.searchsorted(cum, s) - 1
        idx = int(np.clip(idx, 0, len(seg_len) - 1))

        denom = seg_len[idx]
        if denom <= 1e-6:
            t = 0.0
        else:
            t = (s - cum[idx]) / denom

        p = work[idx] * (1.0 - t) + work[idx + 1] * t
        out.append(p)

    out = np.asarray(out, dtype=np.float32)

    if closed and np.linalg.norm(out[0] - out[-1]) > 1e-6:
        out = np.vstack([out, out[0]])

    return out


def simplify_polyline(pts, eps=SIMPLIFY_EPS):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 3:
        return pts

    closed = is_closed_path(pts)
    arr = pts.reshape(-1, 1, 2)

    approx = cv2.approxPolyDP(arr, epsilon=eps, closed=closed)
    simp = approx.reshape(-1, 2).astype(np.float32)

    if closed and len(simp) > 0:
        if np.linalg.norm(simp[0] - simp[-1]) > 1e-6:
            simp = np.vstack([simp, simp[0]])

    simp = resample_polyline_by_count(simp, MAX_POINTS_PER_STROKE)
    return simp


def chaikin_smooth_open(pts, iterations=2):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 3:
        return pts

    out = pts.copy()

    for _ in range(iterations):
        new_pts = [out[0]]

        for i in range(len(out) - 1):
            p0 = out[i]
            p1 = out[i + 1]

            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1

            new_pts.append(q)
            new_pts.append(r)

        new_pts.append(out[-1])
        out = np.asarray(new_pts, dtype=np.float32)

    return out


def chaikin_smooth_closed(pts, iterations=2):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 4:
        return pts

    if np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        core = pts[:-1]
    else:
        core = pts.copy()

    out = core

    for _ in range(iterations):
        new_pts = []
        n = len(out)

        for i in range(n):
            p0 = out[i]
            p1 = out[(i + 1) % n]

            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1

            new_pts.append(q)
            new_pts.append(r)

        out = np.asarray(new_pts, dtype=np.float32)

    out = np.vstack([out, out[0]])
    return out


def smooth_and_reduce_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 3:
        return pts

    closed = is_closed_path(pts)

    if closed:
        smoothed = chaikin_smooth_closed(pts, iterations=SMOOTH_ITERATIONS)
    else:
        smoothed = chaikin_smooth_open(pts, iterations=SMOOTH_ITERATIONS)

    closed_after = is_closed_path(smoothed)
    arr = smoothed.reshape(-1, 1, 2)

    approx = cv2.approxPolyDP(
        arr,
        epsilon=POST_SMOOTH_SIMPLIFY_EPS,
        closed=closed_after,
    )

    out = approx.reshape(-1, 2).astype(np.float32)

    if closed_after and len(out) > 0:
        if np.linalg.norm(out[0] - out[-1]) > 1e-6:
            out = np.vstack([out, out[0]])

    out = resample_polyline_by_count(out, MAX_POINTS_PER_STROKE)
    return out


# ============================================================
# MERGE-CONNECTOR FILLETING ONLY
# ============================================================

def tangent_at_start(pts, k=3):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    j = min(k, len(pts) - 1)
    v = pts[j] - pts[0]
    n = np.linalg.norm(v)

    if n <= 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    return v / n


def tangent_at_end(pts, k=3):
    pts = np.asarray(pts, dtype=np.float32)

    if len(pts) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)

    j = max(0, len(pts) - 1 - k)
    v = pts[-1] - pts[j]
    n = np.linalg.norm(v)

    if n <= 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    return v / n


def cubic_bezier(p0, p1, p2, p3, n):
    ts = np.linspace(0.0, 1.0, n)
    pts = []

    for t in ts:
        q = (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3
        )
        pts.append(q)

    return np.asarray(pts, dtype=np.float32)


def make_filleted_connector(a_pts, b_pts):
    """
    Creates a small smooth cubic connector between a_pts[-1] and b_pts[0].
    This fillets only the merge connector region, not every corner in the drawing.
    """
    a_pts = np.asarray(a_pts, dtype=np.float32)
    b_pts = np.asarray(b_pts, dtype=np.float32)

    p0 = a_pts[-1]
    p3 = b_pts[0]

    gap = float(np.linalg.norm(p3 - p0))

    if gap <= 1e-6:
        return np.asarray([p0], dtype=np.float32)

    if not USE_FILLETED_MERGE_CONNECTORS:
        return np.asarray([p0, p3], dtype=np.float32)

    ta = tangent_at_end(a_pts)
    tb = tangent_at_start(b_pts)

    # Use tangents to create a continuous-ish bridge.
    # c1 follows outgoing tangent from a.
    # c2 points backward along incoming tangent to b.
    ctrl_len = min(
        MERGE_CONNECTOR_FILLET_RADIUS_PX,
        MERGE_CONNECTOR_MAX_CONTROL_FRAC * gap,
    )

    c1 = p0 + ta * ctrl_len
    c2 = p3 - tb * ctrl_len

    n = int(np.clip(
        MERGE_CONNECTOR_SAMPLES,
        MERGE_CONNECTOR_MIN_SAMPLES,
        max(MERGE_CONNECTOR_SAMPLES, MERGE_CONNECTOR_MIN_SAMPLES),
    ))

    connector = cubic_bezier(p0, c1, c2, p3, n=n)

    return connector


def join_with_filleted_connector(a_pts, b_pts):
    """
    Joins two oriented stroke fragments using a smooth connector.
    Avoids duplicate endpoint points.
    """
    connector = make_filleted_connector(a_pts, b_pts)

    # Avoid duplicating a_pts[-1] and b_pts[0].
    if len(connector) >= 2:
        joined = np.vstack([
            a_pts,
            connector[1:-1],
            b_pts,
        ])
    else:
        joined = np.vstack([a_pts, b_pts])

    return joined.astype(np.float32)


# ============================================================
# STROKE MERGING
# ============================================================

def best_endpoint_join(a_pts, b_pts):
    """
    Returns best way to join two open strokes.

    Output:
        distance, joined_points
    """
    a_pts = np.asarray(a_pts, dtype=np.float32)
    b_pts = np.asarray(b_pts, dtype=np.float32)

    variants = []

    # a_end -> b_start
    variants.append((
        np.linalg.norm(a_pts[-1] - b_pts[0]),
        a_pts,
        b_pts,
        tangent_at_end(a_pts),
        tangent_at_start(b_pts),
    ))

    # a_end -> b_end, reverse b
    b_rev = b_pts[::-1]
    variants.append((
        np.linalg.norm(a_pts[-1] - b_rev[0]),
        a_pts,
        b_rev,
        tangent_at_end(a_pts),
        tangent_at_start(b_rev),
    ))

    # a_start -> b_start, reverse a
    a_rev = a_pts[::-1]
    variants.append((
        np.linalg.norm(a_rev[-1] - b_pts[0]),
        a_rev,
        b_pts,
        tangent_at_end(a_rev),
        tangent_at_start(b_pts),
    ))

    # a_start -> b_end, reverse both
    variants.append((
        np.linalg.norm(a_rev[-1] - b_rev[0]),
        a_rev,
        b_rev,
        tangent_at_end(a_rev),
        tangent_at_start(b_rev),
    ))

    dist, aa, bb, ta, tb = min(variants, key=lambda x: x[0])

    dot = float(np.dot(ta, tb))

    if dist > FORCE_MERGE_CLOSE_DIST and dot < MERGE_DIRECTION_DOT_MIN:
        return dist, None

    joined = join_with_filleted_connector(aa, bb)
    return dist, joined


def should_merge_strokes(a, b):
    if a["feature"] != b["feature"]:
        return False

    if a["feature"] in MERGE_BLOCKED_FEATURES:
        return False

    if a["feature"] not in MERGE_ALLOWED_FEATURES:
        return False

    if a["closed"] or b["closed"]:
        return False

    return True


def merge_strokes_once(strokes):
    used = set()
    merged = []

    for i, a in enumerate(strokes):
        if i in used:
            continue

        best_j = None
        best_joined = None
        best_dist = float("inf")

        for j, b in enumerate(strokes):
            if j == i or j in used:
                continue

            if not should_merge_strokes(a, b):
                continue

            dist, joined = best_endpoint_join(a["points"], b["points"])

            if joined is None:
                continue

            if dist <= MERGE_ENDPOINT_DIST and dist < best_dist:
                best_dist = dist
                best_j = j
                best_joined = joined

        if best_j is not None:
            # Smooth/reduce the merged whole stroke, but the only explicit fillet
            # is the connector curve inserted by join_with_filleted_connector().
            new_pts = smooth_and_reduce_points(best_joined)

            merged.append({
                "feature": a["feature"],
                "source": a["source"] + "+merged_filleted_connector",
                "points": new_pts,
            })

            used.add(i)
            used.add(best_j)

        else:
            merged.append({
                "feature": a["feature"],
                "source": a["source"],
                "points": a["points"],
            })
            used.add(i)

    return merged


def merge_strokes_iterative(strokes, max_passes=MAX_MERGE_PASSES):
    current = strokes

    pbar = tqdm(range(max_passes), desc="merge passes", unit="pass", leave=False)
    for pass_idx in pbar:
        before = len(current)
        current_raw = merge_strokes_once(current)
        current = enrich_stroke_metrics(current_raw)
        after = len(current)
        pbar.set_postfix(strokes=after, delta=before - after)

        if after >= before:
            break

    return current


def apply_smoothing_and_merging(strokes):
    """
    Main post-processing entry point.
    Applies general stroke smoothing, then merges compatible fragments.
    Filleting is applied only to connector regions inserted during merging.
    """
    raw = []

    for s in strokes:
        pts = np.asarray(s["points"], dtype=np.float32)

        if ENABLE_STROKE_SMOOTHING:
            if s["feature"] in {"left_iris", "right_iris"}:
                new_pts = resample_polyline_by_count(pts, IRIS_NUM_POINTS)
            else:
                new_pts = smooth_and_reduce_points(pts)
        else:
            new_pts = resample_polyline_by_count(pts, MAX_POINTS_PER_STROKE)

        raw.append({
            "feature": s["feature"],
            "source": s["source"],
            "points": new_pts,
        })

    processed = enrich_stroke_metrics(raw)

    before_merge_count = len(processed)

    if ENABLE_STROKE_MERGING:
        processed = merge_strokes_iterative(processed, max_passes=MAX_MERGE_PASSES)

    after_merge_count = len(processed)

    # Final point cap only. No global filleting here.
    final_raw = []

    for s in processed:
        pts = np.asarray(s["points"], dtype=np.float32)

        if s["feature"] in {"left_iris", "right_iris"}:
            new_pts = resample_polyline_by_count(pts, IRIS_NUM_POINTS)
        else:
            new_pts = resample_polyline_by_count(pts, MAX_POINTS_PER_STROKE)

        final_raw.append({
            "feature": s["feature"],
            "source": s["source"],
            "points": new_pts,
        })

    final = enrich_stroke_metrics(final_raw)

    print()
    print("Smoothing / merging diagnostics:")
    print(f"  Before merge: {before_merge_count} strokes")
    print(f"  After merge:  {after_merge_count} strokes")
    print("  Filleting:    merge connectors only")

    return final


# ============================================================
# STEP 1: LOAD / BINARIZE
# ============================================================

def load_binary_image(path, threshold=THRESHOLD):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    return gray, mask


# ============================================================
# STEP 2: RASTER CLEANUP
# ============================================================

def remove_small_components(mask, min_area=MIN_COMPONENT_AREA):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def morphological_cleanup(mask):
    out = mask.copy()

    if OPEN_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE),
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    if CLOSE_KERNEL_SIZE > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE),
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    return out


def crop_to_foreground_bbox(mask, pad=BBOX_PAD):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return mask, (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)

    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, mask.shape[1] - 1)
    y1 = min(int(ys.max()) + pad, mask.shape[0] - 1)

    cropped = np.zeros_like(mask)
    cropped[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]

    return cropped, (x0, y0, x1, y1)


# ============================================================
# STEP 3: DETECT FILLED PUPILS AND REPLACE WITH HOLLOW IRIS RINGS
# ============================================================

def detect_pupil_blobs(mask):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupils = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < PUPIL_MIN_AREA or area > PUPIL_MAX_AREA:
            continue

        circ = contour_circularity(cnt)
        if circ < PUPIL_CIRCULARITY_MIN:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)

        if not (0.70 <= aspect <= 1.40):
            continue

        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if not (PUPIL_Y_MIN_FRAC * h <= cy <= PUPIL_Y_MAX_FRAC * h):
            continue

        r_est = math.sqrt(area / math.pi)

        pupils.append({
            "cx": float(cx),
            "cy": float(cy),
            "r_est": float(r_est),
            "area": float(area),
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            "contour": cnt,
        })

    pupils = sorted(pupils, key=lambda d: d["cx"])

    if len(pupils) > 2:
        pupils = sorted(pupils, key=lambda p: p["area"], reverse=True)[:2]
        pupils = sorted(pupils, key=lambda p: p["cx"])

    return pupils


def remove_pupils_from_mask(mask, pupils):
    out = mask.copy()

    for p in pupils:
        cv2.drawContours(out, [p["contour"]], contourIdx=-1, color=0, thickness=-1)

    return out


def build_iris_replacement_strokes(pupils):
    strokes = []

    for i, p in enumerate(pupils):
        r = p["r_est"] * IRIS_SCALE + IRIS_EXTRA_PX
        pts = make_circle_polyline(p["cx"], p["cy"], r, n=IRIS_NUM_POINTS)

        label = "left_iris" if i == 0 else ("right_iris" if i == 1 else f"iris_{i}")

        strokes.append({
            "feature": label,
            "source": "iris_replacement_from_filled_pupil",
            "points": pts,
        })

    return strokes


# ============================================================
# STEP 3B: FALLBACK IRIS FROM GLASSES
# ============================================================

def detect_glasses_from_mask(mask):
    h, w = mask.shape
    img_area = h * w

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = abs(cv2.contourArea(cnt))

        if not (GLASSES_AREA_MIN_FRAC * img_area <= area <= GLASSES_AREA_MAX_FRAC * img_area):
            continue

        circ = contour_circularity(cnt)

        if circ < GLASSES_CIRCULARITY_MIN:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)

        if not (GLASSES_ASPECT_MIN <= aspect <= GLASSES_ASPECT_MAX):
            continue

        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if not (GLASSES_Y_MIN_FRAC * h <= cy <= GLASSES_Y_MAX_FRAC * h):
            continue

        (ecx, ecy), er = cv2.minEnclosingCircle(cnt.astype(np.float32))

        candidates.append({
            "cx": float(ecx),
            "cy": float(ecy),
            "r": float(er),
            "area": float(area),
            "circ": float(circ),
            "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
        })

    if len(candidates) < 2:
        return []

    best_pair = None
    best_score = -1e9

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]

            left, right = (a, b) if a["cx"] < b["cx"] else (b, a)

            dx = right["cx"] - left["cx"]
            dy = abs(right["cy"] - left["cy"])
            rmean = 0.5 * (left["r"] + right["r"])
            rdiff = abs(left["r"] - right["r"])

            if dx < 1.0 * rmean:
                continue
            if dx > 5.0 * rmean:
                continue
            if dy > 0.85 * rmean:
                continue
            if rdiff > 0.75 * rmean:
                continue

            score = (
                left["area"]
                + right["area"]
                + 1500.0 * (left["circ"] + right["circ"])
                - 10.0 * dy
                - 10.0 * rdiff
            )

            if score > best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair is None:
        return []

    return list(best_pair)


def build_iris_from_glasses_strokes(glasses):
    strokes = []

    for i, g in enumerate(glasses):
        r = g["r"] * IRIS_FROM_GLASSES_RATIO + IRIS_EXTRA_PX
        pts = make_circle_polyline(g["cx"], g["cy"], r, n=IRIS_NUM_POINTS)

        label = "left_iris" if i == 0 else "right_iris"

        strokes.append({
            "feature": label,
            "source": "iris_replacement_from_glasses",
            "points": pts,
        })

    return strokes


# ============================================================
# STEP 4: SKELETONIZE + SPUR PRUNING
# ============================================================

def skeletonize_mask(mask):
    binary = (mask > 0).astype(bool)
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def count_skeleton_neighbors(skel):
    binary = (skel > 0).astype(np.uint8)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    return cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def prune_skeleton_spurs(skel, iters=PRUNE_SPUR_ITERS):
    out = skel.copy()

    for _ in range(iters):
        neighbors = count_skeleton_neighbors(out)
        endpoints = ((out > 0) & (neighbors <= 1)).astype(np.uint8) * 255
        out[endpoints > 0] = 0

    return out


# ============================================================
# STEP 5: TRACE SKELETON INTO STROKES
# ============================================================

def build_neighbor_map(skel):
    pts = set(map(tuple, np.argwhere(skel > 0)))  # (row, col)
    nbrs = {}

    for p in pts:
        r, c = p
        nlist = []

        for dr, dc in _NEIGHBOR_OFFSETS:
            q = (r + dr, c + dc)
            if q in pts:
                nlist.append(q)

        nbrs[p] = nlist

    return pts, nbrs


def edge_key(a, b):
    return tuple(sorted((a, b)))


def trace_paths_from_skeleton(skel):
    pts, nbrs = build_neighbor_map(skel)

    if len(pts) == 0:
        return []

    degrees = {p: len(nbrs[p]) for p in pts}
    keypoints = {p for p in pts if degrees[p] != 2}

    used_edges = set()
    paths = []

    for p in keypoints:
        for q in nbrs[p]:
            ek = edge_key(p, q)

            if ek in used_edges:
                continue

            path = [p, q]
            used_edges.add(ek)

            prev = p
            cur = q

            while True:
                if cur in keypoints and cur != p:
                    break

                nexts = [n for n in nbrs[cur] if n != prev]

                if len(nexts) == 0:
                    break

                unused_nexts = [n for n in nexts if edge_key(cur, n) not in used_edges]

                if len(unused_nexts) == 0:
                    break

                nxt = unused_nexts[0]
                used_edges.add(edge_key(cur, nxt))
                path.append(nxt)

                prev, cur = cur, nxt

            paths.append(path)

    for start in pts:
        if degrees[start] != 2:
            continue

        unused_neighbors = [n for n in nbrs[start] if edge_key(start, n) not in used_edges]

        if not unused_neighbors:
            continue

        q = unused_neighbors[0]
        path = [start, q]
        used_edges.add(edge_key(start, q))

        prev = start
        cur = q

        while True:
            nexts = [n for n in nbrs[cur] if n != prev]

            if len(nexts) == 0:
                break

            candidates = [n for n in nexts if edge_key(cur, n) not in used_edges]

            if len(candidates) == 0:
                if start in nexts and edge_key(cur, start) not in used_edges:
                    used_edges.add(edge_key(cur, start))
                    path.append(start)
                break

            nxt = candidates[0]
            used_edges.add(edge_key(cur, nxt))
            path.append(nxt)

            prev, cur = cur, nxt

            if cur == start:
                break

        paths.append(path)

    strokes = []

    for path in paths:
        pts_xy = np.array([[p[1], p[0]] for p in path], dtype=np.float32)

        if len(pts_xy) < MIN_STROKE_POINTS:
            continue

        simp = simplify_polyline(pts_xy, eps=SIMPLIFY_EPS)

        if len(simp) < MIN_STROKE_POINTS:
            continue

        L = path_length(simp)
        x0, y0, x1, y1 = stroke_bbox(simp)
        bbox_area = (x1 - x0) * (y1 - y0)

        if L < MIN_STROKE_LENGTH:
            continue

        if bbox_area < MIN_STROKE_BBOX_AREA:
            continue

        strokes.append({
            "feature": None,
            "source": "skeleton",
            "points": simp,
        })

    return strokes


# ============================================================
# STEP 6: METRICS + FEATURE GROUPING
# ============================================================

def enrich_stroke_metrics(strokes):
    enriched = []

    iterator = tqdm(
        list(enumerate(strokes)),
        desc="enrich strokes",
        unit="stroke",
        leave=False,
        disable=len(strokes) < 50,
    )
    for i, s in iterator:
        pts = np.asarray(s["points"], dtype=np.float32)

        x0, y0, x1, y1 = stroke_bbox(pts)
        cx, cy = stroke_centroid(pts)
        length = path_length(pts)
        closed = is_closed_path(pts)
        area = stroke_area_if_closed(pts)
        circ = stroke_circularity_if_closed(pts)

        enriched.append({
            "id": i,
            "feature": s["feature"],
            "source": s["source"],
            "points": pts,
            "closed": bool(closed),
            "length_px": float(length),
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "bbox_w": float(x1 - x0),
            "bbox_h": float(y1 - y0),
            "bbox_area": float((x1 - x0) * (y1 - y0)),
            "centroid": [float(cx), float(cy)],
            "area_closed": float(area),
            "circularity_closed": float(circ),
        })

    return enriched


def find_eye_strokes(strokes, image_shape):
    h, w = image_shape[:2]
    eye_candidates = []

    for s in strokes:
        if s["feature"] is not None:
            continue

        cx, cy = s["centroid"]

        if (
            s["closed"]
            and s["area_closed"] > 6000
            and s["circularity_closed"] > 0.45
            and 0.18 * h <= cy <= 0.70 * h
            and 0.08 * w <= cx <= 0.92 * w
        ):
            eye_candidates.append(s)

    eye_candidates = sorted(eye_candidates, key=lambda s: s["area_closed"], reverse=True)[:2]
    eye_candidates = sorted(eye_candidates, key=lambda s: s["centroid"][0])

    return eye_candidates


def remove_tiny_strokes_near_eyes(strokes, image_shape):
    if not DELETE_TINY_STROKES_NEAR_EYES:
        return strokes

    eye_candidates = find_eye_strokes(strokes, image_shape)

    if len(eye_candidates) < 2:
        return strokes

    eye_regions = []

    for e in eye_candidates:
        cx, cy = e["centroid"]
        r = max(e["bbox_w"], e["bbox_h"]) * 0.55
        eye_regions.append((cx, cy, r, e["id"]))

    keep = []

    for s in strokes:
        keep_this = True

        for cx, cy, r, eid in eye_regions:
            if s["id"] == eid:
                continue

            sx, sy = s["centroid"]
            d = math.hypot(sx - cx, sy - cy)

            if d < 1.10 * r:
                if (
                    s["length_px"] < TINY_NEAR_EYE_LENGTH
                    or s["bbox_area"] < TINY_NEAR_EYE_BBOX_AREA
                ):
                    keep_this = False

        if keep_this:
            keep.append(s)

    return keep


def remove_top_border_artifacts(strokes, image_shape):
    if not DELETE_TOP_BORDER_ARTIFACTS:
        return strokes

    h, w = image_shape[:2]
    keep = []

    for s in strokes:
        x0, y0, x1, y1 = s["bbox"]

        if y1 < TOP_BORDER_FRAC * h and s["length_px"] < TOP_BORDER_MIN_LENGTH_KEEP:
            continue

        keep.append(s)

    return keep


def classify_features(strokes, image_shape):
    h, w = image_shape[:2]

    eye_candidates = find_eye_strokes(strokes, image_shape)

    if len(eye_candidates) >= 1:
        eye_candidates[0]["feature"] = "left_eye"

    if len(eye_candidates) >= 2:
        eye_candidates[1]["feature"] = "right_eye"

    remaining = [s for s in strokes if s["feature"] is None]

    if remaining:
        candidates = []

        for s in remaining:
            x0, y0, x1, y1 = s["bbox"]

            if y0 < 0.18 * h and s["bbox_area"] > 0.10 * w * h:
                candidates.append(s)

        if candidates:
            outer = max(candidates, key=lambda s: s["bbox_area"])
            outer["feature"] = "outer_contour"

    remaining = [s for s in strokes if s["feature"] is None]
    mouth_candidates = []

    for s in remaining:
        cx, cy = s["centroid"]

        if (
            not s["closed"]
            and s["length_px"] > 80
            and 0.52 * h <= cy <= 0.75 * h
            and 0.20 * w <= cx <= 0.85 * w
        ):
            mouth_candidates.append(s)

    if mouth_candidates:
        mouth = max(mouth_candidates, key=lambda s: s["length_px"])
        mouth["feature"] = "mouth"

    remaining = [s for s in strokes if s["feature"] is None]
    nose_candidates = []

    for s in remaining:
        cx, cy = s["centroid"]

        if (
            0.35 * h <= cy <= 0.66 * h
            and 0.45 * w <= cx <= 0.88 * w
            and s["length_px"] > 30
        ):
            nose_candidates.append(s)

    if nose_candidates:
        nose = max(nose_candidates, key=lambda s: s["length_px"])
        nose["feature"] = "nose"

    remaining = [s for s in strokes if s["feature"] is None]

    for s in remaining:
        cx, cy = s["centroid"]

        if cy < 0.43 * h:
            s["feature"] = "hair"

    remaining = [s for s in strokes if s["feature"] is None]

    for s in remaining:
        cx, cy = s["centroid"]

        if 0.38 * h <= cy <= 0.68 * h and cx < 0.33 * w:
            s["feature"] = "ear"

    remaining = [s for s in strokes if s["feature"] is None]

    for s in remaining:
        cx, cy = s["centroid"]

        if cy > 0.74 * h:
            if s["bbox_w"] > 0.35 * w:
                s["feature"] = "shoulders"
            else:
                s["feature"] = "neck_collar"

    for s in strokes:
        if s["feature"] is None:
            s["feature"] = "misc"

    return strokes


def stroke_order_priority(feature):
    priorities = {
        "outer_contour": 0,
        "hair": 1,
        "left_eye": 2,
        "right_eye": 3,
        "left_iris": 4,
        "right_iris": 5,
        "nose": 6,
        "mouth": 7,
        "ear": 8,
        "neck_collar": 9,
        "shoulders": 10,
        "misc": 11,
    }

    return priorities.get(feature, 99)


def sort_strokes_for_drawing(strokes):
    return sorted(
        strokes,
        key=lambda s: (stroke_order_priority(s["feature"]), -s["length_px"]),
    )


# ============================================================
# STEP 7: SAVE JSON
# ============================================================

def serialize_strokes(strokes):
    out = []

    for order_idx, s in enumerate(strokes):
        out.append({
            "order": order_idx,
            "id": int(s["id"]),
            "feature": s["feature"],
            "source": s["source"],
            "closed": bool(s["closed"]),
            "length_px": float(s["length_px"]),
            "num_points": int(len(s["points"])),
            "bbox": [float(v) for v in s["bbox"]],
            "centroid": [float(v) for v in s["centroid"]],
            "points": [[float(x), float(y)] for x, y in s["points"]],
        })

    return out


def save_stroke_json(strokes, out_path):
    groups = defaultdict(list)

    for idx, s in enumerate(strokes):
        groups[s["feature"]].append(idx)

    payload = {
        "num_strokes": len(strokes),
        "groups": dict(groups),
        "strokes": serialize_strokes(strokes),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# STEP 8: RENDERING
# ============================================================

def save_binary_debug(mask, out_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def render_strokes_to_image(strokes, image_shape, thickness):
    h, w = image_shape[:2]
    canvas = np.full((h, w), 255, dtype=np.uint8)

    for s in strokes:
        draw_polyline(canvas, s["points"], thickness=thickness, color=0)

    return canvas


def save_layered_plot(strokes, image_shape, out_path):
    h, w = image_shape[:2]

    plt.figure(figsize=(8, 8))
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.gca().set_aspect("equal")
    plt.gca().set_facecolor("white")

    for s in strokes:
        pts = np.asarray(s["points"])
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1.5, color="black")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay_plot(gray, strokes, out_path):
    h, w = gray.shape[:2]

    plt.figure(figsize=(8, 8))
    plt.imshow(gray, cmap="gray")
    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.gca().set_aspect("equal")

    for s in strokes:
        pts = np.asarray(s["points"])
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1.5)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


# ============================================================
# STEP 9: ANIMATION
# ============================================================

def build_animation_frames(strokes, image_shape):
    h, w = image_shape[:2]
    canvas = np.full((h, w), 255, dtype=np.uint8)
    frames = [cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2RGB)]

    pbar = tqdm(strokes, desc="animation frames", unit="stroke", leave=False)
    for s in pbar:
        pts = np.asarray(s["points"], dtype=np.float32)
        n = len(pts)

        if n < 2:
            continue

        step = max(1, n // ANIM_PROGRESS_STEPS_PER_STROKE)

        for k in range(2, n + 1, step):
            frame = canvas.copy()
            draw_polyline(frame, pts[:k], thickness=ANIM_LINE_THICKNESS, color=0)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))

        draw_polyline(canvas, pts, thickness=ANIM_LINE_THICKNESS, color=0)
        frames.append(cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2RGB))
        pbar.set_postfix(frames=len(frames))

    return frames


def save_animation_gif(frames, out_path):
    imageio.mimsave(out_path, frames, duration=ANIM_FRAME_DURATION)


def save_animation_mp4(frames, out_path, fps=16):
    """Write an mp4 via imageio's ffmpeg plugin. Surfaces the real error on failure."""
    if not frames:
        print(f"[pipeline]   mp4 skipped (no frames): {out_path}", flush=True)
        return

    # h264 requires even width/height; macro_block_size=1 makes ffmpeg pad
    # to the next even number instead of erroring on odd dimensions.
    writer_kwargs = dict(
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    try:
        with imageio.get_writer(out_path, **writer_kwargs) as writer:
            for fr in frames:
                writer.append_data(fr)
        print(
            f"[pipeline]   mp4 written: {out_path} ({len(frames)} frames @ {fps} fps)",
            flush=True,
        )
    except Exception as exc:
        import traceback
        print(f"[pipeline]   MP4 export FAILED for {out_path}: {exc!r}", flush=True)
        traceback.print_exc()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    input_path,
    *,
    jsons_dir=DEFAULT_JSONS_DIR,
    renders_dir=DEFAULT_RENDERS_DIR,
    stem=None,
):
    """Run the stroke-extraction pipeline on a single image.

    Args:
        input_path: Path to the input portrait image (e.g. a cartoonized PNG).
        jsons_dir: Directory where the stroke JSON file is written.
            The JSON is saved as ``<jsons_dir>/<stem>.json``.
        renders_dir: Directory where all rendered debug/preview/animation
            artifacts are written. A per-input subfolder ``<renders_dir>/<stem>/``
            is created so multiple inputs don't collide.
        stem: Output basename used for the JSON file and the render subfolder.
            Defaults to the input file's stem.

    Returns:
        A dict with the final ``strokes`` list and the resolved output paths.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if stem is None:
        stem = input_path.stem

    jsons_dir = Path(jsons_dir)
    renders_dir = Path(renders_dir)
    render_out_dir = renders_dir / stem

    ensure_dir(jsons_dir)
    ensure_dir(render_out_dir)

    json_path = jsons_dir / f"{stem}.json"

    _set_total_steps(11)
    pipeline_start = time.perf_counter()
    _log(f"input:       {input_path}")
    _log(f"json output: {json_path}")
    _log(f"render dir:  {render_out_dir}")

    with _step("load + binarize image"):
        gray, mask0 = load_binary_image(str(input_path))
        h, w = gray.shape[:2]
        _log(f"  image size: {w} x {h}")

    with _step("raster cleanup"):
        mask1 = remove_small_components(mask0, MIN_COMPONENT_AREA)
        mask2 = morphological_cleanup(mask1)
        mask3, bbox = crop_to_foreground_bbox(mask2, pad=BBOX_PAD)
        _log(f"  foreground bbox: {bbox}")

    with _step("detect pupils / glasses for iris replacement"):
        pupils = detect_pupil_blobs(mask3)
        _log(f"  pupils detected: {len(pupils)}")

        if len(pupils) >= 2:
            mask_no_pupils = remove_pupils_from_mask(mask3, pupils[:2])
            iris_strokes = build_iris_replacement_strokes(pupils[:2])
            iris_mode = "filled_pupil_detection"
            glasses = []
        else:
            mask_no_pupils = mask3.copy()
            glasses = detect_glasses_from_mask(mask3)
            _log(f"  glasses fallback detected: {len(glasses)}")

            if USE_GLASSES_FALLBACK_IRIS and len(glasses) == 2:
                iris_strokes = build_iris_from_glasses_strokes(glasses)
                iris_mode = "glasses_fallback"
            else:
                iris_strokes = []
                iris_mode = "none"
        _log(f"  iris mode: {iris_mode} ({len(iris_strokes)} iris strokes)")

    with _step("skeletonize + spur pruning"):
        skel_raw = skeletonize_mask(mask_no_pupils)
        skel_pruned = prune_skeleton_spurs(skel_raw, PRUNE_SPUR_ITERS)
        _log(f"  skeleton pixels (raw -> pruned): "
             f"{int(np.count_nonzero(skel_raw))} -> {int(np.count_nonzero(skel_pruned))}")

    with _step("trace skeleton into vector strokes"):
        traced_strokes = trace_paths_from_skeleton(skel_pruned)
        _log(f"  traced strokes: {len(traced_strokes)}")

    with _step("compute stroke metrics + filter artifacts"):
        raw_strokes = traced_strokes + iris_strokes
        strokes = enrich_stroke_metrics(raw_strokes)

        original_stroke_count = len(strokes)
        original_point_count = sum(len(s["points"]) for s in strokes)

        strokes = remove_top_border_artifacts(strokes, gray.shape)
        strokes = remove_tiny_strokes_near_eyes(strokes, gray.shape)

        raw_again = [
            {
                "feature": s["feature"],
                "source": s["source"],
                "points": s["points"],
            }
            for s in strokes
        ]

        strokes = enrich_stroke_metrics(raw_again)
        _log(f"  strokes after artifact filtering: {len(strokes)}")

    with _step("classify features + smooth/merge"):
        strokes = classify_features(strokes, gray.shape)
        strokes = apply_smoothing_and_merging(strokes)
        strokes = sort_strokes_for_drawing(strokes)
        for new_order, s in enumerate(strokes):
            s["draw_order"] = new_order
        _log(f"  final strokes: {len(strokes)}")

    final_stroke_count = len(strokes)
    final_point_count = sum(len(s["points"]) for s in strokes)

    with _step("save mask/skeleton debug PNGs"):
        save_binary_debug(mask0, str(render_out_dir / "01_binary_input.png"))
        save_binary_debug(mask3, str(render_out_dir / "02_cleaned_mask.png"))
        save_binary_debug(mask_no_pupils, str(render_out_dir / "03_mask_without_pupils.png"))
        save_binary_debug(skel_raw, str(render_out_dir / "04_skeleton_raw.png"))
        save_binary_debug(skel_pruned, str(render_out_dir / "05_skeleton_pruned.png"))

    with _step("save stroke preview renders"):
        thin = render_strokes_to_image(strokes, gray.shape, CENTERLINE_PREVIEW_THICKNESS)
        thick = render_strokes_to_image(strokes, gray.shape, MARKER_PREVIEW_THICKNESS)
        cv2.imwrite(str(render_out_dir / "06_centerline_preview.png"), thin)
        cv2.imwrite(str(render_out_dir / "07_marker_thickness_preview.png"), thick)
        save_layered_plot(strokes, gray.shape, str(render_out_dir / "08_layered_strokes.png"))
        save_overlay_plot(gray, strokes, str(render_out_dir / "09_overlay_on_input.png"))

    with _step(f"write stroke JSON ({final_stroke_count} strokes, {final_point_count} pts)"):
        save_stroke_json(strokes, str(json_path))

    with _step("build + save stroke animation (GIF + MP4)"):
        frames = build_animation_frames(strokes, gray.shape)
        _log(f"  animation frames: {len(frames)}")
        save_animation_gif(frames, str(render_out_dir / "10_stroke_animation.gif"))
        save_animation_mp4(frames, str(render_out_dir / "11_stroke_animation.mp4"))

    total_elapsed = time.perf_counter() - pipeline_start

    print("===================================================")
    print("PIPELINE COMPLETE")
    print("===================================================")
    print(f"Input:           {input_path}")
    print(f"JSON output:     {json_path}")
    print(f"Renders dir:     {render_out_dir}")
    print(f"Total wall time: {total_elapsed:.2f}s")
    print(f"Foreground bbox: {bbox}")
    print(f"Pupils detected: {len(pupils)}")
    print(f"Glasses fallback detected: {len(glasses)}")
    print(f"Iris mode: {iris_mode}")
    print()
    print("Stroke reduction:")
    print(f"  Initial strokes after trace: {original_stroke_count}")
    print(f"  Final strokes:              {final_stroke_count}")
    print(f"  Initial vector points:       {original_point_count}")
    print(f"  Final vector points:         {final_point_count}")
    print(f"  Avg points per final stroke: {final_point_count / max(final_stroke_count, 1):.1f}")
    print()

    groups = defaultdict(int)

    for s in strokes:
        groups[s["feature"]] += 1

    print("Stroke groups:")
    for k in sorted(groups.keys(), key=stroke_order_priority):
        print(f"  {k}: {groups[k]}")

    print()
    print(f"Saved files in {render_out_dir}:")
    print("  01_binary_input.png")
    print("  02_cleaned_mask.png")
    print("  03_mask_without_pupils.png")
    print("  04_skeleton_raw.png")
    print("  05_skeleton_pruned.png")
    print("  06_centerline_preview.png")
    print("  07_marker_thickness_preview.png")
    print("  08_layered_strokes.png")
    print("  09_overlay_on_input.png")
    print("  10_stroke_animation.gif")
    print("  11_stroke_animation.mp4")
    print(f"Saved JSON: {json_path}")

    print()
    print("Per-stroke point counts:")
    for i, s in enumerate(strokes):
        print(
            f"{i:02d} | {s['feature']:14s} | "
            f"{len(s['points']):3d} pts | "
            f"{s['length_px']:7.1f}px | "
            f"{s['source']}"
        )

    return {
        "strokes": strokes,
        "json_path": json_path,
        "render_dir": render_out_dir,
        "stem": stem,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract drawable strokes from a cartoonized portrait."
    )
    parser.add_argument("input", help="Path to the input image (e.g. an art-renders PNG).")
    parser.add_argument(
        "--jsons-dir",
        default=str(DEFAULT_JSONS_DIR),
        help=f"Directory for the output stroke JSON (default: {DEFAULT_JSONS_DIR}).",
    )
    parser.add_argument(
        "--renders-dir",
        default=str(DEFAULT_RENDERS_DIR),
        help=f"Directory for rendered debug/preview/animation files (default: {DEFAULT_RENDERS_DIR}).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Override the output basename (defaults to the input file stem).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        args.input,
        jsons_dir=args.jsons_dir,
        renders_dir=args.renders_dir,
        stem=args.stem,
    )