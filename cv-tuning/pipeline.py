"""Class-based line-art stroke extraction pipeline.

Single-file port of computer-vision/run_pipeline.py wrapped in a
``LineArtStrokePipeline`` class so configs can be cloned/overridden per
prompt without rerunning the whole module. Use ``run_strokes_only`` for
fast tuning loops (saves just the layered strokes PNG) and ``run`` for
full debug output.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class LineArtPipelineConfig:
    # Binary / raster cleanup
    threshold: int = 210
    min_component_area: int = 100
    open_kernel_size: int = 2
    close_kernel_size: int = 1
    bbox_pad: int = 20

    # Skeleton cleanup
    prune_spur_iters: int = 18
    min_branch_pixels: int = 8  # retained for experiments; pruning currently uses endpoint peeling

    # Pupil -> iris replacement
    pupil_min_area: int = 120
    pupil_max_area: int = 5000
    pupil_circularity_min: float = 0.70
    pupil_y_min_frac: float = 0.20
    pupil_y_max_frac: float = 0.70
    iris_scale: float = 1.35
    iris_extra_px: float = 5.0
    iris_num_points: int = 36

    # Glasses fallback iris placement
    use_glasses_fallback_iris: bool = True
    glasses_y_min_frac: float = 0.25
    glasses_y_max_frac: float = 0.62
    glasses_area_min_frac: float = 0.006
    glasses_area_max_frac: float = 0.15
    glasses_circularity_min: float = 0.45
    glasses_aspect_min: float = 0.60
    glasses_aspect_max: float = 1.60
    iris_from_glasses_ratio: float = 1.0 / 3.0

    # Stroke extraction / simplification
    min_stroke_points: int = 4
    min_stroke_length: float = 28.0
    min_stroke_bbox_area: float = 30.0
    simplify_eps: float = 2.4
    close_thresh: float = 4.0

    # Post-processing: smoothing + stroke merging
    enable_stroke_smoothing: bool = True
    smooth_iterations: int = 2
    post_smooth_simplify_eps: float = 2.0
    max_points_per_stroke: int = 60
    enable_stroke_merging: bool = True
    merge_endpoint_dist: float = 16.0
    connect_gaps_when_merging: bool = True
    force_merge_close_dist: float = 6.0
    merge_direction_dot_min: float = -0.25
    max_merge_passes: int = 6

    # Which feature labels can / cannot be endpoint-merged
    merge_allowed_features: Tuple[str, ...] = (
        "outer_contour", "hair", "nose", "mouth", "ear", "neck_collar", "shoulders", "misc"
    )
    merge_blocked_features: Tuple[str, ...] = (
        "left_eye", "right_eye", "left_iris", "right_iris"
    )

    # Geometric junk deletion
    delete_tiny_strokes_near_eyes: bool = True
    tiny_near_eye_length: float = 45.0
    tiny_near_eye_bbox_area: float = 100.0
    delete_top_border_artifacts: bool = True
    top_border_frac: float = 0.06
    top_border_min_length_keep: float = 80.0

    # Rendering
    centerline_preview_thickness: int = 1
    marker_preview_thickness: int = 7
    anim_line_thickness: int = 7
    anim_frame_duration: float = 0.045
    anim_progress_steps_per_stroke: int = 10
    mp4_fps: int = 16

    # Output controls
    save_debug_images: bool = True
    save_gif: bool = True
    save_mp4: bool = True
    verbose: bool = True

    def clone(self, **kwargs):
        return replace(self, **kwargs)


class LineArtStrokePipeline:
    _NEIGHBOR_OFFSETS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    def __init__(self, cfg: Optional[LineArtPipelineConfig] = None):
        self.cfg = cfg or LineArtPipelineConfig()
        self.last_result: Optional[Dict[str, Any]] = None

    # ============================================================
    # Basic utils
    # ============================================================
    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def contour_circularity(cnt) -> float:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * area / (perim * perim))

    @staticmethod
    def path_length(pts) -> float:
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def is_closed_path(self, pts) -> bool:
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3:
            return False
        return np.linalg.norm(pts[0] - pts[-1]) <= self.cfg.close_thresh

    @staticmethod
    def stroke_bbox(pts):
        pts = np.asarray(pts, dtype=np.float32)
        x0 = float(np.min(pts[:, 0]))
        y0 = float(np.min(pts[:, 1]))
        x1 = float(np.max(pts[:, 0]))
        y1 = float(np.max(pts[:, 1]))
        return x0, y0, x1, y1

    @staticmethod
    def stroke_centroid(pts):
        pts = np.asarray(pts, dtype=np.float32)
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    def stroke_area_if_closed(self, pts) -> float:
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3 or not self.is_closed_path(pts):
            return 0.0
        return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))

    def stroke_circularity_if_closed(self, pts) -> float:
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3 or not self.is_closed_path(pts):
            return 0.0
        area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
        perim = cv2.arcLength(pts.reshape(-1, 1, 2), True)
        if perim <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * area / (perim * perim))

    @staticmethod
    def draw_polyline(canvas, pts, thickness=2, color=0):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts_i], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def make_circle_polyline(cx, cy, r, n=36):
        ts = np.linspace(0, 2 * np.pi, n, endpoint=True)
        xs = cx + r * np.cos(ts)
        ys = cy + r * np.sin(ts)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.vstack([pts, pts[0]])
        return pts

    # ============================================================
    # Point reduction / smoothing
    # ============================================================
    def resample_polyline_by_count(self, pts, max_points: Optional[int] = None):
        cfg = self.cfg
        max_points = max_points or cfg.max_points_per_stroke
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) <= max_points:
            return pts

        closed = self.is_closed_path(pts)
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
            t = 0.0 if denom <= 1e-6 else (s - cum[idx]) / denom
            p = work[idx] * (1.0 - t) + work[idx + 1] * t
            out.append(p)
        out = np.asarray(out, dtype=np.float32)
        if closed and np.linalg.norm(out[0] - out[-1]) > 1e-6:
            out = np.vstack([out, out[0]])
        return out

    def simplify_polyline(self, pts, eps: Optional[float] = None):
        eps = self.cfg.simplify_eps if eps is None else eps
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3:
            return pts
        closed = self.is_closed_path(pts)
        approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), epsilon=eps, closed=closed)
        simp = approx.reshape(-1, 2).astype(np.float32)
        if closed and len(simp) > 0 and np.linalg.norm(simp[0] - simp[-1]) > 1e-6:
            simp = np.vstack([simp, simp[0]])
        return self.resample_polyline_by_count(simp, self.cfg.max_points_per_stroke)

    @staticmethod
    def chaikin_smooth_open(pts, iterations=2):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3:
            return pts
        out = pts.copy()
        for _ in range(iterations):
            new_pts = [out[0]]
            for i in range(len(out) - 1):
                p0, p1 = out[i], out[i + 1]
                new_pts.append(0.75 * p0 + 0.25 * p1)
                new_pts.append(0.25 * p0 + 0.75 * p1)
            new_pts.append(out[-1])
            out = np.asarray(new_pts, dtype=np.float32)
        return out

    @staticmethod
    def chaikin_smooth_closed(pts, iterations=2):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 4:
            return pts
        core = pts[:-1] if np.linalg.norm(pts[0] - pts[-1]) < 1e-6 else pts.copy()
        out = core
        for _ in range(iterations):
            new_pts = []
            n = len(out)
            for i in range(n):
                p0, p1 = out[i], out[(i + 1) % n]
                new_pts.append(0.75 * p0 + 0.25 * p1)
                new_pts.append(0.25 * p0 + 0.75 * p1)
            out = np.asarray(new_pts, dtype=np.float32)
        return np.vstack([out, out[0]])

    def smooth_and_reduce_points(self, pts):
        cfg = self.cfg
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 3:
            return pts
        if self.is_closed_path(pts):
            smoothed = self.chaikin_smooth_closed(pts, iterations=cfg.smooth_iterations)
        else:
            smoothed = self.chaikin_smooth_open(pts, iterations=cfg.smooth_iterations)
        closed_after = self.is_closed_path(smoothed)
        approx = cv2.approxPolyDP(
            smoothed.reshape(-1, 1, 2),
            epsilon=cfg.post_smooth_simplify_eps,
            closed=closed_after,
        )
        out = approx.reshape(-1, 2).astype(np.float32)
        if closed_after and len(out) > 0 and np.linalg.norm(out[0] - out[-1]) > 1e-6:
            out = np.vstack([out, out[0]])
        return self.resample_polyline_by_count(out, cfg.max_points_per_stroke)

    # ============================================================
    # Stroke merging
    # ============================================================
    @staticmethod
    def tangent_at_start(pts, k=3):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return np.array([0.0, 0.0], dtype=np.float32)
        j = min(k, len(pts) - 1)
        v = pts[j] - pts[0]
        n = np.linalg.norm(v)
        return np.array([0.0, 0.0], dtype=np.float32) if n <= 1e-6 else v / n

    @staticmethod
    def tangent_at_end(pts, k=3):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) < 2:
            return np.array([0.0, 0.0], dtype=np.float32)
        j = max(0, len(pts) - 1 - k)
        v = pts[-1] - pts[j]
        n = np.linalg.norm(v)
        return np.array([0.0, 0.0], dtype=np.float32) if n <= 1e-6 else v / n

    def best_endpoint_join(self, a_pts, b_pts):
        cfg = self.cfg
        a_pts = np.asarray(a_pts, dtype=np.float32)
        b_pts = np.asarray(b_pts, dtype=np.float32)
        b_rev = b_pts[::-1]
        a_rev = a_pts[::-1]
        variants = [
            (np.linalg.norm(a_pts[-1] - b_pts[0]), a_pts, b_pts, self.tangent_at_end(a_pts), self.tangent_at_start(b_pts)),
            (np.linalg.norm(a_pts[-1] - b_rev[0]), a_pts, b_rev, self.tangent_at_end(a_pts), self.tangent_at_start(b_rev)),
            (np.linalg.norm(a_rev[-1] - b_pts[0]), a_rev, b_pts, self.tangent_at_end(a_rev), self.tangent_at_start(b_pts)),
            (np.linalg.norm(a_rev[-1] - b_rev[0]), a_rev, b_rev, self.tangent_at_end(a_rev), self.tangent_at_start(b_rev)),
        ]
        dist, aa, bb, ta, tb = min(variants, key=lambda x: x[0])
        dot = float(np.dot(ta, tb))
        if dist > cfg.force_merge_close_dist and dot < cfg.merge_direction_dot_min:
            return dist, None
        joined = np.vstack([aa, bb]) if cfg.connect_gaps_when_merging else np.vstack([aa, bb])
        return dist, joined

    def should_merge_strokes(self, a, b) -> bool:
        cfg = self.cfg
        if a["feature"] != b["feature"]:
            return False
        if a["feature"] in set(cfg.merge_blocked_features):
            return False
        if a["feature"] not in set(cfg.merge_allowed_features):
            return False
        if a["closed"] or b["closed"]:
            return False
        return True

    def merge_strokes_once(self, strokes):
        cfg = self.cfg
        used = set()
        merged = []
        for i, a in enumerate(strokes):
            if i in used:
                continue
            best_j, best_joined, best_dist = None, None, float("inf")
            for j, b in enumerate(strokes):
                if j == i or j in used:
                    continue
                if not self.should_merge_strokes(a, b):
                    continue
                dist, joined = self.best_endpoint_join(a["points"], b["points"])
                if joined is not None and dist <= cfg.merge_endpoint_dist and dist < best_dist:
                    best_dist, best_j, best_joined = dist, j, joined
            if best_j is not None:
                merged.append({
                    "feature": a["feature"],
                    "source": a["source"] + "+merged",
                    "points": self.smooth_and_reduce_points(best_joined),
                })
                used.add(i)
                used.add(best_j)
            else:
                merged.append({"feature": a["feature"], "source": a["source"], "points": a["points"]})
                used.add(i)
        return merged

    def merge_strokes_iterative(self, strokes, max_passes: Optional[int] = None):
        max_passes = self.cfg.max_merge_passes if max_passes is None else max_passes
        current = strokes
        for _ in range(max_passes):
            before = len(current)
            current_raw = self.merge_strokes_once(current)
            current = self.enrich_stroke_metrics(current_raw)
            after = len(current)
            if after >= before:
                break
        return current

    def apply_smoothing_and_merging(self, strokes):
        cfg = self.cfg
        raw = []
        for s in strokes:
            pts = np.asarray(s["points"], dtype=np.float32)
            if cfg.enable_stroke_smoothing:
                if s["feature"] in {"left_iris", "right_iris"}:
                    new_pts = self.resample_polyline_by_count(pts, cfg.iris_num_points)
                else:
                    new_pts = self.smooth_and_reduce_points(pts)
            else:
                new_pts = self.resample_polyline_by_count(pts, cfg.max_points_per_stroke)
            raw.append({"feature": s["feature"], "source": s["source"], "points": new_pts})

        processed = self.enrich_stroke_metrics(raw)
        before_merge_count = len(processed)
        if cfg.enable_stroke_merging:
            processed = self.merge_strokes_iterative(processed, max_passes=cfg.max_merge_passes)
        after_merge_count = len(processed)

        final_raw = []
        for s in processed:
            pts = np.asarray(s["points"], dtype=np.float32)
            if s["feature"] in {"left_iris", "right_iris"}:
                new_pts = self.resample_polyline_by_count(pts, cfg.iris_num_points)
            else:
                new_pts = self.smooth_and_reduce_points(pts)
            final_raw.append({"feature": s["feature"], "source": s["source"], "points": new_pts})
        final = self.enrich_stroke_metrics(final_raw)
        return final, {"before_merge_count": before_merge_count, "after_merge_count": after_merge_count}

    # ============================================================
    # Step 1-4: image prep / cleanup / skeleton
    # ============================================================
    def load_binary_image(self, path):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        _, mask = cv2.threshold(gray, self.cfg.threshold, 255, cv2.THRESH_BINARY_INV)
        return gray, mask

    def remove_small_components(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.cfg.min_component_area:
                cleaned[labels == label] = 255
        return cleaned

    def morphological_cleanup(self, mask):
        cfg = self.cfg
        out = mask.copy()
        if cfg.open_kernel_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel_size, cfg.open_kernel_size))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        if cfg.close_kernel_size > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel_size, cfg.close_kernel_size))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        return out

    def crop_to_foreground_bbox(self, mask):
        pad = self.cfg.bbox_pad
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

    def detect_pupil_blobs(self, mask):
        cfg = self.cfg
        h, w = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pupils = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.pupil_min_area or area > cfg.pupil_max_area:
                continue
            circ = self.contour_circularity(cnt)
            if circ < cfg.pupil_circularity_min:
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
            if not (cfg.pupil_y_min_frac * h <= cy <= cfg.pupil_y_max_frac * h):
                continue
            r_est = math.sqrt(area / math.pi)
            pupils.append({
                "cx": float(cx), "cy": float(cy), "r_est": float(r_est), "area": float(area),
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)], "contour": cnt,
            })
        pupils = sorted(pupils, key=lambda d: d["cx"])
        if len(pupils) > 2:
            pupils = sorted(pupils, key=lambda p: p["area"], reverse=True)[:2]
            pupils = sorted(pupils, key=lambda p: p["cx"])
        return pupils

    @staticmethod
    def remove_pupils_from_mask(mask, pupils):
        out = mask.copy()
        for p in pupils:
            cv2.drawContours(out, [p["contour"]], contourIdx=-1, color=0, thickness=-1)
        return out

    def build_iris_replacement_strokes(self, pupils):
        cfg = self.cfg
        strokes = []
        for i, p in enumerate(pupils):
            r = p["r_est"] * cfg.iris_scale + cfg.iris_extra_px
            pts = self.make_circle_polyline(p["cx"], p["cy"], r, n=cfg.iris_num_points)
            label = "left_iris" if i == 0 else ("right_iris" if i == 1 else f"iris_{i}")
            strokes.append({"feature": label, "source": "iris_replacement_from_filled_pupil", "points": pts})
        return strokes

    def detect_glasses_from_mask(self, mask):
        cfg = self.cfg
        h, w = mask.shape
        img_area = h * w
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = abs(cv2.contourArea(cnt))
            if not (cfg.glasses_area_min_frac * img_area <= area <= cfg.glasses_area_max_frac * img_area):
                continue
            circ = self.contour_circularity(cnt)
            if circ < cfg.glasses_circularity_min:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / max(bh, 1)
            if not (cfg.glasses_aspect_min <= aspect <= cfg.glasses_aspect_max):
                continue
            M = cv2.moments(cnt)
            if abs(M["m00"]) < 1e-6:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            if not (cfg.glasses_y_min_frac * h <= cy <= cfg.glasses_y_max_frac * h):
                continue
            (ecx, ecy), er = cv2.minEnclosingCircle(cnt.astype(np.float32))
            candidates.append({
                "cx": float(ecx), "cy": float(ecy), "r": float(er), "area": float(area), "circ": float(circ),
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
            })
        if len(candidates) < 2:
            return []
        best_pair, best_score = None, -1e9
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                left, right = (a, b) if a["cx"] < b["cx"] else (b, a)
                dx = right["cx"] - left["cx"]
                dy = abs(right["cy"] - left["cy"])
                rmean = 0.5 * (left["r"] + right["r"])
                rdiff = abs(left["r"] - right["r"])
                if dx < 1.0 * rmean or dx > 5.0 * rmean or dy > 0.85 * rmean or rdiff > 0.75 * rmean:
                    continue
                score = left["area"] + right["area"] + 1500.0 * (left["circ"] + right["circ"]) - 10.0 * dy - 10.0 * rdiff
                if score > best_score:
                    best_score, best_pair = score, (left, right)
        return [] if best_pair is None else list(best_pair)

    def build_iris_from_glasses_strokes(self, glasses):
        cfg = self.cfg
        strokes = []
        for i, g in enumerate(glasses):
            r = g["r"] * cfg.iris_from_glasses_ratio + cfg.iris_extra_px
            pts = self.make_circle_polyline(g["cx"], g["cy"], r, n=cfg.iris_num_points)
            label = "left_iris" if i == 0 else "right_iris"
            strokes.append({"feature": label, "source": "iris_replacement_from_glasses", "points": pts})
        return strokes

    @staticmethod
    def skeletonize_mask(mask):
        binary = (mask > 0).astype(bool)
        skel = skeletonize(binary)
        return (skel.astype(np.uint8) * 255)

    @staticmethod
    def count_skeleton_neighbors(skel):
        binary = (skel > 0).astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        return cv2.filter2D(binary, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    def prune_skeleton_spurs(self, skel):
        out = skel.copy()
        for _ in range(self.cfg.prune_spur_iters):
            neighbors = self.count_skeleton_neighbors(out)
            endpoints = ((out > 0) & (neighbors <= 1)).astype(np.uint8) * 255
            out[endpoints > 0] = 0
        return out

    # ============================================================
    # Step 5: trace skeleton into strokes
    # ============================================================
    def build_neighbor_map(self, skel):
        pts = set(map(tuple, np.argwhere(skel > 0)))  # (row, col)
        nbrs = {}
        for p in pts:
            r, c = p
            nlist = []
            for dr, dc in self._NEIGHBOR_OFFSETS:
                q = (r + dr, c + dc)
                if q in pts:
                    nlist.append(q)
            nbrs[p] = nlist
        return pts, nbrs

    @staticmethod
    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def trace_paths_from_skeleton(self, skel):
        cfg = self.cfg
        pts, nbrs = self.build_neighbor_map(skel)
        if len(pts) == 0:
            return []
        degrees = {p: len(nbrs[p]) for p in pts}
        keypoints = {p for p in pts if degrees[p] != 2}
        used_edges = set()
        paths = []

        # Paths between keypoints
        for p in keypoints:
            for q in nbrs[p]:
                ek = self.edge_key(p, q)
                if ek in used_edges:
                    continue
                path = [p, q]
                used_edges.add(ek)
                prev, cur = p, q
                while True:
                    if cur in keypoints and cur != p:
                        break
                    nexts = [n for n in nbrs[cur] if n != prev]
                    if len(nexts) == 0:
                        break
                    unused_nexts = [n for n in nexts if self.edge_key(cur, n) not in used_edges]
                    if len(unused_nexts) == 0:
                        break
                    nxt = unused_nexts[0]
                    used_edges.add(self.edge_key(cur, nxt))
                    path.append(nxt)
                    prev, cur = cur, nxt
                paths.append(path)

        # Closed loops with all degree-2 pixels
        for start in pts:
            if degrees[start] != 2:
                continue
            unused_neighbors = [n for n in nbrs[start] if self.edge_key(start, n) not in used_edges]
            if not unused_neighbors:
                continue
            q = unused_neighbors[0]
            path = [start, q]
            used_edges.add(self.edge_key(start, q))
            prev, cur = start, q
            while True:
                nexts = [n for n in nbrs[cur] if n != prev]
                if len(nexts) == 0:
                    break
                candidates = [n for n in nexts if self.edge_key(cur, n) not in used_edges]
                if len(candidates) == 0:
                    if start in nexts and self.edge_key(cur, start) not in used_edges:
                        used_edges.add(self.edge_key(cur, start))
                        path.append(start)
                    break
                nxt = candidates[0]
                used_edges.add(self.edge_key(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == start:
                    break
            paths.append(path)

        strokes = []
        for path in paths:
            pts_xy = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
            if len(pts_xy) < cfg.min_stroke_points:
                continue
            simp = self.simplify_polyline(pts_xy, eps=cfg.simplify_eps)
            if len(simp) < cfg.min_stroke_points:
                continue
            L = self.path_length(simp)
            x0, y0, x1, y1 = self.stroke_bbox(simp)
            bbox_area = (x1 - x0) * (y1 - y0)
            if L < cfg.min_stroke_length or bbox_area < cfg.min_stroke_bbox_area:
                continue
            strokes.append({"feature": None, "source": "skeleton", "points": simp})
        return strokes

    # ============================================================
    # Step 6: metrics + feature grouping
    # ============================================================
    def enrich_stroke_metrics(self, strokes):
        enriched = []
        for i, s in enumerate(strokes):
            pts = np.asarray(s["points"], dtype=np.float32)
            x0, y0, x1, y1 = self.stroke_bbox(pts)
            cx, cy = self.stroke_centroid(pts)
            length = self.path_length(pts)
            closed = self.is_closed_path(pts)
            area = self.stroke_area_if_closed(pts)
            circ = self.stroke_circularity_if_closed(pts)
            enriched.append({
                "id": i,
                "feature": s.get("feature"),
                "source": s.get("source", "unknown"),
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

    def find_eye_strokes(self, strokes, image_shape):
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
        return sorted(eye_candidates, key=lambda s: s["centroid"][0])

    def remove_tiny_strokes_near_eyes(self, strokes, image_shape):
        cfg = self.cfg
        if not cfg.delete_tiny_strokes_near_eyes:
            return strokes
        eye_candidates = self.find_eye_strokes(strokes, image_shape)
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
                    if s["length_px"] < cfg.tiny_near_eye_length or s["bbox_area"] < cfg.tiny_near_eye_bbox_area:
                        keep_this = False
            if keep_this:
                keep.append(s)
        return keep

    def remove_top_border_artifacts(self, strokes, image_shape):
        cfg = self.cfg
        if not cfg.delete_top_border_artifacts:
            return strokes
        h, _ = image_shape[:2]
        keep = []
        for s in strokes:
            _, _, _, y1 = s["bbox"]
            if y1 < cfg.top_border_frac * h and s["length_px"] < cfg.top_border_min_length_keep:
                continue
            keep.append(s)
        return keep

    def classify_features(self, strokes, image_shape):
        h, w = image_shape[:2]
        eye_candidates = self.find_eye_strokes(strokes, image_shape)
        if len(eye_candidates) >= 1:
            eye_candidates[0]["feature"] = "left_eye"
        if len(eye_candidates) >= 2:
            eye_candidates[1]["feature"] = "right_eye"

        remaining = [s for s in strokes if s["feature"] is None]
        if remaining:
            candidates = []
            for s in remaining:
                _, y0, _, _ = s["bbox"]
                if y0 < 0.18 * h and s["bbox_area"] > 0.10 * w * h:
                    candidates.append(s)
            if candidates:
                max(candidates, key=lambda s: s["bbox_area"])["feature"] = "outer_contour"

        remaining = [s for s in strokes if s["feature"] is None]
        mouth_candidates = []
        for s in remaining:
            cx, cy = s["centroid"]
            if (not s["closed"] and s["length_px"] > 80 and 0.52 * h <= cy <= 0.75 * h and 0.20 * w <= cx <= 0.85 * w):
                mouth_candidates.append(s)
        if mouth_candidates:
            max(mouth_candidates, key=lambda s: s["length_px"])["feature"] = "mouth"

        remaining = [s for s in strokes if s["feature"] is None]
        nose_candidates = []
        for s in remaining:
            cx, cy = s["centroid"]
            if 0.35 * h <= cy <= 0.66 * h and 0.45 * w <= cx <= 0.88 * w and s["length_px"] > 30:
                nose_candidates.append(s)
        if nose_candidates:
            max(nose_candidates, key=lambda s: s["length_px"])["feature"] = "nose"

        for s in [s for s in strokes if s["feature"] is None]:
            _, cy = s["centroid"]
            if cy < 0.43 * h:
                s["feature"] = "hair"

        for s in [s for s in strokes if s["feature"] is None]:
            cx, cy = s["centroid"]
            if 0.38 * h <= cy <= 0.68 * h and cx < 0.33 * w:
                s["feature"] = "ear"

        for s in [s for s in strokes if s["feature"] is None]:
            _, cy = s["centroid"]
            if cy > 0.74 * h:
                s["feature"] = "shoulders" if s["bbox_w"] > 0.35 * w else "neck_collar"

        for s in strokes:
            if s["feature"] is None:
                s["feature"] = "misc"
        return strokes

    @staticmethod
    def stroke_order_priority(feature):
        priorities = {
            "outer_contour": 0, "hair": 1, "left_eye": 2, "right_eye": 3,
            "left_iris": 4, "right_iris": 5, "nose": 6, "mouth": 7,
            "ear": 8, "neck_collar": 9, "shoulders": 10, "misc": 11,
        }
        return priorities.get(feature, 99)

    def sort_strokes_for_drawing(self, strokes):
        return sorted(strokes, key=lambda s: (self.stroke_order_priority(s["feature"]), -s["length_px"]))

    # ============================================================
    # Save / render / animation
    # ============================================================
    @staticmethod
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

    def save_stroke_json(self, strokes, out_path, extra_metadata=None):
        groups = defaultdict(list)
        for idx, s in enumerate(strokes):
            groups[s["feature"]].append(idx)
        payload = {
            "num_strokes": len(strokes),
            "groups": dict(groups),
            "config": asdict(self.cfg),
            "metadata": extra_metadata or {},
            "strokes": self.serialize_strokes(strokes),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def save_binary_debug(mask, out_path):
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    def render_strokes_to_image(self, strokes, image_shape, thickness):
        h, w = image_shape[:2]
        canvas = np.full((h, w), 255, dtype=np.uint8)
        for s in strokes:
            self.draw_polyline(canvas, s["points"], thickness=thickness, color=0)
        return canvas

    def save_layered_plot(self, strokes, image_shape, out_path):
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

    @staticmethod
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

    def build_animation_frames(self, strokes, image_shape):
        cfg = self.cfg
        h, w = image_shape[:2]
        canvas = np.full((h, w), 255, dtype=np.uint8)
        frames = [cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2RGB)]
        for s in strokes:
            pts = np.asarray(s["points"], dtype=np.float32)
            n = len(pts)
            if n < 2:
                continue
            step = max(1, n // cfg.anim_progress_steps_per_stroke)
            for k in range(2, n + 1, step):
                frame = canvas.copy()
                self.draw_polyline(frame, pts[:k], thickness=cfg.anim_line_thickness, color=0)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
            self.draw_polyline(canvas, pts, thickness=cfg.anim_line_thickness, color=0)
            frames.append(cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2RGB))
        return frames

    def save_animation_gif(self, frames, out_path):
        imageio.mimsave(out_path, frames, duration=self.cfg.anim_frame_duration)

    def save_animation_mp4(self, frames, out_path):
        try:
            writer = imageio.get_writer(out_path, fps=self.cfg.mp4_fps)
            for fr in frames:
                writer.append_data(fr)
            writer.close()
        except Exception as e:
            print(f"[warning] MP4 export skipped: {e}")

    # ============================================================
    # Main pipeline
    # ============================================================
    def run(self, input_path: str, out_dir: str = "stroke_pipeline_output") -> Dict[str, Any]:
        cfg = self.cfg
        self.ensure_dir(out_dir)

        gray, mask0 = self.load_binary_image(input_path)
        mask1 = self.remove_small_components(mask0)
        mask2 = self.morphological_cleanup(mask1)
        mask3, bbox = self.crop_to_foreground_bbox(mask2)

        pupils = self.detect_pupil_blobs(mask3)
        if len(pupils) >= 2:
            mask_no_pupils = self.remove_pupils_from_mask(mask3, pupils[:2])
            iris_strokes = self.build_iris_replacement_strokes(pupils[:2])
            iris_mode = "filled_pupil_detection"
            glasses = []
        else:
            mask_no_pupils = mask3.copy()
            glasses = self.detect_glasses_from_mask(mask3)
            if cfg.use_glasses_fallback_iris and len(glasses) == 2:
                iris_strokes = self.build_iris_from_glasses_strokes(glasses)
                iris_mode = "glasses_fallback"
            else:
                iris_strokes = []
                iris_mode = "none"

        skel_raw = self.skeletonize_mask(mask_no_pupils)
        skel_pruned = self.prune_skeleton_spurs(skel_raw)
        traced_strokes = self.trace_paths_from_skeleton(skel_pruned)

        raw_strokes = traced_strokes + iris_strokes
        strokes = self.enrich_stroke_metrics(raw_strokes)
        original_stroke_count = len(strokes)
        original_point_count = sum(len(s["points"]) for s in strokes)

        strokes = self.remove_top_border_artifacts(strokes, gray.shape)
        strokes = self.remove_tiny_strokes_near_eyes(strokes, gray.shape)
        raw_again = [{"feature": s["feature"], "source": s["source"], "points": s["points"]} for s in strokes]
        strokes = self.enrich_stroke_metrics(raw_again)
        strokes = self.classify_features(strokes, gray.shape)
        strokes, merge_info = self.apply_smoothing_and_merging(strokes)
        strokes = self.sort_strokes_for_drawing(strokes)

        for new_order, s in enumerate(strokes):
            s["draw_order"] = new_order

        final_stroke_count = len(strokes)
        final_point_count = sum(len(s["points"]) for s in strokes)

        if cfg.save_debug_images:
            self.save_binary_debug(mask0, os.path.join(out_dir, "01_binary_input.png"))
            self.save_binary_debug(mask3, os.path.join(out_dir, "02_cleaned_mask.png"))
            self.save_binary_debug(mask_no_pupils, os.path.join(out_dir, "03_mask_without_pupils.png"))
            self.save_binary_debug(skel_raw, os.path.join(out_dir, "04_skeleton_raw.png"))
            self.save_binary_debug(skel_pruned, os.path.join(out_dir, "05_skeleton_pruned.png"))

        thin = self.render_strokes_to_image(strokes, gray.shape, cfg.centerline_preview_thickness)
        thick = self.render_strokes_to_image(strokes, gray.shape, cfg.marker_preview_thickness)
        cv2.imwrite(os.path.join(out_dir, "06_centerline_preview.png"), thin)
        cv2.imwrite(os.path.join(out_dir, "07_marker_thickness_preview.png"), thick)
        self.save_layered_plot(strokes, gray.shape, os.path.join(out_dir, "08_layered_strokes.png"))
        self.save_overlay_plot(gray, strokes, os.path.join(out_dir, "09_overlay_on_input.png"))

        groups = defaultdict(int)
        for s in strokes:
            groups[s["feature"]] += 1
        groups = dict(groups)

        metadata = {
            "input_path": input_path,
            "out_dir": out_dir,
            "foreground_bbox": bbox,
            "pupils_detected": len(pupils),
            "glasses_detected": len(glasses),
            "iris_mode": iris_mode,
            "original_stroke_count": original_stroke_count,
            "final_stroke_count": final_stroke_count,
            "original_point_count": original_point_count,
            "final_point_count": final_point_count,
            "avg_points_per_final_stroke": final_point_count / max(final_stroke_count, 1),
            "groups": groups,
            **merge_info,
        }
        self.save_stroke_json(strokes, os.path.join(out_dir, "strokes.json"), extra_metadata=metadata)

        frames = None
        if cfg.save_gif or cfg.save_mp4:
            frames = self.build_animation_frames(strokes, gray.shape)
            if cfg.save_gif:
                self.save_animation_gif(frames, os.path.join(out_dir, "10_stroke_animation.gif"))
            if cfg.save_mp4:
                self.save_animation_mp4(frames, os.path.join(out_dir, "11_stroke_animation.mp4"))

        result = {
            "strokes": strokes,
            "metadata": metadata,
            "out_dir": out_dir,
            "paths": {
                "binary_input": os.path.join(out_dir, "01_binary_input.png"),
                "cleaned_mask": os.path.join(out_dir, "02_cleaned_mask.png"),
                "mask_without_pupils": os.path.join(out_dir, "03_mask_without_pupils.png"),
                "skeleton_raw": os.path.join(out_dir, "04_skeleton_raw.png"),
                "skeleton_pruned": os.path.join(out_dir, "05_skeleton_pruned.png"),
                "centerline_preview": os.path.join(out_dir, "06_centerline_preview.png"),
                "marker_preview": os.path.join(out_dir, "07_marker_thickness_preview.png"),
                "layered_strokes": os.path.join(out_dir, "08_layered_strokes.png"),
                "overlay": os.path.join(out_dir, "09_overlay_on_input.png"),
                "gif": os.path.join(out_dir, "10_stroke_animation.gif"),
                "mp4": os.path.join(out_dir, "11_stroke_animation.mp4"),
                "json": os.path.join(out_dir, "strokes.json"),
            },
        }
        self.last_result = result

        if cfg.verbose:
            self.print_summary(result)
        return result

    # ============================================================
    # Strokes-only fast path (for tuning loops)
    # ============================================================
    def _compute_strokes(self, input_path: str):
        """Run all stroke-extraction steps but skip all file writes.

        Returns ``(strokes, gray)`` so callers can render however they want.
        """
        cfg = self.cfg

        gray, mask0 = self.load_binary_image(input_path)
        mask1 = self.remove_small_components(mask0)
        mask2 = self.morphological_cleanup(mask1)
        mask3, _bbox = self.crop_to_foreground_bbox(mask2)

        pupils = self.detect_pupil_blobs(mask3)
        if len(pupils) >= 2:
            mask_no_pupils = self.remove_pupils_from_mask(mask3, pupils[:2])
            iris_strokes = self.build_iris_replacement_strokes(pupils[:2])
        else:
            mask_no_pupils = mask3.copy()
            glasses = self.detect_glasses_from_mask(mask3)
            if cfg.use_glasses_fallback_iris and len(glasses) == 2:
                iris_strokes = self.build_iris_from_glasses_strokes(glasses)
            else:
                iris_strokes = []

        skel_raw = self.skeletonize_mask(mask_no_pupils)
        skel_pruned = self.prune_skeleton_spurs(skel_raw)
        traced_strokes = self.trace_paths_from_skeleton(skel_pruned)

        strokes = self.enrich_stroke_metrics(traced_strokes + iris_strokes)
        strokes = self.remove_top_border_artifacts(strokes, gray.shape)
        strokes = self.remove_tiny_strokes_near_eyes(strokes, gray.shape)
        raw_again = [
            {"feature": s["feature"], "source": s["source"], "points": s["points"]}
            for s in strokes
        ]
        strokes = self.enrich_stroke_metrics(raw_again)
        strokes = self.classify_features(strokes, gray.shape)
        strokes, _ = self.apply_smoothing_and_merging(strokes)
        strokes = self.sort_strokes_for_drawing(strokes)

        for new_order, s in enumerate(strokes):
            s["draw_order"] = new_order

        return strokes, gray

    def run_strokes_only(self, input_path: str, out_png_path: str) -> str:
        """Run the pipeline and save ONLY the layered strokes PNG.

        Used by the tuning loop where rendering 11 debug artifacts per image
        is wasteful. Returns the output path.
        """
        strokes, gray = self._compute_strokes(input_path)
        parent = os.path.dirname(out_png_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.save_layered_plot(strokes, gray.shape, out_png_path)
        return out_png_path

    def print_summary(self, result: Dict[str, Any]):
        m = result["metadata"]
        strokes = result["strokes"]
        print("===================================================")
        print("PIPELINE COMPLETE")
        print("===================================================")
        print(f"Input: {m['input_path']}")
        print(f"Output dir: {m['out_dir']}")
        print(f"Foreground bbox: {m['foreground_bbox']}")
        print(f"Pupils detected: {m['pupils_detected']}")
        print(f"Glasses fallback detected: {m['glasses_detected']}")
        print(f"Iris mode: {m['iris_mode']}")
        print()
        print("Stroke reduction:")
        print(f"  Initial strokes after trace: {m['original_stroke_count']}")
        print(f"  Before merge:                {m['before_merge_count']}")
        print(f"  After merge:                 {m['after_merge_count']}")
        print(f"  Final strokes:               {m['final_stroke_count']}")
        print(f"  Initial vector points:        {m['original_point_count']}")
        print(f"  Final vector points:          {m['final_point_count']}")
        print(f"  Avg points/final stroke:      {m['avg_points_per_final_stroke']:.1f}")
        print()
        print("Stroke groups:")
        for k in sorted(m["groups"].keys(), key=self.stroke_order_priority):
            print(f"  {k}: {m['groups'][k]}")
        print()
        print("Saved files:")
        for _, p in result["paths"].items():
            print(f"  {p}")
        print()
        print("Per-stroke point counts:")
        for i, s in enumerate(strokes):
            print(f"{i:02d} | {s['feature']:14s} | {len(s['points']):3d} pts | {s['length_px']:7.1f}px | {s['source']}")