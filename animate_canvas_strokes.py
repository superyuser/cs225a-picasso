import os
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ============================================================
# CONFIG
# ============================================================

INPUT_JSON = "strokes_canvas_plane.json"
OUT_DIR = "canvas_plane_simulation"

# Animation output
SAVE_GIF = True
SAVE_MP4 = True
GIF_NAME = "strokes_canvas_world.gif"
MP4_NAME = "strokes_canvas_world.mp4"
STATIC_NAME = "strokes_canvas_world_final.png"

FPS = 20
DPI = 160

# Animation pacing
POINT_STRIDE = 1              # draw this many extra points per animation frame
HOLD_FRAMES_BETWEEN_STROKES = 4
HOLD_FRAMES_AT_END = 25

# 3D view
VIEW_ELEV = 18
VIEW_AZIM = -62

# Visual styling
SHOW_RAW_QUAD = True
SHOW_CLEAN_CANVAS = True
SHOW_CLEAN_CANVAS_EDGES = True
SHOW_RAW_CORNER_LABELS = True
SHOW_CLEAN_CORNER_LABELS = True
SHOW_STROKE_START_DOT = False

RAW_QUAD_COLOR = "tab:orange"
RAW_QUAD_ALPHA = 0.22
RAW_QUAD_LINEWIDTH = 1.4

CLEAN_CANVAS_FACE_COLOR = "#87aade"
CLEAN_CANVAS_ALPHA = 0.18
CLEAN_CANVAS_EDGE_COLOR = "#4a6fa5"
CLEAN_CANVAS_EDGE_WIDTH = 1.8

STROKE_COLOR = "black"
STROKE_LINEWIDTH = 2.8

CORNER_MARKER_SIZE = 35
TEXT_OFFSET_SCALE = 0.018     # label offset relative to scene span


# ============================================================
# JSON LOADING
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def arr3(x):
    return np.asarray(x, dtype=float).reshape(3)


def get_canvas_geometry(payload):
    cm = payload["canvas_mapping"]

    raw = cm["raw_corners_xyz"]
    clean = cm["clean_corners_xyz"]

    raw_corners = {
        "TL": arr3(raw["TL"]),
        "TR": arr3(raw["TR"]),
        "BL": arr3(raw["BL"]),
        "BR": arr3(raw["BR"]),
    }

    clean_corners = {
        "TL": arr3(clean["TL"]),
        "TR": arr3(clean["TR"]),
        "BL": arr3(clean["BL"]),
        "BR": arr3(clean["BR"]),
    }

    frame = cm["canvas_frame"]
    origin = arr3(frame["origin_TL_xyz"])
    u_axis = arr3(frame["u_axis_TL_to_TR"])
    v_axis = arr3(frame["v_axis_TL_to_BL"])
    normal = arr3(frame["normal_u_cross_v"])
    width_m = float(frame["width_m"])
    height_m = float(frame["height_m"])

    return raw_corners, clean_corners, origin, u_axis, v_axis, normal, width_m, height_m


def get_strokes_xyz(payload):
    strokes_xyz = []

    for stroke in payload["strokes"]:
        if "points_xyz_m" in stroke:
            pts = np.asarray(stroke["points_xyz_m"], dtype=float)
        else:
            pts = np.asarray(stroke["points"], dtype=float)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Expected stroke points to be Nx3 XYZ coordinates.")

        strokes_xyz.append({
            "feature": stroke.get("feature", "unknown"),
            "order": stroke.get("order", None),
            "points": pts,
        })

    return strokes_xyz


# ============================================================
# SCENE HELPERS
# ============================================================

def polygon_closed(points_dict, order=("TL", "TR", "BR", "BL")):
    pts = [points_dict[k] for k in order]
    pts.append(points_dict[order[0]])
    return np.asarray(pts, dtype=float)


def compute_scene_points(raw_corners, clean_corners, strokes):
    pts = []

    for p in raw_corners.values():
        pts.append(p)

    for p in clean_corners.values():
        pts.append(p)

    for s in strokes:
        pts.extend(list(s["points"]))

    return np.asarray(pts, dtype=float)


def set_axes_equal_3d(ax, points, pad_frac=0.10):
    """
    Make XYZ scales equal so the canvas plane doesn't look distorted.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)

    pad = pad_frac * span
    half = 0.5 * span + pad

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def add_labeled_corner_points(ax, corners, label_prefix="", color="tab:red", text_offset=None):
    for name, p in corners.items():
        ax.scatter([p[0]], [p[1]], [p[2]], s=CORNER_MARKER_SIZE, color=color, depthshade=True)
        label = f"{label_prefix}{name}" if label_prefix else name
        tp = p + text_offset
        ax.text(tp[0], tp[1], tp[2], label, color=color, fontsize=9, weight="bold")


def add_shaded_canvas(ax, clean_corners):
    verts = [[
        clean_corners["TL"],
        clean_corners["TR"],
        clean_corners["BR"],
        clean_corners["BL"],
    ]]

    poly = Poly3DCollection(
        verts,
        facecolors=CLEAN_CANVAS_FACE_COLOR,
        edgecolors=CLEAN_CANVAS_EDGE_COLOR if SHOW_CLEAN_CANVAS_EDGES else "none",
        linewidths=CLEAN_CANVAS_EDGE_WIDTH,
        alpha=CLEAN_CANVAS_ALPHA,
    )
    ax.add_collection3d(poly)
    return poly


def add_raw_quad(ax, raw_corners):
    raw_loop = polygon_closed(raw_corners)
    ax.plot(
        raw_loop[:, 0],
        raw_loop[:, 1],
        raw_loop[:, 2],
        color=RAW_QUAD_COLOR,
        linewidth=RAW_QUAD_LINEWIDTH,
        alpha=0.9,
        linestyle="--",
    )


def make_frame_schedule(strokes, point_stride=1, hold_between=4, hold_end=25):
    """
    Returns a list of tuples:
        (completed_strokes_count, active_stroke_index, active_point_count)

    Interpretation:
        - all strokes < completed_strokes_count are fully drawn
        - stroke active_stroke_index is partially drawn up to active_point_count
        - all later strokes are hidden
    """
    schedule = []

    for si, s in enumerate(strokes):
        n = len(s["points"])

        if n <= 1:
            schedule.append((si, si, n))
            for _ in range(hold_between):
                schedule.append((si + 1, None, None))
            continue

        start_k = 2 if n >= 2 else 1

        for k in range(start_k, n + 1, max(1, point_stride)):
            schedule.append((si, si, k))

        if (n - start_k) % max(1, point_stride) != 0:
            schedule.append((si, si, n))

        for _ in range(hold_between):
            schedule.append((si + 1, None, None))

    for _ in range(hold_end):
        schedule.append((len(strokes), None, None))

    return schedule


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def create_simulation(input_json=INPUT_JSON, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    payload = load_json(input_json)

    raw_corners, clean_corners, origin, u_axis, v_axis, normal, width_m, height_m = get_canvas_geometry(payload)
    strokes = get_strokes_xyz(payload)

    all_scene_pts = compute_scene_points(raw_corners, clean_corners, strokes)
    scene_span = np.max(all_scene_pts.max(axis=0) - all_scene_pts.min(axis=0))
    text_offset = TEXT_OFFSET_SCALE * scene_span * (u_axis + v_axis + 0.7 * normal)

    # --------------------------------------------------------
    # Figure setup
    # --------------------------------------------------------
    fig = plt.figure(figsize=(8.8, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    # Background canvas plane (clean/readjusted rectangle)
    if SHOW_CLEAN_CANVAS:
        add_shaded_canvas(ax, clean_corners)

    # Raw hand-calibrated quad
    if SHOW_RAW_QUAD:
        add_raw_quad(ax, raw_corners)

    # Corner labels
    if SHOW_RAW_CORNER_LABELS:
        add_labeled_corner_points(ax, raw_corners, label_prefix="raw_", color="tab:orange", text_offset=text_offset)

    if SHOW_CLEAN_CORNER_LABELS:
        add_labeled_corner_points(ax, clean_corners, label_prefix="clean_", color="tab:blue", text_offset=text_offset)

    # Pre-create one line artist per stroke
    line_artists = []
    start_dots = []

    for _ in strokes:
        line, = ax.plot([], [], [], color=STROKE_COLOR, linewidth=STROKE_LINEWIDTH)
        line_artists.append(line)

        if SHOW_STROKE_START_DOT:
            dot = ax.scatter([], [], [], s=10, color="red")
        else:
            dot = None
        start_dots.append(dot)

    # Clean canvas edges emphasized
    if SHOW_CLEAN_CANVAS_EDGES:
        clean_loop = polygon_closed(clean_corners)
        ax.plot(
            clean_loop[:, 0],
            clean_loop[:, 1],
            clean_loop[:, 2],
            color=CLEAN_CANVAS_EDGE_COLOR,
            linewidth=CLEAN_CANVAS_EDGE_WIDTH,
            alpha=0.9,
        )

    # Axis formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Stroke Animation on Slanted Canvas Plane", pad=18)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    set_axes_equal_3d(ax, all_scene_pts)

    # Optional pane cleanup
    ax.xaxis.pane.set_alpha(0.06)
    ax.yaxis.pane.set_alpha(0.06)
    ax.zaxis.pane.set_alpha(0.06)
    ax.grid(True, alpha=0.25)

    # --------------------------------------------------------
    # Frame schedule
    # --------------------------------------------------------
    schedule = make_frame_schedule(
        strokes,
        point_stride=POINT_STRIDE,
        hold_between=HOLD_FRAMES_BETWEEN_STROKES,
        hold_end=HOLD_FRAMES_AT_END,
    )

    # --------------------------------------------------------
    # Animation callbacks
    # --------------------------------------------------------
    def init():
        for line in line_artists:
            line.set_data([], [])
            line.set_3d_properties([])

        return tuple(line_artists)

    def update(frame_idx):
        completed_count, active_idx, active_pts = schedule[frame_idx]

        for i, s in enumerate(strokes):
            pts = s["points"]

            if i < completed_count:
                draw_pts = pts
            elif active_idx is not None and i == active_idx:
                draw_pts = pts[:active_pts]
            else:
                draw_pts = np.empty((0, 3), dtype=float)

            line_artists[i].set_data(draw_pts[:, 0] if len(draw_pts) else [],
                                     draw_pts[:, 1] if len(draw_pts) else [])
            line_artists[i].set_3d_properties(draw_pts[:, 2] if len(draw_pts) else [])

        return tuple(line_artists)

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(schedule),
        interval=1000.0 / FPS,
        blit=False,
        repeat=False,
    )

    # --------------------------------------------------------
    # Save final static image
    # --------------------------------------------------------
    for i, s in enumerate(strokes):
        pts = s["points"]
        line_artists[i].set_data(pts[:, 0], pts[:, 1])
        line_artists[i].set_3d_properties(pts[:, 2])

    static_path = os.path.join(out_dir, STATIC_NAME)
    fig.savefig(static_path, dpi=DPI, bbox_inches="tight")

    # Reset for animation export
    init()

    # --------------------------------------------------------
    # Save GIF
    # --------------------------------------------------------
    gif_path = os.path.join(out_dir, GIF_NAME)
    if SAVE_GIF:
        print(f"Saving GIF to: {gif_path}")
        anim.save(gif_path, writer=PillowWriter(fps=FPS), dpi=DPI)

    # --------------------------------------------------------
    # Save MP4
    # --------------------------------------------------------
    mp4_path = os.path.join(out_dir, MP4_NAME)
    if SAVE_MP4:
        try:
            print(f"Saving MP4 to: {mp4_path}")
            anim.save(mp4_path, writer="ffmpeg", fps=FPS, dpi=DPI)
        except Exception as e:
            print(f"[warning] MP4 export skipped: {e}")

    plt.close(fig)

    print("===================================================")
    print("3D CANVAS STROKE SIMULATION COMPLETE")
    print("===================================================")
    print(f"Input JSON:          {input_json}")
    print(f"Output directory:    {out_dir}")
    print(f"Static preview:      {static_path}")
    if SAVE_GIF:
        print(f"GIF animation:       {gif_path}")
    if SAVE_MP4:
        print(f"MP4 animation:       {mp4_path}")
    print()
    print("Canvas geometry:")
    print(f"  width:             {width_m:.4f} m")
    print(f"  height:            {height_m:.4f} m")
    print(f"  origin (TL):       {origin}")
    print(f"  u_axis:            {u_axis}")
    print(f"  v_axis:            {v_axis}")
    print(f"  normal:            {normal}")
    print()
    print(f"Number of strokes:   {len(strokes)}")
    print(f"Animation frames:    {len(schedule)}")

    return {
        "payload": payload,
        "strokes": strokes,
        "schedule_len": len(schedule),
        "static_path": static_path,
        "gif_path": gif_path if SAVE_GIF else None,
        "mp4_path": mp4_path if SAVE_MP4 else None,
    }


if __name__ == "__main__":
    create_simulation()