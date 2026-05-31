import json

import numpy as np

from config import MIN_POINT_DIM


def extract_stroke_points(stroke):
    """
    Extract world-space stroke points from one stroke object.

    Preferred keys:
      1. points_xyz_m
      2. points
    """
    if "points_xyz_m" in stroke:
        pts = stroke["points_xyz_m"]
    elif "points" in stroke:
        pts = stroke["points"]
    else:
        raise ValueError(f"Stroke missing points_xyz_m / points. Stroke keys: {list(stroke.keys())}")

    pts = np.array(pts, dtype=float)

    if len(pts.shape) != 2 or pts.shape[1] != MIN_POINT_DIM:
        raise ValueError(f"Invalid stroke points shape: {pts.shape}")

    return pts[:, :3]


def load_strokes(json_path):
    """
    Load strokes from strokes_canvas_plane.json.

    Expected structure:
      {
        "strokes": [
          {
            "order": ...,
            "points_xyz_m": [[x, y, z], ...]
          },
          ...
        ]
      }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if "strokes" not in data:
        raise ValueError("JSON file does not contain key 'strokes'.")

    strokes_raw = data["strokes"]

    # Sort by order if available.
    strokes_raw = sorted(
        strokes_raw,
        key=lambda s: s.get("order", 0)
    )

    parsed_strokes = []

    for stroke in strokes_raw:
        pts = extract_stroke_points(stroke)

        if pts.shape[0] < 1:
            continue

        parsed_strokes.append({
            "order": stroke.get("order", len(parsed_strokes)),
            "id": stroke.get("id", None),
            "feature": stroke.get("feature", "unknown"),
            "closed": stroke.get("closed", False),
            "points": pts
        })

    if len(parsed_strokes) == 0:
        raise ValueError("No usable strokes found in JSON.")

    return parsed_strokes, data
