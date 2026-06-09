"""Run the demo-day calibration/setup sequence.

Pipeline:
    1. robot/calibrate_canvas.py writes canvas_corners.json.
    2. robot/visit_corners.py visits the calibrated drawable canvas corners.

Run this before robot/run_portrait_pipeline.py on demo day.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent

CALIBRATE_CANVAS_SCRIPT = SCRIPT_DIR / "calibrate_canvas.py"
VISIT_CORNERS_SCRIPT = SCRIPT_DIR / "visit_corners.py"

DEFAULT_CORNERS_JSON = DEMO_DAY_DIR / "canvas_corners.json"
DEFAULT_INTRINSICS_JSON = SCRIPT_DIR / "camera_intrinsics.json"
DEFAULT_CONFIG_JSON = DEMO_DAY_DIR / "config.json"


def log(message: str) -> None:
    print(f"[calibration-pipeline] {message}", flush=True)


def demo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return DEMO_DAY_DIR / path


def run_command(command: list[str], *, label: str) -> None:
    log(f"starting {label}")
    log("command: " + " ".join(command))
    subprocess.run(command, cwd=DEMO_DAY_DIR, check=True)
    log(f"finished {label}")


def require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")


def load_canvas_border_offsets() -> tuple[float, float]:
    with DEFAULT_CONFIG_JSON.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return (
        float(config.get("canvas_border_horizontal_offset_m", 0.0)),
        float(config.get("canvas_border_vertical_offset_m", 0.0)),
    )


def run_calibrate_canvas(args: argparse.Namespace) -> Path:
    corners_json = demo_path(args.corners_json)
    command = [
        sys.executable,
        str(CALIBRATE_CANVAS_SCRIPT),
        "--camera-index",
        str(args.camera_index),
        "--intrinsics-json",
        str(demo_path(args.intrinsics_json)),
        "--output-json",
        str(corners_json),
        "--tag-size-mm",
        str(args.canvas_tag_size_mm),
        "--dwell-s",
        str(args.canvas_dwell_s),
    ]
    if args.no_preview:
        command.append("--no-preview")
    if args.no_canvas_move:
        command.append("--no-move")

    run_command(command, label="canvas calibration")
    require_file(corners_json, label="canvas calibration JSON")
    return corners_json


def run_visit_corners(args: argparse.Namespace, corners_json: Path) -> None:
    command = [
        sys.executable,
        str(VISIT_CORNERS_SCRIPT),
        "--corners-json",
        str(corners_json),
        "--dwell-s",
        str(args.corner_dwell_s),
    ]
    if args.skip_final_init:
        command.append("--skip-final-init")

    run_command(command, label="canvas corner visit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrate_canvas -> visit_corners."
    )

    flow = parser.add_argument_group("flow control")
    flow.add_argument(
        "--skip-canvas-calibration",
        action="store_true",
        help="Use the existing corners JSON and skip robot/calibrate_canvas.py.",
    )
    flow.add_argument(
        "--with_offsets",
        "--with-offsets",
        dest="with_offsets",
        action="store_true",
        help=(
            "Second-pass mode: reuse the existing corners JSON and visit "
            "corners with the current config.json canvas border offsets."
        ),
    )
    flow.add_argument(
        "--skip-corner-visit",
        action="store_true",
        help="Skip robot/visit_corners.py.",
    )

    canvas = parser.add_argument_group("canvas calibration")
    canvas.add_argument("--camera-index", type=int, default=0)
    canvas.add_argument("--no-preview", action="store_true")
    canvas.add_argument(
        "--no-canvas-move",
        action="store_true",
        help="Pass --no-move to robot/calibrate_canvas.py.",
    )
    canvas.add_argument(
        "--intrinsics-json",
        default=str(DEFAULT_INTRINSICS_JSON),
    )
    canvas.add_argument(
        "--corners-json",
        default=str(DEFAULT_CORNERS_JSON),
        help="Canvas calibration JSON path shared by calibration and corner visit.",
    )
    canvas.add_argument("--canvas-tag-size-mm", type=float, default=35.0)
    canvas.add_argument("--canvas-dwell-s", type=float, default=0.5)

    corners = parser.add_argument_group("corner visit")
    corners.add_argument("--corner-dwell-s", type=float, default=1.0)
    corners.add_argument("--skip-final-init", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corners_json = demo_path(args.corners_json)

    try:
        if args.skip_canvas_calibration or args.with_offsets:
            require_file(corners_json, label="existing canvas calibration JSON")
            if args.with_offsets:
                horizontal_offset_m, vertical_offset_m = load_canvas_border_offsets()
                log(f"with_offsets mode: using existing canvas calibration JSON: {corners_json}")
                log(
                    "current config border offsets: "
                    f"horizontal={horizontal_offset_m:+.4f} m, "
                    f"vertical={vertical_offset_m:+.4f} m"
                )
            else:
                log(f"using existing canvas calibration JSON: {corners_json}")
        else:
            corners_json = run_calibrate_canvas(args)

        if args.skip_corner_visit:
            log("skipping canvas corner visit")
        else:
            run_visit_corners(args, corners_json)

    except subprocess.CalledProcessError as exc:
        log(f"{exc.cmd[1]} exited with status {exc.returncode}")
        return exc.returncode
    except Exception as exc:
        log(str(exc))
        return 1

    log("calibration pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
