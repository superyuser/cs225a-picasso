"""Run the portrait robot workflow from an existing capture.

Pipeline:
    1. Read an existing raw capture image.
    2. cv/portraitfy.py converts the capture to clean line art.
    3. cv/run_pipeline_robust_bst.py converts the line art to a stroke JSON.
    4. robot/draw_strokes.py draws that stroke JSON on the calibrated canvas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_portrait_pipeline as pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run existing capture -> portrait -> stroke JSON -> draw_strokes, "
            "skipping orient_camera.py."
        )
    )
    parser.add_argument(
        "capture",
        help="Existing raw capture image to convert and draw.",
    )

    portrait = parser.add_argument_group("portrait generation")
    portrait.add_argument("--portrait-output", default=None)
    portrait.add_argument(
        "--portraits-dir",
        default=str(pipeline.DEFAULT_PORTRAITS_DIR),
    )
    portrait.add_argument("--portrait-model", default="gpt-image-1")
    portrait.add_argument("--portrait-size", default="1024x1024")

    strokes = parser.add_argument_group("stroke generation")
    strokes.add_argument(
        "--strokes-dir",
        default=str(pipeline.DEFAULT_STROKES_DIR),
    )
    strokes.add_argument(
        "--stroke-jsons-dir",
        default=str(pipeline.DEFAULT_STROKE_JSONS_DIR),
    )
    strokes.add_argument("--canvas-w-mm", dest="canvas_w_mm", type=float, default=254.0)
    strokes.add_argument("--canvas-h-mm", dest="canvas_h_mm", type=float, default=254.0)
    strokes.add_argument("--seed", type=int, default=42)
    strokes.add_argument("--fill-brush-mm", type=float, default=5.5)
    strokes.add_argument("--outline-brush-mm", type=float, default=3.5)
    strokes.add_argument("--barrier-dilate-px", type=int, default=3)
    strokes.add_argument("--no-skeleton-outline", action="store_true")
    strokes.add_argument("--fillet-angle-deg", type=float, default=150.0)
    strokes.add_argument("--fillet-ratio", type=float, default=0.25)
    strokes.add_argument("--fillet-iterations", type=int, default=2)
    strokes.add_argument(
        "--with-animation",
        action="store_true",
        help="Generate the optional stroke animation from run_pipeline_robust_bst.py.",
    )
    strokes.add_argument("--animation-output", default=None)
    strokes.add_argument("--animation-fps", type=int, default=24)
    strokes.add_argument("--animation-strokes-per-frame", type=int, default=1)
    strokes.add_argument("--animation-hold-seconds", type=float, default=1.0)

    drawing = parser.add_argument_group("robot drawing")
    drawing.add_argument(
        "--no-draw",
        action="store_true",
        help="Stop after generating the stroke JSON.",
    )
    drawing.add_argument(
        "--draw-dry-run",
        action="store_true",
        help="Pass --dry-run to robot/draw_strokes.py.",
    )
    drawing.add_argument(
        "--corners-json",
        default=str(pipeline.DEFAULT_CORNERS_JSON),
    )
    drawing.add_argument("--max-step-m", type=float, default=0.001)
    drawing.add_argument("--hover-x-offset-m", type=float, default=-0.015)
    drawing.add_argument("--init-dwell-s", type=float, default=0.25)
    drawing.add_argument("--hover-dwell-s", type=float, default=0.1)
    drawing.add_argument("--stroke-dwell-s", type=float, default=0.0)
    drawing.add_argument("--dip-dwell-s", type=float, default=0.5)
    drawing.add_argument("--timeout-s", type=float, default=120.0)
    drawing.add_argument("--status-period-s", type=float, default=0.5)
    drawing.add_argument("--max-strokes-per-layer", type=int, default=None)
    drawing.add_argument("--first-outline-stroke-settle-s", type=float, default=0.75)
    drawing.add_argument("--rebuild-tool-init-path", action="store_true")
    drawing.add_argument("--no-cached-tool-init-path", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    portrait_process: subprocess.Popen | None = None

    try:
        capture_path = pipeline.demo_path(args.capture)
        pipeline.require_file(capture_path, label="raw capture")
        pipeline.log(f"using existing raw capture: {capture_path}")

        portrait_process, portrait_path = pipeline.start_portraitfy(args, capture_path)

        preloaded_initial_paint = pipeline.should_preload_initial_paint(args)
        if preloaded_initial_paint:
            pipeline.log(
                "portrait generation is running; dipping paint 1 before waiting "
                "for the portrait output"
            )
            pipeline.run_initial_paint_dip(args)

        pipeline.wait_for_portraitfy(portrait_process, portrait_path)
        pipeline.log(f"portrait image: {portrait_path}")

        stroke_json_path = pipeline.run_stroke_pipeline(args, portrait_path)
        pipeline.log(f"stroke JSON: {stroke_json_path}")

        if args.no_draw:
            pipeline.log("stopping before robot drawing because --no-draw was set")
        else:
            pipeline.run_draw_strokes(
                args,
                stroke_json_path,
                skip_initial_dip=preloaded_initial_paint,
            )

    except subprocess.CalledProcessError as exc:
        if portrait_process is not None:
            pipeline.stop_process(portrait_process, label="portrait generation")
        pipeline.log(
            f"{pipeline.command_display_name(exc.cmd)} exited with status "
            f"{exc.returncode}"
        )
        return exc.returncode
    except Exception as exc:
        if portrait_process is not None:
            pipeline.stop_process(portrait_process, label="portrait generation")
        pipeline.log(str(exc))
        return 1

    pipeline.log("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
