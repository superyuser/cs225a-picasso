"""Run the full camera-to-portrait robot workflow.

Pipeline:
    1. robot/orient_camera.py captures a centered selfie into raw-captures.
    2. cv/portraitfy.py converts the selfie to clean line art.
    3. cv/run_pipeline_robust_bst.py converts the line art to a stroke JSON.
    4. robot/draw_strokes.py draws that stroke JSON on the calibrated canvas.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent

ORIENT_CAMERA_SCRIPT = SCRIPT_DIR / "orient_camera.py"
DRAW_STROKES_SCRIPT = SCRIPT_DIR / "draw_strokes.py"
PORTRAITFY_SCRIPT = DEMO_DAY_DIR / "cv" / "portraitfy.py"
ROBUST_PIPELINE_SCRIPT = DEMO_DAY_DIR / "cv" / "run_pipeline_robust_bst.py"

DEFAULT_RAW_CAPTURES_DIR = DEMO_DAY_DIR / "raw-captures"
DEFAULT_PORTRAITS_DIR = DEMO_DAY_DIR / "portraits"
DEFAULT_STROKES_DIR = DEMO_DAY_DIR / "strokes"
DEFAULT_STROKE_JSONS_DIR = DEMO_DAY_DIR / "stroke-jsons"
DEFAULT_CORNERS_JSON = DEMO_DAY_DIR / "canvas_corners.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAPTURE_LOG_PREFIX = "Saved centered raw capture:"


def log(message: str) -> None:
    print(f"[portrait-pipeline] {message}", flush=True)


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def newest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))


def demo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return DEMO_DAY_DIR / path


def find_new_capture(
    raw_captures_dir: Path,
    *,
    before_paths: set[Path],
    started_at_s: float,
) -> Path | None:
    after_paths = image_files(raw_captures_dir)
    new_paths = [path for path in after_paths if path not in before_paths]
    if new_paths:
        return newest_file(new_paths)

    # Fallback for unusual timestamp/set behavior. orient_camera.py writes a
    # unique filename, so this should rarely be needed.
    recent_paths = [
        path
        for path in after_paths
        if path.stat().st_mtime >= started_at_s - 1.0
    ]
    return newest_file(recent_paths)


def run_command(command: list[str], *, label: str) -> None:
    log(f"starting {label}")
    log("command: " + " ".join(command))
    subprocess.run(command, cwd=DEMO_DAY_DIR, check=True)
    log(f"finished {label}")


def require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not created: {path}")


def portrait_output_path(args: argparse.Namespace, capture_path: Path) -> Path:
    if args.portrait_output is None:
        return demo_path(args.portraits_dir) / f"{capture_path.stem}_cartoon.png"
    return demo_path(args.portrait_output)


def build_portraitfy_command(
    args: argparse.Namespace,
    *,
    capture_path: Path,
    portrait_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(PORTRAITFY_SCRIPT),
        "--input",
        str(capture_path),
        "--output",
        str(portrait_path),
        "--model",
        args.portrait_model,
        "--size",
        args.portrait_size,
    ]


def start_portraitfy(
    args: argparse.Namespace,
    capture_path: Path,
) -> tuple[subprocess.Popen, Path]:
    portrait_path = portrait_output_path(args, capture_path)
    command = build_portraitfy_command(
        args,
        capture_path=capture_path,
        portrait_path=portrait_path,
    )
    log("starting portrait generation")
    log("command: " + " ".join(command))
    process = subprocess.Popen(command, cwd=DEMO_DAY_DIR)
    return process, portrait_path


def wait_for_process(process: subprocess.Popen, *, label: str) -> None:
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, process.args)
    log(f"finished {label}")


def stop_process(process: subprocess.Popen, *, label: str) -> None:
    if process.poll() is not None:
        return
    log(f"terminating {label}")
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        log(f"killing {label}")
        process.kill()
        process.wait()


def command_display_name(command) -> str:
    if isinstance(command, (list, tuple)):
        parts = [str(part) for part in command]
        if len(parts) >= 3 and parts[1] == "-u":
            return parts[2]
        if len(parts) >= 2:
            return parts[1]
        if parts:
            return parts[0]
    return str(command)


def capture_path_from_log_line(line: str) -> Path | None:
    if CAPTURE_LOG_PREFIX not in line:
        return None
    raw_path = line.split(CAPTURE_LOG_PREFIX, 1)[1].strip()
    if not raw_path:
        return None
    return demo_path(raw_path)


def run_camera_capture_and_portrait(
    args: argparse.Namespace,
) -> tuple[Path, subprocess.Popen, Path]:
    raw_captures_dir = demo_path(args.raw_captures_dir)
    before_paths = set(image_files(raw_captures_dir))
    started_at_s = time.time()

    command = [
        sys.executable,
        "-u",
        str(ORIENT_CAMERA_SCRIPT),
        "--camera-index",
        str(args.camera_index),
        "--config-file",
        args.config_file,
        "--joint-arrival-threshold",
        str(args.joint_arrival_threshold),
        "--joint-max-step-deg",
        str(args.joint_max_step_deg),
        "--joint-controller-settle-s",
        str(args.joint_controller_settle_s),
        "--raw-captures-dir",
        str(raw_captures_dir),
    ]
    if args.no_preview:
        command.append("--no-preview")

    log("starting camera capture")
    log("command: " + " ".join(command))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    camera_process = subprocess.Popen(
        command,
        cwd=DEMO_DAY_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    capture_path: Path | None = None
    portrait_process: subprocess.Popen | None = None
    portrait_path: Path | None = None

    assert camera_process.stdout is not None
    try:
        for line in camera_process.stdout:
            print(line, end="", flush=True)
            logged_capture_path = capture_path_from_log_line(line)
            if logged_capture_path is None or portrait_process is not None:
                continue

            capture_path = logged_capture_path
            require_file(capture_path, label="raw capture")
            log(f"selected raw capture: {capture_path}")
            log("photo saved; starting portrait generation while robot returns to INIT_POS")
            portrait_process, portrait_path = start_portraitfy(args, capture_path)

        camera_returncode = camera_process.wait()
    except Exception:
        stop_process(camera_process, label="camera capture")
        if portrait_process is not None:
            stop_process(portrait_process, label="portrait generation")
        raise

    if camera_returncode != 0:
        if portrait_process is not None:
            stop_process(portrait_process, label="portrait generation")
        raise subprocess.CalledProcessError(camera_returncode, command)

    log("finished camera capture")

    if capture_path is None:
        capture_path = find_new_capture(
            raw_captures_dir,
            before_paths=before_paths,
            started_at_s=started_at_s,
        )
        if capture_path is None:
            raise RuntimeError(
                "camera capture completed but no new raw capture was found in "
                f"{raw_captures_dir}"
            )
        log(f"selected raw capture: {capture_path}")

    if portrait_process is None:
        log("starting portrait generation after camera capture completed")
        portrait_process, portrait_path = start_portraitfy(args, capture_path)
    else:
        assert portrait_path is not None

    return capture_path, portrait_process, portrait_path


def run_portraitfy(args: argparse.Namespace, capture_path: Path) -> Path:
    portrait_path = portrait_output_path(args, capture_path)
    command = build_portraitfy_command(
        args,
        capture_path=capture_path,
        portrait_path=portrait_path,
    )
    run_command(command, label="portrait generation")
    require_file(portrait_path, label="portrait image")
    return portrait_path


def wait_for_portraitfy(process: subprocess.Popen, portrait_path: Path) -> None:
    wait_for_process(process, label="portrait generation")
    require_file(portrait_path, label="portrait image")


def should_preload_initial_paint(args: argparse.Namespace) -> bool:
    return not args.no_draw and not args.draw_dry_run


def run_initial_paint_dip(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(DRAW_STROKES_SCRIPT),
        "--initial-dip-only",
        "--init-dwell-s",
        str(args.init_dwell_s),
        "--hover-dwell-s",
        str(args.hover_dwell_s),
        "--dip-dwell-s",
        str(args.dip_dwell_s),
        "--timeout-s",
        str(args.timeout_s),
        "--status-period-s",
        str(args.status_period_s),
    ]
    if args.rebuild_tool_init_path:
        command.append("--rebuild-tool-init-path")
    if args.no_cached_tool_init_path:
        command.append("--no-cached-tool-init-path")

    run_command(command, label="initial paint dip")


def run_stroke_pipeline(args: argparse.Namespace, portrait_path: Path) -> Path:
    stroke_jsons_dir = demo_path(args.stroke_jsons_dir)
    stroke_json_path = stroke_jsons_dir / f"{portrait_path.stem}.json"

    command = [
        sys.executable,
        str(ROBUST_PIPELINE_SCRIPT),
        "--input",
        str(portrait_path),
        "--strokes-dir",
        str(demo_path(args.strokes_dir)),
        "--stroke-jsons-dir",
        str(stroke_jsons_dir),
        "--canvas_w_mm",
        str(args.canvas_w_mm),
        "--canvas_h_mm",
        str(args.canvas_h_mm),
        "--seed",
        str(args.seed),
        "--fill-brush-mm",
        str(args.fill_brush_mm),
        "--outline-brush-mm",
        str(args.outline_brush_mm),
        "--barrier-dilate-px",
        str(args.barrier_dilate_px),
        "--fillet-angle-deg",
        str(args.fillet_angle_deg),
        "--fillet-ratio",
        str(args.fillet_ratio),
        "--fillet-iterations",
        str(args.fillet_iterations),
    ]
    if args.no_skeleton_outline:
        command.append("--no-skeleton-outline")
    if args.with_animation:
        if args.animation_output is not None:
            command.extend(["--animation-output", str(demo_path(args.animation_output))])
        command.extend(
            [
                "--animation-fps",
                str(args.animation_fps),
                "--animation-strokes-per-frame",
                str(args.animation_strokes_per_frame),
                "--animation-hold-seconds",
                str(args.animation_hold_seconds),
            ]
        )
    else:
        command.append("--no-animation")

    run_command(command, label="stroke generation")
    require_file(stroke_json_path, label="stroke JSON")
    return stroke_json_path


def run_draw_strokes(
    args: argparse.Namespace,
    stroke_json_path: Path,
    *,
    skip_initial_dip: bool,
) -> None:
    command = [
        sys.executable,
        str(DRAW_STROKES_SCRIPT),
        str(stroke_json_path),
        "--corners-json",
        str(demo_path(args.corners_json)),
        "--max-step-m",
        str(args.max_step_m),
        "--hover-x-offset-m",
        str(args.hover_x_offset_m),
        "--init-dwell-s",
        str(args.init_dwell_s),
        "--hover-dwell-s",
        str(args.hover_dwell_s),
        "--stroke-dwell-s",
        str(args.stroke_dwell_s),
        "--dip-dwell-s",
        str(args.dip_dwell_s),
        "--timeout-s",
        str(args.timeout_s),
        "--status-period-s",
        str(args.status_period_s),
        "--first-outline-stroke-settle-s",
        str(args.first_outline_stroke_settle_s),
    ]
    if args.draw_dry_run:
        command.append("--dry-run")
    if skip_initial_dip:
        command.append("--skip-initial-dip")
    if args.max_strokes_per_layer is not None:
        command.extend(
            ["--max-strokes-per-layer", str(args.max_strokes_per_layer)]
        )
    if args.rebuild_tool_init_path:
        command.append("--rebuild-tool-init-path")
    if args.no_cached_tool_init_path:
        command.append("--no-cached-tool-init-path")

    run_command(command, label="robot drawing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run capture -> portrait -> stroke JSON -> draw_strokes."
    )

    capture = parser.add_argument_group("camera capture")
    capture.add_argument(
        "--capture",
        default=None,
        help="Use an existing raw capture image and skip orient_camera.py.",
    )
    capture.add_argument("--camera-index", type=int, default=0)
    capture.add_argument("--no-preview", action="store_true")
    capture.add_argument("--config-file", default="basket.xml")
    capture.add_argument("--joint-arrival-threshold", type=float, default=0.28)
    capture.add_argument("--joint-max-step-deg", type=float, default=0.5)
    capture.add_argument("--joint-controller-settle-s", type=float, default=0.25)
    capture.add_argument(
        "--raw-captures-dir",
        default=str(DEFAULT_RAW_CAPTURES_DIR),
    )

    portrait = parser.add_argument_group("portrait generation")
    portrait.add_argument("--portrait-output", default=None)
    portrait.add_argument("--portraits-dir", default=str(DEFAULT_PORTRAITS_DIR))
    portrait.add_argument("--portrait-model", default="gpt-image-1")
    portrait.add_argument("--portrait-size", default="1024x1024")

    strokes = parser.add_argument_group("stroke generation")
    strokes.add_argument("--strokes-dir", default=str(DEFAULT_STROKES_DIR))
    strokes.add_argument(
        "--stroke-jsons-dir",
        default=str(DEFAULT_STROKE_JSONS_DIR),
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
    drawing.add_argument("--corners-json", default=str(DEFAULT_CORNERS_JSON))
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
        if args.capture is None:
            capture_path, portrait_process, portrait_path = run_camera_capture_and_portrait(args)
        else:
            capture_path = demo_path(args.capture)
            require_file(capture_path, label="raw capture")
            log(f"using existing raw capture: {capture_path}")
            portrait_process, portrait_path = start_portraitfy(args, capture_path)

        preloaded_initial_paint = should_preload_initial_paint(args)
        if preloaded_initial_paint:
            log(
                "portrait generation is running; dipping paint 1 before waiting "
                "for the portrait output"
            )
            run_initial_paint_dip(args)

        wait_for_portraitfy(portrait_process, portrait_path)

        log(f"portrait image: {portrait_path}")

        stroke_json_path = run_stroke_pipeline(args, portrait_path)
        log(f"stroke JSON: {stroke_json_path}")

        if args.no_draw:
            log("stopping before robot drawing because --no-draw was set")
        else:
            run_draw_strokes(
                args,
                stroke_json_path,
                skip_initial_dip=preloaded_initial_paint,
            )

    except subprocess.CalledProcessError as exc:
        if portrait_process is not None:
            stop_process(portrait_process, label="portrait generation")
        log(f"{command_display_name(exc.cmd)} exited with status {exc.returncode}")
        return exc.returncode
    except Exception as exc:
        if portrait_process is not None:
            stop_process(portrait_process, label="portrait generation")
        log(str(exc))
        return 1

    log("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
