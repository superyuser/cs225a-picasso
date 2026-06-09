from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from portraitfy import cartoonize_to_line_art


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_CAPTURE_DIR = DEMO_DAY_DIR / "webcam-captures"
DEFAULT_PORTRAIT_DIR = DEMO_DAY_DIR / "portraits"
DEFAULT_STROKES_DIR = DEMO_DAY_DIR / "strokes"
DEFAULT_STROKE_JSONS_DIR = DEMO_DAY_DIR / "stroke-jsons"


def backend_value(name: str) -> int | None:
    if name == "any":
        return None
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "v4l2":
        return cv2.CAP_V4L2
    raise ValueError(f"Unsupported backend: {name}")


def open_capture(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def capture_photo(
    output_path: Path,
    camera_index: int,
    backend: int | None,
    warmup_frames: int,
    width: int | None,
    height: int | None,
    no_preview: bool,
) -> Path:
    cap = open_capture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_name = "Webcam Capture"
    last_frame = None

    try:
        for _ in range(max(0, warmup_frames)):
            ok, frame = cap.read()
            if ok:
                last_frame = frame
            time.sleep(0.03)

        if no_preview:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera opened but no frame could be read")
            last_frame = frame
        else:
            print("Press Space/Enter/c to capture, or q/Esc to quit.")
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera opened but no frame could be read")
                last_frame = frame

                preview = frame.copy()
                cv2.putText(
                    preview,
                    "Space/Enter/c: capture   q/Esc: quit",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    raise SystemExit("Capture cancelled.")
                if key in (13, 10, 32, ord("c")):
                    break

        if last_frame is None:
            raise RuntimeError("No camera frame was captured")
        if not cv2.imwrite(str(output_path), last_frame):
            raise RuntimeError(f"Could not write captured image to {output_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a webcam photo, convert it with portraitfy.py, then run "
            "run_pipeline_better_strokes.py and generate the MP4 stroke animation."
        )
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("any", "dshow", "v4l2"), default="dshow")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--capture-output", type=str, default=None)
    parser.add_argument("--portrait-output", type=str, default=None)
    parser.add_argument("--strokes-dir", type=str, default=str(DEFAULT_STROKES_DIR))
    parser.add_argument("--stroke-jsons-dir", type=str, default=str(DEFAULT_STROKE_JSONS_DIR))
    parser.add_argument("--animation-output", type=str, default=None)
    parser.add_argument("--animation-fps", type=int, default=24)
    parser.add_argument("--animation-strokes-per-frame", type=int, default=1)
    parser.add_argument("--animation-hold-seconds", type=float, default=1.0)
    parser.add_argument("--portrait-model", type=str, default="gpt-image-1")
    parser.add_argument("--portrait-size", type=str, default="1024x1024")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    capture_path = (
        Path(args.capture_output)
        if args.capture_output is not None
        else DEFAULT_CAPTURE_DIR / f"webcam_{timestamp}.png"
    )
    portrait_path = (
        Path(args.portrait_output)
        if args.portrait_output is not None
        else DEFAULT_PORTRAIT_DIR / f"{capture_path.stem}_cartoon.png"
    )

    print(f"Capturing webcam photo to {capture_path}")
    capture_photo(
        capture_path,
        camera_index=args.camera_index,
        backend=backend_value(args.backend),
        warmup_frames=args.warmup_frames,
        width=args.width,
        height=args.height,
        no_preview=args.no_preview,
    )

    print(f"Running portraitfy.py -> {portrait_path}")
    cartoonize_to_line_art(
        input_path=str(capture_path),
        output_path=str(portrait_path),
        model=args.portrait_model,
        size=args.portrait_size,
    )

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_pipeline_robust_bst.py"),
        "--input",
        str(portrait_path),
        "--strokes-dir",
        args.strokes_dir,
        "--stroke-jsons-dir",
        args.stroke_jsons_dir,
        "--animation-fps",
        str(args.animation_fps),
        "--animation-strokes-per-frame",
        str(args.animation_strokes_per_frame),
        "--animation-hold-seconds",
        str(args.animation_hold_seconds),
    ]
    if args.animation_output is not None:
        cmd.extend(["--animation-output", args.animation_output])

    print("Running stroke pipeline:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    stem = portrait_path.stem
    animation_path = (
        Path(args.animation_output)
        if args.animation_output is not None
        else Path(args.strokes_dir) / stem / "08_paint_stroke_animation.mp4"
    )
    json_path = Path(args.stroke_jsons_dir) / f"{stem}.json"

    print("Done.")
    print(f"Capture:       {capture_path}")
    print(f"Line art:      {portrait_path}")
    print(f"Painting plan: {json_path}")
    print(f"Animation:     {animation_path}")


if __name__ == "__main__":
    main()
