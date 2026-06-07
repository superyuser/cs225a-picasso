"""Try camera indexes and preview any stream that opens."""

from __future__ import annotations

import argparse
import glob
import os
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np


WINDOW_NAME = "Camera Index Probe"


def available_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def open_capture(index: int, backend: int | None) -> cv2.VideoCapture:
    if backend is None:
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def draw_label(frame: np.ndarray, index: int) -> np.ndarray:
    labelled = frame.copy()
    cv2.putText(
        labelled,
        f"Camera index: {index}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        labelled,
        "Space/Enter/n: next   q/Esc: quit",
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return labelled


def preview_index(index: int, *, backend: int | None, read_timeout_s: float) -> str:
    print(f"Trying camera index {index}...")
    cap = open_capture(index, backend)

    if not cap.isOpened():
        cap.release()
        print(f"  index {index}: could not open")
        return "next"

    start_time = time.monotonic()
    first_frame_seen = False
    print(f"  index {index}: opened. Showing preview.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                if not first_frame_seen and time.monotonic() - start_time < read_timeout_s:
                    time.sleep(0.02)
                    continue

                print(f"  index {index}: could not read frame")
                return "next"

            first_frame_seen = True
            cv2.imshow(WINDOW_NAME, draw_label(frame, index))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return "quit"
            if key in (13, 10, 32, ord("n")):
                return "next"

    finally:
        cap.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try OpenCV camera indexes and show a stream for each camera that opens."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First camera index to try (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=10,
        help="Last camera index to try, inclusive (default: 10).",
    )
    parser.add_argument(
        "--read-timeout-s",
        type=float,
        default=1.0,
        help="Seconds to wait for the first readable frame after opening (default: 1.0).",
    )
    parser.add_argument(
        "--backend",
        choices=("any", "dshow", "v4l2"),
        default="any",
        help="OpenCV capture backend to use (default: any).",
    )
    return parser.parse_args()


def backend_value(name: str) -> int | None:
    if name == "any":
        return None
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "v4l2":
        return cv2.CAP_V4L2
    raise ValueError(f"Unsupported backend: {name}")


def main() -> None:
    args = parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be greater than or equal to --start")

    devices = available_video_devices()
    if devices:
        print("Video devices visible to this process:")
        for device in devices:
            print(f"  {device}")
    else:
        print("No /dev/video* devices are visible to this process.")
        print("If a camera is connected, it is not exposed to this Linux environment.")

    backend = backend_value(args.backend)

    try:
        for index in range(args.start, args.end + 1):
            action = preview_index(
                index,
                backend=backend,
                read_timeout_s=args.read_timeout_s,
            )
            if action == "quit":
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
