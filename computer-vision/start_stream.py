"""Live-webcam face-and-shoulder capture.

Can be used as a module:

    from start_stream import capture_portrait
    saved_path = capture_portrait()

Or invoked directly from the command line:

    python start_stream.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "captures"

CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
AUTO_CAPTURE_SECONDS = 3.0

FACE_TO_SHOULDER_WIDTH_MULT = 3.0
FACE_TO_SHOULDER_HEIGHT_MULT = 3.2
FACE_TOP_PADDING_MULT = 0.6
FACE_CENTER_Y_SHIFT_MULT = 0.9


def make_capture_id(now: Optional[datetime] = None) -> str:
    """Return a compact, sortable id at seconds resolution (e.g. ``20260531T111037``)."""
    return (now or datetime.now()).strftime("%Y%m%dT%H%M%S")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _make_square_face_shoulder_bbox(face_box, frame_shape):
    """Convert a face bbox into a larger square head+shoulders bbox."""
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_box

    face_cx = x + w / 2
    face_cy = y + h / 2

    crop_w = w * FACE_TO_SHOULDER_WIDTH_MULT
    crop_h = h * FACE_TO_SHOULDER_HEIGHT_MULT
    side = max(crop_w, crop_h)

    crop_cx = face_cx
    crop_cy = face_cy + h * FACE_CENTER_Y_SHIFT_MULT

    x1 = int(crop_cx - side / 2)
    y1 = int(crop_cy - side / 2 - h * FACE_TOP_PADDING_MULT)
    x2 = int(x1 + side)
    y2 = int(y1 + side)

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift
        x2 = frame_w
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift
        y2 = frame_h

    x1 = _clamp(x1, 0, frame_w - 1)
    y1 = _clamp(y1, 0, frame_h - 1)
    x2 = _clamp(x2, x1 + 1, frame_w)
    y2 = _clamp(y2, y1 + 1, frame_h)

    final_side = min(x2 - x1, y2 - y1)
    x2 = x1 + final_side
    y2 = y1 + final_side

    return x1, y1, x2, y2


def _load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade face detector.")
    return detector


def capture_portrait(
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    *,
    camera_index: int = CAMERA_INDEX,
    auto_capture_seconds: float = AUTO_CAPTURE_SECONDS,
    capture_id: Optional[str] = None,
    show_window: bool = True,
) -> Optional[Path]:
    """Stream the webcam, detect a face+shoulders bbox, and save a square crop.

    Args:
        output_dir: Directory to save the crop into. Created if missing.
        camera_index: OpenCV camera index.
        auto_capture_seconds: Seconds a face must stay visible before auto-saving.
        capture_id: Override the saved file's id. Defaults to a new datetime id.
        show_window: Show a preview window (requires OpenCV with GUI support).

    Returns:
        Path to the saved crop, or ``None`` if the user quit without saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_detector = _load_face_detector()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    bbox_first_seen_time: Optional[float] = None
    latest_crop = None
    saved_path: Optional[Path] = None

    if show_window:
        print("Webcam opened.")
        print("Press ENTER to save crop immediately.")
        print("Press Q or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            display = frame.copy() if show_window else None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            if len(faces) > 0:
                largest_face = max(faces, key=lambda box: box[2] * box[3])
                x1, y1, x2, y2 = _make_square_face_shoulder_bbox(
                    largest_face, frame.shape
                )
                latest_crop = frame[y1:y2, x1:x2].copy()

                if bbox_first_seen_time is None:
                    bbox_first_seen_time = time.time()
                elapsed = time.time() - bbox_first_seen_time
                remaining = max(0.0, auto_capture_seconds - elapsed)

                if show_window and display is not None:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        display,
                        f"Closest face+shoulders | Auto capture in {remaining:.1f}s",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if elapsed >= auto_capture_seconds:
                    saved_path = _save_crop(latest_crop, output_dir, capture_id)
                    break
            else:
                bbox_first_seen_time = None
                latest_crop = None
                if show_window and display is not None:
                    cv2.putText(
                        display,
                        "No face detected",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if show_window and display is not None:
                cv2.imshow("Webcam Face + Shoulder Crop", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (13, 10):
                    if latest_crop is not None:
                        saved_path = _save_crop(latest_crop, output_dir, capture_id)
                        break
                    print("No bounding box available yet.")
                if key == ord("q") or key == 27:
                    print("Quit without saving.")
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    return saved_path


def _save_crop(
    crop, output_dir: Path, capture_id: Optional[str]
) -> Path:
    cid = capture_id or make_capture_id()
    path = output_dir / f"{cid}.jpg"
    cv2.imwrite(str(path), crop)
    print(f"Saved crop to: {path}")
    return path


def main() -> None:
    capture_portrait()


if __name__ == "__main__":
    main()
