"""Live-stream from a USB webcam, detect a face, snap a photo after 3 s.

Flow:
    1. Open the USB camera (auto-probes if --index is wrong).
    2. Stream live preview with face detection overlay.
    3. As soon as at least one face is detected, lock onto the CLOSEST face
       (largest bounding-box area = closest to camera) and start a 3-second
       countdown shown on-screen.
    4. When the countdown hits zero, snap the current frame and save it to
       ``demo-day/captures/<timestamp>.jpg``.
    5. Show the captured frame briefly, then exit (or loop with --loop).

If the face is lost during the countdown, the countdown resets.

Note: this file is named ``mediapipe.py`` for historical reasons; we use
OpenCV's bundled Haar cascade instead of the actual ``mediapipe`` package
to avoid a self-import collision (``import mediapipe`` from this file
would re-import itself).

Usage:
    python mediapipe.py                 # default USB cam at index 1
    python mediapipe.py --index 0       # built-in webcam
    python mediapipe.py --probe         # scan indices 0-5 and exit
    python mediapipe.py --countdown 5   # wait 5 s instead of 3
    python mediapipe.py --loop          # keep capturing instead of exiting
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_CAPTURES_DIR = DEMO_DAY_DIR / "captures"

DEFAULT_INDEX = 1
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_COUNTDOWN_SECONDS = 3.0
DEFAULT_MIN_FACE_AREA = 4000          # ignore faces this small (px^2) -- false positives
DEFAULT_LOCK_FRAMES = 3               # need N consecutive face frames before counting down
PROBE_RANGE = range(6)
PREVIEW_SECONDS_AFTER_CAPTURE = 1.5

# Square head+shoulders crop geometry. Matches computer-vision/start_stream.py
# so all downstream pipeline stages see a consistent framing.
FACE_TO_SHOULDER_WIDTH_MULT = 3.0
FACE_TO_SHOULDER_HEIGHT_MULT = 3.2
FACE_TOP_PADDING_MULT = 0.6
FACE_CENTER_Y_SHIFT_MULT = 0.9


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------
def preferred_backend() -> int:
    return cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY


def open_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index, preferred_backend())
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def probe_cameras() -> list[int]:
    found: list[int] = []
    backend = preferred_backend()
    for i in PROBE_RANGE:
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  index {i}: open, {w}x{h}")
            found.append(i)
        else:
            print(f"  index {i}: opened but no frame")
        cap.release()
    return found


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return detector


def detect_faces(detector: cv2.CascadeClassifier, frame, min_face_area: int):
    """Run Haar cascade and return list of (x, y, w, h) tuples."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(80, 80),
    )
    out = []
    for (x, y, w, h) in faces:
        if w * h >= min_face_area:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def closest_face(faces):
    """Return the face with the largest bbox area (= closest to camera)."""
    if not faces:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


# ---------------------------------------------------------------------------
# Head + shoulders crop
# ---------------------------------------------------------------------------
def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def head_shoulders_bbox(face_box, frame_shape):
    """Convert a face bbox into a square head+shoulders crop bbox.

    Same geometry as ``computer-vision/start_stream.py`` so the framing
    matches whatever the rest of the pipeline expects. Returns
    ``(x1, y1, x2, y2)`` clamped to the frame. The bbox is guaranteed to
    be square (within rounding) and to fit inside the frame.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_box

    face_cx = x + w / 2.0
    face_cy = y + h / 2.0

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
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > frame_w:
        shift = x2 - frame_w
        x1 -= shift; x2 = frame_w
    if y2 > frame_h:
        shift = y2 - frame_h
        y1 -= shift; y2 = frame_h

    x1 = _clamp(x1, 0, frame_w - 1)
    y1 = _clamp(y1, 0, frame_h - 1)
    x2 = _clamp(x2, x1 + 1, frame_w)
    y2 = _clamp(y2, y1 + 1, frame_h)

    final_side = min(x2 - x1, y2 - y1)
    x2 = x1 + final_side
    y2 = y1 + final_side

    return x1, y1, x2, y2


def crop_head_shoulders(frame, face_box):
    """Return a square head+shoulders crop of ``frame`` around ``face_box``."""
    x1, y1, x2, y2 = head_shoulders_bbox(face_box, frame.shape)
    return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def _put_text(frame, text, org, *, scale=0.7, color=(255, 255, 255),
              thickness=2, outline=True):
    if outline:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_status(frame, *, status_text, fps):
    h, w = frame.shape[:2]
    _put_text(frame, status_text, (12, 32), scale=0.8)
    _put_text(frame, f"{w}x{h}  {fps:5.1f} fps",
              (12, h - 14), scale=0.55, color=(220, 220, 220), thickness=1)


def draw_big_number(frame, n: int):
    h, w = frame.shape[:2]
    text = str(n)
    scale = 8.0
    thickness = 18
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    org = ((w - size[0]) // 2, (h + size[1]) // 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 8, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (60, 220, 60), thickness, cv2.LINE_AA)


def draw_face_boxes(frame, faces, selected):
    for box in faces:
        x, y, w, h = box
        if selected is not None and box == selected:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 3)
            _put_text(frame, "closest", (x, max(0, y - 8)),
                      scale=0.55, color=(0, 220, 0), thickness=2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 1)


def draw_crop_preview(frame, face_box):
    """Outline the head+shoulders crop rect on ``frame`` so the user can
    see what will actually be saved."""
    x1, y1, x2, y2 = head_shoulders_bbox(face_box, frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    _put_text(frame, "crop", (x1 + 6, y1 + 22),
              scale=0.55, color=(255, 255, 0), thickness=2)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def make_capture_name() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S") + ".jpg"


def save_capture(frame, captures_dir: Path) -> Path:
    captures_dir.mkdir(parents=True, exist_ok=True)
    out_path = captures_dir / make_capture_name()
    # If two captures happen in the same second, disambiguate.
    suffix = 1
    while out_path.exists():
        out_path = captures_dir / (out_path.stem + f"_{suffix}.jpg")
        suffix += 1
    cv2.imwrite(str(out_path), frame)
    return out_path


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
def run_capture(cap, *, window_title: str, captures_dir: Path,
                countdown_seconds: float, min_face_area: int,
                lock_frames: int, loop: bool) -> list[Path]:
    detector = load_face_detector()
    print("Streaming. Press 'q' or ESC in the video window to quit.")
    print(f"Saving captures to: {captures_dir}")

    state = "idle"               # "idle" -> "countdown" -> "captured"
    countdown_start = 0.0
    consecutive_face_frames = 0
    last_save_path: Path | None = None
    captured_frame = None
    captured_at = 0.0
    saved_paths: list[Path] = []

    frame_count = 0
    fps_window_start = time.perf_counter()
    fps_estimate = 0.0

    while True:
        now = time.perf_counter()

        if state == "captured":
            # Hold the snapped image on screen, no new camera reads.
            if now - captured_at < PREVIEW_SECONDS_AFTER_CAPTURE:
                preview = captured_frame.copy()
                _put_text(preview, f"Saved {last_save_path.name}",
                          (12, 40), scale=0.9, color=(0, 220, 0), thickness=2)
                cv2.imshow(window_title, preview)
                if (cv2.waitKey(15) & 0xFF) in (ord("q"), 27):
                    break
                continue
            else:
                if not loop:
                    break
                state = "idle"
                consecutive_face_frames = 0

        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame.")
            break

        clean = frame.copy()    # we save the un-annotated version
        frame_count += 1
        elapsed = now - fps_window_start
        if elapsed >= 0.5:
            fps_estimate = frame_count / elapsed
            frame_count = 0
            fps_window_start = now

        faces = detect_faces(detector, frame, min_face_area)
        target = closest_face(faces)
        draw_face_boxes(frame, faces, target)
        if target is not None:
            draw_crop_preview(frame, target)

        if target is None:
            consecutive_face_frames = 0
            if state == "countdown":
                state = "idle"
                draw_status(frame, status_text="Face lost. Looking...", fps=fps_estimate)
            else:
                draw_status(frame, status_text="Looking for face...", fps=fps_estimate)
        else:
            consecutive_face_frames += 1
            if state == "idle" and consecutive_face_frames >= lock_frames:
                state = "countdown"
                countdown_start = now
            if state == "countdown":
                remaining = countdown_seconds - (now - countdown_start)
                if remaining <= 0.0:
                    crop = crop_head_shoulders(clean, target)
                    last_save_path = save_capture(crop, captures_dir)
                    saved_paths.append(last_save_path)
                    h_c, w_c = crop.shape[:2]
                    print(f"Captured -> {last_save_path}  ({w_c}x{h_c} square crop)")
                    captured_frame = crop
                    captured_at = now
                    state = "captured"
                    continue
                draw_big_number(frame, int(remaining) + 1)
                draw_status(frame, status_text="Hold still...", fps=fps_estimate)
            else:
                draw_status(frame, status_text="Face detected. Hold still...",
                            fps=fps_estimate)

        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX,
                        help=f"Camera index (default: {DEFAULT_INDEX}).")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN_SECONDS,
                        help=f"Seconds after lock before capture (default: {DEFAULT_COUNTDOWN_SECONDS}).")
    parser.add_argument("--min-face-area", type=int, default=DEFAULT_MIN_FACE_AREA,
                        help=f"Reject faces smaller than this area in px^2 (default: {DEFAULT_MIN_FACE_AREA}).")
    parser.add_argument("--lock-frames", type=int, default=DEFAULT_LOCK_FRAMES,
                        help=f"Consecutive face frames needed before countdown (default: {DEFAULT_LOCK_FRAMES}).")
    parser.add_argument("--captures-dir", default=str(DEFAULT_CAPTURES_DIR),
                        help=f"Where to save snapped photos (default: {DEFAULT_CAPTURES_DIR}).")
    parser.add_argument("--loop", action="store_true",
                        help="Keep going after each capture instead of exiting.")
    parser.add_argument("--probe", action="store_true",
                        help="Probe camera indices 0..5 and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.probe:
        print("Probing camera indices...")
        found = probe_cameras()
        if not found:
            print("No cameras found.")
            return 1
        print(f"Found: {found}")
        return 0

    print(f"Opening camera index {args.index} "
          f"({args.width}x{args.height} @ {args.fps}fps) ...")
    cap = open_camera(args.index, args.width, args.height, args.fps)
    if cap is None:
        print(f"Could not open camera at index {args.index}. Probing alternatives...")
        found = probe_cameras()
        if not found:
            print("No working cameras found. Check that the USB camera is plugged in "
                  "and not in use by another app.")
            return 1
        alt = next((i for i in found if i != args.index), found[0])
        print(f"Falling back to camera index {alt}.")
        cap = open_camera(alt, args.width, args.height, args.fps)
        if cap is None:
            return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened: {actual_w}x{actual_h} @ {actual_fps:.1f}fps "
          f"(backend = {cap.getBackendName()})")

    try:
        saved = run_capture(
            cap,
            window_title=f"USB Camera (index {args.index})",
            captures_dir=Path(args.captures_dir),
            countdown_seconds=args.countdown,
            min_face_area=args.min_face_area,
            lock_frames=args.lock_frames,
            loop=args.loop,
        )
    except cv2.error as exc:
        if "not implemented" in str(exc).lower():
            print(
                "\nERROR: this OpenCV build has no GUI support "
                "(probably `opencv-python-headless`). Install the GUI version:\n"
                "    pip uninstall -y opencv-python-headless\n"
                "    pip install opencv-python\n"
            )
            return 2
        raise

    print(f"\nCaptured {len(saved)} photo(s).")
    for p in saved:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
