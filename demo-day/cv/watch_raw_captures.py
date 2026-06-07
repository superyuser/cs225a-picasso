"""Watch ``demo-day/raw-captures`` and segment new captures into ``demo-day/captures``.

For every new image in the raw-captures directory:
  1. Detect faces with OpenCV's Haar cascade.
  2. Pick the "target" person -- a weighted score of how centered the face is
     and how close the person is (bbox area). This handles the case where
     several people are in frame.
  3. Build a generous head + upper-body crop around the target face.
  4. Run background removal (rembg with the ``u2net_human_seg`` model) on the
     crop to produce a transparent PNG of just the target person.
  5. Save the BGRA PNG as ``demo-day/captures/<stem>.png``.

If ``rembg`` is not installed, the script still runs but saves a plain
(un-segmented) crop with an opaque alpha channel, and prints install
instructions.

Usage:
    python watch_raw_captures.py                 # watch indefinitely
    python watch_raw_captures.py --process-existing  # also process files
                                                     # already in raw-captures
    python watch_raw_captures.py --once          # process current files and exit
    python watch_raw_captures.py --input <jpg>   # one-shot on a single file

First run will download a ~170 MB ONNX model (rembg ``u2net_human_seg``).
Subsequent runs use the cached model.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_RAW_DIR = DEMO_DAY_DIR / "raw-captures"
DEFAULT_OUT_DIR = DEMO_DAY_DIR / "captures"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Face-detection params (same as the rest of the project).
FACE_SCALE_FACTOR = 1.15
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)
MIN_FACE_AREA_PX = 4000

# Target-person scoring weights.
SCORE_WEIGHT_CLOSENESS = 1.0   # larger face = closer
SCORE_WEIGHT_CENTERED = 1.0    # face near image center

# Square head + shoulders crop geometry. Matches computer-vision/start_stream.py
# and demo-day/cv/mediapipe.py so every downstream stage sees the same framing.
FACE_TO_SHOULDER_WIDTH_MULT = 3.0
FACE_TO_SHOULDER_HEIGHT_MULT = 3.2
FACE_TOP_PADDING_MULT = 0.6
FACE_CENTER_Y_SHIFT_MULT = 0.9

POLL_INTERVAL_S = 1.0
FILE_SETTLE_POLLS = 1  # how many consecutive identical sizes before processing


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def detect_faces(detector, image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE,
    )
    out = []
    for (x, y, w, h) in faces:
        if w * h >= MIN_FACE_AREA_PX:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def pick_target_face(faces, frame_shape):
    """Score each face by closeness + centeredness and return the best one."""
    if not faces:
        return None
    h, w = frame_shape[:2]
    image_cx, image_cy = w / 2.0, h / 2.0
    diag = math.hypot(w, h)
    image_area = float(w * h)

    def score(box):
        x, y, fw, fh = box
        fcx, fcy = x + fw / 2.0, y + fh / 2.0
        closeness = (fw * fh) / image_area              # 0..1
        dist = math.hypot(fcx - image_cx, fcy - image_cy) / (diag / 2.0)
        centered = max(0.0, 1.0 - dist)                  # 0..1
        return SCORE_WEIGHT_CLOSENESS * closeness + SCORE_WEIGHT_CENTERED * centered

    return max(faces, key=score)


# ---------------------------------------------------------------------------
# Crop geometry
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def head_shoulders_bbox(face_box, frame_shape):
    """Square head + shoulders crop bbox around ``face_box``.

    Same geometry as ``demo-day/cv/mediapipe.py`` so downstream stages see a
    consistent framing. Returns ``(x1, y1, x2, y2)`` clamped to the frame.
    The bbox is guaranteed to be square (within rounding) and to fit inside
    the frame -- if the face is near an edge, the bbox slides inward rather
    than clipping outside.
    """
    frame_h, frame_w = frame_shape[:2]
    x, y, fw, fh = face_box
    fcx, fcy = x + fw / 2.0, y + fh / 2.0

    crop_w = fw * FACE_TO_SHOULDER_WIDTH_MULT
    crop_h = fh * FACE_TO_SHOULDER_HEIGHT_MULT
    side = max(crop_w, crop_h)

    crop_cx = fcx
    crop_cy = fcy + fh * FACE_CENTER_Y_SHIFT_MULT

    x1 = int(crop_cx - side / 2)
    y1 = int(crop_cy - side / 2 - fh * FACE_TOP_PADDING_MULT)
    x2 = int(x1 + side)
    y2 = int(y1 + side)

    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > frame_w:
        x1 -= x2 - frame_w; x2 = frame_w
    if y2 > frame_h:
        y1 -= y2 - frame_h; y2 = frame_h

    x1 = _clamp(x1, 0, frame_w - 1)
    y1 = _clamp(y1, 0, frame_h - 1)
    x2 = _clamp(x2, x1 + 1, frame_w)
    y2 = _clamp(y2, y1 + 1, frame_h)

    final_side = min(x2 - x1, y2 - y1)
    x2 = x1 + final_side
    y2 = y1 + final_side
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------
class Segmenter:
    """Lazy-loaded rembg session. Falls back to opaque-alpha crop if missing."""

    def __init__(self):
        self._session = None
        self._available: Optional[bool] = None  # tri-state: None=untried

    def _ensure_session(self):
        if self._available is False:
            return None
        if self._session is not None:
            return self._session
        try:
            from rembg import new_session  # type: ignore
            self._session = new_session("u2net_human_seg")
            self._available = True
            print("[segmenter] rembg u2net_human_seg session ready.")
            return self._session
        except ImportError:
            print(
                "[segmenter] rembg not installed -- saving plain crops instead.\n"
                "             To enable background removal: pip install rembg onnxruntime"
            )
            self._available = False
            return None
        except Exception as exc:  # noqa: BLE001
            print(f"[segmenter] could not initialize rembg ({exc}); saving plain crops.")
            self._available = False
            return None

    def segment(self, image_bgr):
        """Return a BGRA numpy array.

        If rembg is available, run human segmentation. Otherwise return the
        crop with a fully opaque alpha channel so the pipeline still produces
        something usable.
        """
        session = self._ensure_session()
        if session is None:
            h, w = image_bgr.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=np.uint8)
            return np.concatenate([image_bgr, alpha], axis=2)

        from rembg import remove  # type: ignore
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = remove(rgb, session=session)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"rembg returned unexpected shape: {rgba.shape}")
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------
def process_image(
    src_path: Path,
    out_dir: Path,
    detector,
    segmenter: Segmenter,
) -> Optional[Path]:
    image_bgr = cv2.imread(str(src_path))
    if image_bgr is None:
        print(f"  could not read {src_path}; skipping.")
        return None

    faces = detect_faces(detector, image_bgr)
    if not faces:
        print(f"  no faces detected in {src_path.name}; segmenting the whole frame.")
        bgra = segmenter.segment(image_bgr)
    else:
        target = pick_target_face(faces, image_bgr.shape)
        x1, y1, x2, y2 = head_shoulders_bbox(target, image_bgr.shape)
        crop = image_bgr[y1:y2, x1:x2]
        print(
            f"  {len(faces)} face(s); target={target}; "
            f"head+shoulders crop=({x1},{y1})-({x2},{y2}) -> {crop.shape[1]}x{crop.shape[0]}"
        )
        bgra = segmenter.segment(crop)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_path.stem}.png"
    cv2.imwrite(str(out_path), bgra)
    return out_path


# ---------------------------------------------------------------------------
# Watcher loop
# ---------------------------------------------------------------------------
def _eligible_files(d: Path):
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def watch_loop(
    raw_dir: Path,
    out_dir: Path,
    *,
    skip_existing: bool,
    once: bool,
    poll_interval: float,
):
    detector = load_face_detector()
    segmenter = Segmenter()
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    existing = list(_eligible_files(raw_dir))
    # Files that existed at startup are treated as already-settled (no in-flight
    # writes), so the size-stability check only applies to files that appear
    # after the watcher is running.
    startup_files: set[str] = {p.name for p in existing}

    if skip_existing:
        seen = {p.name for p in existing}
        print(f"Ignoring {len(seen)} existing file(s) in {raw_dir} (--skip-existing).")
    else:
        already_done = [p for p in existing if (out_dir / f"{p.stem}.png").exists()]
        to_do = [p for p in existing if not (out_dir / f"{p.stem}.png").exists()]
        seen = {p.name for p in already_done}
        if already_done:
            print(
                f"Skipping {len(already_done)} existing file(s) that already have a "
                f"PNG in {out_dir}."
            )
        if to_do:
            print(f"Found {len(to_do)} existing file(s) in {raw_dir} to process.")
        if not existing:
            print(f"No existing files in {raw_dir}.")

    pending: dict[str, tuple[int, int]] = {}  # name -> (size, polls_unchanged)

    if once:
        print(f"One-shot scan of {raw_dir} ...")
    else:
        print(f"Watching {raw_dir}. Press Ctrl+C to stop.")
        print(f"Writing PNGs to {out_dir}.")

    while True:
        try:
            for p in _eligible_files(raw_dir):
                if p.name in seen:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size == 0:
                    continue

                # Files that existed at startup are already settled.
                # New files must show stable size across FILE_SETTLE_POLLS
                # consecutive polls before being processed.
                if p.name not in startup_files:
                    prev_size, polls = pending.get(p.name, (-1, 0))
                    if size == prev_size:
                        polls += 1
                    else:
                        polls = 0
                    pending[p.name] = (size, polls)
                    if polls < FILE_SETTLE_POLLS:
                        continue

                seen.add(p.name)
                pending.pop(p.name, None)
                print(f"\nNew capture: {p.name}")
                try:
                    out_path = process_image(p, out_dir, detector, segmenter)
                    if out_path is not None:
                        print(f"  -> {out_path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR while processing {p.name}: {exc}")

            if once:
                return
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help=f"Watch this directory (default: {DEFAULT_RAW_DIR}).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Write segmented PNGs here (default: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--input", default=None,
                        help="Process a single file and exit (skips watching).")
    parser.add_argument("--process-existing", action="store_true",
                        help="(deprecated; now the default) Process existing files on startup.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not process files already present in --raw-dir on startup. "
                             "By default, existing files are processed unless their matching "
                             "<stem>.png is already in --out-dir.")
    parser.add_argument("--once", action="store_true",
                        help="Process current files once and exit, don't keep watching.")
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_S,
                        help=f"Seconds between directory polls (default: {POLL_INTERVAL_S}).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.input is not None:
        detector = load_face_detector()
        segmenter = Segmenter()
        out_path = process_image(Path(args.input), out_dir, detector, segmenter)
        if out_path is None:
            return 1
        print(f"-> {out_path}")
        return 0

    watch_loop(
        raw_dir,
        out_dir,
        skip_existing=args.skip_existing,
        once=args.once,
        poll_interval=args.poll_interval,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
