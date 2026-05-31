"""Pipeline entry point.

Waits for the phone camera stream and watches two directories:

* ``captures/``    - new photos here are cartoonized via ``cartoonize`` and
                     saved to ``art-renders/`` with the same id.
* ``art-renders/`` - new renders here are passed to ``run_pipeline`` and
                     produce ``stroke-jsons/<id>.json`` plus
                     ``stroke-renders/<id>/``.

A ``metadata.json`` file records the chain capture -> render -> strokes JSON.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from cartoonize import cartoonize
from run_pipeline import (
    DEFAULT_JSONS_DIR as STROKE_JSONS_DIR,
    DEFAULT_RENDERS_DIR as STROKE_RENDERS_DIR,
    run_pipeline,
)
from start_stream import _load_face_detector, _make_square_face_shoulder_bbox
from phone_camera_server import DEFAULT_PORT as PHONE_SERVER_PORT
from phone_camera_server import serve as serve_phone_camera


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
PHONE_LATEST_FRAME_PATH = CAPTURES_DIR / "phone_latest.jpg"
RENDERS_DIR = SCRIPT_DIR / "art-renders"
METADATA_PATH = SCRIPT_DIR / "metadata.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
POLL_INTERVAL_SECONDS = 0.5
PIPELINE_WAIT_SECONDS = 180.0
PHONE_FRAME_POLL_SECONDS = 0.05
PHONE_WINDOW_NAME = "Phone Stream - Detected Crop"
PHONE_CROP_WINDOW_NAME = "Phone Crop Preview"
PHONE_SERVER_HOST = "0.0.0.0"


def _load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"Warning: {METADATA_PATH.name} is not valid JSON; starting fresh.")
        return []


def _save_metadata(entries: list[dict]) -> None:
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(SCRIPT_DIR).as_posix())
    except ValueError:
        return str(path.as_posix())


def _processed_capture_keys(entries: list[dict]) -> set[str]:
    return {entry["capture"] for entry in entries if "capture" in entry}


def _existing_image_keys(target_dir: Path) -> set[str]:
    if not target_dir.exists():
        return set()
    return {
        _relative(path)
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _processed_render_keys() -> set[str]:
    """Renders that already have a matching ``stroke-jsons/<stem>.json``."""
    if not STROKE_JSONS_DIR.exists():
        return set()
    processed_stems = {f.stem for f in STROKE_JSONS_DIR.glob("*.json")}
    keys: set[str] = set()
    for stem in processed_stems:
        for ext in IMAGE_EXTENSIONS:
            keys.add(_relative(RENDERS_DIR / f"{stem}{ext}"))
    return keys


def _phone_server_is_running(port: int = PHONE_SERVER_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_phone_server_if_needed() -> Optional[threading.Thread]:
    if _phone_server_is_running():
        print(f"Phone camera server already running on port {PHONE_SERVER_PORT}.")
        return None

    def run_server() -> None:
        try:
            serve_phone_camera(PHONE_SERVER_HOST, PHONE_SERVER_PORT, use_https=True)
        except OSError as exc:
            print(f"Phone camera server could not start: {exc}")

    thread = threading.Thread(
        target=run_server,
        daemon=True,
        name="phone-camera-server",
    )
    thread.start()
    time.sleep(0.5)
    return thread


def _upsert_metadata_by_id(
    entry_id: str,
    fields: dict,
    metadata_lock: threading.Lock,
) -> None:
    with metadata_lock:
        entries = _load_metadata()
        for entry in entries:
            if entry.get("id") == entry_id:
                entry.update(fields)
                break
        else:
            entries.append({"id": entry_id, **fields})
        _save_metadata(entries)


def process_capture(capture_path: Path, metadata_lock: threading.Lock) -> None:
    """Cartoonize ``capture_path`` and update the metadata file."""
    capture_id = capture_path.stem
    render_path = RENDERS_DIR / f"{capture_id}.png"
    print(f"[{capture_id}] cartoonizing {_relative(capture_path)} ...")
    try:
        cartoonize(capture_path, render_path)
    except Exception as exc:
        print(f"[{capture_id}] cartoonize failed: {exc}")
        return

    _upsert_metadata_by_id(
        capture_id,
        {
            "capture": _relative(capture_path),
            "render": _relative(render_path),
            "rendered_at": datetime.now().isoformat(timespec="seconds"),
        },
        metadata_lock,
    )
    print(f"[{capture_id}] saved render to {_relative(render_path)}")


def process_render(render_path: Path, metadata_lock: threading.Lock) -> None:
    """Run the stroke pipeline on ``render_path`` and update metadata."""
    stem = render_path.stem
    print(f"[{stem}] running stroke pipeline on {_relative(render_path)} ...")
    try:
        result = run_pipeline(render_path)
    except Exception as exc:
        print(f"[{stem}] run_pipeline failed: {exc}")
        return

    json_path = Path(result["json_path"])
    render_out_dir = Path(result["render_dir"])

    _upsert_metadata_by_id(
        stem,
        {
            "render": _relative(render_path),
            "strokes_json": _relative(json_path),
            "stroke_renders_dir": _relative(render_out_dir),
            "strokes_at": datetime.now().isoformat(timespec="seconds"),
        },
        metadata_lock,
    )
    print(f"[{stem}] saved strokes to {_relative(json_path)}")


def _is_stable_image(path: Path, sizes: dict[Path, int]) -> bool:
    """Return True once the file's size is stable across two polls."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        sizes.pop(path, None)
        return False
    if size <= 0:
        sizes[path] = size
        return False
    previous = sizes.get(path)
    sizes[path] = size
    return previous == size


def _read_image_when_complete(path: Path):
    try:
        image_bytes = path.read_bytes()
    except FileNotFoundError:
        return None
    if not image_bytes:
        return None

    data = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _window_was_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _newest_stable_capture(
    output_dir: Path,
    already_seen: set[str],
    pending_sizes: dict[Path, int],
) -> Optional[Path]:
    if not output_dir.exists():
        return None

    newest: Optional[Path] = None
    for path in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime_ns):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path == PHONE_LATEST_FRAME_PATH:
            continue
        key = _relative(path)
        if key in already_seen:
            continue
        if not _is_stable_image(path, pending_sizes):
            continue
        newest = path

    return newest


def capture_from_phone_stream(
    output_dir: Path = CAPTURES_DIR,
    latest_frame_path: Path = PHONE_LATEST_FRAME_PATH,
    show_windows: bool = True,
) -> Optional[Path]:
    """Preview the mobile live feed until the phone UI saves a capture."""
    face_detector = _load_face_detector()

    print(f"Waiting for mobile stream at {_relative(latest_frame_path)} ...")
    print("Open phone_camera_server.py on your phone, start the camera, then press Save Capture.")

    try:
        last_mtime_ns: Optional[int] = latest_frame_path.stat().st_mtime_ns
    except FileNotFoundError:
        last_mtime_ns = None
    captures_seen = _existing_image_keys(output_dir)
    captures_seen.add(_relative(latest_frame_path))
    pending_capture_sizes: dict[Path, int] = {}
    saved_path: Optional[Path] = None
    preview_window_shown = False

    while True:
        new_capture = _newest_stable_capture(
            output_dir,
            captures_seen,
            pending_capture_sizes,
        )
        if new_capture is not None:
            captures_seen.add(_relative(new_capture))
            pending_capture_sizes.pop(new_capture, None)
            if saved_path is None:
                saved_path = new_capture
                print(f"Using phone capture {_relative(saved_path)}")
                if not show_windows:
                    return saved_path

        try:
            stat = latest_frame_path.stat()
        except FileNotFoundError:
            if saved_path is not None and not preview_window_shown:
                return saved_path
            if show_windows:
                if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                    return saved_path
                if saved_path is not None and _window_was_closed(PHONE_WINDOW_NAME):
                    return saved_path
            else:
                time.sleep(PHONE_FRAME_POLL_SECONDS)
            continue

        if stat.st_mtime_ns == last_mtime_ns:
            if show_windows:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return saved_path
                if saved_path is not None and _window_was_closed(PHONE_WINDOW_NAME):
                    return saved_path
            else:
                time.sleep(PHONE_FRAME_POLL_SECONDS)
            continue
        last_mtime_ns = stat.st_mtime_ns

        frame = _read_image_when_complete(latest_frame_path)
        if frame is None:
            time.sleep(PHONE_FRAME_POLL_SECONDS)
            continue

        display = frame.copy()
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
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if show_windows:
                cv2.imshow(PHONE_CROP_WINDOW_NAME, latest_crop)
        else:
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

        label = (
            f"Captured {_relative(saved_path)}"
            if saved_path is not None
            else "Press Save Capture on phone"
        )
        cv2.putText(
            display,
            label,
            (30, max(90, display.shape[0] - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if show_windows:
            cv2.imshow(PHONE_WINDOW_NAME, display)
            preview_window_shown = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return saved_path
            if saved_path is not None and _window_was_closed(PHONE_WINDOW_NAME):
                return saved_path


def watch_directory(
    target_dir: Path,
    on_new_file: Callable[[Path], None],
    stop_event: threading.Event,
    already_seen: set[str],
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    """Poll ``target_dir`` and call ``on_new_file`` for each new stable image."""
    pending_sizes: dict[Path, int] = {}
    while not stop_event.is_set():
        if target_dir.exists():
            for path in sorted(target_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                key = _relative(path)
                if key in already_seen:
                    continue
                if not _is_stable_image(path, pending_sizes):
                    continue
                already_seen.add(key)
                pending_sizes.pop(path, None)
                try:
                    on_new_file(path)
                except Exception as exc:
                    print(f"Error processing {path.name}: {exc}")
        stop_event.wait(poll_interval)


def main() -> None:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    STROKE_JSONS_DIR.mkdir(parents=True, exist_ok=True)
    STROKE_RENDERS_DIR.mkdir(parents=True, exist_ok=True)

    start_phone_server_if_needed()

    metadata_lock = threading.Lock()
    stop_event = threading.Event()

    captures_seen = _processed_capture_keys(_load_metadata())
    captures_seen.update(_existing_image_keys(CAPTURES_DIR))
    captures_seen.add(_relative(PHONE_LATEST_FRAME_PATH))
    renders_seen = _processed_render_keys()

    def on_new_capture(path: Path) -> None:
        process_capture(path, metadata_lock)

    def on_new_render(path: Path) -> None:
        process_render(path, metadata_lock)

    capture_watcher = threading.Thread(
        target=watch_directory,
        args=(CAPTURES_DIR, on_new_capture, stop_event, captures_seen),
        daemon=True,
        name="capture-watcher",
    )
    render_watcher = threading.Thread(
        target=watch_directory,
        args=(RENDERS_DIR, on_new_render, stop_event, renders_seen),
        daemon=True,
        name="render-watcher",
    )
    capture_watcher.start()
    render_watcher.start()

    print(f"Watching {_relative(CAPTURES_DIR)} -> cartoonize")
    print(f"Watching {_relative(RENDERS_DIR)} -> run_pipeline")

    try:
        saved_path: Optional[Path] = capture_from_phone_stream(output_dir=CAPTURES_DIR)
        if saved_path is None:
            print("No capture saved; exiting.")
            return

        target_id = saved_path.stem
        expected_json = STROKE_JSONS_DIR / f"{target_id}.json"
        print(f"Waiting for full pipeline output: {_relative(expected_json)}")

        deadline = time.time() + PIPELINE_WAIT_SECONDS
        while time.time() < deadline:
            if expected_json.exists():
                print(f"Pipeline complete for {target_id}.")
                break
            time.sleep(0.5)
        else:
            print("Timed out waiting for pipeline to finish.")
    finally:
        stop_event.set()
        capture_watcher.join(timeout=2.0)
        render_watcher.join(timeout=2.0)


if __name__ == "__main__":
    main()
