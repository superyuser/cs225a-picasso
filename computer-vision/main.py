"""Pipeline entry point.

Runs the webcam capture (``start_stream``) and watches two directories:

* ``captures/``    - new photos here are cartoonized via ``cartoonize`` and
                     saved to ``art-renders/`` with the same id.
* ``art-renders/`` - new renders here are passed to ``run_pipeline`` and
                     produce ``stroke-jsons/<id>.json`` plus
                     ``stroke-renders/<id>/``.

A ``metadata.json`` file records the chain capture -> render -> strokes JSON.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from cartoonize import cartoonize
from run_pipeline import (
    DEFAULT_JSONS_DIR as STROKE_JSONS_DIR,
    DEFAULT_RENDERS_DIR as STROKE_RENDERS_DIR,
    run_pipeline,
)
from start_stream import capture_portrait


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = SCRIPT_DIR / "captures"
RENDERS_DIR = SCRIPT_DIR / "art-renders"
METADATA_PATH = SCRIPT_DIR / "metadata.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
POLL_INTERVAL_SECONDS = 0.5
PIPELINE_WAIT_SECONDS = 180.0


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

    metadata_lock = threading.Lock()
    stop_event = threading.Event()

    captures_seen = _processed_capture_keys(_load_metadata())
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
        saved_path: Optional[Path] = capture_portrait(output_dir=CAPTURES_DIR)
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
