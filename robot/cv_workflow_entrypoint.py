"""Bridge computer-vision stroke JSONs into the robot drawing workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = Path(__file__).resolve().parent
CV_STROKE_JSONS_DIR = REPO_ROOT / "computer-vision" / "stroke-jsons"
MAPPED_STROKES_DIR = ROBOT_DIR / "mapped-strokes"
CANVAS_SIMULATIONS_DIR = ROBOT_DIR / "canvas-simulations"
METADATA_PATH = ROBOT_DIR / "cv_robot_metadata.json"
IMAGE_POPUP = REPO_ROOT / "image_popup.py"
POLL_INTERVAL_SECONDS = 0.5
CANVAS_X_OFFSET_M = 0.03
CANVAS_OFFSET_VEC = [CANVAS_X_OFFSET_M, 0.0, 0.0]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOT_DIR))

from map_to_canvas_strokes import convert_strokes_to_canvas_plane, default_canvas_output_path
from animate_canvas_strokes import create_simulation
from main import draw_strokes


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path)


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"Warning: {relative(METADATA_PATH)} is invalid JSON; starting fresh.")
        return []


def save_metadata(entries: list[dict]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = METADATA_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    tmp_path.replace(METADATA_PATH)


def upsert_metadata(source_path: Path, fields: dict) -> dict:
    source_key = relative(source_path)
    entries = load_metadata()
    for entry in entries:
        if entry.get("source_strokes_json") == source_key:
            entry.update(fields)
            save_metadata(entries)
            return entry

    entry = {
        "source_strokes_json": source_key,
        **fields,
    }
    entries.append(entry)
    save_metadata(entries)
    return entry


def processed_sources() -> set[str]:
    return {
        entry["source_strokes_json"]
        for entry in load_metadata()
        if entry.get("source_strokes_json") and entry.get("robot_status") == "complete"
    }


def is_stable_file(path: Path, sizes: dict[Path, int]) -> bool:
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


def is_cv_stroke_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("strokes"), list) and "canvas_mapping" not in payload


def launch_image_popup(image_path: Path, title: str) -> None:
    subprocess.Popen(
        [
            sys.executable,
            str(IMAGE_POPUP),
            str(image_path),
            "--title",
            title,
        ],
        cwd=REPO_ROOT,
    )


def map_and_draw(source_path: Path, *, run_robot: bool = True) -> Path:
    MAPPED_STROKES_DIR.mkdir(parents=True, exist_ok=True)
    mapped_path = default_canvas_output_path(source_path, MAPPED_STROKES_DIR)

    print(f"[{source_path.stem}] mapping {relative(source_path)}")
    upsert_metadata(
        source_path,
        {
            "mapped_strokes_json": relative(mapped_path),
            "mapping_status": "running",
            "mapping_started_at": iso_now(),
            "robot_status": "pending",
        },
    )

    try:
        convert_strokes_to_canvas_plane(
            input_json_path=source_path,
            output_json_path=mapped_path,
        )
    except Exception as exc:
        upsert_metadata(
            source_path,
            {
                "mapping_status": "failed",
                "mapping_failed_at": iso_now(),
                "mapping_error": str(exc),
            },
        )
        raise

    upsert_metadata(
        source_path,
        {
            "mapped_strokes_json": relative(mapped_path),
            "mapping_status": "complete",
            "mapping_completed_at": iso_now(),
        },
    )

    simulation_dir = CANVAS_SIMULATIONS_DIR / source_path.stem
    print(f"[{source_path.stem}] building canvas animation preview")
    simulation = create_simulation(input_json=mapped_path, out_dir=simulation_dir)
    static_preview_path = Path(simulation["static_path"])
    launch_image_popup(
        static_preview_path,
        f"{source_path.stem} canvas world final",
    )
    upsert_metadata(
        source_path,
        {
            "canvas_simulation_static": relative(static_preview_path),
            "canvas_simulation_dir": relative(simulation_dir),
        },
    )

    if not run_robot:
        upsert_metadata(
            source_path,
            {
                "robot_status": "skipped",
                "robot_completed_at": iso_now(),
            },
        )
        return mapped_path

    print(f"[{source_path.stem}] drawing {relative(mapped_path)}")
    upsert_metadata(
        source_path,
        {
            "robot_status": "running",
            "robot_started_at": iso_now(),
            "robot_canvas_offset_xyz_m": CANVAS_OFFSET_VEC,
        },
    )

    ok = draw_strokes(
        str(mapped_path),
        hold_done=False,
        canvas_offset_vec=CANVAS_OFFSET_VEC,
    )
    if ok:
        upsert_metadata(
            source_path,
            {
                "robot_status": "complete",
                "robot_completed_at": iso_now(),
            },
        )
    else:
        upsert_metadata(
            source_path,
            {
                "robot_status": "failed",
                "robot_failed_at": iso_now(),
            },
        )
        raise RuntimeError(f"Robot drawing failed for {mapped_path}")

    return mapped_path


def watch_directory(
    directory: Path,
    on_stable_file: Callable[[Path], None],
    *,
    process_existing: bool = False,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pending_sizes: dict[Path, int] = {}
    seen = set() if process_existing else {relative(path) for path in directory.glob("*.json")}
    seen.update(processed_sources())

    print(f"Watching {relative(directory)} for CV stroke JSONs.")
    print(f"Mapped outputs: {relative(MAPPED_STROKES_DIR)}")
    print(f"Metadata: {relative(METADATA_PATH)}")

    while True:
        for path in sorted(directory.glob("*.json")):
            key = relative(path)
            if key in seen:
                continue
            if not is_stable_file(path, pending_sizes):
                continue
            if not is_cv_stroke_json(path):
                seen.add(key)
                pending_sizes.pop(path, None)
                continue

            try:
                on_stable_file(path)
                seen.add(key)
                pending_sizes.pop(path, None)
            except Exception as exc:
                print(f"Error processing {relative(path)}: {exc}")

        time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch CV stroke JSONs, map them to canvas coordinates, and run the robot."
    )
    parser.add_argument(
        "--stroke-jsons-dir",
        default=str(CV_STROKE_JSONS_DIR),
        help=f"Directory to watch (default: {CV_STROKE_JSONS_DIR}).",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process existing uncompleted JSONs instead of only future files.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Only map files and update metadata; do not call robot/main.py.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_SECONDS,
        help=f"Polling interval in seconds (default: {POLL_INTERVAL_SECONDS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.stroke_jsons_dir)

    def handle_file(path: Path) -> None:
        map_and_draw(path, run_robot=not args.no_robot)

    try:
        watch_directory(
            source_dir,
            handle_file,
            process_existing=args.process_existing,
            poll_interval=args.poll_interval,
        )
    except KeyboardInterrupt:
        print("Stopping CV robot workflow entrypoint.")


if __name__ == "__main__":
    main()
