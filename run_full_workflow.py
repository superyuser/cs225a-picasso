"""Run the full CV-to-robot workflow from one entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CV_MAIN = REPO_ROOT / "computer-vision" / "main.py"
CV_STROKE_JSONS_DIR = REPO_ROOT / "computer-vision" / "stroke-jsons"
ROBOT_BRIDGE = REPO_ROOT / "robot" / "cv_workflow_entrypoint.py"
ROBOT_METADATA = REPO_ROOT / "robot" / "cv_robot_metadata.json"
IMAGE_POPUP = REPO_ROOT / "image_popup.py"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TERMINAL_ROBOT_STATUSES = {"complete", "skipped", "failed"}


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return str(path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_cv_stroke_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("strokes"), list) and "canvas_mapping" not in payload


def existing_cv_stroke_jsons() -> set[Path]:
    if not CV_STROKE_JSONS_DIR.exists():
        return set()
    return {
        path.resolve()
        for path in CV_STROKE_JSONS_DIR.glob("*.json")
        if is_cv_stroke_json(path)
    }


def newest_new_cv_stroke_json(baseline: set[Path]) -> Path | None:
    candidates = [
        path
        for path in existing_cv_stroke_jsons()
        if path.resolve() not in baseline
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def load_robot_metadata() -> list[dict]:
    if not ROBOT_METADATA.exists():
        return []
    try:
        data = load_json(ROBOT_METADATA)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def robot_entry_for(source_path: Path) -> dict | None:
    source_key = relative(source_path)
    for entry in load_robot_metadata():
        if entry.get("source_strokes_json") == source_key:
            return entry
    return None


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def launch_image_popup(python: str, image_path: Path, title: str) -> None:
    if not image_path.exists():
        print(f"Preview image not found, skipping popup: {relative(image_path)}")
        return
    subprocess.Popen(
        [
            python,
            str(IMAGE_POPUP),
            str(image_path),
            "--title",
            title,
        ],
        cwd=REPO_ROOT,
    )


def launch_cv_preview_popups(python: str, source_path: Path) -> None:
    stem = source_path.stem
    art_render = REPO_ROOT / "computer-vision" / "art-renders" / f"{stem}.png"
    layered_strokes = (
        REPO_ROOT
        / "computer-vision"
        / "stroke-renders"
        / stem
        / "08_layered_strokes.png"
    )

    launch_image_popup(python, art_render, f"{stem} art render")
    launch_image_popup(python, layered_strokes, f"{stem} layered strokes")


def wait_for_robot_completion(
    source_path: Path,
    robot_proc: subprocess.Popen,
    timeout_seconds: float | None,
    poll_interval: float,
) -> dict:
    deadline = None if timeout_seconds is None else time.time() + timeout_seconds
    source_key = relative(source_path)
    print(f"Waiting for robot workflow metadata for {source_key}")

    while True:
        entry = robot_entry_for(source_path)
        if entry is not None:
            status = entry.get("robot_status")
            if status in TERMINAL_ROBOT_STATUSES:
                return entry

        return_code = robot_proc.poll()
        if return_code is not None:
            raise RuntimeError(f"Robot bridge exited early with status {return_code}.")

        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for robot workflow for {source_key}.")

        time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run computer-vision/main.py, then map and draw its stroke JSON with the robot."
    )
    parser.add_argument(
        "--python",
        default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)),
        help="Python interpreter for child scripts (default: repo .venv if present).",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run through mapping only; do not send commands to the robot.",
    )
    parser.add_argument(
        "--robot-timeout",
        type=float,
        default=0.0,
        help="Seconds to wait for robot completion after CV finishes. 0 means wait forever.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval for metadata checks and the robot bridge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = existing_cv_stroke_jsons()
    timeout = None if args.robot_timeout <= 0 else args.robot_timeout

    robot_cmd = [
        args.python,
        str(ROBOT_BRIDGE),
        "--poll-interval",
        str(args.poll_interval),
    ]
    if args.no_robot:
        robot_cmd.append("--no-robot")

    cv_cmd = [args.python, str(CV_MAIN)]

    robot_proc: subprocess.Popen | None = None
    try:
        print("Starting robot bridge watcher.")
        robot_proc = subprocess.Popen(robot_cmd, cwd=REPO_ROOT)

        print("Starting computer vision workflow.")
        cv_result = subprocess.run(cv_cmd, cwd=REPO_ROOT)
        if cv_result.returncode != 0:
            raise RuntimeError(f"Computer vision workflow exited with status {cv_result.returncode}.")

        source_path = newest_new_cv_stroke_json(baseline)
        if source_path is None:
            raise RuntimeError("Computer vision workflow finished, but no new stroke JSON was found.")

        print(f"Computer vision produced {relative(source_path)}")
        launch_cv_preview_popups(args.python, source_path)

        entry = wait_for_robot_completion(
            source_path,
            robot_proc,
            timeout,
            args.poll_interval,
        )

        status = entry.get("robot_status")
        mapped = entry.get("mapped_strokes_json")
        print(f"Robot workflow status: {status}")
        if mapped:
            print(f"Mapped strokes: {mapped}")
        if status == "failed":
            raise RuntimeError(f"Robot workflow failed for {relative(source_path)}.")

    except KeyboardInterrupt:
        print("Stopping full workflow.")
        raise
    finally:
        if robot_proc is not None:
            terminate_process(robot_proc)


if __name__ == "__main__":
    main()
