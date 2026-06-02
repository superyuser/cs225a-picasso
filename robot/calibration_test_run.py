"""Move through calibrated canvas corners as a position-only test run.

Sequence:
    current position -> INIT -> TL -> TR -> BR -> BL -> INIT

The corner positions are read from ``robot/canvas_calibration.json`` so this
can be run immediately after recalibrating the canvas values.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import redis

from config import (
    DT,
    DWELL_AT_INIT_S,
    INIT_POS,
    POS_TOL_M,
    config_file_for_this_example,
    controller_to_use,
)
from redis_io import decode_redis_value, position_error, read_np, redis_keys, send_position


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_JSON = SCRIPT_DIR / "canvas_calibration.json"
CORNER_ORDER = ("TL", "TR", "BR", "BL")


def load_corner_targets(calibration_json_path: str | Path) -> dict[str, np.ndarray]:
    calibration_json_path = Path(calibration_json_path)
    with calibration_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_corners = payload.get("raw_corners_xyz")
    if not isinstance(raw_corners, dict):
        raise ValueError(f"{calibration_json_path} missing raw_corners_xyz object.")

    targets: dict[str, np.ndarray] = {}
    for name in CORNER_ORDER:
        value = raw_corners.get(name)
        arr = np.array(value, dtype=float)
        if arr.shape != (3,):
            raise ValueError(
                f"{calibration_json_path} raw_corners_xyz.{name} must be [x, y, z]."
            )
        targets[name] = arr

    return targets


def ensure_robot_ready(redis_client: redis.Redis) -> bool:
    config_raw = redis_client.get(redis_keys.config_file_name)
    if config_raw is None:
        print("Could not read config file name from Redis.")
        print("Missing key:", redis_keys.config_file_name)
        return False

    config_file_name = decode_redis_value(config_raw)
    if config_file_name != config_file_for_this_example:
        print("This script is meant to be used with config file:", config_file_for_this_example)
        print("Current config file:", config_file_name)
        return False

    while True:
        active_raw = redis_client.get(redis_keys.active_controller)
        active_controller = decode_redis_value(active_raw) if active_raw is not None else None
        if active_controller == controller_to_use:
            break
        redis_client.set(redis_keys.active_controller, controller_to_use)
        time.sleep(0.05)

    print("Using controller:", controller_to_use)
    return True


def run_calibration_test(
    calibration_json_path: str | Path = DEFAULT_CALIBRATION_JSON,
    *,
    hold_done: bool = True,
    dwell_s: float = DWELL_AT_INIT_S,
) -> bool:
    corners = load_corner_targets(calibration_json_path)
    targets = [("INIT", INIT_POS), *((name, corners[name]) for name in CORNER_ORDER), ("INIT", INIT_POS)]

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return False

    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,),
    )

    print("Current position:", current_position)
    print("Calibration JSON:", calibration_json_path)
    print("Target sequence:")
    for label, target in targets:
        print(f"  {label}: {target}")

    target_index = 0
    label, target = targets[target_index]
    done = False
    send_position(redis_client, target)
    print(f"Starting calibration test run. Going to {label}.")

    loop_time = 0.0
    time.sleep(0.01)
    init_time = time.perf_counter_ns() * 1e-9

    try:
        while True:
            loop_time += DT
            time.sleep(max(0.0, loop_time - (time.perf_counter_ns() * 1e-9 - init_time)))

            current_position = read_np(
                redis_client,
                redis_keys.cartesian_task_current_position,
                (3,),
            )

            if done:
                send_position(redis_client, INIT_POS)
                print("DONE. Holding INIT.")
                time.sleep(0.25)
                continue

            label, target = targets[target_index]
            err = position_error(current_position, target)

            print(
                "CALIBRATION_TEST",
                target_index + 1,
                "/",
                len(targets),
                "| target:",
                label,
                "| pos_error:",
                round(err, 5),
            )

            send_position(redis_client, target)

            if err >= POS_TOL_M:
                continue

            print(f"Reached {label}.")
            time.sleep(dwell_s)
            target_index += 1

            if target_index >= len(targets):
                print("Finished calibration test sequence.")
                if not hold_done:
                    return True
                done = True
                print("DONE. Holding INIT.")
            else:
                label, target = targets[target_index]
                send_position(redis_client, target)
                print(f"Going to {label}.")

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        return False

    except Exception as exc:
        print("Exception occurred:")
        print(exc)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move through INIT and calibrated canvas corners as a robot test run."
    )
    parser.add_argument(
        "--calibration-json",
        default=str(DEFAULT_CALIBRATION_JSON),
        help=f"Canvas calibration JSON (default: {DEFAULT_CALIBRATION_JSON}).",
    )
    parser.add_argument(
        "--return-when-done",
        action="store_true",
        help="Exit after returning to INIT instead of holding INIT forever.",
    )
    parser.add_argument(
        "--dwell-s",
        type=float,
        default=DWELL_AT_INIT_S,
        help=f"Seconds to dwell at each target (default: {DWELL_AT_INIT_S}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok = run_calibration_test(
        args.calibration_json,
        hold_done=not args.return_when_done,
        dwell_s=args.dwell_s,
    )
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
