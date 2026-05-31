import sys
import time
import argparse

import redis

from config import (
    DEFAULT_JSON_PATH,
    DT,
    DWELL_AFTER_RETRACT_S,
    DWELL_AT_INIT_S,
    DWELL_BEFORE_CONTACT_S,
    INIT_POS,
    POS_TOL_M,
    RETRACT_VEC,
    config_file_for_this_example,
    controller_to_use,
)
from path_builder import build_drawing_path
from redis_io import decode_redis_value, position_error, read_np, redis_keys, send_position
from state import State
from strokes import load_strokes


def draw_strokes(json_path=DEFAULT_JSON_PATH, hold_done=True, canvas_offset_vec=None):
    print("Loading stroke JSON:", json_path)

    strokes, metadata = load_strokes(json_path, offset_vec=canvas_offset_vec)

    print("Loaded strokes:", len(strokes))
    if "num_strokes" in metadata:
        print("JSON num_strokes:", metadata["num_strokes"])

    total_points = sum(len(s["points"]) for s in strokes)
    print("Total raw stroke points:", total_points)

    redis_client = redis.Redis()

    # Check config file.
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

    # Set active controller.
    while redis_client.get(redis_keys.active_controller).decode("utf-8") != controller_to_use:
        redis_client.set(redis_keys.active_controller, controller_to_use)
        time.sleep(0.05)

    print("Using controller:", controller_to_use)

    # Read current robot position.
    current_position = read_np(
        redis_client,
        redis_keys.cartesian_task_current_position,
        (3,)
    )

    print("Current position:", current_position)
    print("INIT_POS:", INIT_POS)
    print("RETRACT_VEC:", RETRACT_VEC)
    if canvas_offset_vec is not None:
        print("Canvas stroke offset vector (m):", canvas_offset_vec)

    # Build the drawing path after INIT. Startup travel to INIT is a single goal
    # so the Cartesian controller can handle the motion internally.
    path, labels = build_drawing_path(current_position, strokes)

    print("Total commanded waypoints:", len(path))

    if len(path) == 0:
        print("Path is empty. Exiting.")
        return False

    # Send INIT once, then keep holding it until the controller reaches it.
    path_index = 0
    send_position(redis_client, INIT_POS)

    state = State.GOING_INIT

    print("Starting position-only stroke drawing.")
    print("Going directly to INIT position.")

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
                (3,)
            )

            if state == State.GOING_INIT:
                err = position_error(current_position, INIT_POS)

                print(
                    "GOING_INIT",
                    "| pos_error:", round(err, 5)
                )

                send_position(redis_client, INIT_POS)

                if err < POS_TOL_M:
                    time.sleep(DWELL_AT_INIT_S)
                    send_position(redis_client, path[path_index])
                    state = State.EXECUTING_PATH
                    print("Reached INIT position.")
                    print("First label:", labels[path_index])

            elif state == State.EXECUTING_PATH:
                target = path[path_index]
                err = position_error(current_position, target)

                if path_index % 25 == 0:
                    print(
                        "EXECUTING_PATH",
                        path_index + 1, "/", len(path),
                        "| label:", labels[path_index],
                        "| pos_error:", round(err, 5)
                    )

                if err < POS_TOL_M:
                    # Optional micro-dwells at important contact transitions.
                    label = labels[path_index]

                    if label.endswith("_approach_contact"):
                        time.sleep(DWELL_BEFORE_CONTACT_S)
                    elif label.endswith("_retract"):
                        time.sleep(DWELL_AFTER_RETRACT_S)

                    path_index += 1

                    if path_index >= len(path):
                        send_position(redis_client, INIT_POS)
                        state = State.DONE
                        print("Finished all strokes. Holding INIT.")
                        if not hold_done:
                            return True
                    else:
                        send_position(redis_client, path[path_index])

            elif state == State.DONE:
                send_position(redis_client, INIT_POS)
                print("DONE. Holding INIT.")
                time.sleep(0.25)

    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
        return False

    except Exception as e:
        print("Exception occurred:")
        print(e)
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Draw mapped canvas stroke JSON with the robot.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help=f"Mapped canvas stroke JSON (default: {DEFAULT_JSON_PATH}).",
    )
    parser.add_argument(
        "--return-when-done",
        action="store_true",
        help="Exit after finishing all strokes instead of holding INIT forever.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ok = draw_strokes(args.json_path, hold_done=not args.return_when_done)
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
