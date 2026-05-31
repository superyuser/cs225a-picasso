import json
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class RedisKeys:
    cartesian_task_goal_position: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::goal_position"
    cartesian_task_current_position: str = "opensai::controllers::Titania::cartesian_controller::cartesian_task::current_position"

    active_controller: str = "opensai::controllers::Titania::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


redis_keys = RedisKeys()


def decode_redis_value(val):
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def read_np(redis_client, key, expected_shape):
    """
    Read numpy array from Redis.

    expected_shape:
        (3,) for Cartesian position
    """
    while True:
        val = redis_client.get(key)

        if val is not None:
            try:
                val = decode_redis_value(val)
                arr = np.array(json.loads(val), dtype=float)

                if arr.shape == expected_shape:
                    return arr

            except Exception:
                pass

        time.sleep(0.01)


def send_position(redis_client, pos):
    redis_client.set(
        redis_keys.cartesian_task_goal_position,
        json.dumps(np.array(pos, dtype=float).tolist())
    )


def position_error(current_pos, goal_pos):
    return np.linalg.norm(np.array(goal_pos, dtype=float) - np.array(current_pos, dtype=float))
