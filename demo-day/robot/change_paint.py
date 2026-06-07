"""Run a single paint change from the robot home position.

Sequence (see helpers/paint_change.py):
    home -> INIT_POS -> TOOL_INIT -> rinse in water -> wipe napkin
    -> dip selected paint -> TOOL_INIT -> reverse path -> INIT_POS

Usage:
    python change_paint.py --paint 2
    python change_paint.py --paint p3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import redis
except ImportError:
    redis = None

from helpers.paint_change import normalize_paint_number, run_paint_change_sequence
from helpers.tool_station_coords import ensure_robot_ready

CONFIG_FILE_FOR_THIS_SCRIPT = "basket.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paint",
        required=True,
        help="Paint cup number to load (1, 2, 3, or p1/p2/p3).",
    )
    parser.add_argument(
        "--config-file",
        default=CONFIG_FILE_FOR_THIS_SCRIPT,
        help=f"Expected Sai config file (default: {CONFIG_FILE_FOR_THIS_SCRIPT}).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Maximum seconds to wait for each motion segment (default: 120).",
    )
    parser.add_argument(
        "--rebuild-tool-init-path",
        action="store_true",
        help=(
            "Rebuild and save demo-day/tool_init_joint_path.json instead of "
            "using the cached joint path."
        ),
    )
    parser.add_argument(
        "--no-cached-tool-init-path",
        action="store_true",
        help="Do not load the cached INIT_POS -> TOOL_INIT joint path JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if redis is None:
        print("`redis` package is not installed.")
        return 1

    try:
        paint_number = normalize_paint_number(args.paint)
    except ValueError as exc:
        print(exc)
        return 1

    redis_client = redis.Redis()
    if not ensure_robot_ready(redis_client):
        return 1

    print(f"Starting paint change for paint {paint_number}.")
    ok = run_paint_change_sequence(
        redis_client,
        paint_number,
        timeout_s=args.timeout_s,
        use_cached_tool_init_path=not args.no_cached_tool_init_path,
        rebuild_tool_init_path=args.rebuild_tool_init_path,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
