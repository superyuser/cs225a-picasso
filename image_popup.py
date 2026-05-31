"""Show an image in a persistent OpenCV window until the user closes it."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def show_image(path: str | Path, *, title: str | None = None) -> None:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    window_title = title or image_path.name
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, image)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), 27):
            break
        try:
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyWindow(window_title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show an image until its window is closed.")
    parser.add_argument("image_path")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_image(args.image_path, title=args.title)


if __name__ == "__main__":
    main()
