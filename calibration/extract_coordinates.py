"""Read X, Y, Z robot coordinates off a phone photo using OpenAI vision.

The phone photo is expected to show some visible display of the robot's
end-effector pose (e.g. teach pendant screen, controller readout). The
vision model OCRs/parses the three axis values and returns them as
``(x, y, z)`` floats in meters.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You read robot end-effector coordinates from a photograph. "
    "The image shows a teach pendant, controller screen, or similar "
    "display with numerical X, Y, Z position values in meters. "
    "Return ONLY a JSON object of the form "
    '{"x": <number>, "y": <number>, "z": <number>}. '
    "All three must be numbers in meters. Do not include units, prose, "
    "or any other keys. If you genuinely cannot read an axis, use null."
)

USER_PROMPT = (
    "Extract the X, Y, Z end-effector coordinates from this image. "
    "Reply with strict JSON only."
)


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or put it in a .env file."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is not installed. "
            "Install it with `pip install openai`."
        ) from exc

    _client = OpenAI(api_key=api_key)
    return _client


def _encode_image(image_path: Path) -> str:
    data = Path(image_path).read_bytes()
    suffix = Path(image_path).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else (suffix or "jpeg")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def extract_xyz_from_image(
    image_path,
    *,
    model: str = DEFAULT_MODEL,
) -> Tuple[float, float, float]:
    """Call the OpenAI vision model and return ``(x, y, z)`` floats in meters."""
    client = _get_client()
    image_url = _encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model returned non-JSON: {content!r}") from exc

    try:
        x = float(payload["x"])
        y = float(payload["y"])
        z = float(payload["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Model response missing or non-numeric x/y/z: {payload!r}"
        ) from exc

    return x, y, z


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a calibration photo.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    x, y, z = extract_xyz_from_image(args.image, model=args.model)
    print(json.dumps({"x": x, "y": y, "z": z}, indent=2))
