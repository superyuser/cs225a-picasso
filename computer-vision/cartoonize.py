"""Cartoonize a portrait photo using OpenAI's image edit API.

Can be used as a module:

    from cartoonize import cartoonize
    output_path = cartoonize("input.jpg", "output.png")

Or invoked directly from the command line:

    python cartoonize.py input.jpg output.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


def _log(msg: str) -> None:
    print(f"[cartoonize] {msg}", flush=True)


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"

DEFAULT_PROMPT_KEY = "realistic"
DEFAULT_MODEL = "gpt-image-1-mini"
DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "low"


@lru_cache(maxsize=1)
def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, str]:
    """Load and cache the prompts mapping from ``prompts.json``."""
    _log(f"loading prompts from {path}")
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    _log(f"loaded {len(prompts)} prompt(s): {sorted(prompts)}")
    return prompts


def get_prompt(key: str = DEFAULT_PROMPT_KEY) -> str:
    """Return a named prompt from ``prompts.json``."""
    prompts = load_prompts()
    if key not in prompts:
        available = ", ".join(sorted(prompts)) or "(none)"
        raise KeyError(f"Prompt '{key}' not found. Available: {available}")
    return prompts[key]


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )
    _log("OpenAI client initialized")
    return OpenAI(api_key=api_key)


def cartoonize(
    input_path: str | os.PathLike[str],
    output_path: Optional[str | os.PathLike[str]] = None,
    *,
    prompt: Optional[str] = None,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    quality: str = DEFAULT_QUALITY,
) -> Path:
    """Cartoonize ``input_path`` and write the result to ``output_path``.

    Args:
        input_path: Path to the source portrait image.
        output_path: Where to save the cartoonized PNG. Defaults to
            ``<input_stem>_cartoon.png`` next to the input.
        prompt: Explicit prompt text. Overrides ``prompt_key`` when provided.
        prompt_key: Key to look up in ``prompts.json``.
        model: OpenAI image-edit model name.
        size: Output image size string, e.g. ``"1024x1024"``.
        quality: Image quality setting, e.g. ``"low"`` or ``"high"``.

    Returns:
        The path to the saved cartoonized image.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cartoon.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.perf_counter()
    in_size_kb = input_path.stat().st_size / 1024.0
    _log(f"input:  {input_path} ({in_size_kb:.1f} KB)")
    _log(f"output: {output_path}")
    _log(f"model={model} size={size} quality={quality} prompt_key={prompt_key}")

    prompt_text = prompt if prompt is not None else get_prompt(prompt_key)
    _log(f"prompt length: {len(prompt_text)} chars")

    client = _get_client()
    with input_path.open("rb") as image_file:
        kwargs = {
            "model": model,
            "image": image_file,
            "prompt": prompt_text,
            "size": size,
            "quality": quality,
        }
        _log("calling OpenAI images.edit ...")
        api_start = time.perf_counter()
        try:
            result = client.images.edit(**kwargs)
        except TypeError as exc:
            if "quality" not in str(exc):
                raise
            _log(
                "installed openai SDK doesn't support 'quality' on images.edit; "
                "retrying without it. Run `pip install --upgrade openai` to enable it."
            )
            image_file.seek(0)
            kwargs.pop("quality", None)
            result = client.images.edit(**kwargs)
        api_elapsed = time.perf_counter() - api_start
        _log(f"API responded in {api_elapsed:.2f}s")

    _log("decoding base64 response ...")
    image_bytes = base64.b64decode(result.data[0].b64_json)
    output_path.write_bytes(image_bytes)
    out_size_kb = output_path.stat().st_size / 1024.0
    total_elapsed = time.perf_counter() - pipeline_start
    _log(f"wrote {output_path} ({out_size_kb:.1f} KB) in {total_elapsed:.2f}s total")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cartoonize a portrait image.")
    parser.add_argument("input", help="Path to the input portrait image.")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to write the cartoonized PNG (default: <input>_cartoon.png).",
    )
    parser.add_argument("--prompt-key", default=DEFAULT_PROMPT_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--quality", default=DEFAULT_QUALITY)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved = cartoonize(
        args.input,
        args.output,
        prompt_key=args.prompt_key,
        model=args.model,
        size=args.size,
        quality=args.quality,
    )
    print(f"Saved to {saved}")


if __name__ == "__main__":
    main()
