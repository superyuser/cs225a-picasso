"""Thin OpenAI image-edit wrapper used by the cv-tuning batch runner.

Exposes a single function ``cartoonize`` that takes a source image path
and a fully-formed prompt string, calls ``client.images.edit`` with the
preset model/size/quality, and returns the decoded PNG bytes.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def cartoonize(
    image_path,
    prompt_text: str,
    model: str = "gpt-image-1-mini",
    size: str = "1024x1024",
    quality: str = "low",
) -> bytes:
    """Call OpenAI's image edit endpoint and return the decoded PNG bytes."""
    client = get_client()
    with open(image_path, "rb") as image_file:
        kwargs = {
            "model": model,
            "image": image_file,
            "prompt": prompt_text,
            "size": size,
            "quality": quality,
        }
        try:
            result = client.images.edit(**kwargs)
        except TypeError as exc:
            if "quality" not in str(exc):
                raise
            image_file.seek(0)
            kwargs.pop("quality", None)
            result = client.images.edit(**kwargs)

    return base64.b64decode(result.data[0].b64_json)
