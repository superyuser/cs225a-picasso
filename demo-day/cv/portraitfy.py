import argparse
import base64
import os
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = REPO_ROOT / ".env"

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DAY_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = DEMO_DAY_DIR / "portraits"


def _load_env() -> None:
    """Load OPENAI_API_KEY from the repo-root ``.env`` if dotenv is available.

    Falls back silently if ``python-dotenv`` isn't installed -- in that case
    the caller is expected to have ``OPENAI_API_KEY`` exported in the shell.
    """
    if load_dotenv is None:
        return
    if DOTENV_PATH.is_file():
        load_dotenv(DOTENV_PATH)
    else:
        load_dotenv()


_load_env()


LINE_ART_PROMPT = """
Transform the uploaded portrait into a cute, simplified black-and-white line-art portrait for robotic acrylic painting.

Preserve the person's recognizable identity, hairstyle, glasses if present, and main facial features, but stylize the portrait to be cuter, softer, and more charming. Slightly enlarge the eyes, soften the cheeks and jawline, simplify the nose and mouth, and make the expression gentle and friendly. Keep the likeness recognizable while making the result feel like an adorable clean avatar.

Composition requirements:
- front-facing head-and-shoulders portrait
- centered subject
- upright face
- no crop through the top of the head
- no background objects
- pure white background

Style requirements:
- clean black line art only
- white fill/background only
- no grayscale shading
- no color
- no hatching
- no stippling
- no realistic skin texture
- no painterly shading
- no filled black regions except tiny pupils if necessary
- no complex hair strand texture
- no tiny decorative details

Line-art requirements:
- use smooth, bold, continuous black outlines
- use sparse, simple, readable strokes
- simplify hair into large contour shapes instead of many strands
- glasses should be clearly outlined with simple closed curves
- eyes should be cute, simple, and slightly enlarged
- eyebrows should be simple single strokes or simple filled curves
- nose should be minimal, using only a few simple lines
- mouth should be a simple cute line or small simple curve
- shirt neckline should be simple and clear
- shoulders should be lightly indicated with minimal lines

Robot painting constraints:
- line work must be drawable by a robot brush/marker
- avoid overlapping strokes
- avoid doubled parallel outlines
- avoid extremely thin, tiny, or noisy details
- avoid dense texture
- keep all important lines separated enough to be vectorized
- make the portrait recognizable using as few strokes as possible

Very important:
The output should be a black line drawing on a pure white background. It should NOT be a colored portrait. It should NOT include skin-tone fill, hair fill, shirt fill, or background texture. The fill colors will be added later by a separate robot painting pipeline.

Overall goal:
Create a cute, simplified, exaggerated-but-recognizable portrait line drawing with clean black outlines, large expressive eyes, soft rounded features, and a pure white background.
""".strip()


def decode_b64_image(b64_data: str) -> bytes:
    """
    Handles both raw base64 strings and data URL style image strings.
    """
    if "," in b64_data and b64_data.strip().startswith("data:"):
        b64_data = b64_data.split(",", 1)[1]
    return base64.b64decode(b64_data)


def save_image_from_response(result, output_path: str):
    """
    Saves the first returned image from an OpenAI image response.
    This expects base64 image output.
    """
    if not result.data:
        raise RuntimeError("Image API returned no data.")

    image_b64 = result.data[0].b64_json
    if image_b64 is None:
        raise RuntimeError("Image API response did not include b64_json.")

    image_bytes = decode_b64_image(image_b64)

    with open(output_path, "wb") as f:
        f.write(image_bytes)


def cartoonize_to_line_art(
    input_path: str,
    output_path: str,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
):
    """
    Converts an input portrait into cute, simplified black-and-white line art.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in "
            f"{DOTENV_PATH} or export it in your shell. "
            "If dotenv isn't installed, run `pip install python-dotenv`."
        )
    client = OpenAI()

    input_path = str(input_path)
    output_path = str(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "rb") as image_file:
        result = client.images.edit(
            model=model,
            image=image_file,
            prompt=LINE_ART_PROMPT,
            size=size,
        )

    save_image_from_response(result, output_path)

    print("Done.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input portrait image.",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=(
            "Path to save output line-art image. "
            f"Defaults to {DEFAULT_OUTPUT_DIR}/<input_stem>_cartoon.png."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-image-1",
        type=str,
        help="OpenAI image model.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        type=str,
        help="Output image size, e.g. 1024x1024.",
    )

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        input_stem = Path(args.input).stem
        output_path = DEFAULT_OUTPUT_DIR / f"{input_stem}_cartoon.png"

    cartoonize_to_line_art(
        input_path=args.input,
        output_path=output_path,
        model=args.model,
        size=args.size,
    )


if __name__ == "__main__":
    main()