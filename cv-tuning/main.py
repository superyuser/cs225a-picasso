"""CV-tuning batch runner.

Reads a folder of input portraits and, for a chosen prompt:
  1. Sends each image through OpenAI's image-edit API (``generate.cartoonize``)
     using settings from ``openai-model-presets.json``. Results are written to
     ``generated-renders/<stem>__<prompt>.png`` and cached on disk.
  2. Runs the stroke-extraction pipeline (``pipeline.LineArtStrokePipeline``)
     with a per-prompt config from ``prompt_name_to_pipeline_mapping.json``
     plus any ad-hoc CLI ``--override key=value`` flags.
  3. Saves ONLY the layered strokes PNG to
     ``stroke-pngs/<stem>__<prompt>[<suffix>].png``.

Efficiency features designed for tuning loops
---------------------------------------------
- Generated renders are cached on disk and never regenerated unless
  ``--regenerate`` is passed -- so iterating on ``pipeline.py`` parameters
  costs zero OpenAI calls.
- ``--strokes-only`` skips the OpenAI step entirely and just re-runs the
  pipeline over the existing ``generated-renders/<stem>__<prompt>.png``
  files. This is the main knob for parameter tuning.
- ``--override key=value`` lets you sweep config values without editing JSON,
  e.g. ``--override threshold=200 --override simplify_eps=3.0``.
- ``--out-suffix _t200`` tags the stroke PNG filename so you can keep
  multiple parameter variants side-by-side for visual comparison.
- ``--image foo.jpg`` restricts the run to a single input for fast
  back-and-forth on one face.

Examples
--------
  # First time: stock cv-tuning/images/, then run the full thing
  python main.py --prompt simpsons

  # Tune pipeline params without burning OpenAI credits
  python main.py --prompt simpsons --strokes-only --override threshold=190
  python main.py --prompt simpsons --strokes-only --override simplify_eps=3.5 \\
                                   --out-suffix _eps3.5

  # Force fresh cartoonization
  python main.py --prompt simpsons --regenerate

  # Just one image, fast loop
  python main.py --prompt simple --image alice.jpg --strokes-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline import LineArtPipelineConfig, LineArtStrokePipeline


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "images"
GENERATED_DIR = SCRIPT_DIR / "generated-renders"
STROKES_DIR = SCRIPT_DIR / "stroke-pngs"
PROMPTS_PATH = SCRIPT_DIR / "prompts.json"
PROMPT_MAPPING_PATH = SCRIPT_DIR / "prompt_name_to_pipeline_mapping.json"
PRESETS_PATH = SCRIPT_DIR / "openai-model-presets.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CONFIG_FIELD_NAMES = {f.name for f in LineArtPipelineConfig.__dataclass_fields__.values()}


def _log(msg: str) -> None:
    print(f"[cv-tuning] {msg}", flush=True)


def load_json(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"Missing config file: {path}")
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def coerce_value(raw: str) -> Any:
    """Best-effort cast for CLI override strings: int -> float -> bool -> str."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--override expects key=value, got: {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if key not in CONFIG_FIELD_NAMES:
            raise ValueError(
                f"Unknown pipeline config field: {key!r}. "
                f"Valid fields: {sorted(CONFIG_FIELD_NAMES)}"
            )
        out[key] = coerce_value(value)
    return out


def build_config(prompt_name: str, mapping: Dict[str, Any],
                 cli_overrides: Dict[str, Any]) -> Tuple[LineArtPipelineConfig, Dict[str, Any]]:
    raw_overrides = mapping.get(prompt_name, {}) or {}
    overrides: Dict[str, Any] = {
        k: v for k, v in raw_overrides.items()
        if not k.startswith("_") and k in CONFIG_FIELD_NAMES
    }
    overrides.update(cli_overrides)
    overrides.setdefault("save_debug_images", False)
    overrides.setdefault("save_gif", False)
    overrides.setdefault("save_mp4", False)
    overrides.setdefault("verbose", False)
    return LineArtPipelineConfig(**overrides), overrides


def list_input_images(images_dir: Path, single: Optional[str]) -> List[Path]:
    if single:
        candidate = Path(single)
        if not candidate.is_absolute():
            candidate = images_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {candidate}")
        return [candidate]
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def generate_render(input_path: Path, prompt_text: str, presets: Dict[str, Any],
                    out_path: Path, *, force: bool) -> Tuple[Path, bool]:
    """Ensure ``out_path`` exists; return (path, did_call_api)."""
    if out_path.exists() and not force:
        return out_path, False
    from generate import cartoonize
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = cartoonize(str(input_path), prompt_text, **presets)
    out_path.write_bytes(image_bytes)
    return out_path, True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-run prompts + stroke pipeline over cv-tuning/images/.",
    )
    parser.add_argument("--prompt", help="Prompt name from prompts.json.")
    parser.add_argument("--image", help="Single image filename (under images/) to process.")
    parser.add_argument("--images-dir", default=str(IMAGES_DIR),
                        help="Override input directory (default: cv-tuning/images/).")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force re-cartoonize even if cached render exists.")
    parser.add_argument("--strokes-only", action="store_true",
                        help="Skip OpenAI; only rerun pipeline on existing generated renders.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to stroke-png filenames (e.g. '_t190').")
    parser.add_argument("--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Pipeline config override; repeatable.")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompt names and exit.")
    args = parser.parse_args()

    prompts = load_json(PROMPTS_PATH)
    if args.list_prompts:
        for name in sorted(prompts):
            print(name)
        return 0

    if not args.prompt:
        parser.error("--prompt is required (or use --list-prompts).")

    if args.prompt not in prompts:
        _log(f"Prompt '{args.prompt}' not found. Available: {sorted(prompts)}")
        return 2

    mapping = load_json(PROMPT_MAPPING_PATH, default={})
    presets = load_json(PRESETS_PATH, default={})
    cli_overrides = parse_overrides(args.override)
    cfg, applied_overrides = build_config(args.prompt, mapping, cli_overrides)

    pipeline = LineArtStrokePipeline(cfg)

    GENERATED_DIR.mkdir(exist_ok=True)
    STROKES_DIR.mkdir(exist_ok=True)

    images_dir = Path(args.images_dir).resolve()
    image_paths = list_input_images(images_dir, args.image)
    if not image_paths:
        _log(f"No images found in {images_dir}. Drop .jpg/.png files there and rerun.")
        return 1

    _log(f"prompt:       {args.prompt}")
    _log(f"images:       {len(image_paths)}  (from {images_dir})")
    _log(f"presets:      {presets}")
    _log(f"overrides:    {applied_overrides}")
    _log(f"strokes-only: {args.strokes_only}   regenerate: {args.regenerate}")
    _log("")

    prompt_text = prompts[args.prompt]
    api_calls = 0
    cached = 0
    pipeline_runs = 0
    pipeline_failures: List[Tuple[Path, str]] = []

    overall_start = time.perf_counter()
    for idx, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        gen_path = GENERATED_DIR / f"{stem}__{args.prompt}.png"
        strokes_path = STROKES_DIR / f"{stem}__{args.prompt}{args.out_suffix}.png"

        _log(f"[{idx}/{len(image_paths)}] {img_path.name}")

        if args.strokes_only:
            if not gen_path.exists():
                _log(f"    skip (no cached render at {gen_path.name}; rerun without --strokes-only)")
                continue
            cached += 1
            _log(f"    generate: cached -> generated-renders/{gen_path.name}")
        else:
            t0 = time.perf_counter()
            try:
                _, did_call = generate_render(
                    img_path, prompt_text, presets, gen_path, force=args.regenerate,
                )
            except Exception as exc:
                _log(f"    generate FAILED: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            if did_call:
                api_calls += 1
                _log(f"    generate: NEW    -> generated-renders/{gen_path.name}  ({elapsed:.2f}s)")
            else:
                cached += 1
                _log(f"    generate: cached -> generated-renders/{gen_path.name}")

        t0 = time.perf_counter()
        try:
            pipeline.run_strokes_only(str(gen_path), str(strokes_path))
            pipeline_runs += 1
            _log(f"    strokes:  stroke-pngs/{strokes_path.name}  ({time.perf_counter() - t0:.2f}s)")
        except Exception as exc:
            pipeline_failures.append((img_path, str(exc)))
            _log(f"    strokes FAILED: {exc}")

    total = time.perf_counter() - overall_start
    _log("")
    _log("=== Summary ===")
    _log(f"  total time:        {total:.2f}s")
    _log(f"  images processed:  {len(image_paths)}")
    _log(f"  openai calls:      {api_calls}")
    _log(f"  cached renders:    {cached}")
    _log(f"  pipeline runs:     {pipeline_runs}")
    if pipeline_failures:
        _log(f"  failures:          {len(pipeline_failures)}")
        for path, err in pipeline_failures:
            _log(f"    - {path.name}: {err}")
    _log(f"  stroke-pngs dir:   {STROKES_DIR}")

    return 0 if not pipeline_failures else 3


if __name__ == "__main__":
    sys.exit(main())
