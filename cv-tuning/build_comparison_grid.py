"""Build per-person comparison grids of every stroke PNG.

For each input portrait (identified by the filename stem before the first
``__`` separator), collects every stroke PNG that was generated from that
person -- across all prompts and parameter trial suffixes -- and lays them
out on a single grid for direct visual comparison.

Filename convention (from cv-tuning/main.py):
    <stem>__<prompt><suffix>.png

Examples:
    william__simpsons.png        -> stem=william, trial=simpsons
    william__simpsons_spur8.png  -> stem=william, trial=simpsons_spur8
    william__simple.png          -> stem=william, trial=simple

Output:
    cv-tuning/stroke-comparisons/<stem>__grid.png

Usage:
    python build_comparison_grid.py
    python build_comparison_grid.py --filter-prompt simpsons
    python build_comparison_grid.py --stems khatib william --cols 3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STROKES_DIR = SCRIPT_DIR / "stroke-pngs"
OUT_DIR = SCRIPT_DIR / "stroke-comparisons"
SEP = "__"


def parse_filename(name: str) -> Optional[Tuple[str, str]]:
    """Return ``(stem, trial_label)`` or ``None`` if the filename is off-convention."""
    if not name.lower().endswith(".png"):
        return None
    base = name[:-4]
    if SEP not in base:
        return None
    stem, trial = base.split(SEP, 1)
    if not stem or not trial:
        return None
    return stem, trial


def collect_files(
    strokes_dir: Path,
    filter_prompt: Optional[str],
) -> Dict[str, List[Tuple[str, Path]]]:
    """Group stroke PNGs by stem. Returns ``{stem: [(trial_label, path), ...]}``."""
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for p in sorted(strokes_dir.iterdir()):
        if not p.is_file():
            continue
        parsed = parse_filename(p.name)
        if parsed is None:
            continue
        stem, trial = parsed
        if filter_prompt and not (
            trial == filter_prompt or trial.startswith(filter_prompt + "_")
        ):
            continue
        groups.setdefault(stem, []).append((trial, p))
    for trials in groups.values():
        trials.sort(key=lambda t: (len(t[0]), t[0]))  # baseline (shorter) first, sweeps after
    return groups


def build_grid(
    stem: str,
    trials: List[Tuple[str, Path]],
    out_path: Path,
    cols: Optional[int],
    cell_size: float,
    dpi: int,
) -> None:
    n = len(trials)
    if n == 0:
        return
    if cols is None:
        cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (label, path) in zip(axes_flat, trials):
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{stem}   ({n} trial{'s' if n != 1 else ''})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strokes-dir", default=str(STROKES_DIR),
                        help="Directory to read stroke PNGs from.")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Where to write grid PNGs.")
    parser.add_argument("--stems", nargs="+",
                        help="Only build grids for these stems (e.g. --stems khatib william).")
    parser.add_argument("--filter-prompt",
                        help="Only include trials whose label starts with this prompt name "
                             "(e.g. 'simpsons' matches 'simpsons' and 'simpsons_spur8').")
    parser.add_argument("--cols", type=int,
                        help="Override grid column count (default: ceil(sqrt(n))).")
    parser.add_argument("--cell-size", type=float, default=4.0,
                        help="Per-cell size in inches (default: 4).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    strokes_dir = Path(args.strokes_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not strokes_dir.exists():
        print(f"No stroke-pngs directory found at {strokes_dir}")
        return 1

    groups = collect_files(strokes_dir, args.filter_prompt)
    if args.stems:
        wanted = set(args.stems)
        groups = {k: v for k, v in groups.items() if k in wanted}

    if not groups:
        print("No matching stroke PNGs found.")
        return 0

    print(f"Building grids for {len(groups)} stem(s) -> {out_dir}")
    for stem, trials in sorted(groups.items()):
        suffix = f"__{args.filter_prompt}_grid" if args.filter_prompt else "__grid"
        out_path = out_dir / f"{stem}{suffix}.png"
        print(f"  {stem}: {len(trials)} trial(s) -> {out_path.name}")
        build_grid(stem, trials, out_path, args.cols, args.cell_size, args.dpi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
