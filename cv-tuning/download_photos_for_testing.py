"""Download test portraits into cv-tuning/images/ by shelling out to curl.

Reads ``photos_to_download.csv`` (columns: ``name,url``) and saves each URL
to ``cv-tuning/images/<name><ext>`` where ``<ext>`` is inferred from the URL
(defaults to ``.jpg``). Existing files are skipped unless ``--force`` is set.

Usage:
    python download_photos_for_testing.py
    python download_photos_for_testing.py --force
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "photos_to_download.csv"
IMAGES_DIR = SCRIPT_DIR / "images"


def infer_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix else ".jpg"


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"name", "url"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"{csv_path.name} is missing required columns: {sorted(missing)}. "
                f"Expected header: name,url"
            )
        for row in reader:
            name = (row.get("name") or "").strip()
            url = (row.get("url") or "").strip()
            if not name or not url:
                continue
            yield name, url


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "-o", str(dest),
        url,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr or completed.stdout or "curl failed\n")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(CSV_PATH),
                        help="Path to the CSV file (default: photos_to_download.csv).")
    parser.add_argument("--out-dir", default=str(IMAGES_DIR),
                        help="Destination directory (default: cv-tuning/images/).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the destination file already exists.")
    args = parser.parse_args()

    if shutil.which("curl") is None:
        raise SystemExit("curl is not on PATH. Install it or use a different downloader.")

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(load_rows(csv_path))
    if not rows:
        print(f"No rows in {csv_path}")
        return 0

    print(f"Downloading {len(rows)} file(s) -> {out_dir}")
    downloaded = skipped = failed = 0
    for name, url in rows:
        dest = out_dir / f"{name}{infer_extension(url)}"
        if dest.exists() and not args.force:
            print(f"  skip  {dest.name}  (already exists)")
            skipped += 1
            continue
        print(f"  curl  {url} -> {dest.name}")
        if download(url, dest):
            downloaded += 1
        else:
            failed += 1

    print()
    print(f"Done. downloaded={downloaded} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
