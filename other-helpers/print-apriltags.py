#!/usr/bin/env python3
"""
Generate an A4 PDF of repeated AprilTag collections for a painting station + canvas.

Default physical geometry:
  - AprilTag black square: 35 mm x 35 mm
  - White border:          5 mm on each side
  - Total patch/cut size:  45 mm x 45 mm

Note: 35 mm black square + 5 mm white border EACH SIDE cannot be 40 mm total.
If you need exactly 40 mm total, set TOTAL_PATCH_MM = 40.0 and BORDER_MM = 2.5.

Install:
  pip install opencv-contrib-python reportlab pillow numpy

Run:
  python generate_apriltag_collections.py

Print PDF at 100% scale / actual size. Do not fit/shrink to page.
"""

import io
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# =========================
# USER CONFIG
# =========================

OUTPUT_PDF = "apriltag_collections_paint_station_canvas.pdf"
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
DPI = 600

# Physical tag geometry.
BLACK_TAG_MM = 35.0
BORDER_MM = 5.0                    # border on EACH side
TOTAL_PATCH_MM = BLACK_TAG_MM + 2 * BORDER_MM

# To force exactly 40 mm total with a 35 mm black tag, use this instead:
# BORDER_MM = 2.5
# TOTAL_PATCH_MM = 40.0

COPIES_OF_COLLECTION = 4           # 3-4 copies recommended
CUT_LINE_WIDTH_PT = 0.45
LABEL_FONT_SIZE = 7
TITLE_FONT_SIZE = 12

@dataclass(frozen=True)
class TagSpec:
    tag_id: int
    name: str
    group: str

# Use distinct IDs so canvas tags and paint/water tags are never confused.
TAGS: List[TagSpec] = [
    TagSpec(0,  "CANVAS_TL", "canvas"),
    TagSpec(1,  "CANVAS_TR", "canvas"),
    TagSpec(2,  "CANVAS_BR", "canvas"),
    TagSpec(3,  "CANVAS_BL", "canvas"),
    TagSpec(10, "PAINT_1",   "station"),
    TagSpec(11, "PAINT_2",   "station"),
    TagSpec(12, "PAINT_3",   "station"),
    TagSpec(13, "WATER",     "station"),
]

# Layout: A4 landscape, one repeated collection per page.
PAGE_W, PAGE_H = landscape(A4)
MARGIN_X_MM = 15.0
MARGIN_Y_MM = 12.0
COL_GAP_MM = 12.0
ROW_GAP_MM = 15.0
LABEL_GAP_MM = 3.0

# =========================
# HELPERS
# =========================

def mm_to_px(x_mm: float, dpi: int = DPI) -> int:
    return int(round(x_mm / 25.4 * dpi))


def make_marker_patch(tag_id: int) -> ImageReader:
    """Return a ReportLab ImageReader containing white patch + centered AprilTag."""
    black_px = mm_to_px(BLACK_TAG_MM)
    border_px = mm_to_px(BORDER_MM)
    total_px = mm_to_px(TOTAL_PATCH_MM)

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILY)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, black_px, borderBits=1)

    patch = np.ones((total_px, total_px), dtype=np.uint8) * 255

    # Center exactly in case rounding makes total_px slightly different from black+2border.
    y0 = (total_px - black_px) // 2
    x0 = (total_px - black_px) // 2
    patch[y0:y0 + black_px, x0:x0 + black_px] = marker

    pil = Image.fromarray(patch).convert("L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


def draw_tag(c: canvas.Canvas, spec: TagSpec, x_pt: float, y_pt: float) -> None:
    patch_size_pt = TOTAL_PATCH_MM * mm
    img = make_marker_patch(spec.tag_id)

    # Image patch.
    c.drawImage(img, x_pt, y_pt, width=patch_size_pt, height=patch_size_pt, mask=None)

    # Thin black cut/scissor boundary around the whole patch.
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(CUT_LINE_WIDTH_PT)
    c.rect(x_pt, y_pt, patch_size_pt, patch_size_pt, stroke=1, fill=0)

    # Label below each physical copy.
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", LABEL_FONT_SIZE)
    c.drawCentredString(x_pt + patch_size_pt / 2, y_pt - LABEL_GAP_MM * mm, f"ID {spec.tag_id} - {spec.name}")

    c.setFont("Helvetica", LABEL_FONT_SIZE - 1)
    c.drawCentredString(
        x_pt + patch_size_pt / 2,
        y_pt - (LABEL_GAP_MM + 3.2) * mm,
        f"{BLACK_TAG_MM:.0f}mm black + {BORDER_MM:.1f}mm border"
    )


def draw_collection_page(c: canvas.Canvas, copy_idx: int) -> None:
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", TITLE_FONT_SIZE)
    c.drawString(MARGIN_X_MM * mm, PAGE_H - MARGIN_Y_MM * mm, f"AprilTag collection copy {copy_idx}/{COPIES_OF_COLLECTION}")

    c.setFont("Helvetica", 8)
    c.drawString(
        MARGIN_X_MM * mm,
        PAGE_H - (MARGIN_Y_MM + 5) * mm,
        "Print at 100% actual size. Cut on thin outer black lines. One complete collection is repeated per page."
    )

    # Scale check line: 50 mm exact.
    sx = PAGE_W - (MARGIN_X_MM + 60) * mm
    sy = PAGE_H - (MARGIN_Y_MM + 2) * mm
    c.setLineWidth(1.0)
    c.line(sx, sy, sx + 50 * mm, sy)
    c.setFont("Helvetica", 7)
    c.drawCentredString(sx + 25 * mm, sy - 3.5 * mm, "50 mm scale check")

    patch = TOTAL_PATCH_MM * mm
    x0 = MARGIN_X_MM * mm
    y_top = PAGE_H - (MARGIN_Y_MM + 20) * mm

    # Two rows, four columns:
    # row 1: canvas tags, row 2: station tags.
    for idx, spec in enumerate(TAGS):
        row = 0 if idx < 4 else 1
        col = idx if idx < 4 else idx - 4
        x = x0 + col * (patch + COL_GAP_MM * mm)
        y = y_top - patch - row * (patch + ROW_GAP_MM * mm)
        draw_tag(c, spec, x, y)

    # Group labels at left of each row.
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x0, y_top + 2.5 * mm, "CANVAS CORNERS: TL, TR, BR, BL")
    c.drawString(x0, y_top - patch - ROW_GAP_MM * mm + 2.5 * mm, "PAINT/WATER STATION: paint1, paint2, paint3, water")


def build_pdf() -> None:
    c = canvas.Canvas(OUTPUT_PDF, pagesize=landscape(A4))
    c.setTitle("AprilTag Collections - Paint Station and Canvas")

    for copy_idx in range(1, COPIES_OF_COLLECTION + 1):
        draw_collection_page(c, copy_idx)
        if copy_idx != COPIES_OF_COLLECTION:
            c.showPage()

    c.save()
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Physical patch size: {TOTAL_PATCH_MM:.1f} mm x {TOTAL_PATCH_MM:.1f} mm")
    print(f"Black AprilTag square: {BLACK_TAG_MM:.1f} mm x {BLACK_TAG_MM:.1f} mm")
    print(f"White border: {BORDER_MM:.1f} mm on each side")


if __name__ == "__main__":
    build_pdf()
