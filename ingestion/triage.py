"""
ingestion/triage.py
--------------------
Chart / figure candidate detection for PDF pages.

For each page, after table extraction, this module:
1. Calls ``page.get_images()`` to find embedded raster images.
2. Skips image regions already covered by text or table chunks.
3. For each remaining image region, crops the page at 200 DPI and saves
   the result as ``./output/chart_crops/{document_id}_{page}_{index}.png``.
4. Looks for nearby caption text (within ~30 pt of the image bbox, containing
   figure/chart keywords) and attaches it as the chunk's ``text`` field.

No assumptions are made about specific document structure or content.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from ingestion.layout import _overlaps_any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Caption-keyword pattern — intentionally generic
_CAPTION_RE = re.compile(
    r"\b(figure|fig\.?|chart|diagram|exhibit|graph|plot|image|photo)\b",
    re.IGNORECASE,
)

# How far (in pts) above or below the image bbox to look for captions
_CAPTION_PROXIMITY_PTS = 30.0

# Output directory for cropped images
_CROP_DIR = Path("./output/chart_crops")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_image_bboxes(
    page: fitz.Page,
) -> List[Tuple[float, float, float, float]]:
    """Return raw (un-normalised) bounding boxes of embedded images on *page*.

    Uses ``page.get_image_rects()`` when available (PyMuPDF ≥ 1.23) and
    falls back to ``page.get_images()`` + ``page.get_image_bbox()`` for older
    versions.
    """
    bboxes = []
    try:
        # Preferred: get_image_rects returns Rect objects directly
        for item in page.get_images(full=True):
            xref = item[0]
            rects = page.get_image_rects(xref)
            for r in rects:
                bboxes.append((r.x0, r.y0, r.x1, r.y1))
    except Exception:
        pass

    # Deduplicate
    seen = set()
    unique: List[Tuple[float, float, float, float]] = []
    for b in bboxes:
        key = tuple(round(v, 2) for v in b)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def _norm_bbox(
    x0: float, y0: float, x1: float, y1: float,
    pw: float, ph: float,
) -> Tuple[float, float, float, float]:
    return (x0 / pw, y0 / ph, x1 / pw, y1 / ph)


def _find_caption(
    page: fitz.Page,
    img_bbox_raw: Tuple[float, float, float, float],
    proximity_pts: float = _CAPTION_PROXIMITY_PTS,
) -> Optional[str]:
    """Search for caption text near the image bounding box.

    Looks for text blocks whose y-range is within *proximity_pts* of the top
    or bottom edge of *img_bbox_raw*, and that contain a caption keyword.

    Returns the stripped caption string, or None if not found.
    """
    x0, y0, x1, y1 = img_bbox_raw
    blocks = page.get_text("blocks")
    candidates = []
    for b in blocks:
        if b[6] != 0:
            continue
        bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], b[4]
        text = text.strip()
        if not text:
            continue
        # Check vertical proximity: block is just above or just below image
        above = y0 - proximity_pts <= by1 <= y0
        below = y1 <= by0 <= y1 + proximity_pts
        if not (above or below):
            continue
        # Check horizontal overlap with image
        h_overlap = min(bx1, x1) - max(bx0, x0)
        if h_overlap <= 0:
            continue
        if _CAPTION_RE.search(text):
            candidates.append(text)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chart_candidates(
    page: fitz.Page,
    document_id: str,
    page_number: int,
    existing_norm_bboxes: List[Tuple[float, float, float, float]],
    crop_dir: Path = _CROP_DIR,
) -> List[dict]:
    """Detect chart / figure candidates on *page* and return raw dicts.

    Parameters
    ----------
    page:
        PyMuPDF page object.
    document_id:
        Identifier used to name the cropped image files.
    page_number:
        1-based page index (for file naming and chunk metadata).
    existing_norm_bboxes:
        Normalised bboxes of already-identified text and table chunks.
        Image regions that substantially overlap these are skipped.
    crop_dir:
        Directory where cropped PNGs will be saved.

    Returns
    -------
    List of dicts with keys: ``"bbox"`` (normalised), ``"image_path"`` (str),
    ``"text"`` (Optional[str]).
    """
    pw, ph = page.rect.width, page.rect.height
    if pw == 0 or ph == 0:
        return []

    raw_image_bboxes = _get_image_bboxes(page)
    if not raw_image_bboxes:
        return []

    crop_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, raw_bbox in enumerate(raw_image_bboxes):
        raw_x0, raw_y0, raw_x1, raw_y1 = raw_bbox

        # Skip degenerate bboxes
        if raw_x1 <= raw_x0 or raw_y1 <= raw_y0:
            continue

        norm = _norm_bbox(raw_x0, raw_y0, raw_x1, raw_y1, pw, ph)

        # Skip if already covered by a text or table chunk
        if _overlaps_any(norm, existing_norm_bboxes, min_overlap=0.5):
            continue

        # Crop page region at 200 DPI
        clip = fitz.Rect(raw_x0, raw_y0, raw_x1, raw_y1)
        pixmap = page.get_pixmap(clip=clip, dpi=200)
        fname = f"{document_id}_{page_number}_{idx}.png"
        out_path = crop_dir / fname
        pixmap.save(str(out_path))

        # Look for nearby caption
        caption = _find_caption(page, raw_bbox)

        results.append(
            {
                "bbox": norm,
                "image_path": str(out_path),
                "text": caption,
            }
        )

    return results
