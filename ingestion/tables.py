"""
ingestion/tables.py
-------------------
Table extraction using pdfplumber as a secondary pass over each page.

For every table detected by pdfplumber:
* A ``Chunk`` with ``chunk_type="table"`` is created.
* ``table_data`` is a list of rows (each row is a list of cell strings;
  None cells are replaced with empty strings).
* ``bbox`` is the normalised union bounding box of the detected table region.

Text blocks that spatially overlap a detected table's bbox are **excluded**
from the text-chunk output to prevent double-extraction.

No assumptions are made about specific document structures or content.

Detection strategy (in order):
1. ``lines_strict`` — ruled-border tables (highest precision, zero prose FP).
2. ``text/lines`` within candidate regions — horizontally-ruled tables.
3. Two-stage candidate-region + crop — borderless whitespace-aligned tables:
   a. STAGE 1: scan words into text lines; find runs of ≥3 consecutive short
      lines (each <60 chars, contains a digit, consistent left-x within 5pt).
      Build a tight union bbox over just those lines — never page-spanning.
   b. STAGE 2: crop the page to each candidate bbox, then call
      extract_table(text strategy) within the crop. Keep if ≥2 cols & ≥2 rows.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

import pdfplumber


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A pdfplumber table bounding box: (x0, top, x1, bottom) in page coordinates
PlumberBBox = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_bbox(
    x0: float, top: float, x1: float, bottom: float,
    page_width: float, page_height: float,
) -> Tuple[float, float, float, float]:
    """Convert pdfplumber page-space bbox to normalised [0,1] coordinates."""
    return (
        x0 / page_width,
        top / page_height,
        x1 / page_width,
        bottom / page_height,
    )


def _sanitise_row(row: List[Optional[str]]) -> List[str]:
    """Replace None cells with empty strings."""
    return [cell if cell is not None else "" for cell in row]


def _candidate_regions(plumber_page: pdfplumber.page.Page) -> List[PlumberBBox]:
    """
    STAGE 1: Scan word objects to find tight candidate bboxes for tabular regions.

    Strategy: group words into visual lines by vertical position, then for each
    line compute per-x-cluster word groups (to handle two-column page layouts
    where table text sits right of prose text at a different x0). Within each
    x-cluster, tag a line as 'tabular' if it is short (<60 chars) and contains
    a digit. Then find runs of ≥3 consecutive tabular lines within the same
    x-cluster and build a tight union bbox from just those lines.

    Returns a list of (x0, top, x1, bottom) bboxes in page coordinates.
    """
    words = plumber_page.extract_words(keep_blank_chars=False)
    if not words:
        return []

    pw = plumber_page.width

    # Group words into visual lines by rounding vertical midpoint to 1pt
    keyfunc = lambda w: round((w["top"] + w["bottom"]) / 2)
    words_sorted = sorted(words, key=keyfunc)

    # For each line, split words into x-clusters (left-column vs right-column)
    # by clustering on x0 with a gap threshold: if x0 jumps by > 30% of page
    # width, treat as a new column cluster.
    _X_COLUMN_GAP = pw * 0.25

    # Build a list of (y_mid, x0, top, bottom, x1, text) per word-cluster-in-line
    cluster_lines: List[dict] = []
    for y_mid, group in itertools.groupby(words_sorted, key=keyfunc):
        line_words = sorted(list(group), key=lambda w: w["x0"])
        # Split into x-clusters by gap
        clusters: List[List[dict]] = [[]]
        for w in line_words:
            if clusters[-1] and (w["x0"] - clusters[-1][-1]["x1"]) > _X_COLUMN_GAP:
                clusters.append([])
            clusters[-1].append(w)
        for cl in clusters:
            if not cl:
                continue
            text = " ".join(w["text"] for w in cl)
            x0 = min(w["x0"] for w in cl)
            x1 = max(w["x1"] for w in cl)
            top = min(w["top"] for w in cl)
            bottom = max(w["bottom"] for w in cl)
            cluster_lines.append({
                "y_mid": y_mid,
                "x0": x0, "x1": x1, "top": top, "bottom": bottom,
                "text": text,
                "col_anchor": round(x0 / 20) * 20,  # bucket by 20pt for grouping
            })

    # Tag each cluster-line as tabular.
    # A line qualifies if it is short AND (has a digit OR is very short — the
    # latter catches label/key cells in label-value tables which have no digits).
    _SHORT_LINE_CHARS = 60
    _VERY_SHORT_CHARS = 20  # pure key cells like "n_layers", "head_dim"
    for cl in cluster_lines:
        has_digit = any(c.isdigit() for c in cl["text"])
        char_count = len(cl["text"])
        is_short = char_count < _SHORT_LINE_CHARS
        is_very_short = char_count < _VERY_SHORT_CHARS
        cl["tabular"] = is_short and (has_digit or is_very_short)

    # Group cluster-lines by their x column anchor, then find runs of ≥3 tabular
    from collections import defaultdict
    by_col: dict = defaultdict(list)
    for cl in cluster_lines:
        by_col[cl["col_anchor"]].append(cl)

    _MIN_RUN = 3
    _X_TOLERANCE = 15.0  # pt — allow slight x drift within a column
    margin = 4.0
    regions = []

    for col_anchor, col_lines in by_col.items():
        # Sort by vertical position
        col_lines_sorted = sorted(col_lines, key=lambda c: c["y_mid"])
        i = 0
        while i < len(col_lines_sorted):
            if not col_lines_sorted[i]["tabular"]:
                i += 1
                continue
            run = [col_lines_sorted[i]]
            j = i + 1
            while j < len(col_lines_sorted) and col_lines_sorted[j]["tabular"]:
                if abs(col_lines_sorted[j]["x0"] - run[0]["x0"]) <= _X_TOLERANCE:
                    run.append(col_lines_sorted[j])
                else:
                    break
                j += 1
            if len(run) >= _MIN_RUN:
                rx0 = min(ln["x0"] for ln in run)
                rtop = min(ln["top"] for ln in run)
                rx1 = max(ln["x1"] for ln in run)
                rbottom = max(ln["bottom"] for ln in run)
                regions.append((
                    max(0, rx0 - margin),
                    max(0, rtop - margin),
                    min(pw, rx1 + margin),
                    min(plumber_page.height, rbottom + margin),
                ))
            i = j if j > i else i + 1

    # Merge vertically-overlapping regions (handles label-col + value-col from
    # the same table being detected as separate x-clusters).
    # Two regions are merged if their vertical spans overlap by ≥50%.
    if len(regions) > 1:
        merged = True
        while merged:
            merged = False
            new_regions = []
            used = [False] * len(regions)
            for a_idx in range(len(regions)):
                if used[a_idx]:
                    continue
                ax0, atop, ax1, abottom = regions[a_idx]
                a_height = abottom - atop
                for b_idx in range(a_idx + 1, len(regions)):
                    if used[b_idx]:
                        continue
                    bx0, btop, bx1, bbottom = regions[b_idx]
                    # Vertical overlap ratio
                    ov_top = max(atop, btop)
                    ov_bottom = min(abottom, bbottom)
                    ov = max(0.0, ov_bottom - ov_top)
                    b_height = bbottom - btop
                    overlap_ratio = ov / min(a_height, b_height) if min(a_height, b_height) > 0 else 0.0
                    if overlap_ratio >= 0.5:
                        # Merge: take union
                        ax0 = min(ax0, bx0)
                        atop = min(atop, btop)
                        ax1 = max(ax1, bx1)
                        abottom = max(abottom, bbottom)
                        a_height = abottom - atop
                        used[b_idx] = True
                        merged = True
                new_regions.append((ax0, atop, ax1, abottom))
                used[a_idx] = True
            regions = new_regions

    return regions



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tables(
    plumber_page: pdfplumber.page.Page,
) -> List[dict]:
    """Extract all tables from a pdfplumber page object.

    Parameters
    ----------
    plumber_page:
        A single pdfplumber ``Page`` instance.

    Returns
    -------
    List of dicts, each with:
    * ``"bbox"``       – normalised ``(x0, y0, x1, y1)``
    * ``"table_data"`` – list of rows (list of cell strings)
    * ``"raw_bbox"``   – un-normalised bbox for downstream overlap tests
    """
    pw = plumber_page.width
    ph = plumber_page.height

    if pw == 0 or ph == 0:
        return []

    results = []

    # -----------------------------------------------------------------------
    # Pass 1: lines_strict — ruled-border tables (highest precision)
    # -----------------------------------------------------------------------
    _SETTINGS = {
        "vertical_strategy": "lines_strict",
        "horizontal_strategy": "lines_strict",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
    }
    try:
        found = plumber_page.find_tables(table_settings=_SETTINGS)
    except Exception:
        found = []

    if found:
        for tbl in found:
            x0, top, x1, bottom = tbl.bbox
            rows = tbl.extract()
            if not rows:
                continue
            sanitised = [_sanitise_row(row) for row in rows]
            norm = _norm_bbox(x0, top, x1, bottom, pw, ph)
            results.append({
                "bbox": norm,
                "raw_bbox": (x0, top, x1, bottom),
                "table_data": sanitised,
            })
        return results

    # -----------------------------------------------------------------------
    # Pass 2: text/lines — horizontally-ruled tables (no vertical lines)
    # -----------------------------------------------------------------------
    try:
        found_lines = plumber_page.find_tables(table_settings={
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 2,
        })
        for t in found_lines:
            rows = t.extract()
            if not rows or len(rows[0] or []) < 2 or len(rows) < 2:
                continue
            x0, top, x1, bottom = t.bbox
            # Sanity guard: still reject a clear page-spanning false positive
            if ((x1 - x0) * (bottom - top)) / (pw * ph) > 0.6:
                continue
            sanitised = [_sanitise_row(row) for row in rows]
            norm = _norm_bbox(x0, top, x1, bottom, pw, ph)
            results.append({
                "bbox": norm,
                "raw_bbox": (x0, top, x1, bottom),
                "table_data": sanitised,
            })
    except Exception:
        pass

    if results:
        return results

    # -----------------------------------------------------------------------
    # Pass 3: Two-stage candidate-region + crop (borderless tables)
    # -----------------------------------------------------------------------
    candidate_bboxes = _candidate_regions(plumber_page)

    for cand_bbox in candidate_bboxes:
        cx0, ctop, cx1, cbottom = cand_bbox
        try:
            crop = plumber_page.crop(cand_bbox)
        except Exception:
            continue

        _TEXT_SETTINGS = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        }
        try:
            tables_in_crop = crop.find_tables(table_settings=_TEXT_SETTINGS)
        except Exception:
            continue

        for tbl in tables_in_crop:
            rows = tbl.extract()
            if not rows or len(rows[0] or []) < 2 or len(rows) < 2:
                continue
            # tbl.bbox from a cropped page is already in original page coordinates
            px0, ptop, px1, pbottom = tbl.bbox
            sanitised = [_sanitise_row(row) for row in rows]
            norm = _norm_bbox(px0, ptop, px1, pbottom, pw, ph)
            results.append({
                "bbox": norm,
                "raw_bbox": (px0, ptop, px1, pbottom),
                "table_data": sanitised,
            })

    return results
