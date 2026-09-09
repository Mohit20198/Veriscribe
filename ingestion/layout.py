"""
ingestion/layout.py
-------------------
Layout-aware block ordering for PDF pages extracted with PyMuPDF.

Responsibilities
~~~~~~~~~~~~~~~~
* Detect 1-, 2-, or 3-column page layouts using x-midpoint clustering.
* Sort blocks column-left-to-right, then top-to-bottom within each column.
* Strip repeating headers / footers (blocks whose normalised y-position
  repeats across the majority of pages within a small tolerance).
* Merge vertically adjacent blocks in the same column into paragraph-level
  chunks to avoid single-line fragmentation.

No assumptions are made about any specific document's structure, content,
or filename.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

# A raw PyMuPDF block: (x0, y0, x1, y1, text, block_no, block_type)
RawBlock = Tuple[float, float, float, float, str, int, int]

# A normalised block after layout processing
NormBlock = Dict  # keys: bbox, text, col_index


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _cluster_midpoints(
    midpoints: List[float],
    page_width: float,
    max_cols: int = 3,
) -> List[float]:
    """Return sorted column-centre x-positions using simple gap analysis.

    We look for large horizontal gaps between sorted x-midpoints and treat
    each contiguous cluster as a column.  The number of detected columns is
    capped at *max_cols* (default 3).
    """
    if not midpoints:
        return []

    sorted_mids = sorted(set(midpoints))
    if len(sorted_mids) == 1:
        return sorted_mids

    # Compute gaps between consecutive midpoints
    gaps = [sorted_mids[i + 1] - sorted_mids[i] for i in range(len(sorted_mids) - 1)]
    avg_gap = statistics.mean(gaps)
    # A gap is "large" if it is significantly bigger than average,
    # OR if it's an absolute large jump (>= 15% of page width) which protects
    # against the case where there's only 1 gap total (exactly 2 unique points).
    threshold = avg_gap * 1.5

    # Build clusters
    clusters: List[List[float]] = [[sorted_mids[0]]]
    for i, gap in enumerate(gaps):
        if gap > threshold or gap >= 0.15:
            clusters.append([])
        clusters[-1].append(sorted_mids[i + 1])

    # Cap at max_cols by merging smallest-gap pairs when over the limit
    while len(clusters) > max_cols:
        # Find adjacent pair with smallest gap between their means and merge
        cluster_means = [statistics.mean(c) for c in clusters]
        inter_gaps = [cluster_means[i + 1] - cluster_means[i] for i in range(len(cluster_means) - 1)]
        merge_idx = inter_gaps.index(min(inter_gaps))
        clusters[merge_idx] = clusters[merge_idx] + clusters[merge_idx + 1]
        del clusters[merge_idx + 1]

    return [statistics.mean(c) for c in clusters]


def _assign_column(x_mid: float, col_centres: List[float]) -> int:
    """Return the index of the nearest column centre."""
    return min(range(len(col_centres)), key=lambda i: abs(col_centres[i] - x_mid))


# ---------------------------------------------------------------------------
# Header / footer detection
# ---------------------------------------------------------------------------

def detect_header_footer_zones(
    pages: List[fitz.Page],
    y_tolerance: float = 0.02,   # normalised — ~14 px on a 792-pt page
    presence_ratio: float = 0.5, # must appear on at least 50 % of pages
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (header_zones, footer_zones) as lists of (y_low, y_high) bands.

    A "repeating" y-band is one where at least *presence_ratio* of pages
    have a text block whose normalised y-midpoint falls within *y_tolerance*
    of a common value.

    Returns two lists of (y_low, y_high) normalised bands to exclude.
    """
    n_pages = len(pages)
    # A header/footer must repeat across multiple pages. If there is only 1 page,
    # we cannot confidently distinguish a header from a genuine document title.
    if n_pages < 2:
        return [], []

    # Collect normalised y-midpoints from each page
    all_y_mids: List[float] = []
    page_y_mids: List[List[float]] = []

    for page in pages:
        pw, ph = page.rect.width, page.rect.height
        if pw == 0 or ph == 0:
            page_y_mids.append([])
            continue
        blocks = page.get_text("blocks")
        y_mids = []
        for b in blocks:
            if b[6] != 0:  # skip non-text blocks
                continue
            y_mid = ((b[1] + b[3]) / 2) / ph
            y_mids.append(y_mid)
            all_y_mids.append(y_mid)
        page_y_mids.append(y_mids)

    if not all_y_mids:
        return [], []

    # Cluster global y-mids and find those that appear on many pages
    # We use a simple scan: sort global mids, form clusters, count page hits
    sorted_global = sorted(all_y_mids)
    clusters: List[List[float]] = [[sorted_global[0]]]
    for val in sorted_global[1:]:
        if val - clusters[-1][-1] <= y_tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])

    repeating_y_bands: List[Tuple[float, float]] = []
    for cluster in clusters:
        centre = statistics.mean(cluster)
        y_low = centre - y_tolerance
        y_high = centre + y_tolerance
        # Count pages that have a block in this band
        hits = sum(
            1 for page_mids in page_y_mids
            if any(y_low <= ym <= y_high for ym in page_mids)
        )
        # Must appear on multiple pages (>=2) AND meet the presence ratio
        if hits >= 2 and hits >= presence_ratio * n_pages:
            repeating_y_bands.append((y_low, y_high))

    # Classify as header or footer by y-position
    headers = [(lo, hi) for lo, hi in repeating_y_bands if statistics.mean([lo, hi]) < 0.15]
    footers = [(lo, hi) for lo, hi in repeating_y_bands if statistics.mean([lo, hi]) > 0.85]

    return headers, footers


def _in_excluded_zone(
    norm_y_mid: float,
    header_bands: List[Tuple[float, float]],
    footer_bands: List[Tuple[float, float]],
) -> bool:
    for lo, hi in header_bands + footer_bands:
        if lo <= norm_y_mid <= hi:
            return True
    return False


# ---------------------------------------------------------------------------
# Block merging — with spanning-block support
# ---------------------------------------------------------------------------

_PARA_MERGE_GAP_NORM = 0.015       # ~12 px on a 792-pt page in normalised units
_SPAN_OVERLAP_THRESHOLD = 0.70     # block must overlap >70% with one band to be a column block
_MIN_COL_BLOCK_WIDTH = 0.08        # blocks narrower than 8% of page width are excluded from
                                    # column-centre computation (they are narrow floats, watermarks,
                                    # or margin annotations that should not anchor a column).  They
                                    # still appear as chunks in the final output.
_MAX_COL_BLOCK_WIDTH = 0.50        # blocks wider than 50% are excluded from column-centre
                                    # computation to prevent spanning headings from disrupting
                                    # 2-col detection. If all blocks are excluded, we fall back to all.


def _col_bands_from_centres(
    col_centres: List[float],
) -> List[Tuple[float, float]]:
    """Return the normalised (x_lo, x_hi) corridor for each column centre.

    ``col_centres`` are already normalised [0, 1] values (x-midpoints as a
    fraction of page width).  Boundaries are placed at the midpoint between
    adjacent centres; the leftmost band starts at 0.0 and the rightmost ends
    at 1.0.
    """
    n = len(col_centres)
    bands: List[Tuple[float, float]] = []
    for i, centre in enumerate(col_centres):
        lo = (col_centres[i - 1] + centre) / 2 if i > 0 else 0.0
        hi = (centre + col_centres[i + 1]) / 2 if i < n - 1 else 1.0
        bands.append((lo, hi))
    return bands


def _band_overlap_fraction(
    bx0: float, bx1: float,
    band_lo: float, band_hi: float,
) -> float:
    """Fraction of a block's x-width that falls within a column band.

    Returns a value in [0, 1].  A block that lies entirely within the band
    returns 1.0; a block entirely outside returns 0.0.
    """
    bw = bx1 - bx0
    if bw <= 0:
        return 0.0
    inter = max(0.0, min(bx1, band_hi) - max(bx0, band_lo))
    return inter / bw


def _classify_block_span(
    blk: Dict,
    col_bands: List[Tuple[float, float]],
    threshold: float = _SPAN_OVERLAP_THRESHOLD,
) -> None:
    """Label ``blk`` as a column block or a spanning block in-place.

    A block is a *column block* if >``threshold`` of its x-width falls within
    a single column band.  Otherwise it is a *spanning block* — a full-width
    element such as a heading, abstract, or caption that genuinely covers
    multiple column bands and must not be clamped to any one column.

    Sets ``blk["spanning"]`` (bool) and updates ``blk["col_index"]`` to -1
    for spanning blocks.
    """
    bx0, _, bx1, _ = blk["bbox"]
    best_frac = 0.0
    best_idx = blk["col_index"]
    for i, (lo, hi) in enumerate(col_bands):
        frac = _band_overlap_fraction(bx0, bx1, lo, hi)
        if frac > best_frac:
            best_frac = frac
            best_idx = i
    if best_frac >= threshold:
        # Genuine column block — keep its assigned column
        blk["col_index"] = best_idx
        blk["col_band"] = col_bands[best_idx]
        blk["spanning"] = False
    else:
        # Spanning block — covers multiple columns; do not assign to any column
        blk["col_index"] = -1
        blk["col_band"] = (0.0, 1.0)   # full-width sentinel
        blk["spanning"] = True


def _merge_column_blocks(
    blocks: List[Dict],
) -> List[Dict]:
    """Merge vertically adjacent blocks within the SAME column.

    Only merges genuine column blocks (``spanning == False`` and same
    ``col_index``).  Spanning blocks are never merged with column blocks.

    The merged bbox uses the NATURAL x-extent of the constituent blocks
    (no clamping needed — only same-column blocks are merged, so their
    x-extents legitimately belong to that column).
    """
    if not blocks:
        return []

    merged: List[Dict] = [dict(blocks[0])]
    for blk in blocks[1:]:
        prev = merged[-1]
        # Never merge across column boundaries or spanning-block boundaries
        if blk["col_index"] != prev["col_index"]:
            merged.append(dict(blk))
            continue
        if blk.get("spanning") or prev.get("spanning"):
            merged.append(dict(blk))
            continue
        gap = blk["bbox"][1] - prev["bbox"][3]   # y gap (normalised)
        if gap < _PARA_MERGE_GAP_NORM:
            # Merge — natural x-union stays within the column
            new_bbox = (
                min(prev["bbox"][0], blk["bbox"][0]),
                prev["bbox"][1],
                max(prev["bbox"][2], blk["bbox"][2]),
                blk["bbox"][3],
            )
            prev["bbox"] = new_bbox
            sep = " " if prev["text"].rstrip() and blk["text"].lstrip() else ""
            prev["text"] = prev["text"].rstrip() + sep + blk["text"].lstrip()
        else:
            merged.append(dict(blk))

    return merged


def _interleave_spanning(
    col_blocks: List[Dict],
    spanning_blocks: List[Dict],
) -> List[Dict]:
    """Merge column blocks and spanning blocks into a single reading order.

    Column blocks are already sorted (col_index, y0).  Spanning blocks are
    inserted at their natural y0 position, breaking into column sequences
    wherever a spanning block interrupts them — matching how a human reader
    reads a two-column page with a full-width heading or caption.
    """
    # Assign each column block a sort key of (y0_of_column_top, x_col, y0)
    # so entire column reads before next column, but spanning blocks can
    # interrupt mid-column by their actual y0.
    if not spanning_blocks:
        return col_blocks
    if not col_blocks:
        return sorted(spanning_blocks, key=lambda b: b["bbox"][1])

    # Build a timeline: each entry is (y0, is_spanning, block)
    timeline = []
    for blk in col_blocks:
        timeline.append((blk["bbox"][1], 1, blk))   # 1 = column (secondary sort)
    for blk in spanning_blocks:
        timeline.append((blk["bbox"][1], 0, blk))   # 0 = spanning (appears first at same y)

    # Sort: y0 ascending; at equal y, spanning blocks come first
    timeline.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in timeline]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def order_blocks(
    page: fitz.Page,
    header_bands: List[Tuple[float, float]],
    footer_bands: List[Tuple[float, float]],
    excluded_bboxes: List[Tuple[float, float, float, float]] | None = None,
) -> List[Dict]:
    """Return layout-ordered, merged, header/footer-stripped text blocks.

    Parameters
    ----------
    page:
        The PyMuPDF page to process.
    header_bands:
        Normalised ``(y_low, y_high)`` bands to exclude as headers.
    footer_bands:
        Normalised ``(y_low, y_high)`` bands to exclude as footers.
    excluded_bboxes:
        List of normalised bounding boxes (e.g. from detected tables / images)
        whose text blocks should be suppressed.

    Returns
    -------
    List of dicts with keys ``bbox`` (normalised tuple), ``text`` (str),
    ``col_index`` (int, or -1 for spanning blocks), and
    ``spanning`` (bool).  The list is in correct reading order.
    """
    pw, ph = page.rect.width, page.rect.height
    if pw == 0 or ph == 0:
        return []

    excluded_bboxes = excluded_bboxes or []

    raw_blocks: List[RawBlock] = page.get_text("blocks")
    text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]

    def norm(x0, y0, x1, y1):
        return (x0 / pw, y0 / ph, x1 / pw, y1 / ph)

    normed = []
    for b in text_blocks:
        nb = norm(b[0], b[1], b[2], b[3])
        y_mid = (nb[1] + nb[3]) / 2
        if _in_excluded_zone(y_mid, header_bands, footer_bands):
            continue
        if _overlaps_any(nb, excluded_bboxes, min_overlap=0.3):
            continue
        normed.append({"bbox": nb, "text": b[4], "col_index": -1, "spanning": False})

    if not normed:
        return []

    # ── Detect columns ────────────────────────────────────────────────────
    # Exclude narrow blocks (watermarks, margin annotations, single-word floats)
    # and wide blocks (spanning headings, abstracts) from the x-midpoint set
    # used to compute column centres.  They remain in `normed` and will be
    # classified/chunked normally afterward.
    wide_mids = [
        (b["bbox"][0] + b["bbox"][2]) / 2
        for b in normed
        if _MIN_COL_BLOCK_WIDTH <= (b["bbox"][2] - b["bbox"][0]) <= _MAX_COL_BLOCK_WIDTH
    ]
    # Fall back to all blocks if every block was excluded (e.g. 1-col layout)
    x_mids_for_clustering = wide_mids if wide_mids else [
        (b["bbox"][0] + b["bbox"][2]) / 2 for b in normed
    ]
    col_centres = _cluster_midpoints(x_mids_for_clustering, pw)
    col_bands = _col_bands_from_centres(col_centres)

    # ── Assign column index; detect spanning blocks ───────────────────────
    for blk in normed:
        x_mid = (blk["bbox"][0] + blk["bbox"][2]) / 2
        blk["col_index"] = _assign_column(x_mid, col_centres)
        _classify_block_span(blk, col_bands)

    # ── Separate, sort, merge column blocks ───────────────────────────────
    col_blks = [b for b in normed if not b["spanning"]]
    span_blks = [b for b in normed if b["spanning"]]

    col_blks.sort(key=lambda b: (b["col_index"], b["bbox"][1]))
    col_blks = _merge_column_blocks(col_blks)

    # ── Interleave spanning blocks by y-position ──────────────────────────
    result = _interleave_spanning(col_blks, span_blks)
    return result


# ---------------------------------------------------------------------------
# Overlap helper (reused by tables.py / triage.py via import)
# ---------------------------------------------------------------------------

def _bbox_overlap_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Return the intersection area of two normalised bboxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _overlaps_any(
    bbox: Tuple[float, float, float, float],
    others: List[Tuple[float, float, float, float]],
    min_overlap: float = 0.3,
) -> bool:
    """Return True if *bbox* overlaps any bbox in *others* by >= *min_overlap*
    fraction of *bbox*'s own area."""
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    area = bw * bh
    if area <= 0:
        return False
    for other in others:
        if _bbox_overlap_area(bbox, other) / area >= min_overlap:
            return True
    return False
