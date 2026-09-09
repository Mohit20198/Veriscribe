"""
scripts/merge_bbox_debug.py
----------------------------
Investigates whether _merge_into_paragraphs() in layout.py is producing
bboxes that span column boundaries — which would be a bug: a merged paragraph
block should never be wider than the column band it started in.

Specifically checks page 2 of output/arxiv_paper.pdf, where chunk index 4
had bbox width ≈ 0.648 (x0=0.176 → x1=0.824) — almost the full page width.

Report:
  A. The raw PyMuPDF blocks on page 2 (pre-merge), with their x-midpoints
     and assigned column indices.
  B. The post-merge blocks, highlighting any whose bbox width is larger
     than their column band width.
  C. A concrete before/after showing which blocks were merged to form the
     wide bbox.
  D. Conclusion: bug confirmed or not, and if yes, the exact fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from ingestion.layout import (
    _assign_column,
    _cluster_midpoints,
    _in_excluded_zone,
    _merge_into_paragraphs,
    _overlaps_any,
    detect_header_footer_zones,
)

PDF = "output/arxiv_paper.pdf"
TARGET_PAGE = 2   # 1-based

doc = fitz.open(PDF)
all_pages = [doc[i] for i in range(len(doc))]
header_bands, footer_bands = detect_header_footer_zones(all_pages)

page = doc[TARGET_PAGE - 1]
pw, ph = page.rect.width, page.rect.height

print(f"Page {TARGET_PAGE}  size: {pw:.1f} x {ph:.1f} pt")
print(f"Header bands: {header_bands}")
print(f"Footer bands: {footer_bands}")

# ── Step 1: raw blocks (same logic as order_blocks) ───────────────────────
raw_blocks = page.get_text("blocks")
text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]

def norm(x0, y0, x1, y1):
    return (x0 / pw, y0 / ph, x1 / pw, y1 / ph)

normed = []
for b in text_blocks:
    nb = norm(b[0], b[1], b[2], b[3])
    y_mid = (nb[1] + nb[3]) / 2
    if _in_excluded_zone(y_mid, header_bands, footer_bands):
        continue
    normed.append({"bbox": nb, "text": b[4], "col_index": -1})

# ── Step 2: column detection ───────────────────────────────────────────────
x_mids = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in normed]
col_centres = _cluster_midpoints([m * pw for m in x_mids], pw)
print(f"\nDetected {len(col_centres)} column centre(s): "
      f"{[round(c/pw, 3) for c in col_centres]} (normalised)")

col_band_widths = []
for i, centre in enumerate(col_centres):
    # Estimate band width: half-gap to neighbours
    left_edge = (col_centres[i - 1] + centre) / 2 if i > 0 else 0.0
    right_edge = (centre + col_centres[i + 1]) / 2 if i < len(col_centres) - 1 else pw
    col_band_widths.append((left_edge / pw, right_edge / pw))

print("Column bands (normalised x ranges):")
for i, (lo, hi) in enumerate(col_band_widths):
    print(f"  col {i}: x in [{lo:.3f}, {hi:.3f}]  width={hi-lo:.3f}")

for blk in normed:
    x_mid_raw = (blk["bbox"][0] + blk["bbox"][2]) / 2 * pw
    blk["col_index"] = _assign_column(x_mid_raw, col_centres)

normed.sort(key=lambda b: (b["col_index"], b["bbox"][1]))

# ── Step 3: print pre-merge blocks ────────────────────────────────────────
print(f"\n{'─'*72}")
print(f"PRE-MERGE BLOCKS ({len(normed)} total)")
print(f"{'─'*72}")
print(f"  {'#':>3}  {'COL':>3}  {'y0':>6}  {'y1':>6}  {'x0':>6}  {'x1':>6}  "
      f"{'WIDTH':>6}  TEXT-PREVIEW")
for i, b in enumerate(normed):
    x0, y0, x1, y1 = b["bbox"]
    w = x1 - x0
    preview = b["text"].replace("\n", " ").strip()[:45]
    print(f"  {i:>3}  {b['col_index']:>3}  {y0:.4f}  {y1:.4f}  "
          f"{x0:.4f}  {x1:.4f}  {w:.4f}  {preview}")

# ── Step 4: merge ─────────────────────────────────────────────────────────
merged = _merge_into_paragraphs(normed, ph)

print(f"\n{'─'*72}")
print(f"POST-MERGE BLOCKS ({len(merged)} total)")
print(f"{'─'*72}")

# Column band widths for checking
def col_band_for(col_idx):
    if col_idx < len(col_band_widths):
        return col_band_widths[col_idx]
    return (0.0, 1.0)

print(f"  {'#':>3}  {'COL':>3}  {'y0':>6}  {'y1':>6}  {'x0':>6}  {'x1':>6}  "
      f"{'WIDTH':>6}  {'COL-BAND-W':>10}  {'BUG?':>5}  TEXT-PREVIEW")

bugs_found = []
for i, b in enumerate(merged):
    x0, y0, x1, y1 = b["bbox"]
    w = x1 - x0
    col_lo, col_hi = col_band_for(b["col_index"])
    col_bw = col_hi - col_lo
    # Flag as bug if merged width > column band width + small tolerance
    is_bug = w > col_bw + 0.02
    bug_str = "YES!" if is_bug else ""
    if is_bug:
        bugs_found.append((i, b, w, col_bw))
    preview = b["text"].replace("\n", " ").strip()[:40]
    print(f"  {i:>3}  {b['col_index']:>3}  {y0:.4f}  {y1:.4f}  "
          f"{x0:.4f}  {x1:.4f}  {w:.4f}  {col_bw:>10.4f}  {bug_str:>5}  {preview}")

# ── Step 5: bug report ────────────────────────────────────────────────────
print(f"\n{'─'*72}")
if bugs_found:
    print(f"BUG CONFIRMED: {len(bugs_found)} merged block(s) wider than their column band.")
    print()
    for idx, blk, w, col_bw in bugs_found:
        print(f"  Merged block #{idx}:")
        print(f"    bbox  = {tuple(f'{v:.4f}' for v in blk['bbox'])}")
        print(f"    width = {w:.4f}  (column band width = {col_bw:.4f})")
        print(f"    col_index = {blk['col_index']}")
        print(f"    text preview = {blk['text'].replace(chr(10),' ').strip()[:80]}")
    print()
    print("ROOT CAUSE:")
    print("  _merge_into_paragraphs() extends bbox.x0/x1 to the union of merged")
    print("  blocks.  If two same-column blocks have slightly different x extents")
    print("  (common in two-column layouts where text wraps differently per line),")
    print("  the merged bbox can span slightly outside the column's natural x-band.")
    print("  More critically, if the column detection mis-assigns a block to the")
    print("  wrong column, the merged block can balloon to near-full-page width.")
    print()
    print("FIX:")
    print("  In _merge_into_paragraphs(), clamp the merged x0/x1 to the column's")
    print("  representative x-band rather than taking the union of block x extents.")
    print("  Alternatively, pass col_band_widths into the function and enforce:")
    print()
    print("    new_x0 = max(col_lo, min(prev['bbox'][0], blk['bbox'][0]))")
    print("    new_x1 = min(col_hi, max(prev['bbox'][2], blk['bbox'][2]))")
else:
    print("No bbox-width bug detected on this page: all merged blocks stay")
    print("within their column bands (within 0.02 normalised tolerance).")
    print()
    print("The wide bbox on chunk index 4 is likely caused by a single")
    print("un-merged block that spans both columns (e.g., a section heading")
    print("or an abstract that genuinely fills the full page width).")

# ── Step 6: zoom into the specific wide chunk ─────────────────────────────
from collections import defaultdict
from ingestion.pipeline import parse_document

print(f"\n{'─'*72}")
print("ZOOM: All page-2 chunks from parse_document(), sorted by reading_order")
print(f"{'─'*72}")
all_chunks = parse_document(PDF, "arxiv_paper")
p2_chunks = [c for c in all_chunks if c.page_number == TARGET_PAGE]
p2_chunks.sort(key=lambda c: c.reading_order_index)

print(f"  {'IDX':>3}  {'TYPE':<18}  {'x0':>6} {'y0':>6} {'x1':>6} {'y1':>6}  "
      f"{'W':>6}  TEXT-PREVIEW")
for c in p2_chunks:
    x0, y0, x1, y1 = c.bbox
    w = x1 - x0
    preview = (c.text or "").replace("\n", " ").strip()[:45]
    print(f"  {c.reading_order_index:>3}  {c.chunk_type:<18}  "
          f"{x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f}  {w:.4f}  {preview}")

doc.close()
