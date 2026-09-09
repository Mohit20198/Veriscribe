"""
scripts/diagnose.py
--------------------
Comprehensive diagnostic for Veriscribe Phase-1 against a real PDF.

Reports:
  1. Total chunks + breakdown by type
  2. Per-page column count (detects multi-column pages)
  3. Header/footer bands detected + sample raw block text that was stripped
  4. Table chunks with table_data sample
  5. Chart candidate chunks
  6. First 5 chunks (JSON) for TWO specific pages:
       - best candidate for multi-column or table page
       - best candidate for chart/image page
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.layout import (
    _cluster_midpoints,
    detect_header_footer_zones,
    order_blocks,
)
from ingestion.pipeline import parse_document

PDF = "output/arxiv_paper.pdf"
DOC_ID = "arxiv_paper"

print("=" * 70)
print(f"  Veriscribe Phase-1 Diagnostic")
print(f"  PDF : {PDF}")
print("=" * 70)

# --------------------------------------------------------------------------
# Run parse_document
# --------------------------------------------------------------------------
chunks = parse_document(PDF, DOC_ID)

type_counts = Counter(c.chunk_type for c in chunks)
pages_seen = sorted({c.page_number for c in chunks})

print(f"\nTotal chunks : {len(chunks)}")
print(f"Pages        : {len(pages_seen)} ({min(pages_seen)}–{max(pages_seen)})")
print("By type:")
for ct, n in sorted(type_counts.items()):
    print(f"  {ct:<22} {n}")

# --------------------------------------------------------------------------
# A. Column detection per page
# --------------------------------------------------------------------------
print("\n" + "─" * 70)
print("A. COLUMN DETECTION (per page)")
print("─" * 70)

fitz_doc = fitz.open(PDF)
plumber_doc = pdfplumber.open(PDF)
all_fitz_pages = [fitz_doc[i] for i in range(len(fitz_doc))]
header_bands, footer_bands = detect_header_footer_zones(all_fitz_pages)

multi_col_pages = []
page_col_counts = {}
for pg_idx, fitz_page in enumerate(all_fitz_pages[:20]):   # inspect first 20 pages
    page_num = pg_idx + 1
    pw, ph = fitz_page.rect.width, fitz_page.rect.height
    if pw == 0 or ph == 0:
        continue
    raw = fitz_page.get_text("blocks")
    text_blocks = [b for b in raw if b[6] == 0 and b[4].strip()]
    x_mids = [((b[0] + b[2]) / 2) / pw for b in text_blocks]   # normalised
    if x_mids:
        col_centres = _cluster_midpoints([m * pw for m in x_mids], pw)
        n_cols = len(col_centres)
    else:
        n_cols = 0
    page_col_counts[page_num] = n_cols
    if n_cols >= 2:
        multi_col_pages.append((page_num, n_cols))

if multi_col_pages:
    print(f"Multi-column pages detected: {multi_col_pages[:10]}")
    best_multi_col_page = multi_col_pages[0][0]
else:
    print("No multi-column pages detected in first 20 pages — all single-column.")
    best_multi_col_page = None

for pg, nc in list(page_col_counts.items())[:10]:
    print(f"  page {pg:2d}: {nc} column(s)")

# --------------------------------------------------------------------------
# B. Header / footer stripping
# --------------------------------------------------------------------------
print("\n" + "─" * 70)
print("B. HEADER / FOOTER STRIPPING")
print("─" * 70)
print(f"Header bands (normalised y): {header_bands}")
print(f"Footer bands (normalised y): {footer_bands}")

# Find raw blocks that fall in these zones — show sample from page 2
SAMPLE_PAGE_IDX = 1  # page 2 (0-based)
sample_page = all_fitz_pages[SAMPLE_PAGE_IDX]
pw_s, ph_s = sample_page.rect.width, sample_page.rect.height
raw_s = sample_page.get_text("blocks")

print(f"\nBlocks on page 2 that were STRIPPED by header/footer detection:")
stripped_count = 0
for b in raw_s:
    if b[6] != 0:
        continue
    y_mid = ((b[1] + b[3]) / 2) / ph_s
    in_header = any(lo <= y_mid <= hi for lo, hi in header_bands)
    in_footer = any(lo <= y_mid <= hi for lo, hi in footer_bands)
    if in_header or in_footer:
        zone = "HEADER" if in_header else "FOOTER"
        text_preview = b[4].replace("\n", " ").strip()[:80]
        print(f"  [{zone}] y_mid={y_mid:.4f}  text='{text_preview}'")
        stripped_count += 1

if stripped_count == 0:
    # Try all pages
    found_strip_example = False
    for pg_idx, fitz_page in enumerate(all_fitz_pages):
        pw_p, ph_p = fitz_page.rect.width, fitz_page.rect.height
        if pw_p == 0 or ph_p == 0:
            continue
        for b in fitz_page.get_text("blocks"):
            if b[6] != 0:
                continue
            y_mid = ((b[1] + b[3]) / 2) / ph_p
            in_header = any(lo <= y_mid <= hi for lo, hi in header_bands)
            in_footer = any(lo <= y_mid <= hi for lo, hi in footer_bands)
            if in_header or in_footer:
                zone = "HEADER" if in_header else "FOOTER"
                text_preview = b[4].replace("\n", " ").strip()[:80]
                print(f"  [page {pg_idx+1}][{zone}] y_mid={y_mid:.4f}  text='{text_preview}'")
                found_strip_example = True
                if stripped_count >= 5:
                    break
                stripped_count += 1
        if found_strip_example and stripped_count >= 5:
            break
    if not found_strip_example:
        print("  (No blocks matched the detected bands on first pass — "
              "bands may be at page extremes with no text there)")

# --------------------------------------------------------------------------
# C. Table chunks + table_data
# --------------------------------------------------------------------------
print("\n" + "─" * 70)
print("C. TABLE CHUNKS")
print("─" * 70)

table_chunks = [c for c in chunks if c.chunk_type == "table"]
print(f"Table chunks found: {len(table_chunks)}")

if table_chunks:
    tc = table_chunks[0]
    print(f"\nFirst table — page {tc.page_number}, bbox={tuple(f'{v:.4f}' for v in tc.bbox)}")
    print(f"  Rows x Cols: {len(tc.table_data)} x {max(len(r) for r in tc.table_data)}")
    print("  table_data (first 4 rows):")
    for row in tc.table_data[:4]:
        print(f"    {row}")
else:
    print("  lines_strict found NO tables in this PDF.")
    print("  → This is the known limitation: borderless/whitespace-aligned")
    print("    tables are invisible to pdfplumber's lines_strict strategy.")
    print("    arXiv papers typically use LaTeX booktabs (no vertical lines)")
    print("    which lines_strict cannot detect.")

# --------------------------------------------------------------------------
# D. Chart candidate chunks
# --------------------------------------------------------------------------
print("\n" + "─" * 70)
print("D. CHART CANDIDATES")
print("─" * 70)
chart_chunks = [c for c in chunks if c.chunk_type == "chart_candidate"]
print(f"Chart candidate chunks found: {len(chart_chunks)}")
for cc in chart_chunks[:3]:
    cap = (cc.text or "")[:60] if cc.text else "(no caption)"
    print(f"  page {cc.page_number}  bbox={tuple(f'{v:.4f}' for v in cc.bbox)}")
    print(f"  caption='{cap}'")
    print(f"  image_path={cc.image_path}")

# --------------------------------------------------------------------------
# E. First 5 chunks for TWO selected pages (JSON)
# --------------------------------------------------------------------------
by_page = defaultdict(list)
for c in chunks:
    by_page[c.page_number].append(c)

# Pick page with a table OR multi-column
if table_chunks:
    page_a = table_chunks[0].page_number
    page_a_label = f"page {page_a} (has table)"
elif best_multi_col_page:
    page_a = best_multi_col_page
    page_a_label = f"page {page_a} (multi-column)"
else:
    page_a = pages_seen[0]
    page_a_label = f"page {page_a} (first page)"

# Pick page with a chart
if chart_chunks:
    page_b = chart_chunks[0].page_number
    page_b_label = f"page {page_b} (has chart_candidate)"
else:
    # Just pick a different page
    page_b = pages_seen[min(5, len(pages_seen) - 1)]
    page_b_label = f"page {page_b} (sample)"

print("\n" + "─" * 70)
print(f"E. FIRST 5 CHUNKS (JSON) — {page_a_label}")
print("─" * 70)
page_a_chunks = sorted(by_page[page_a], key=lambda c: c.reading_order_index)[:5]
for c in page_a_chunks:
    d = c.model_dump()
    # Truncate long text for readability
    if d.get("text") and len(d["text"]) > 200:
        d["text"] = d["text"][:200] + "…[truncated]"
    # Truncate table_data
    if d.get("table_data"):
        d["table_data"] = d["table_data"][:3]
    print(json.dumps(d, indent=2, default=str))
    print()

print("─" * 70)
print(f"E2. FIRST 5 CHUNKS (JSON) — {page_b_label}")
print("─" * 70)
page_b_chunks = sorted(by_page[page_b], key=lambda c: c.reading_order_index)[:5]
for c in page_b_chunks:
    d = c.model_dump()
    if d.get("text") and len(d["text"]) > 200:
        d["text"] = d["text"][:200] + "…[truncated]"
    if d.get("table_data"):
        d["table_data"] = d["table_data"][:3]
    print(json.dumps(d, indent=2, default=str))
    print()

fitz_doc.close()
plumber_doc.close()
