"""
scripts/raw_table_debug.py
--------------------------
Dumps pdfplumber's raw page.lines and page.rects objects for pages 2 and 6
of output/arxiv_paper.pdf.  This is the ground truth for whether the PDF
contains vector line/rect operators that pdfplumber's lines_strict strategy
can use to detect tables.

Also runs find_tables() with every available strategy and reports results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pdfplumber

PDF = "output/arxiv_paper.pdf"
TARGET_PAGES = [2, 6]   # 1-based

print("=" * 72)
print(f"  Raw pdfplumber lines/rects debug — {PDF}")
print("=" * 72)

with pdfplumber.open(PDF) as doc:
    for page_1based in TARGET_PAGES:
        page = doc.pages[page_1based - 1]

        print(f"\n{'─'*72}")
        print(f"PAGE {page_1based}  (size: {page.width:.1f} x {page.height:.1f} pt)")
        print(f"{'─'*72}")

        # ── 1. Raw lines ──────────────────────────────────────────────────
        lines = page.lines
        print(f"\n[1] page.lines  →  {len(lines)} line object(s)")
        if lines:
            print("     Sample (first 10):")
            for ln in lines[:10]:
                print(f"       x0={ln['x0']:7.2f}  y0={ln['top']:7.2f}  "
                      f"x1={ln['x1']:7.2f}  y1={ln['bottom']:7.2f}  "
                      f"width={ln.get('width',ln.get('linewidth','?'))!r}  "
                      f"stroke={ln.get('stroking_color','?')}")
        else:
            print("     (none — no PDF line operators on this page)")

        # ── 2. Raw rects ──────────────────────────────────────────────────
        rects = page.rects
        print(f"\n[2] page.rects  →  {len(rects)} rect object(s)")
        if rects:
            print("     Sample (first 10):")
            for r in rects[:10]:
                print(f"       x0={r['x0']:7.2f}  y0={r['top']:7.2f}  "
                      f"x1={r['x1']:7.2f}  y1={r['bottom']:7.2f}  "
                      f"fill={r.get('fill','?')!r}  "
                      f"stroke={r.get('stroking_color','?')!r}")
        else:
            print("     (none — no PDF rect operators on this page)")

        # ── 3. Curve objects (sometimes used for rules) ───────────────────
        curves = page.curves
        print(f"\n[3] page.curves →  {len(curves)} curve object(s)")

        # ── 4. Edges (what table finder actually uses) ────────────────────
        edges = page.edges
        print(f"\n[4] page.edges  →  {len(edges)} edge object(s)"
              "  (horizontal + vertical ruling lines used by find_tables)")
        if edges:
            h_edges = [e for e in edges if e.get("orientation") == "h"]
            v_edges = [e for e in edges if e.get("orientation") == "v"]
            print(f"       horizontal edges: {len(h_edges)}")
            print(f"       vertical edges  : {len(v_edges)}")
            print("       First 5 edges:")
            for e in edges[:5]:
                print(f"         {e}")

        # ── 5. find_tables with every strategy ───────────────────────────
        print(f"\n[5] find_tables() results per strategy:")
        strategies = [
            ("lines_strict", "lines_strict"),
            ("lines",        "lines"),
            ("text",         "text"),
        ]
        for vs, hs in strategies:
            try:
                found = page.find_tables(table_settings={
                    "vertical_strategy":   vs,
                    "horizontal_strategy": hs,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "edge_min_length": 3,
                })
                print(f"       v={vs:<14} h={hs:<14} → {len(found)} table(s) found")
                for i, t in enumerate(found[:3]):
                    rows = t.extract()
                    nrows = len(rows) if rows else 0
                    ncols = len(rows[0]) if rows and rows[0] else 0
                    print(f"          Table {i}: bbox={t.bbox}  {nrows}R x {ncols}C")
                    if rows:
                        print(f"          Row 0: {rows[0][:4]}")
            except Exception as exc:
                print(f"       v={vs:<14} h={hs:<14} → ERROR: {exc}")

print("\nDone.")
