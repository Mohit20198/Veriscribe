"""
scripts/debug_viz.py
--------------------
Renders pages of a PDF at 150 DPI with colored bbox overlays per chunk_type:
  blue  → text
  red   → table
  green → chart_candidate

Each box is labeled with:
  [IDX type]
  x:[x0-x1] y:[y0-y1]

readable directly from the image.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
from ingestion.pipeline import parse_document

PDF_PATH = "output/arxiv_paper.pdf"
DOC_ID   = "arxiv_paper"
DPI      = 150
OUT_DIR  = Path("output")

COLORS = {
    "text":            (0.10, 0.40, 0.85),
    "table":           (0.85, 0.10, 0.10),
    "chart_candidate": (0.05, 0.72, 0.20),
}
FILL_OPACITY = 0.12
STROKE_WIDTH = 1.5

print(f"Running parse_document …")
chunks = parse_document(PDF_PATH, DOC_ID)
print(f"  Total chunks: {len(chunks)}")

doc = fitz.open(PDF_PATH)

for page_number in [1, 2]:
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height

    shape = page.new_shape()
    page_chunks = sorted(
        [c for c in chunks if c.page_number == page_number],
        key=lambda c: c.reading_order_index,
    )

    print(f"\nPage {page_number}: {len(page_chunks)} chunk(s)")
    for c in page_chunks:
        x0n, y0n, x1n, y1n = c.bbox
        x0, y0, x1, y1 = x0n*pw, y0n*ph, x1n*pw, y1n*ph
        col = COLORS[c.chunk_type]
        r = fitz.Rect(x0, y0, x1, y1)
        shape.draw_rect(r)
        shape.finish(color=col, fill=col, fill_opacity=FILL_OPACITY, width=STROKE_WIDTH)

        # ── coordinate label inside the box ─────────────────────────────
        line1 = f"[{c.reading_order_index} {c.chunk_type[:4]}]"
        line2 = f"x:[{x0n:.3f}-{x1n:.3f}]"
        line3 = f"y:[{y0n:.3f}-{y1n:.3f}]"
        line4 = f"w={x1n-x0n:.3f}"

        label_x = x0 + 2
        label_y = y0 + 8   # first baseline
        fs = 5.5            # font size — small but legible at 150 DPI

        for i, line in enumerate([line1, line2, line3, line4]):
            page.insert_text(
                fitz.Point(label_x, label_y + i * (fs + 1)),
                line,
                fontsize=fs,
                color=col,
            )

        w_pct = (x1n - x0n) * 100
        flag = " ← SPANNING?" if w_pct > 55 else ""
        print(f"  [{c.reading_order_index:2d}] {c.chunk_type:<18} "
              f"x:[{x0n:.3f}–{x1n:.3f}] y:[{y0n:.3f}–{y1n:.3f}] "
              f"w={x1n-x0n:.3f} ({w_pct:.0f}%){flag}")

    shape.commit()

    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path = OUT_DIR / f"debug_page{page_number}.png"
    pix.save(str(out_path))
    print(f"  → Saved {out_path}")

doc.close()
print("\nDone.")
