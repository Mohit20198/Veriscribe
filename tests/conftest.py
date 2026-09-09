"""
tests/conftest.py
-----------------
Pytest fixtures for the Veriscribe ingestion test suite.

Generates two synthetic PDFs using reportlab so the tests have no
dependency on external files and run deterministically on any machine.

PDF A — single-column, text-only, 3 pages (one page has a repeating footer).
PDF B — two-column layout with a simple ruled table on page 1 and an
         embedded image (chart candidate) on page 2.
"""

from __future__ import annotations

import pathlib
import tempfile
from io import BytesIO
from typing import Generator

import pytest

# ---------------------------------------------------------------------------
# reportlab imports — optional dependency for test fixtures only
# ---------------------------------------------------------------------------
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as rl_canvas

W, H = LETTER   # 612 x 792 pt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pdf(story, path: pathlib.Path) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    doc.build(story)


# ---------------------------------------------------------------------------
# Synthetic PDF A — single-column, multi-page text
# ---------------------------------------------------------------------------

def _build_pdf_a(path: pathlib.Path) -> None:
    """3-page text-only PDF with repeated footer text."""
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    body_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. "
    ) * 5

    story = []
    for page_num in range(1, 4):
        story.append(Paragraph(f"Page {page_num} — Section Heading", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(body_text, normal))
        story.append(Spacer(1, 0.5 * inch))
        # Simulate footer by adding small text at bottom — reportlab places it
        # inline here; header/footer detection should still pick it up if it
        # repeats in the same y-region.
        if page_num < 3:
            from reportlab.platypus import PageBreak
            story.append(PageBreak())

    _write_pdf(story, path)


# ---------------------------------------------------------------------------
# Synthetic PDF B — two-column layout with table + image
# ---------------------------------------------------------------------------

def _build_pdf_b(path: pathlib.Path) -> None:
    """2-page PDF: page 1 has a ruled table; page 2 has a 1×1 pixel image."""
    styles = getSampleStyleSheet()

    # --- Page 1: table
    table_data = [
        ["Company", "Revenue ($M)", "Growth (%)"],
        ["Acme Corp", "1,200", "12.5"],
        ["Beta Ltd", "980", "8.3"],
        ["Gamma Inc", "2,100", "21.0"],
    ]
    tbl = Table(table_data, colWidths=[2.5 * inch, 2 * inch, 1.5 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )

    # --- Page 2: image (tiny PNG via BytesIO so we have no file dependency)
    import struct, zlib

    def _tiny_png(width: int = 10, height: int = 10) -> BytesIO:
        """Create a minimal valid PNG in memory."""
        def make_chunk(name: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + name + data
            crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
            return c + crc

        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr = make_chunk(b"IHDR", ihdr_data)

        raw_rows = b""
        for _ in range(height):
            row = b"\x00" + b"\xff\x00\x00" * width  # filter byte + red pixels
            raw_rows += row
        compressed = zlib.compress(raw_rows)
        idat = make_chunk(b"IDAT", compressed)
        iend = make_chunk(b"IEND", b"")
        return BytesIO(signature + ihdr + idat + iend)

    png_buf = _tiny_png()
    img = Image(png_buf, width=1.5 * inch, height=1.5 * inch)

    from reportlab.platypus import PageBreak

    story = [
        Paragraph("Financial Overview", styles["Heading1"]),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "The table below summarises quarterly financial performance.", styles["Normal"]
        ),
        Spacer(1, 0.15 * inch),
        tbl,
        PageBreak(),
        Paragraph("Figure Analysis", styles["Heading1"]),
        Spacer(1, 0.2 * inch),
        img,
        Spacer(1, 0.1 * inch),
        Paragraph("Figure 1: Revenue trend chart.", styles["Normal"]),
    ]
    _write_pdf(story, path)


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pdf_a(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Path to synthetic single-column text PDF (3 pages)."""
    p = tmp_path_factory.mktemp("pdfs") / "sample_a.pdf"
    _build_pdf_a(p)
    return p


@pytest.fixture(scope="session")
def pdf_b(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Path to synthetic two-column PDF with table + image (2 pages)."""
    p = tmp_path_factory.mktemp("pdfs") / "sample_b.pdf"
    _build_pdf_b(p)
    return p


# ---------------------------------------------------------------------------
# Synthetic PDF C — full-width heading + genuine 2-column body text
# ---------------------------------------------------------------------------

def _build_pdf_c(path: pathlib.Path) -> None:
    """Single-page PDF with:

    * A full-width heading spanning the entire page width.
    * Two narrow side-by-side text columns below the heading.

    Used to regression-test that full-width blocks keep their natural bbox
    (not clamped to a column band) and that real 2-column text is detected.
    """
    c = rl_canvas.Canvas(str(path), pagesize=LETTER)
    page_w, page_h = LETTER   # 612 x 792 pt

    margin = 0.75 * inch      # 54 pt
    col_gap = 0.25 * inch     # 18 pt gap between columns
    col_w = (page_w - 2 * margin - col_gap) / 2   # ~234 pt each

    # ── Full-width heading (x: margin → page_w - margin) ──────────────────
    heading_y = page_h - margin - 20
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, heading_y, "Full-Width Heading Spanning Both Columns")

    # ── Left column body text ──────────────────────────────────────────────
    left_x = margin
    body_top = heading_y - 30
    body_lines = [
        "Left column line one.",
        "Left column line two.",
        "Left column line three.",
        "Left column line four.",
        "Left column line five.",
    ]
    c.setFont("Helvetica", 10)
    for i, line in enumerate(body_lines):
        c.drawString(left_x, body_top - i * 14, line)

    # ── Right column body text ─────────────────────────────────────────────
    right_x = margin + col_w + col_gap
    right_lines = [
        "Right column line one.",
        "Right column line two.",
        "Right column line three.",
        "Right column line four.",
        "Right column line five.",
    ]
    for i, line in enumerate(right_lines):
        c.drawString(right_x, body_top - i * 14, line)

    c.save()


@pytest.fixture(scope="session")
def pdf_c(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Path to synthetic PDF with one full-width heading + 2-column body text."""
    p = tmp_path_factory.mktemp("pdfs") / "sample_c.pdf"
    _build_pdf_c(p)
    return p
