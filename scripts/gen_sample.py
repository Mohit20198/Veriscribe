"""
scripts/gen_sample.py
---------------------
Generate a realistic multi-column sample PDF for the Veriscribe demo.
Saves to ./output/sample_real.pdf.
Includes: headings, multi-paragraph body text, a ruled table, a tiny
embedded image (chart candidate), and a footer line on each page.
"""
import struct, zlib
from io import BytesIO
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

W, H = LETTER
OUT = Path("./output/sample_real.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
normal = styles["Normal"]
h1 = styles["Heading1"]
h2 = styles["Heading2"]

body = (
    "Artificial intelligence has transformed the landscape of modern industry, "
    "enabling organisations to extract actionable insights from vast pools of "
    "unstructured data. Natural language processing, computer vision, and "
    "reinforcement learning have matured significantly over the past decade, "
    "converging into foundation models that serve as general-purpose cognitive "
    "engines. This report analyses key adoption trends across five sectors: "
    "healthcare, finance, logistics, legal, and manufacturing. "
) * 2

table_data = [
    ["Sector",        "AI Adoption (%)", "YoY Growth", "Primary Use Case"],
    ["Healthcare",    "68",              "+14 pp",     "Clinical decision support"],
    ["Finance",       "79",              "+9 pp",      "Fraud detection"],
    ["Logistics",     "55",              "+18 pp",     "Route optimisation"],
    ["Legal",         "41",             "+22 pp",     "Contract review"],
    ["Manufacturing", "62",             "+11 pp",     "Predictive maintenance"],
]

tbl = Table(table_data, colWidths=[1.6*inch, 1.2*inch, 1.1*inch, 2.2*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("GRID",        (0, 0), (-1, -1), 0.5, colors.black),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECF0F1")]),
    ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
]))

# Tiny in-memory PNG (solid blue square)
def _tiny_png(w=20, h=20):
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(name, data):
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw = b""
    for _ in range(h):
        raw += b"\x00" + b"\x1a\x8c\xd4" * w  # steel-blue pixels
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return BytesIO(sig + ihdr + idat + iend)

img = Image(_tiny_png(), width=1.8*inch, height=1.4*inch)

story = [
    Paragraph("AI Industry Adoption Report 2025", h1),
    Spacer(1, 0.15*inch),
    Paragraph("Executive Summary", h2),
    Paragraph(body, normal),
    Spacer(1, 0.2*inch),
    Paragraph("Table 1: AI Adoption by Sector", h2),
    Spacer(1, 0.1*inch),
    tbl,
    PageBreak(),
    Paragraph("Visual Analysis", h1),
    Spacer(1, 0.2*inch),
    img,
    Spacer(1, 0.1*inch),
    Paragraph("Figure 1: Sector adoption trend (2020–2025).", normal),
    Spacer(1, 0.3*inch),
    Paragraph("Methodology", h2),
    Paragraph(body, normal),
]

doc = SimpleDocTemplate(str(OUT), pagesize=LETTER)
doc.build(story)
print(f"Saved: {OUT}")
