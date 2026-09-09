"""
scripts/debug_table_bbox.py
----------------------------
Renders a debug overlay image for each tested page, drawing the
detected table bboxes in red. Run with:
    python scripts/debug_table_bbox.py
"""
import sys
import pdfplumber
from ingestion.tables import extract_tables, _candidate_regions

def draw_tables_on_page(pdf_path, page_num, output_path):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]

        # Show candidate regions before table extraction (in blue)
        candidates = _candidate_regions(page)
        print(f"\nPage {page_num}: {len(candidates)} candidate region(s):")
        for i, cbox in enumerate(candidates):
            x0, top, x1, bottom = cbox
            area = ((x1 - x0) * (bottom - top)) / (page.width * page.height)
            print(f"  Candidate {i+1}: bbox={cbox}, area={area:.1%}")

        # Detected tables (in red)
        tables = extract_tables(page)
        print(f"Page {page_num}: {len(tables)} table(s) detected:")
        for i, tbl in enumerate(tables):
            raw = tbl["raw_bbox"]
            norm = tbl["bbox"]
            x0, top, x1, bottom = raw
            area = ((x1 - x0) * (bottom - top)) / (page.width * page.height)
            rows = tbl["table_data"]
            print(f"  Table {i+1}: norm_bbox={norm}, area={area:.1%}, "
                  f"{len(rows)} rows x {len(rows[0]) if rows else 0} cols")

        im = page.to_image(resolution=150)

        # Draw candidate regions in blue
        for cbox in candidates:
            im.draw_rect(cbox, stroke="blue", stroke_width=2, fill=None)

        # Draw detected tables in red (on top)
        for tbl in tables:
            im.draw_rect(tbl["raw_bbox"], stroke="red", stroke_width=3, fill=None)

        im.save(output_path, format="PNG")
        print(f"  -> Saved {output_path}  (blue=candidates, red=detected tables)")

def main():
    pdf_path = "output/arxiv_paper.pdf"
    print("=" * 60)
    draw_tables_on_page(pdf_path, 2, "output/debug_page2_tables.png")
    print("=" * 60)
    draw_tables_on_page(pdf_path, 4, "output/debug_page4_tables.png")

if __name__ == "__main__":
    main()
