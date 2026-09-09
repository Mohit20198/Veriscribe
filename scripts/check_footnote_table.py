"""
Inspect the footnote 'table' detected on page 2 and run extraction against it.
"""
import os, sys, json
os.environ.setdefault("OPENROUTER_API_KEY", open(".env").read().split("OPENROUTER_API_KEY=")[1].split()[0])

import pdfplumber
from ingestion.tables import extract_tables
from ingestion.models import Chunk
from extraction.extractor import extract_facts_from_chunk
from openai import OpenAI

def main():
    with pdfplumber.open("output/arxiv_paper.pdf") as pdf:
        page = pdf.pages[1]  # page 2 (0-indexed)
        pw, ph = page.width, page.height
        tables = extract_tables(page)

    print(f"Tables detected on page 2: {len(tables)}")
    for i, tbl in enumerate(tables):
        x0, top, x1, bottom = tbl["raw_bbox"]
        area = ((x1 - x0) * (bottom - top)) / (pw * ph)
        print(f"\n--- Table {i+1} ---")
        print(f"  raw_bbox: {tbl['raw_bbox']}")
        print(f"  area: {area:.1%}")
        print(f"  rows x cols: {len(tbl['table_data'])} x {len(tbl['table_data'][0]) if tbl['table_data'] else 0}")
        print(f"  table_data:")
        for row in tbl["table_data"]:
            print(f"    {row}")

    # Now run extraction on the footnote table (the one with URLs)
    # Identify it as the one with high area or URL content
    for i, tbl in enumerate(tables):
        flat_cells = " ".join(cell for row in tbl["table_data"] for cell in row)
        url_count = flat_cells.count("http")
        print(f"\nTable {i+1}: url_count={url_count}, flat_cells preview: {flat_cells[:120]}")

        chunk = Chunk(
            document_id="arxiv_doc",
            page_number=2,
            chunk_type="table",
            text=None,
            bbox=tbl["bbox"],
            reading_order_index=i,
            table_data=tbl["table_data"],
        )
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        print(f"  -> Running extraction on Table {i+1}...")
        facts = extract_facts_from_chunk(chunk, context_window="", client=client)
        print(f"  -> Got {len(facts)} fact(s):")
        for f in facts:
            print(f"     {f.statement!r} | entity={f.canonical_entity!r} | attr={f.raw_attribute!r} | val={f.value!r} | evidence={f.evidence_quote!r}")

if __name__ == "__main__":
    main()
