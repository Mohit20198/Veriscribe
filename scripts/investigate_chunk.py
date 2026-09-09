"""
investigate_chunk.py
--------------------
Re-runs extraction on the Q4 earnings presentation,
then for every extracted fact whose value looks like "-452" / "(452)",
prints the source chunk's raw text or table_data verbatim.

This lets us confirm whether the source was a comparative FY23/FY24 table
and therefore whether the "-452" is an extraction attribution bug
(wrong column assigned to wrong year).
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv()

import openai
from ingestion.pipeline import parse_document
from extraction.pipeline import extract_document_facts

DOC_PATH   = "data/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"
DOC_ID     = "03-delhivery-q4-fy24-earnings-presentation"
KEYWORDS   = ["ebitda", "revenue", "profit", "pat"]

def main():
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    print("=== STEP 1: Parse document ===")
    chunks = parse_document(DOC_PATH, DOC_ID)
    # Index by chunk_id for quick lookup
    chunk_by_id = {c.chunk_id: c for c in chunks}
    print(f"Total chunks: {len(chunks)}")

    # Filter to EBITDA-relevant chunks only (keep extraction fast)
    filtered = [
        c for c in chunks
        if any(kw in (c.text or "").lower() or
               any(kw in str(c.table_data).lower() if c.table_data else False
                   for kw in KEYWORDS)
               for kw in KEYWORDS)
    ][:20]
    print(f"Filtered to {len(filtered)} chunks for extraction.\n")

    print("=== STEP 2: Extract facts ===")
    facts = extract_document_facts(filtered, client)
    print(f"Extracted {len(facts)} facts.\n")

    print("=== STEP 3: Hunt for the -452 / (452) fact ===")
    targets = [
        f for f in facts
        if "452" in str(f.value).replace(",", "").replace(" ", "")
    ]

    if not targets:
        print("No fact with '452' in value found. Printing ALL facts with EBITDA attribute:")
        targets = [f for f in facts if "ebitda" in f.raw_attribute.lower()]

    for fact in targets:
        print(f"\n{'='*60}")
        print(f"Fact:")
        print(f"  entity         : {fact.canonical_entity}")
        print(f"  raw_attribute  : {fact.raw_attribute}")
        print(f"  value          : {fact.value}")
        print(f"  temporal_scope : {fact.temporal_scope}")
        print(f"  evidence_quote : {fact.evidence_quote}")
        print(f"  source_chunk_id: {fact.source_chunk_id}")

        chunk = chunk_by_id.get(fact.source_chunk_id)
        if chunk is None:
            print("  [SOURCE CHUNK NOT FOUND IN INDEX]")
            continue

        print(f"\nSource chunk (page {chunk.page_number}, type={chunk.chunk_type}):")
        if chunk.chunk_type == "text":
            print(f"  raw text:\n{chunk.text}")
        elif chunk.chunk_type == "table":
            print(f"  table caption: {chunk.text or '(none)'}")
            print(f"  table_data ({len(chunk.table_data)} rows):")
            for row in chunk.table_data:
                print("    " + " | ".join(str(cell or "") for cell in row))
        else:
            print(f"  image_path: {chunk.image_path}")

if __name__ == "__main__":
    main()
