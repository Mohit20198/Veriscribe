"""
scripts/run_phase2.py
---------------------
Run Phase 2 (Fact Extraction) on the chunks from Phase 1.
"""

import sys
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

from ingestion.pipeline import parse_document
from extraction.pipeline import extract_document_facts

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_phase2.py path/to/pdf")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    load_dotenv()
    
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)
        
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    # 1. Ingestion Phase
    print(f"Running Phase 1 ingestion on {pdf_path}...")
    chunks = parse_document(pdf_path, document_id="demo_doc")
    print(f"Found {len(chunks)} chunks.")
    
    # Check if there are any table or chart chunks to ensure we demonstrate them
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    chart_chunks = [c for c in chunks if c.chunk_type == "chart_candidate"]
    
    print(f"Includes {len(table_chunks)} table chunks and {len(chart_chunks)} chart chunks.")
    
    # Limit chunks for the demo so we don't blow through tokens, but ensure we 
    # grab at least one of each type if available.
    demo_chunks = []
    
    if table_chunks:
        demo_chunks.append(table_chunks[0])
    if chart_chunks:
        demo_chunks.append(chart_chunks[0])
        
    text_chunks = [c for c in chunks if c.chunk_type == "text"]
    demo_chunks.extend(text_chunks[:3]) # 3 text chunks
    
    # 2. Extraction Phase
    print(f"\nRunning Phase 2 extraction on {len(demo_chunks)} sampled chunks...")
    facts = extract_document_facts(demo_chunks, client)
    
    print(f"\nExtracted {len(facts)} facts.")
    for i, fact in enumerate(facts):
        print(f"\n--- FACT {i+1} ---")
        print(json.dumps(fact.model_dump(), indent=2))
        
if __name__ == "__main__":
    main()
