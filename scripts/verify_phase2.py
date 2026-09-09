import sys
import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding='utf-8')

from ingestion.pipeline import parse_document
from extraction.extractor import extract_facts_from_chunk

logging.basicConfig(level=logging.INFO)

def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        sys.exit(1)
        
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    
    pdf_path = "output/arxiv_paper.pdf"
    print(f"Running Phase 1 ingestion on {pdf_path}...")
    chunks = parse_document(pdf_path, document_id="arxiv_doc")
    print(f"Found {len(chunks)} chunks.")
    
    # Find specific chunks
    # a) Qualifying clause test: page 3 text containing "Wikipedia contexts"
    qualifying_chunk = next((c for c in chunks if c.chunk_type == "text" and "Wikipedia contexts" in (c.text or "")), None)
    
    # b) Chart test: Figure 4 grouped bar chart
    chart_chunk = next((c for c in chunks if c.chunk_type == "chart_candidate" and "Performance of Mistral 7B" in (c.text or "")), None)
    if not chart_chunk:
         chart_chunk = next((c for c in chunks if c.chunk_type == "chart_candidate" and c.page_number == 3), None)
    
    # c) Table test: Table 1 (page 2) or Table 2 (page 3)
    table_chunk = next((c for c in chunks if c.chunk_type == "table" and c.page_number in (2, 3)), None)
    if not table_chunk:
        # Fallback if lines_strict missed it (as documented in README)
        table_chunk = next((c for c in chunks if c.chunk_type == "text" and c.page_number in (2, 3) and ("Model Architecture" in (c.text or "") or "Table 1" in (c.text or ""))), None)
        if not table_chunk:
             table_chunk = next((c for c in chunks if c.chunk_type == "text" and "dim" in (c.text or "") and "n_layers" in (c.text or "")), None)
        
    targets = [
        ("QUALIFYING CLAUSE TEST", qualifying_chunk),
        ("CHART TEST", chart_chunk),
        ("TABLE TEST", table_chunk)
    ]
    
    for test_name, chunk in targets:
        print(f"\n==========================================")
        print(f"--- {test_name} ---")
        if not chunk:
            print(f"WARNING: Could not find target chunk for {test_name}")
            continue
            
        print(f"Chunk info: type={chunk.chunk_type}, page={chunk.page_number}, text snippet: {chunk.text[:100] if chunk.text else 'None'}")
        
        # Build a pseudo context window from surrounding text chunks
        context_window = ""
        idx = chunks.index(chunk)
        if idx > 0 and chunks[idx-1].text:
            context_window += chunks[idx-1].text + "\n\n"
        if idx < len(chunks) - 1 and chunks[idx+1].text:
            context_window += chunks[idx+1].text
            
        try:
            facts = extract_facts_from_chunk(chunk, context_window=context_window, client=client)
            if not facts:
                print("-> 0 facts extracted. Check if LLM returned empty list or malformed JSON.")
            for i, fact in enumerate(facts):
                print(f"Fact {i+1}:")
                print(json.dumps(fact.model_dump(), indent=2))
        except Exception as e:
            print(f"Exception during extraction: {e}")

if __name__ == "__main__":
    main()
