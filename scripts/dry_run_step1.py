import os
from collections import Counter
from ingestion.pipeline import parse_document
from scripts.debug_table_bbox import draw_tables_on_page

def main():
    docs = [
        "data/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "data/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"
    ]
    
    for doc_path in docs:
        print(f"\nParsing: {doc_path}")
        doc_id = os.path.basename(doc_path).split('.')[0]
        chunks = parse_document(doc_path, doc_id)
        print(f"Total chunks: {len(chunks)}")
        
        counts = Counter(c.chunk_type for c in chunks)
        for k, v in counts.items():
            print(f"  {k}: {v}")
            
    # Debug overlay for annual report. Let's find a page with a table.
    print("\nGenerating debug overlay for annual report...")
    doc_path = docs[0]
    doc_id = os.path.basename(doc_path).split('.')[0]
    chunks = parse_document(doc_path, doc_id)
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    if table_chunks:
        # page is 0-indexed in our chunk metadata "page"
        target_page = table_chunks[0].page_number
        print(f"Found table on page {target_page}. Rendering debug overlay...")
        draw_tables_on_page(doc_path, target_page, "output/dry_run_table_debug.png")
    else:
        print("No tables found in the document!")

if __name__ == "__main__":
    main()
