"""
demo.py
-------
Quick smoke-test / demo script for the Veriscribe ingestion pipeline.

Usage:
    python demo.py <path_to_pdf>

Prints:
  * Total chunk count
  * Breakdown by chunk_type
  * First 3 chunks as pretty JSON
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

# Make sure the project root is on the Python path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.pipeline import parse_document


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python demo.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_id = Path(pdf_path).stem  # use filename stem as document_id

    print(f"\n{'='*60}")
    print(f"  Veriscribe Phase-1 Demo")
    print(f"  PDF      : {pdf_path}")
    print(f"  Document : {doc_id}")
    print(f"{'='*60}\n")

    chunks = parse_document(pdf_path, document_id=doc_id)

    # Summary
    type_counts = Counter(c.chunk_type for c in chunks)
    print(f"Total chunks : {len(chunks)}")
    print("By type      :")
    for ctype, count in sorted(type_counts.items()):
        print(f"  {ctype:<20} {count}")

    # First 3 chunks
    print("\n--- First 3 chunks (JSON) ---\n")
    for i, chunk in enumerate(chunks[:3]):
        print(json.dumps(chunk.model_dump(), indent=2, default=str))
        if i < 2:
            print()


if __name__ == "__main__":
    main()
