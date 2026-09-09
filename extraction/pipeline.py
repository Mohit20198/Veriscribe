"""
extraction/pipeline.py
----------------------
Orchestrates fact extraction across the document.
"""

from typing import Any, List
import logging
from ingestion.models import Chunk
from extraction.models import Fact
from extraction.extractor import extract_facts_from_chunk
from extraction.dedup import deduplicate_facts

log = logging.getLogger(__name__)

def extract_document_facts(chunks: List[Chunk], client: Any) -> List[Fact]:
    """Orchestrate the extraction pipeline over a list of chunks."""
    all_facts: List[Fact] = []
    
    # Group by page to build context window
    page_to_chunks = {}
    for c in chunks:
        page_to_chunks.setdefault(c.page_number, []).append(c)
        
    for page, p_chunks in page_to_chunks.items():
        # Ensure chunks are sorted by reading_order_index
        p_chunks.sort(key=lambda x: x.reading_order_index)
        
        for i, chunk in enumerate(p_chunks):
            # Build context window from previous and next chunk text
            context_parts = []
            if i > 0 and p_chunks[i-1].chunk_type == "text" and p_chunks[i-1].text:
                context_parts.append(p_chunks[i-1].text)
            if chunk.chunk_type == "text" and chunk.text:
                context_parts.append(f">>> {chunk.text} <<<")
            if i < len(p_chunks) - 1 and p_chunks[i+1].chunk_type == "text" and p_chunks[i+1].text:
                context_parts.append(p_chunks[i+1].text)
                
            context_window = "\n\n".join(context_parts)
            # Cap at ~300 chars roughly if needed, but doing it by adjacent chunks is often better.
            if len(context_window) > 1000:
                # Basic crop just to be safe if chunks are huge
                context_window = context_window[:1000]
                
            try:
                facts = extract_facts_from_chunk(chunk, context_window, client)
                all_facts.extend(facts)
            except Exception as e:
                log.error(f"Unexpected error extracting chunk {chunk.chunk_id}: {e}")

    # Deduplicate facts globally across the document
    deduped = deduplicate_facts(all_facts)
    
    log.info(f"extract_document_facts finished: {len(deduped)} facts from {len(chunks)} chunks.")
    return deduped
