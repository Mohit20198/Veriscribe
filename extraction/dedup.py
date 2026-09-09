"""
extraction/dedup.py
-------------------
Deduplication logic for extracted facts.
"""

from typing import List
from extraction.models import Fact

def deduplicate_facts(facts: List[Fact]) -> List[Fact]:
    """Group facts by content_hash within a single document.
    
    Keeps the first occurrence as the canonical Fact. For subsequent duplicates,
    appends their (page_number, bbox) to the `occurrences` list on the kept fact.
    """
    seen_hashes = {}
    deduped = []
    
    for fact in facts:
        # Recompute hash just to be safe
        h = fact.compute_hash()
        fact.content_hash = h
        
        occ = {
            "page_number": fact.occurrences[0]["page_number"] if fact.occurrences else 0,
            "bbox": fact.evidence_bbox
        }
        
        if h in seen_hashes:
            # Duplicate found, append occurrence to the canonical fact
            canonical_fact = seen_hashes[h]
            # Avoid adding duplicate exact occurrences (same chunk/bbox) if any edge case causes it
            if occ not in canonical_fact.occurrences:
                canonical_fact.occurrences.append(occ)
        else:
            # First time seeing this fact
            if not fact.occurrences:
                fact.occurrences = [occ]
            seen_hashes[h] = fact
            deduped.append(fact)
            
    return deduped
