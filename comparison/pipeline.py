"""
comparison/pipeline.py
----------------------
Orchestrator for comparing new facts against the existing knowledge graph.
"""

import logging
from pathlib import Path
from typing import List, Any, Optional
from extraction.models import Fact
from comparison.models import Relationship, RelationshipType
from comparison.vector_store import VectorStore
from comparison.storage import SQLiteStore
from comparison.clustering import CorroborationClusters
from comparison.deterministic_checks import try_deterministic_reconcile
from comparison.judge import judge_pair
from canonicalization.pipeline import canonicalize_facts

log = logging.getLogger(__name__)

# Tunable constants
COMPARISON_TOP_K = 8

def compare_new_fact(
    fact: Fact,
    store: SQLiteStore,
    vector_store: VectorStore,
    client: Any,
    embedder: Any,
    clusters: CorroborationClusters,
    top_k: int = COMPARISON_TOP_K,
    entity_store: Optional[Path] = None,
    attribute_store: Optional[Path] = None,
) -> List[Relationship]:
    """
    Ingests a single new fact, compares it against the top-k most similar existing facts,
    and records the relationships.

    Step 0 (canonicalization) is performed unconditionally here so callers never need
    to remember to call canonicalize_facts() manually before comparison.  This is the
    correct place to enforce the invariant: every fact that touches the vector store or
    the deterministic reconciler MUST have context["normalized"] populated.

    This function makes incremental ingestion work — adding fact #500 only triggers 
    comparisons against its top-k retrieved candidates, never a full rebuild.
    """
    new_relationships = []

    # 0. Canonicalize the incoming fact — entity resolution, attribute snapping,
    #    and unit normalization.  This MUST happen before vector-store indexing and
    #    before try_deterministic_reconcile() (which reads context["normalized"]).
    #    Calling it here means callers never need to remember to do it upstream.
    canonicalized = canonicalize_facts(
        [fact], embedder,
        entity_store=entity_store,
        attribute_store=attribute_store,
    )
    fact = canonicalized[0]  # replace with the canonical copy

    # 1. Retrieve candidates (query BEFORE adding so the new fact doesn't retrieve itself)
    candidate_ids = vector_store.query_similar(fact, embedder, top_k=top_k, exclude_same_document=True)
    
    # 2. Add fact to vector store and SQLite
    vector_store.add_facts([fact], embedder)
    store.save_fact(fact)
    
    if not candidate_ids:
        return new_relationships

    # 3. Filter candidates to unique cluster representatives to avoid redundant comparisons
    # against multiple facts that all corroborate each other.
    representative_candidates = set()
    for cid in candidate_ids:
        rep_id = clusters.get_cluster_representative(cid)
        representative_candidates.add(rep_id)

    # 4. Compare against each representative
    for candidate_id in representative_candidates:
        candidate_fact = store.get_fact(candidate_id)
        if not candidate_fact:
            log.warning(f"Candidate fact {candidate_id} found in vector store but missing from SQLite.")
            continue
            
        # Try deterministic check first
        judge_result = try_deterministic_reconcile(fact, candidate_fact)
        
        # Fall back to LLM judge if deterministic check returns None
        if not judge_result:
            rel = judge_pair(fact, candidate_fact, client)
        else:
            rel = Relationship(
                fact_id_a=fact.fact_id,
                fact_id_b=candidate_fact.fact_id,
                relationship_type=RelationshipType(judge_result.relationship),
                confidence=judge_result.confidence,
                justification=judge_result.justification
            )

        if rel:
            new_relationships.append(rel)
            store.save_relationship(rel)
            
            # If they corroborate, join their clusters
            if rel.relationship_type == RelationshipType.CORROBORATES:
                clusters.add_corroboration(fact.fact_id, candidate_fact.fact_id)
                
    return new_relationships
