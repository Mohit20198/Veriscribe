"""
orchestration/status.py
-----------------------
Shared logic for retrieving the overall status of the knowledge store.
"""
from typing import Dict, Any, List
from comparison.storage import SQLiteStore

def get_system_status(store: SQLiteStore) -> Dict[str, Any]:
    """
    Returns a summary dictionary of the system's current state.
    """
    doc_ids = store.get_document_ids()
    fact_cnt = store.count_facts()
    rel_cnts = store.count_relationships()

    documents_summary: List[Dict[str, Any]] = []
    for did in doc_ids:
        facts = store.get_facts_by_document(did)
        documents_summary.append({
            "document_id": did,
            "fact_count": len(facts)
        })

    return {
        "documents": documents_summary,
        "total_facts": fact_cnt,
        "relationships_by_type": rel_cnts
    }
