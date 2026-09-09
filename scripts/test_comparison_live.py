"""
scripts/test_comparison_live.py
-------------------------------
Live end-to-end run of Phase 4 comparison against the LLM judge.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from extraction.models import Fact
from comparison.storage import SQLiteStore
from comparison.vector_store import VectorStore
from comparison.clustering import CorroborationClusters
from comparison.pipeline import compare_new_fact

def main():
    db_path = "output/test_veriscribe.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    chroma_dir = "output/test_chroma"
    import shutil
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    store = SQLiteStore(db_path=db_path)
    vector_store = VectorStore(persist_directory=chroma_dir)
    
    # Initialize clusters from existing SQLite relationships
    existing_corroborations = store.get_all_corroborations()
    clusters = CorroborationClusters(initial_pairs=existing_corroborations)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    # 1. Base fact
    fact_a = Fact(
        fact_id="fact-base",
        document_id="mistral-7b-v0.1",
        statement="Mistral 7B has 32 layers.",
        canonical_entity="Mistral_7B",
        raw_attribute="n_layers",
        value="32",
        unit=None,
        temporal_scope={},
        confidence=0.98,
        source_chunk_id="chunk1",
        source_type="text",
        evidence_bbox=(0,0,0,0),
        evidence_context_window="Table 1: Architectural details of Mistral 7B. We use 32 layers and 32 attention heads.",
        content_hash="hash-a"
    )
    # Mock normalizer output
    fact_a.context["normalized"] = {"parse_ok": True, "normalized_unit": None, "normalized_value": 32.0}
    
    # 2. Add base fact to graph
    print("Ingesting base fact...")
    store.save_fact(fact_a)
    vector_store.add_facts([fact_a], embedder)

    def run_comparison(test_name, fact_b):
        print(f"\n=======================================================")
        print(f"RUNNING TEST: {test_name}")
        print(f"=======================================================")
        rels = compare_new_fact(fact_b, store, vector_store, client, embedder, clusters)
        if not rels:
            print("NO RELATIONSHIPS FOUND!")
            return

        for rel in rels:
            print(f"Relationship: {rel.relationship_type.value}")
            print(f"Confidence:   {rel.confidence}")
            print(f"Justification:\n{rel.justification}")
            
    # Case 1: Reconciled (Original test case)
    fact_reconciled = Fact(
        fact_id="fact-reconciled",
        document_id="mistral-draft",
        statement="Mistral 7B has 24 layers.",
        canonical_entity="Mistral_7B",
        raw_attribute="n_layers",
        value="24",
        unit=None,
        temporal_scope={},
        confidence=0.95,
        source_chunk_id="chunk2",
        source_type="text",
        evidence_bbox=(0,0,0,0),
        evidence_context_window="In an earlier draft of this model, Mistral 7B has 24 layers. This was scaled up to 32 before release.",
        content_hash="hash-reconciled"
    )
    fact_reconciled.context["normalized"] = {"parse_ok": True, "normalized_unit": None, "normalized_value": 24.0}
    run_comparison("RECONCILED (Draft vs Final)", fact_reconciled)

    # Case 2: Corroboration
    fact_corroborate = Fact(
        fact_id="fact-corroborate",
        document_id="mistral-blog",
        statement="Mistral 7B's architecture uses 32 transformer layers.",
        canonical_entity="Mistral_7B",
        raw_attribute="n_layers",
        value="32",
        unit=None,
        temporal_scope={},
        confidence=0.99,
        source_chunk_id="chunk3",
        source_type="text",
        evidence_bbox=(0,0,0,0),
        evidence_context_window="As detailed in our architecture overview, Mistral 7B's architecture uses 32 transformer layers and group-query attention.",
        content_hash="hash-corroborate"
    )
    fact_corroborate.context["normalized"] = {"parse_ok": True, "normalized_unit": None, "normalized_value": 32.0}
    run_comparison("CORROBORATION", fact_corroborate)

    # Case 3: Genuine Contradiction
    fact_contradict = Fact(
        fact_id="fact-contradict-genuine",
        document_id="mistral-fake-news",
        statement="Mistral 7B has 40 layers, per the final production build.",
        canonical_entity="Mistral_7B",
        raw_attribute="n_layers",
        value="40",
        unit=None,
        temporal_scope={},
        confidence=0.90,
        source_chunk_id="chunk4",
        source_type="text",
        evidence_bbox=(0,0,0,0),
        evidence_context_window="Despite rumors of a smaller model, Mistral 7B has 40 layers, per the final production build released to enterprise customers.",
        content_hash="hash-contradict-genuine"
    )
    fact_contradict.context["normalized"] = {"parse_ok": True, "normalized_unit": None, "normalized_value": 40.0}
    run_comparison("GENUINE CONTRADICTION", fact_contradict)

if __name__ == "__main__":
    main()
