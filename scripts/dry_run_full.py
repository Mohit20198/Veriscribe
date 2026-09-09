import os
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
from ingestion.pipeline import parse_document
from extraction.pipeline import extract_document_facts
from canonicalization.attribute_taxonomy import AttributeTaxonomy
from comparison.vector_store import VectorStore
from comparison.judge import judge_pair
from sentence_transformers import SentenceTransformer

def main():
    import openai
    from dotenv import load_dotenv
    load_dotenv()
    
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", "dummy_key")
    )
    
    docs = [
        "data/delhivery/02-delhivery-annual-report-fy24-excerpt.pdf",
        "data/delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf"
    ]
    
    keywords = ["revenue", "ebitda", "pat", "employees", "profit"]
    all_facts = []
    
    # ---------------------------------------------------------
    # STEP 2: Extraction
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 2: EXTRACTION (Sample)")
    print("="*50)
    
    doc_chunks_sample = {}
    for doc_path in docs:
        doc_id = os.path.basename(doc_path).split('.')[0]
        print(f"\nExtracting from {doc_id}...")
        
        # We assume chunks are already parsed or we parse them quickly
        # Wait, parsing 100 pages is slow. Let's do it if needed.
        chunks = parse_document(doc_path, doc_id)
        
        # Filter chunks
        filtered = []
        for c in chunks:
            text = (c.text or "").lower()
            if any(k in text for k in keywords):
                filtered.append(c)
        
        # Take up to 15 relevant chunks
        sample = filtered[:15]
        doc_chunks_sample[doc_id] = sample
        print(f"  Selected {len(sample)} relevant chunks out of {len(chunks)} total.")
        
        # Extract facts
        doc_facts = extract_document_facts(sample, client)
        print(f"  Extracted {len(doc_facts)} facts.")
        all_facts.extend(doc_facts)
        
        # Print a few examples
        for i, f in enumerate(doc_facts[:3]):
            print(f"  [Fact {i+1}] {f.canonical_entity} | {f.raw_attribute} = {f.value} {f.unit or ''}")
            
    # ---------------------------------------------------------
    # STEP 3: Canonicalization
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 3: CANONICALIZATION")
    print("="*50)
    
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    taxonomy = AttributeTaxonomy()
    
    for f in all_facts:
        canonical, raw = taxonomy.snap(f.raw_attribute, embedder)
        f.raw_attribute = canonical  # Mutate fact in place for the dry run pipeline
        
    print(f"Taxonomy state: {len(taxonomy.canonical_attributes)} canonical attributes identified.")
    for k in taxonomy.canonical_attributes:
        print(f"  - {k}")
        
    # ---------------------------------------------------------
    # STEP 4: Comparison
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 4: COMPARISON")
    print("="*50)
    
    store = VectorStore()
    # Clean up any existing state
    try:
        store.client.delete_collection("facts")
        store = VectorStore()
    except Exception:
        pass
        
    store.add_facts(all_facts, embedder)
    
    # We will track evaluated pairs to avoid printing A->B and B->A
    evaluated = set()
    found_corroboration = False
    found_contradiction = False
    
    fact_by_id = {f.fact_id: f for f in all_facts}
    print(f"Comparing {len(all_facts)} facts across the graph...")
    
    for fact in all_facts:
        related_ids = store.query_similar(fact, embedder, top_k=5, exclude_same_document=True)
        for cand_id in related_ids:
            if cand_id not in fact_by_id:
                continue
            cand = fact_by_id[cand_id]
            if fact.fact_id == cand.fact_id:
                continue
                
            # Skip intra-document comparisons for this summary (we want cross-document matches)
            if fact.document_id == cand.document_id:
                continue
                
            pair_id = tuple(sorted([fact.fact_id, cand.fact_id]))
            if pair_id in evaluated:
                continue
            evaluated.add(pair_id)
            
            # Run judge — may return None on API timeout or malformed response
            result = judge_pair(fact, cand, client)
            if result is None:
                continue
            
            if result.relationship_type in ["corroborates", "contradicts", "reconciled"]:
                print(f"\n[MATCH FOUND: {result.relationship_type.upper()}] (confidence: {result.confidence})")
                print(f"  Fact A ({fact.document_id}): {fact.canonical_entity} | {fact.raw_attribute} = {fact.value}")
                print(f"  Fact B ({cand.document_id}): {cand.canonical_entity} | {cand.raw_attribute} = {cand.value}")
                print(f"  Justification: {result.justification}")
                if result.relationship_type == "corroborates": found_corroboration = True
                if result.relationship_type == "contradicts": found_contradiction = True
                
    if not found_corroboration and not found_contradiction:
        print("\nNo meaningful cross-document corroboration or contradiction was found in this sample.")
        print(f"Total candidate pairs evaluated by the judge: {len(evaluated)}")

if __name__ == "__main__":
    main()
