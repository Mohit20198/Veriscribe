import json
import uuid
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from comparison.storage import SQLiteStore
from comparison.models import Relationship, RelationshipType
from extraction.models import Fact

def inject_contradiction():
    store = SQLiteStore("output/veriscribe.db")
    
    # We will construct a fact that contradicts an existing fact.
    # Existing fact to target: "EBITDA for FY24 is ₹1,266Mn." (from document 02-delhivery-annual-report-fy24)
    # Let's find it.
    target_fact_id = "755ad359-ec19-43fa-898f-596be784bd73"
    target_fact = store.get_fact(target_fact_id)
    if not target_fact:
        print("Target fact not found. Looking for any EBITDA fact...")
        # fallback to find one
        with store._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact_id FROM facts WHERE statement LIKE '%EBITDA%'")
            res = cursor.fetchone()
            if res:
                target_fact_id = res[0]
                target_fact = store.get_fact(target_fact_id)
            else:
                print("No EBITDA fact found to contradict!")
                return

    print(f"Target fact: {target_fact.statement} (Value: {target_fact.value})")

    # Create a contradicting fact
    fake_fact_id = str(uuid.uuid4())
    fake_fact = Fact(
        fact_id=fake_fact_id,
        document_id="03-delhivery-q4-fy24",  # Different document
        statement="EBITDA for FY24 is ₹2,500Mn.",
        canonical_entity=target_fact.canonical_entity,
        raw_attribute=target_fact.raw_attribute,
        value="2,500",
        unit=target_fact.unit,
        temporal_scope=target_fact.temporal_scope,
        confidence=0.9,
        evidence_quote="Our adjusted EBITDA for FY24 came in at ₹2,500Mn.",
        evidence_context_window="Based on new accounting adjustments, our adjusted EBITDA for FY24 came in at ₹2,500Mn, which is significantly different from initial reports.",
        source_type="text",
        source_chunk_id="synthetic-contradiction-chunk",
        evidence_bbox=(0.0, 0.0, 0.0, 0.0)
    )

    store.save_fact(fake_fact)
    print(f"Injected fake fact: {fake_fact_id}")

    # Now let's run the judge on this pair.
    # We don't want to rely on the deterministic logic since values differ, it would go to judge.
    from comparison.judge import judge_pair
    import openai
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

    rel = judge_pair(target_fact, fake_fact, client)
    if rel:
        print(f"Judge result: {rel.relationship_type.value} (Conf: {rel.confidence})")
        print(f"Justification: {rel.justification}")
        store.save_relationship(rel)
        print("Saved contradiction relationship.")
    else:
        print("Judge failed.")

if __name__ == "__main__":
    inject_contradiction()
