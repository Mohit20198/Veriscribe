from comparison.storage import SQLiteStore
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json

store = SQLiteStore("output/veriscribe.db")

with store._connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT fact_id_a, fact_id_b, relationship_type, justification, confidence FROM relationships WHERE relationship_type = 'reconciled'")
    rows = cursor.fetchall()

print(f"Total 'reconciled' relationships found in DB: {len(rows)}\n")

for i, row in enumerate(rows):
    fid_a, fid_b, rel, justification, conf = row
    
    # Check if the justification starts with the deterministic pattern
    is_deterministic = justification.startswith("Facts report effectively identical values") or justification.startswith("Facts report exactly the same value")
    source = "DETERMINISTIC" if is_deterministic else "LLM JUDGE"
    
    print(f"[{i+1}] Fact A ({fid_a}) <-> Fact B ({fid_b}):")
    print(f"Source: {source}")
    print(f"Confidence: {conf}")
    print(f"Justification: {justification}\n")
