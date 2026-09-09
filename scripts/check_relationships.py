import sqlite3
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect('output/veriscribe.db')
query = """
SELECT r.relationship_type, r.confidence, r.justification, f1.data_json, f2.data_json 
FROM relationships r 
JOIN facts f1 ON r.fact_id_a = f1.fact_id 
JOIN facts f2 ON r.fact_id_b = f2.fact_id
"""
res = conn.execute(query).fetchall()

for row in res:
    rel_type, conf, just, data_a_str, data_b_str = row
    data_a = json.loads(data_a_str)
    data_b = json.loads(data_b_str)
    
    val_a = data_a.get("value")
    unit_a = data_a.get("unit")
    val_b = data_b.get("value")
    unit_b = data_b.get("unit")
    
    print(f"\n--- {rel_type.upper()} (conf={conf}) ---")
    print(f"Fact A ({data_a.get('document_id')}): {val_a} {unit_a}")
    print(f"Fact B ({data_b.get('document_id')}): {val_b} {unit_b}")
    print(f"Justification: {just}")
