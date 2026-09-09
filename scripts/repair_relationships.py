import logging
from comparison.storage import SQLiteStore
from comparison.deterministic_checks import try_deterministic_reconcile

logging.basicConfig(level=logging.INFO)

store = SQLiteStore("output/veriscribe.db")

doc_ids = store.get_document_ids()
all_facts = {}
for did in doc_ids:
    for f in store.get_facts_by_document(did):
        all_facts[f.fact_id] = f

with store._connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT fact_id_a, fact_id_b, relationship_type, confidence, justification FROM relationships WHERE relationship_type = 'reconciled'")
    rows = cursor.fetchall()

fixed_count = 0

with store._connect() as conn:
    cursor = conn.cursor()
    for row in rows:
        fid_a, fid_b, old_rel, old_conf, old_just = row
        f_a = all_facts.get(fid_a)
        f_b = all_facts.get(fid_b)
        
        if not f_a or not f_b:
            continue
            
        new_res = try_deterministic_reconcile(f_a, f_b)
        if new_res:
            rel_val = new_res.relationship.value if hasattr(new_res.relationship, 'value') else new_res.relationship
            if rel_val == 'corroborates':
                cursor.execute("""
                    UPDATE relationships 
                    SET relationship_type = ?, confidence = ?, justification = ?
                    WHERE fact_id_a = ? AND fact_id_b = ?
                """, (rel_val, new_res.confidence, new_res.justification, fid_a, fid_b))
                fixed_count += 1
                print(f"Fixed {fid_a} <-> {fid_b} from 'reconciled' to 'corroborates'.")
    conn.commit()

print(f"\nTotal stale 'reconciled' relationships fixed: {fixed_count}")
