from comparison.storage import SQLiteStore
from comparison.deterministic_checks import try_deterministic_reconcile

store = SQLiteStore("output/veriscribe.db")
doc_ids = store.get_document_ids()
all_facts = {}
for did in doc_ids:
    for f in store.get_facts_by_document(did):
        all_facts[f.fact_id] = f

with store._connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT fact_id_a, fact_id_b, relationship_type FROM relationships WHERE relationship_type = 'reconciled'")
    rows = cursor.fetchall()

print(f"Total 'reconciled' relationships found in DB: {len(rows)}\n")

for row in rows:
    fid_a, fid_b, old_rel = row
    f_a = all_facts.get(fid_a)
    f_b = all_facts.get(fid_b)
    
    new_label = "reconciled (no change)"
    if f_a and f_b:
        new_res = try_deterministic_reconcile(f_a, f_b)
        if new_res:
            rel_val = new_res.relationship.value if hasattr(new_res.relationship, 'value') else new_res.relationship
            if rel_val == 'corroborates':
                new_label = "corroborates (WOULD CHANGE)"
        else:
            new_label = "None (defers to judge)"
            
    print(f"Fact A ({fid_a}) <-> Fact B ({fid_b}):")
    print(f"  Before: {old_rel}")
    print(f"  After (current deterministic logic): {new_label}\n")
