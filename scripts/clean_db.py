import sqlite3

db_path = 'output/veriscribe.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

docs_to_remove = [
    'Mohit_Upadhyay_AIML (7)',
    'REVISED STANDARD OISD-STD-116',
    'superjoin-vit-2026-assignment',
    '30_LLM_Interview_Questions_AmanAI_Lab'
]

print("Deleting from facts and relationships...")
for doc in docs_to_remove:
    print(f"Removing {doc}...")
    # Get all fact IDs for this document
    c.execute("SELECT fact_id FROM facts WHERE document_id = ?", (doc,))
    fact_ids = [row[0] for row in c.fetchall()]
    
    if fact_ids:
        # Delete relationships
        placeholders = ','.join('?' * len(fact_ids))
        c.execute(f"DELETE FROM relationships WHERE fact_id_a IN ({placeholders}) OR fact_id_b IN ({placeholders})", fact_ids + fact_ids)
        
        # Delete facts
        c.execute(f"DELETE FROM facts WHERE fact_id IN ({placeholders})", fact_ids)

conn.commit()
print("Done!")
