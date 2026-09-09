import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from comparison.storage import SQLiteStore

store = SQLiteStore("output/veriscribe.db")

with store._connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT fact_id_a, fact_id_b, relationship_type, confidence, justification FROM relationships ORDER BY relationship_type")
    rows = cursor.fetchall()

from collections import Counter
counts = Counter(r[2] for r in rows)
print(f"Total relationships: {len(rows)}")
print("By type:", dict(counts))

for rel_type in ['contradicts', 'corroborates', 'reconciled']:
    hits = [(r[0], r[1], r[2], r[3], r[4]) for r in rows if r[2] == rel_type]
    if not hits:
        print(f"\n[{rel_type.upper()}] — none found")
        continue
    print(f"\n{'='*60}")
    print(f"[{rel_type.upper()}] — {len(hits)} found")
    print(f"{'='*60}")
    for fa, fb, rt, conf, just in hits[:3]:
        fa_obj = store.get_fact(fa)
        fb_obj = store.get_fact(fb)
        if fa_obj and fb_obj:
            print(f"\n  Fact A ({fa}): {fa_obj.statement[:80]}")
            print(f"  Fact B ({fb}): {fb_obj.statement[:80]}")
        print(f"  Confidence: {conf}")
        print(f"  Justification: {just[:250]}")
