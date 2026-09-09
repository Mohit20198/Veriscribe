import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from comparison.storage import SQLiteStore
from comparison.judge import judge_pair
from orchestration.cli import _build_client

store = SQLiteStore("output/veriscribe.db")
client = _build_client()

f_a = store.get_fact("f0f42dc6-c9ad-4eff-a8ac-12b7b5ccdd8c")
f_b = store.get_fact("b172b716-0ead-4508-be7f-123c72a4e4dc")

# It's a symmetric pair, let's judge both directions
pairs = [(f_a, f_b), (f_b, f_a)]

for a, b in pairs:
    print(f"\nRe-judging: Fact A ({a.fact_id}) <-> Fact B ({b.fact_id})")
    verdict = judge_pair(a, b, client)
    store.save_relationship(verdict)
    print(f"  New Verdict: {verdict.relationship_type.value}")
    print(f"  Confidence: {verdict.confidence}")
    print(f"  Justification: {verdict.justification}")

print("\n--- Final Confirmation ---")
with store._connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT fact_id_a, fact_id_b, justification FROM relationships WHERE relationship_type = 'reconciled'")
    rows = cursor.fetchall()

remaining_bad = 0
for row in rows:
    fid_a, fid_b, just = row
    if just.startswith("Facts report effectively identical values") or just.startswith("Facts report exactly the same value"):
        a_fact = store.get_fact(fid_a)
        b_fact = store.get_fact(fid_b)
        if not a_fact.temporal_scope or not b_fact.temporal_scope:
            remaining_bad += 1

print(f"Number of deterministic 'reconciled' relationships with empty scope remaining: {remaining_bad}")
