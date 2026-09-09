import sys
import os
from dotenv import load_dotenv
import openai
from comparison.storage import SQLiteStore
from comparison.judge import judge_pair

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
store = SQLiteStore("output/veriscribe.db")
client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

# The facts involved in the contradiction
fact_a_id = "755ad359-ec19-43fa-898f-596be784bd73"
fact_b_id = "bc87bafd-1e0e-459f-94fc-1d733c38738a"

fact_a = store.get_fact(fact_a_id)
fact_b = store.get_fact(fact_b_id)

rel = judge_pair(fact_a, fact_b, client)
if rel:
    print(f"Re-judged: {rel.relationship_type.value}")
    print(f"Justification length: {len(rel.justification)}")
    store.save_relationship(rel)
    print("Saved untruncated relationship to DB.")
