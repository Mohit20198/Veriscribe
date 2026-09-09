import sys
import os
from extraction.models import Fact
from comparison.deterministic_checks import try_deterministic_reconcile
from comparison.judge import judge_pair
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy_key")
)

# Pair 2: 1.6% vs 1.6%
fact_c = Fact(
    fact_id="C", document_id="02", canonical_entity="Delhivery", raw_attribute="EBITDA Margin",
    statement="EBITDA Margin was 1.6%", value="1.6", unit="%", 
    evidence_quote="EBITDA Margin was 1.6%", evidence_context_window="In FY24, the EBITDA Margin was 1.6%.",
    temporal_scope={},
    context={"normalized": {"normalized_value": 1.6, "normalized_unit": "%", "parse_ok": True}},
    confidence=1.0, source_chunk_id="x", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

fact_d = Fact(
    fact_id="D", document_id="03", canonical_entity="Delhivery", raw_attribute="EBITDA Margin",
    statement="Margin was 1.6%", value="1.6", unit="None", 
    evidence_quote="Margin was 1.6%", evidence_context_window="FY24 margin was 1.6%.",
    temporal_scope={"year": "FY24", "period": "full year"},
    context={"normalized": {"normalized_value": 1.6, "normalized_unit": "%", "parse_ok": True}},
    confidence=1.0, source_chunk_id="y", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

res2 = try_deterministic_reconcile(fact_c, fact_d)
if res2 is None:
    print("Deterministic reconcile deferred to judge.")
    res2 = judge_pair(fact_c, fact_d, client)
    if res2:
        print(f"1.6% vs 1.6% Result (JUDGE): {res2.relationship_type.value} | {res2.justification}")
    else:
        print("Judge failed.")
else:
    print(f"1.6% vs 1.6% Result (DETERMINISTIC): {res2.relationship} | {res2.justification}")
