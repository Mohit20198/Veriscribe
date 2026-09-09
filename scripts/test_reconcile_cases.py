import sys
from extraction.models import Fact
from comparison.deterministic_checks import try_deterministic_reconcile

# Pair 1: 81,415Mn vs 8,142Cr
fact_a = Fact(
    fact_id="A", document_id="02", canonical_entity="Delhivery", raw_attribute="Revenue",
    statement="Revenue was 81,415 Mn", value="81,415", unit="Mn", evidence_quote="", evidence_context_window="",
    temporal_scope={"period": "FY24"},
    context={"normalized": {"normalized_value": 81415000000.0, "normalized_unit": "mn", "parse_ok": True}},
    confidence=1.0, source_chunk_id="x", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

fact_b = Fact(
    fact_id="B", document_id="03", canonical_entity="Delhivery", raw_attribute="Revenue",
    statement="Revenue was 8,142 Cr", value="8,142", unit="Cr", evidence_quote="", evidence_context_window="",
    temporal_scope={"period": "FY24", "type": "fiscal_year"},
    context={"normalized": {"normalized_value": 81420000000.0, "normalized_unit": "cr", "parse_ok": True}},
    confidence=1.0, source_chunk_id="y", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

res = try_deterministic_reconcile(fact_a, fact_b)
print(f"81,415Mn vs 8,142Cr Result: {res.relationship if res else 'None'}\n{res.justification if res else ''}")

# Pair 2: 1.6% vs 1.6%
fact_c = Fact(
    fact_id="C", document_id="02", canonical_entity="Delhivery", raw_attribute="EBITDA Margin",
    statement="EBITDA Margin was 1.6%", value="1.6", unit="%", evidence_quote="", evidence_context_window="",
    temporal_scope={},
    context={"normalized": {"normalized_value": 1.6, "normalized_unit": "%", "parse_ok": True}},
    confidence=1.0, source_chunk_id="x", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

fact_d = Fact(
    fact_id="D", document_id="03", canonical_entity="Delhivery", raw_attribute="EBITDA Margin",
    statement="Margin was 1.6%", value="1.6", unit="None", evidence_quote="", evidence_context_window="",
    temporal_scope={"year": "FY24", "period": "full year"},
    context={"normalized": {"normalized_value": 1.6, "normalized_unit": "%", "parse_ok": True}},
    confidence=1.0, source_chunk_id="y", source_type="text", evidence_bbox=(0.0, 0.0, 0.0, 0.0)
)

res2 = try_deterministic_reconcile(fact_c, fact_d)
print(f"\n1.6% vs 1.6% Result: {res2.relationship if res2 else 'None'}\n{res2.justification if res2 else ''}")
