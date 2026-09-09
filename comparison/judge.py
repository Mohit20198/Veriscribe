"""
comparison/judge.py
-------------------
LLM-based judge for evaluating the relationship between two facts.
"""

import json
import logging
from typing import Optional
from pydantic import ValidationError

from extraction.models import Fact
from comparison.models import Relationship, RelationshipType, JudgeResponse

log = logging.getLogger(__name__)

# Judge model — named constant so it can be audited/changed in one place.
# Use claude-sonnet-5 (or better) for the judge: the judge must read long
# evidence context windows and reason about scope/reconciliation carefully.
JUDGE_MODEL = "anthropic/claude-sonnet-5"

SYSTEM_PROMPT = """You are a meticulous, evidence-based data judge. Your task is to compare two facts extracted from documents and determine their relationship.

Before declaring a contradiction, you MUST check whether the surrounding context contains a qualifying clause, scope limiter, or different time period that explains the difference. If they refer to different time periods or conditions, they do NOT contradict.
If the two facts are about clearly different entities or attributes despite superficial similarity, return "unrelated".

Only reference details that are literally present in the provided evidence quotes and context windows. Do not infer or invent additional narrative, dates, or explanations not stated in the source text.

Your outputs MUST follow the JSON schema provided. 
- relationship: "corroborates", "contradicts", "reconciled", "ambiguous", or "unrelated"
- confidence: 0.0 to 1.0
- justification: 2-3 sentences. You MUST reference specific evidence from BOTH facts in your justification. Never output a generic "these differ" statement.
"""

def judge_pair(fact_a: Fact, fact_b: Fact, client) -> Optional[Relationship]:
    """
    Passes two facts to the LLM judge to determine their relationship.
    Returns a Relationship object, or None if the LLM call fails or returns malformed JSON.
    """
    
    content = f"""
--- FACT A ---
Entity: {fact_a.canonical_entity}
Attribute: {fact_a.raw_attribute}
Statement: {fact_a.statement}
Value: {fact_a.value} {fact_a.unit or ''}
Temporal Scope: {fact_a.temporal_scope}
Evidence Context:
{fact_a.evidence_context_window or 'N/A'}

--- FACT B ---
Entity: {fact_b.canonical_entity}
Attribute: {fact_b.raw_attribute}
Statement: {fact_b.statement}
Value: {fact_b.value} {fact_b.unit or ''}
Temporal Scope: {fact_b.temporal_scope}
Evidence Context:
{fact_b.evidence_context_window or 'N/A'}
"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": JudgeResponse.model_json_schema(),
                    "strict": True
                }
            },
            max_tokens=1000,
            temperature=0.0
        )
    except Exception as e:
        log.error(f"LLM API call failed for comparing {fact_a.fact_id} and {fact_b.fact_id}: {e}")
        return None

    if not response.choices or not response.choices[0].message.content:
        log.error("LLM returned empty response")
        return None

    content_str = response.choices[0].message.content.strip()
    if content_str.startswith("```"):
        content_str = content_str.split("```")[1]
        if content_str.startswith("json"):
            content_str = content_str[4:]
    content_str = content_str.strip()

    try:
        args = json.loads(content_str)
        result = JudgeResponse(**args)
    except (ValidationError, json.JSONDecodeError) as e:
        log.error(f"LLM returned malformed JSON for pair {fact_a.fact_id}/{fact_b.fact_id}: {e}\nContent: {content_str}")
        return None

    try:
        rel_type = RelationshipType(result.relationship)
    except ValueError:
        log.error(
            f"LLM returned unknown relationship type '{result.relationship}' "
            f"for pair {fact_a.fact_id}/{fact_b.fact_id}"
        )
        return None

    return Relationship(
        fact_id_a=fact_a.fact_id,
        fact_id_b=fact_b.fact_id,
        relationship_type=rel_type,
        confidence=result.confidence,
        justification=result.justification
    )
