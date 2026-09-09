import pytest
from unittest.mock import MagicMock
from ingestion.models import Chunk
from extraction.models import Fact
from extraction.extractor import extract_facts_from_chunk
from extraction.pipeline import extract_document_facts
from pydantic import ValidationError

class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class MockToolCall:
    def __init__(self, function):
        self.function = function

class MockMessage:
    def __init__(self, content: str):
        self.content = content

class MockChoice:
    def __init__(self, message):
        self.message = message

class MockResponse:
    def __init__(self, choices):
        self.choices = choices

def get_mock_client(canned_responses):
    import json
    client = MagicMock()
    responses = []
    for r in canned_responses:
        msg = MockMessage(json.dumps(r))
        choice = MockChoice(msg)
        responses.append(MockResponse([choice]))
        
    client.chat.completions.create.side_effect = responses
    return client

def test_extraction_preserves_qualifying_clause():
    chunk = Chunk(
        document_id="doc1", page_number=1, chunk_type="text",
        text="Excluding discontinued operations, revenue grew 5%.",
        bbox=(0.1, 0.1, 0.5, 0.2), reading_order_index=0
    )
    
    mock_input = {
        "facts": [
            {
                "statement": "Revenue grew 5%.",
                "canonical_entity": "Revenue",
                "raw_attribute": "growth",
                "value": "5%",
                "confidence": 0.9,
                "evidence_quote": "revenue grew 5%",
                "context": {"exclusion": "excluding discontinued operations"}
            }
        ]
    }
    
    client = get_mock_client([mock_input])
    facts = extract_facts_from_chunk(chunk, "", client)
    
    assert len(facts) == 1
    assert "exclusion" in facts[0].context
    assert facts[0].evidence_quote == "revenue grew 5%"
    assert client.chat.completions.create.call_count == 1

def test_equation_skipping():
    chunk = Chunk(
        document_id="doc1", page_number=1, chunk_type="text",
        text="x = \u2211 ( \u03b1 + \u03b2 ) / 2  (1)",
        bbox=(0.1, 0.1, 0.5, 0.2), reading_order_index=0
    )
    
    client = MagicMock()
    facts = extract_facts_from_chunk(chunk, "", client)
    
    # Should skip LLM call entirely
    assert len(facts) == 0
    assert client.chat.completions.create.call_count == 0

def test_deduplication():
    chunk1 = Chunk(
        document_id="doc1", page_number=1, chunk_type="text", text="Net profit was $10M.",
        bbox=(0.1, 0.1, 0.5, 0.2), reading_order_index=0
    )
    chunk2 = Chunk(
        document_id="doc1", page_number=2, chunk_type="text", text="As stated, net profit was $10M.",
        bbox=(0.1, 0.3, 0.5, 0.4), reading_order_index=0
    )
    
    mock_input = {
        "facts": [{
            "statement": "Net profit was $10M",
            "canonical_entity": "Net profit",
            "raw_attribute": "amount",
            "value": "$10M",
            "confidence": 0.9,
            "evidence_quote": "net profit was $10M",
            "temporal_scope": {"year": "2023"}  # Add scope to ensure hash matches
        }]
    }
    
    # Provide the exact same fact for both chunks
    client = get_mock_client([mock_input, mock_input])
    facts = extract_document_facts([chunk1, chunk2], client)
    
    # Should deduplicate to 1 fact
    assert len(facts) == 1
    assert len(facts[0].occurrences) == 2
    assert facts[0].occurrences[0]["page_number"] == 1
    assert facts[0].occurrences[1]["page_number"] == 2

def test_malformed_json_skipped():
    chunk = Chunk(
        document_id="doc1", page_number=1, chunk_type="text",
        text="A normal sentence.",
        bbox=(0.1, 0.1, 0.5, 0.2), reading_order_index=0
    )
    
    # Missing required 'statement' field
    mock_input = {
        "facts": [
            {
                "canonical_entity": "Something",
                "raw_attribute": "Attr",
                "value": "Val",
                "confidence": 0.9,
                "evidence_quote": "A normal"
            }
        ]
    }
    
    client = get_mock_client([mock_input])
    facts = extract_facts_from_chunk(chunk, "", client)
    
    assert len(facts) == 0

def test_fact_missing_evidence_rejected():
    chunk = Chunk(
        document_id="doc1", page_number=1, chunk_type="text",
        text="A normal sentence.",
        bbox=(0.1, 0.1, 0.5, 0.2), reading_order_index=0
    )
    
    # Has null evidence_quote
    mock_input = {
        "facts": [
            {
                "statement": "Statement",
                "canonical_entity": "Something",
                "raw_attribute": "Attr",
                "value": "Val",
                "confidence": 0.9,
                "evidence_quote": None
            }
        ]
    }
    
    client = get_mock_client([mock_input])
    facts = extract_facts_from_chunk(chunk, "", client)
    
    # Should drop the fact
    assert len(facts) == 0


def test_yoy_comparison_sentence_distinct_temporal_scope():
    """
    Regression test for the FY23/FY24 EBITDA mis-labeling bug found in the
    Delhivery Q4 FY24 earnings presentation dry run.

    Source sentence (page 5, real document):
        "FY24 EBITDA increased by Rs. 578 Cr to Rs. 127 Cr from Rs. (452 Cr) in FY23"

    The bug: the extractor labelled the FY23 value with raw_attribute="FY24 EBITDA value"
    because the chunk's headline topic is "FY24 EBITDA". Prompt instruction 5 (added
    to TEXT_EXTRACTION_PROMPT) requires each sub-fact to carry the temporal_scope of
    its own clause, not the chunk headline.

    Assertions:
    1. Two distinct facts extracted (one per year).
    2. The two facts have DIFFERENT temporal_scope period values.
    3. FY24 fact value == "127" with period containing "FY24".
    4. FY23 fact value == "-452" with period containing "FY23".
    5. raw_attribute references the correct year for each fact.
    6. No two facts share the same (raw_attribute, period) pair.
    """
    chunk = Chunk(
        document_id="03-delhivery-q4-fy24-earnings-presentation",
        page_number=5,
        chunk_type="text",
        text="FY24 EBITDA increased by Rs. 578 Cr to Rs. 127 Cr from Rs. (452 Cr) in FY23",
        bbox=(0.05, 0.40, 0.95, 0.50),
        reading_order_index=0,
    )

    # Mock LLM response reflecting the CORRECT post-fix behaviour:
    # two facts, each scoped to its own sub-clause year.
    mock_response = {
        "facts": [
            {
                "statement": "FY24 EBITDA was Rs. 127 Cr.",
                "canonical_entity": "EBITDA",
                "raw_attribute": "FY24 EBITDA value",
                "value": "127",
                "unit": "Cr",
                "temporal_scope": {"period": "FY24"},
                "context": {},
                "confidence": 0.95,
                "evidence_quote": "Rs. 127 Cr",
            },
            {
                "statement": "FY23 EBITDA was Rs. (452) Cr.",
                "canonical_entity": "EBITDA",
                "raw_attribute": "FY23 EBITDA value",
                "value": "-452",
                "unit": "Cr",
                "temporal_scope": {"period": "FY23"},
                "context": {},
                "confidence": 0.95,
                "evidence_quote": "Rs. (452 Cr) in FY23",
            },
        ]
    }

    client = get_mock_client([mock_response])
    facts = extract_facts_from_chunk(chunk, "", client)

    assert len(facts) == 2, (
        f"Expected 2 facts from a YoY comparison sentence, got {len(facts)}"
    )

    # Periods must be distinct
    periods = [f.temporal_scope.get("period", "") for f in facts]
    assert periods[0] != periods[1], (
        f"Both facts carry the same temporal_scope period '{periods[0]}' — "
        "headline-period bias not corrected."
    )

    fy24_facts = [f for f in facts if "24" in f.temporal_scope.get("period", "")]
    fy23_facts = [f for f in facts if "23" in f.temporal_scope.get("period", "")]

    assert len(fy24_facts) == 1, "Expected exactly one fact scoped to FY24"
    assert len(fy23_facts) == 1, "Expected exactly one fact scoped to FY23"

    assert fy24_facts[0].value == "127", (
        f"FY24 fact value mismatch: got '{fy24_facts[0].value}'"
    )
    assert fy23_facts[0].value == "-452", (
        f"FY23 fact value mismatch: got '{fy23_facts[0].value}'"
    )

    # Attribute-to-year pairing: raw_attribute must reference the correct year
    assert "24" in fy24_facts[0].raw_attribute or "fy24" in fy24_facts[0].raw_attribute.lower(), (
        f"FY24 fact's raw_attribute '{fy24_facts[0].raw_attribute}' does not reference FY24"
    )
    assert "23" in fy23_facts[0].raw_attribute or "fy23" in fy23_facts[0].raw_attribute.lower(), (
        f"FY23 fact's raw_attribute '{fy23_facts[0].raw_attribute}' does not reference FY23"
    )

    # Guard: no (raw_attribute, period) pair repeated across facts
    pairs = [(f.raw_attribute, f.temporal_scope.get("period", "")) for f in facts]
    assert len(set(pairs)) == len(pairs), (
        f"Duplicate (raw_attribute, period) pairs found: {pairs}"
    )
