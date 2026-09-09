"""
tests/test_orchestration.py
----------------------------
Unit tests for orchestration/pipeline.py.

All LLM calls (extraction + comparison judge) are mocked.
The embedder uses a real SentenceTransformer loaded once per session.
SQLite uses :memory: so no disk state leaks between tests.
ChromaDB uses an EphemeralClient.

Tests
-----
1. test_idempotency          — ingesting the same document_id twice doesn't
                               duplicate facts.
2. test_incremental_isolation — ingesting document B after A only triggers
                               compare_new_fact() on B's facts, not a full
                               re-scan of A (verified via call-count assertions).
3. test_ingest_result_fields — IngestResult fields are populated correctly.
4. test_force_flag           — force=True re-ingests a document already stored.
"""

from __future__ import annotations

import json
import uuid
from typing import List
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from comparison.storage import SQLiteStore
from comparison.vector_store import VectorStore
from extraction.models import Fact
from orchestration.pipeline import ingest_document, IngestResult

# ---------------------------------------------------------------------------
# Session-scoped embedder (loads the model once for the whole test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Per-test in-memory stores
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return SQLiteStore(":memory:")


@pytest.fixture
def vector_store():
    client = chromadb.EphemeralClient()
    vs = VectorStore(client=client)
    return vs


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_fact(doc_id: str, entity: str = "Revenue", attr: str = "amount",
               value: str = "100", unit: str = "Cr") -> dict:
    """Return a canned LLM extraction response dict for one fact."""
    return {
        "facts": [
            {
                "statement": f"{entity} {attr} is {value} {unit}.",
                "canonical_entity": entity,
                "raw_attribute": attr,
                "value": value,
                "unit": unit,
                "temporal_scope": {"period": "FY24"},
                "context": {},
                "confidence": 0.95,
                "evidence_quote": f"{value} {unit}",
            }
        ]
    }


def _mock_client(responses: List[dict]) -> MagicMock:
    """Create a mock OpenAI client that returns canned JSON responses in sequence."""
    client = MagicMock()
    choices = []
    for r in responses:
        msg = MagicMock()
        msg.content = json.dumps(r)
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        choices.append(resp)
    client.chat.completions.create.side_effect = choices
    return client


# ---------------------------------------------------------------------------
# Test 1 — Idempotency: same document_id twice doesn't duplicate facts
# ---------------------------------------------------------------------------

def test_idempotency(store, vector_store, embedder, tmp_path):
    """Ingesting the same document_id twice must not duplicate facts."""

    fake_pdf = tmp_path / "doc_a.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    fact_response = _make_fact("doc_a")

    # We need parse_document and extract_document_facts to be mockable.
    # patch them at the orchestration.pipeline import path.
    with (
        patch("orchestration.pipeline.parse_document") as mock_parse,
        patch("orchestration.pipeline.extract_document_facts") as mock_extract,
    ):
        # Simulate 1 chunk and 1 extracted fact on first call
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = str(uuid.uuid4())
        mock_chunk.page_number = 1
        mock_chunk.chunk_type = "text"
        mock_chunk.text = "Revenue was 100 Cr in FY24."
        mock_chunk.document_id = "doc_a"
        mock_chunk.reading_order_index = 0
        mock_chunk.bbox = (0.0, 0.0, 1.0, 0.1)
        mock_chunk.table_data = None
        mock_chunk.image_path = None
        mock_parse.return_value = [mock_chunk]

        fact = Fact(
            document_id="doc_a",
            statement="Revenue amount is 100 Cr.",
            canonical_entity="Revenue",
            raw_attribute="amount",
            value="100",
            unit="Cr",
            temporal_scope={"period": "FY24"},
            confidence=0.95,
            source_chunk_id=mock_chunk.chunk_id,
            source_type="text",
            evidence_quote="100 Cr",
            evidence_bbox=(0.0, 0.0, 1.0, 0.1),
            occurrences=[{"page_number": 1, "bbox": (0.0, 0.0, 1.0, 0.1)}],
        )
        fact.content_hash = fact.compute_hash()
        mock_extract.return_value = [fact]

        client = MagicMock()

        # First ingest
        r1 = ingest_document(
            pdf_path=str(fake_pdf),
            document_id="doc_a",
            store=store,
            vector_store=vector_store,
            client=client,
            embedder=embedder,
        )

        assert not r1.skipped
        assert r1.fact_count == 1

        # Second ingest — same document_id, no force
        r2 = ingest_document(
            pdf_path=str(fake_pdf),
            document_id="doc_a",
            store=store,
            vector_store=vector_store,
            client=client,
            embedder=embedder,
        )

        assert r2.skipped, "Second ingest should be skipped (idempotency)"
        # parse and extract must NOT have been called a second time
        assert mock_parse.call_count == 1, (
            f"parse_document called {mock_parse.call_count} times; expected 1"
        )
        assert mock_extract.call_count == 1, (
            f"extract_document_facts called {mock_extract.call_count} times; expected 1"
        )

        # Only 1 fact in the DB
        stored = store.get_facts_by_document("doc_a")
        assert len(stored) == 1, f"Expected 1 stored fact, got {len(stored)}"


# ---------------------------------------------------------------------------
# Test 2 — Incremental isolation: ingesting B only compares B's facts
# ---------------------------------------------------------------------------

def test_incremental_isolation(store, vector_store, embedder, tmp_path):
    """
    After doc_a is ingested, ingesting doc_b should only call
    compare_new_fact() for doc_b's facts, not trigger re-processing of doc_a.

    We assert: compare_new_fact() is called exactly N_b times where N_b is
    the number of facts in doc_b.
    """
    fake_pdf_a = tmp_path / "doc_a.pdf"
    fake_pdf_b = tmp_path / "doc_b.pdf"
    fake_pdf_a.write_bytes(b"%PDF-1.4 a")
    fake_pdf_b.write_bytes(b"%PDF-1.4 b")

    def _make_mock_fact(doc_id: str, entity: str) -> Fact:
        chunk_id = str(uuid.uuid4())
        f = Fact(
            document_id=doc_id,
            statement=f"{entity} revenue is 200 Cr.",
            canonical_entity=entity,
            raw_attribute="revenue",
            value="200",
            unit="Cr",
            temporal_scope={"period": "FY24"},
            confidence=0.9,
            source_chunk_id=chunk_id,
            source_type="text",
            evidence_quote="200 Cr",
            evidence_bbox=(0.0, 0.0, 1.0, 0.1),
            occurrences=[{"page_number": 1, "bbox": (0.0, 0.0, 1.0, 0.1)}],
        )
        f.content_hash = f.compute_hash()
        return f

    fact_a = _make_mock_fact("doc_a", "CompanyA")
    fact_b1 = _make_mock_fact("doc_b", "CompanyB_1")
    fact_b2 = _make_mock_fact("doc_b", "CompanyB_2")

    client = MagicMock()

    with (
        patch("orchestration.pipeline.parse_document") as mock_parse,
        patch("orchestration.pipeline.extract_document_facts") as mock_extract,
        patch("orchestration.pipeline.compare_new_fact") as mock_compare,
    ):
        mock_compare.return_value = []  # no relationships for simplicity

        # --- Ingest doc_a (1 fact) ---
        mock_parse.return_value = [MagicMock(chunk_id=str(uuid.uuid4()), page_number=1,
                                             chunk_type="text", text=".", document_id="doc_a",
                                             reading_order_index=0, bbox=(0,0,1,1),
                                             table_data=None, image_path=None)]
        mock_extract.return_value = [fact_a]

        ingest_document(
            pdf_path=str(fake_pdf_a),
            document_id="doc_a",
            store=store,
            vector_store=vector_store,
            client=client,
            embedder=embedder,
        )

        calls_after_a = mock_compare.call_count
        assert calls_after_a == 1, (
            f"Expected 1 compare call for doc_a, got {calls_after_a}"
        )

        # --- Ingest doc_b (2 facts) ---
        mock_parse.return_value = [MagicMock(chunk_id=str(uuid.uuid4()), page_number=1,
                                             chunk_type="text", text=".", document_id="doc_b",
                                             reading_order_index=0, bbox=(0,0,1,1),
                                             table_data=None, image_path=None)]
        mock_extract.return_value = [fact_b1, fact_b2]

        ingest_document(
            pdf_path=str(fake_pdf_b),
            document_id="doc_b",
            store=store,
            vector_store=vector_store,
            client=client,
            embedder=embedder,
        )

        calls_after_b = mock_compare.call_count
        new_calls = calls_after_b - calls_after_a
        assert new_calls == 2, (
            f"Expected exactly 2 new compare calls for doc_b's facts, got {new_calls}. "
            "A's facts must not be re-scanned."
        )


# ---------------------------------------------------------------------------
# Test 3 — IngestResult fields populated correctly
# ---------------------------------------------------------------------------

def test_ingest_result_fields(store, vector_store, embedder, tmp_path):
    """IngestResult.fact_count, chunk_count, elapsed_seconds are populated."""
    fake_pdf = tmp_path / "doc_c.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 c")

    chunk_id = str(uuid.uuid4())
    fact = Fact(
        document_id="doc_c",
        statement="Profit was 50 Cr.",
        canonical_entity="Profit",
        raw_attribute="amount",
        value="50",
        unit="Cr",
        temporal_scope={"period": "FY24"},
        confidence=0.9,
        source_chunk_id=chunk_id,
        source_type="text",
        evidence_quote="50 Cr",
        evidence_bbox=(0.0, 0.0, 1.0, 0.1),
        occurrences=[{"page_number": 1, "bbox": (0.0, 0.0, 1.0, 0.1)}],
    )
    fact.content_hash = fact.compute_hash()

    mock_chunk = MagicMock(chunk_id=chunk_id, page_number=1, chunk_type="text",
                           text="Profit was 50 Cr.", document_id="doc_c",
                           reading_order_index=0, bbox=(0,0,1,1),
                           table_data=None, image_path=None)

    with (
        patch("orchestration.pipeline.parse_document", return_value=[mock_chunk]),
        patch("orchestration.pipeline.extract_document_facts", return_value=[fact]),
        patch("orchestration.pipeline.compare_new_fact", return_value=[]),
    ):
        result = ingest_document(
            pdf_path=str(fake_pdf),
            document_id="doc_c",
            store=store,
            vector_store=vector_store,
            client=MagicMock(),
            embedder=embedder,
        )

    assert result.document_id == "doc_c"
    assert result.fact_count == 1
    assert result.chunk_count == 1
    assert result.elapsed_seconds > 0
    assert result.error_count == 0
    assert not result.skipped


# ---------------------------------------------------------------------------
# Test 4 — force=True re-ingests an already-stored document
# ---------------------------------------------------------------------------

def test_force_flag(store, vector_store, embedder, tmp_path):
    """force=True must bypass the idempotency check and call parse again."""
    fake_pdf = tmp_path / "doc_d.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 d")

    chunk_id = str(uuid.uuid4())
    fact = Fact(
        document_id="doc_d",
        statement="EBITDA was 30 Cr.",
        canonical_entity="EBITDA",
        raw_attribute="amount",
        value="30",
        unit="Cr",
        temporal_scope={"period": "FY24"},
        confidence=0.9,
        source_chunk_id=chunk_id,
        source_type="text",
        evidence_quote="30 Cr",
        evidence_bbox=(0.0, 0.0, 1.0, 0.1),
        occurrences=[{"page_number": 1, "bbox": (0.0, 0.0, 1.0, 0.1)}],
    )
    fact.content_hash = fact.compute_hash()
    mock_chunk = MagicMock(chunk_id=chunk_id, page_number=1, chunk_type="text",
                           text="EBITDA was 30 Cr.", document_id="doc_d",
                           reading_order_index=0, bbox=(0,0,1,1),
                           table_data=None, image_path=None)

    with (
        patch("orchestration.pipeline.parse_document", return_value=[mock_chunk]) as mp,
        patch("orchestration.pipeline.extract_document_facts", return_value=[fact]),
        patch("orchestration.pipeline.compare_new_fact", return_value=[]),
    ):
        # First ingest
        ingest_document(
            pdf_path=str(fake_pdf), document_id="doc_d",
            store=store, vector_store=vector_store,
            client=MagicMock(), embedder=embedder,
        )
        assert mp.call_count == 1

        # Second ingest with force=True
        r = ingest_document(
            pdf_path=str(fake_pdf), document_id="doc_d",
            store=store, vector_store=vector_store,
            client=MagicMock(), embedder=embedder, force=True,
        )

    assert not r.skipped, "force=True must not skip"
    assert mp.call_count == 2, (
        f"parse_document must be called again with force=True (got {mp.call_count})"
    )
