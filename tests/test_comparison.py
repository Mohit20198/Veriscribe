"""
tests/test_comparison.py
------------------------
Test suite for Phase 4 fact comparison.
"""

import os
import pytest
from unittest.mock import MagicMock
from extraction.models import Fact
from comparison.models import RelationshipType
from comparison.storage import SQLiteStore
from comparison.clustering import CorroborationClusters
from comparison.deterministic_checks import try_deterministic_reconcile
from comparison.pipeline import compare_new_fact
from comparison.vector_store import VectorStore
import chromadb

class FakeEmbedder:
    def encode(self, texts):
        return [[0.1] * 384 for _ in texts]

@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return SQLiteStore(db_path=db_path)

@pytest.fixture
def vector_store():
    client = chromadb.EphemeralClient()
    return VectorStore(persist_directory="", client=client)

@pytest.fixture
def clusters():
    return CorroborationClusters()

@pytest.fixture
def embedder():
    return FakeEmbedder()

def create_fact(fact_id, statement, value, unit, scope, doc_id="doc1"):
    return Fact(
        fact_id=fact_id,
        document_id=doc_id,
        statement=statement,
        canonical_entity="TestEntity",
        raw_attribute="TestAttribute",
        value=value,
        unit=unit,
        temporal_scope=scope,
        confidence=0.9,
        source_chunk_id="chunk1",
        source_type="text",
        evidence_bbox=(0, 0, 100, 100),
        evidence_context_window="Some context."
    )

def test_deterministic_reconcile_fires():
    fact_a = create_fact("f1", "Revenue was 100", "100", "USD", {"year": 2023})
    fact_a.context["normalized"] = {"parse_ok": True, "normalized_unit": "USD", "normalized_value": 100.0}
    
    fact_b = create_fact("f2", "Revenue hit 100.5", "100.5", "USD", {"year": 2024}, doc_id="doc2")
    fact_b.context["normalized"] = {"parse_ok": True, "normalized_unit": "USD", "normalized_value": 100.5}
    
    # 100 vs 100.5 -> within 1% (0.5%) -> should reconcile because scopes differ!
    result = try_deterministic_reconcile(fact_a, fact_b)
    
    assert result is not None
    assert result.relationship == RelationshipType.RECONCILED.value
    assert "temporal reconciliation" in result.justification

def test_deterministic_reconcile_fails_same_scope():
    fact_a = create_fact("f1", "Revenue was 100", "100", "USD", {"year": 2023})
    fact_a.context["normalized"] = {"parse_ok": True, "normalized_unit": "USD", "normalized_value": 100.0}
    
    fact_b = create_fact("f2", "Revenue was exactly 100", "100", "USD", {"year": 2023}, doc_id="doc2")
    fact_b.context["normalized"] = {"parse_ok": True, "normalized_unit": "USD", "normalized_value": 100.0}
    
    # Values match and temporal scopes are identical -> CORROBORATES
    result = try_deterministic_reconcile(fact_a, fact_b)
    assert result is not None
    assert result.relationship == "corroborates"

def test_llm_routing_on_same_scope_different_wording(store, vector_store, clusters, embedder):
    fact_a = create_fact("f1", "Revenue was 100", "100", "USD", {"year": 2023})
    fact_b = create_fact("f2", "Total income hit 100", "100", "USD", {"year": 2023}, doc_id="doc2")
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"relationship": "corroborates", "confidence": 0.95, "justification": "They report the same revenue."}'
    mock_client.chat.completions.create.return_value = mock_response

    vector_store.add_facts([fact_a], embedder)
    store.save_fact(fact_a)

    rels = compare_new_fact(fact_b, store, vector_store, mock_client, embedder, clusters)
    
    assert len(rels) == 1
    assert rels[0].relationship_type == RelationshipType.CORROBORATES
    mock_client.chat.completions.create.assert_called_once()

def test_corroboration_clustering(store, vector_store, embedder):
    clusters = CorroborationClusters()
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"relationship": "corroborates", "confidence": 0.95, "justification": "Test."}'
    mock_client.chat.completions.create.return_value = mock_response

    # Fact A
    fact_a = create_fact("fA", "Rev 100", "100", "USD", {"year": 2023}, doc_id="docA")
    vector_store.add_facts([fact_a], embedder)
    store.save_fact(fact_a)
    
    # Fact B corroborates A
    fact_b = create_fact("fB", "Income 100", "100", "USD", {"year": 2023}, doc_id="docB")
    rels = compare_new_fact(fact_b, store, vector_store, mock_client, embedder, clusters)
    assert rels[0].relationship_type == RelationshipType.CORROBORATES
    assert mock_client.chat.completions.create.call_count == 1
    
    # Check cluster transitivity
    assert clusters.get_cluster_representative("fB") == clusters.get_cluster_representative("fA")

    # Fact C comes in. Retrieves A and B. Should only compare against the representative.
    mock_client.chat.completions.create.reset_mock()
    fact_c = create_fact("fC", "Net 100", "100", "USD", {"year": 2023}, doc_id="docC")
    rels = compare_new_fact(fact_c, store, vector_store, mock_client, embedder, clusters)
    
    assert mock_client.chat.completions.create.call_count == 1

def test_malformed_llm_response(store, vector_store, clusters, embedder):
    fact_a = create_fact("f1", "Revenue was 100", "100", "USD", {"year": 2023})
    fact_b = create_fact("f2", "Total income hit 100", "100", "USD", {"year": 2023}, doc_id="doc2")
    
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"broken": json'
    mock_client.chat.completions.create.return_value = mock_response

    vector_store.add_facts([fact_a], embedder)
    store.save_fact(fact_a)

    rels = compare_new_fact(fact_b, store, vector_store, mock_client, embedder, clusters)
    assert len(rels) == 0

def test_deterministic_missing_temporal_scope():
    from comparison.deterministic_checks import try_deterministic_reconcile
    
    fact_a = create_fact("f1", "Rev 100", "100", "USD", {})
    fact_a.context = {"normalized": {"normalized_value": 100.0, "normalized_unit": "usd", "parse_ok": True}}
    
    fact_b = create_fact("f2", "Total income 100", "100", "USD", {"year": 2024})
    fact_b.context = {"normalized": {"normalized_value": 100.0, "normalized_unit": "usd", "parse_ok": True}}

    # One missing temporal scope
    res = try_deterministic_reconcile(fact_a, fact_b)
    assert res is None  # Must defer to judge

    # Both missing temporal scopes
    fact_b.temporal_scope = {}
    res = try_deterministic_reconcile(fact_a, fact_b)
    assert res is None  # Must defer to judge
