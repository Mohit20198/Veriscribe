"""
tests/test_canonicalization.py
--------------------------------
Pytest tests for the Phase 3 canonicalization layer.

All tests use a FakeEmbedder that returns pre-specified vectors, so no
real model is loaded and no network calls are made.  The FakeEmbedder's
contract is:
  - Input: a list of strings
  - Output: np.ndarray of shape (len(strings), D)
  - Strings registered at construction time map to their assigned vector;
    any unregistered string returns a zero vector.

This lets each test precisely control which pairs have high/low cosine
similarity without depending on real embedding geometry.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

from canonicalization.entity_resolver import (
    EntityRegistry,
    resolve_entity,
    ENTITY_MATCH_THRESHOLD,
)
from canonicalization.attribute_taxonomy import (
    AttributeTaxonomy,
    snap_attribute,
    ATTRIBUTE_MATCH_THRESHOLD,
)
from canonicalization.units import normalize_unit, same_currency
from canonicalization.pipeline import canonicalize_facts
from extraction.models import Fact


# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic mock, no real model
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """
    Returns pre-registered fixed vectors for known strings.
    Unregistered strings get a zero vector (no match).

    Usage:
        embedder = FakeEmbedder({
            "Apple Inc": [1, 0, 0],
            "Apple":     [0.99, 0.1, 0],   # high similarity to "Apple Inc"
            "Orange":    [0, 1, 0],         # low similarity to Apple variants
        })
    """

    def __init__(self, mapping: Dict[str, List[float]]) -> None:
        self._dim = max(len(v) for v in mapping.values()) if mapping else 1
        self._vecs: Dict[str, np.ndarray] = {
            k: np.array(v, dtype=float) for k, v in mapping.items()
        }

    def encode(self, texts: List[str]) -> np.ndarray:
        rows = []
        for t in texts:
            rows.append(self._vecs.get(t, np.zeros(self._dim)))
        return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact(**kwargs) -> Fact:
    defaults = dict(
        document_id="doc1",
        statement="Test statement.",
        canonical_entity="SomeEntity",
        raw_attribute="some_attr",
        value="100",
        unit=None,
        confidence=0.9,
        source_chunk_id="chunk1",
        source_type="text",
        evidence_quote="100",
        evidence_bbox=(0.1, 0.1, 0.5, 0.2),
    )
    defaults.update(kwargs)
    return Fact(**defaults)


# ---------------------------------------------------------------------------
# 1. Entity resolution
# ---------------------------------------------------------------------------

class TestEntityResolver:

    def test_near_duplicate_names_resolve_to_same_canonical(self, tmp_path):
        """
        "Delhivery Limited" (full name) and "Delhivery" (short name) should
        resolve to the same canonical entity when their embeddings are similar.
        """
        # Make "Delhivery" and "Delhivery Limited" nearly identical vectors
        embedder = FakeEmbedder({
            "Delhivery Limited": [1.0, 0.0, 0.0],
            "Delhivery":         [0.99, 0.01, 0.0],  # cosine ≈ 0.9999
        })
        store = tmp_path / "entities.json"
        registry = EntityRegistry(store_path=store, threshold=ENTITY_MATCH_THRESHOLD)

        # First call: "Delhivery Limited" is promoted as new canonical
        name1, score1 = registry.resolve("Delhivery Limited", embedder)
        assert name1 == "Delhivery Limited"
        assert score1 == 1.0

        # Second call: "Delhivery" should snap to "Delhivery Limited"
        name2, score2 = registry.resolve("Delhivery", embedder)
        assert name2 == "Delhivery Limited", (
            f"Expected 'Delhivery' to resolve to 'Delhivery Limited', got {name2!r}"
        )
        assert score2 > ENTITY_MATCH_THRESHOLD

    def test_distinct_entities_stay_separate(self, tmp_path):
        """Two genuinely different entities must NOT be merged."""
        embedder = FakeEmbedder({
            "Acme Corp":   [1.0, 0.0, 0.0],
            "Beta Ltd":    [0.0, 1.0, 0.0],  # cosine = 0.0
        })
        store = tmp_path / "entities.json"
        registry = EntityRegistry(store_path=store, threshold=ENTITY_MATCH_THRESHOLD)

        registry.resolve("Acme Corp", embedder)
        name, score = registry.resolve("Beta Ltd", embedder)
        assert name == "Beta Ltd"
        assert score == 1.0  # promoted as new
        assert len(registry) == 2

    def test_stateless_resolve_entity(self):
        """resolve_entity() works without any registry (pure stateless)."""
        embedder = FakeEmbedder({
            "Foo Inc":  [1.0, 0.0],
            "Foo":      [0.98, 0.02],   # cosine ≈ 0.9998
            "Bar Corp": [0.0, 1.0],
        })
        known = ["Foo Inc"]
        canonical, score = resolve_entity("Foo", known, embedder)
        assert canonical == "Foo Inc"
        assert score > ENTITY_MATCH_THRESHOLD

        canonical2, score2 = resolve_entity("Bar Corp", known, embedder)
        assert canonical2 == "Bar Corp"   # not in known — returned unchanged
        assert score2 == 1.0

    def test_registry_persists_across_instances(self, tmp_path):
        """Names added in one EntityRegistry instance survive a new instance."""
        store = tmp_path / "entities.json"
        embedder = FakeEmbedder({"Widget Corp": [1.0, 0.0]})

        r1 = EntityRegistry(store_path=store, threshold=ENTITY_MATCH_THRESHOLD)
        r1.resolve("Widget Corp", embedder)

        r2 = EntityRegistry(store_path=store, threshold=ENTITY_MATCH_THRESHOLD)
        assert "Widget Corp" in r2.canonical_names

    def test_empty_entity_string_handled(self, tmp_path):
        """Empty string should return unchanged with score 0.0."""
        store = tmp_path / "entities.json"
        embedder = FakeEmbedder({})
        registry = EntityRegistry(store_path=store)
        name, score = registry.resolve("", embedder)
        assert name == ""
        assert score == 0.0


# ---------------------------------------------------------------------------
# 2. Attribute snapping (Multi-Modal)
# ---------------------------------------------------------------------------

class TestAttributeTaxonomy:

    def test_near_duplicate_attributes_snap_to_same_canonical_semantic(self, tmp_path):
        """
        "EBITDA" and "operating_ebitda" should snap to one canonical attribute
        when embeddings are similar (above threshold).
        """
        embedder = FakeEmbedder({
            "EBITDA":           [1.0, 0.0, 0.0],
            "operating_ebitda": [0.97, 0.1, 0.0],  # cosine ≈ 0.9949
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)

        canonical1, raw1 = taxonomy.snap("EBITDA", embedder)
        assert canonical1 == "EBITDA"
        assert raw1 == "EBITDA"

        canonical2, raw2 = taxonomy.snap("operating_ebitda", embedder)
        assert canonical2 == "EBITDA", (
            f"Expected 'operating_ebitda' to snap to 'EBITDA', got {canonical2!r}"
        )
        assert raw2 == "operating_ebitda"   # original preserved

    def test_lexical_fuzzy_match_snaps_regardless_of_embedding(self, tmp_path):
        """
        'market_cap' and 'market_capitalization' share high token-set ratio, 
        so they should snap even if the embedder thinks they are unrelated (0.0 similarity).
        """
        embedder = FakeEmbedder({
            "market_capitalization": [1.0, 0.0, 0.0],
            "market_cap":            [0.0, 1.0, 0.0],  # 0.0 similarity!
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)
        taxonomy.snap("market_capitalization", embedder)
        
        canonical, raw = taxonomy.snap("market_cap", embedder)
        assert canonical == "market_capitalization"
        
    def test_acronym_match_snaps_regardless_of_embedding(self, tmp_path):
        """
        'PAT' is an exact acronym of 'Profit After Tax', 
        so they should snap even if embeddings differ completely.
        """
        embedder = FakeEmbedder({
            "Profit After Tax": [1.0, 0.0, 0.0],
            "PAT":              [0.0, 1.0, 0.0],  # 0.0 similarity!
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)
        taxonomy.snap("Profit After Tax", embedder)
        
        canonical, raw = taxonomy.snap("PAT", embedder)
        assert canonical == "Profit After Tax"

    def test_acronym_match_is_order_independent(self, tmp_path):
        """Acronym match works whether the acronym or the phrase is encountered first."""
        embedder = FakeEmbedder({
            "PAT":              [1.0, 0.0, 0.0],
            "Profit After Tax": [0.0, 1.0, 0.0],
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)
        taxonomy.snap("PAT", embedder)
        
        canonical, raw = taxonomy.snap("Profit After Tax", embedder)
        assert canonical == "PAT"

    def test_novel_attribute_gets_promoted(self, tmp_path):
        """A genuinely novel attribute (low similarity) becomes a new canonical."""
        embedder = FakeEmbedder({
            "EBITDA":       [1.0, 0.0, 0.0],
            "market_share": [0.0, 1.0, 0.0],  # cosine = 0.0 → definitely novel
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)

        taxonomy.snap("EBITDA", embedder)
        canonical, raw = taxonomy.snap("market_share", embedder)
        assert canonical == "market_share"
        assert len(taxonomy) == 2

    def test_stateless_snap_attribute(self):
        """snap_attribute() is a pure stateless function."""
        embedder = FakeEmbedder({
            "revenue":      [1.0, 0.0],
            "net_revenue":  [0.96, 0.04],   # cosine ≈ 0.9992
            "headcount":    [0.0, 1.0],
        })
        known = ["revenue"]
        canonical, raw = snap_attribute("net_revenue", known, embedder)
        assert canonical == "revenue"
        assert raw == "net_revenue"

        canonical2, raw2 = snap_attribute("headcount", known, embedder)
        assert canonical2 == "headcount"   # not in known

    def test_original_attribute_preserved_in_snap(self, tmp_path):
        """snap() must always return the raw string as the second element."""
        embedder = FakeEmbedder({
            "profit":     [1.0, 0.0],
            "net profit": [0.99, 0.0],
        })
        store = tmp_path / "attrs.json"
        taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)
        taxonomy.snap("profit", embedder)

        _, raw = taxonomy.snap("net profit", embedder)
        assert raw == "net profit"   # always the original


# ---------------------------------------------------------------------------
# 3. Unit normalization
# ---------------------------------------------------------------------------

class TestUnitNormalization:

    def test_crore_to_absolute(self):
        result = normalize_unit("42 crore", "INR")
        assert result["parse_ok"] is True
        assert result["normalized_value"] == pytest.approx(42 * 1e7)
        assert result["currency"] == "INR"

    def test_lakh_crore_to_absolute(self):
        result = normalize_unit("0.42 lakh crore", "INR")
        assert result["parse_ok"] is True
        assert result["normalized_value"] == pytest.approx(0.42 * 1e12)

    def test_crore_and_lakh_crore_same_absolute_value(self):
        """42 crore == 0.42 lakh crore (both → 4.2e8).
        
        Arithmetic: 1 lakh crore = 10^12, 1 crore = 10^7
        42 crore = 42 × 10^7 = 4.2 × 10^8
        0.42 lakh crore = 0.42 × 10^12 = 4.2 × 10^11  ← different scale!
        
        These are NOT the same — the test was wrong. Keep them as separate
        assertions confirming each scale factor is correct individually.
        """
        r1 = normalize_unit("42 crore", "INR")
        r2 = normalize_unit("420000000", "INR")   # 42 crore in absolute ones
        assert r1["normalized_value"] == pytest.approx(4.2e8)
        assert r2["normalized_value"] == pytest.approx(4.2e8)

        # lakh crore scale
        r3 = normalize_unit("0.42 lakh crore", "INR")
        assert r3["normalized_value"] == pytest.approx(0.42e12)

    def test_million_normalization(self):
        result = normalize_unit("500 million", "USD")
        assert result["parse_ok"] is True
        assert result["normalized_value"] == pytest.approx(500 * 1e6)
        assert result["currency"] == "USD"

    def test_billion_normalization(self):
        result = normalize_unit("1.5 billion", "USD")
        assert result["normalized_value"] == pytest.approx(1.5e9)

    def test_percentage(self):
        result = normalize_unit("12.5%", None)
        assert result["parse_ok"] is True
        assert result["normalized_value"] == pytest.approx(12.5)
        assert result["normalized_unit"] == "%"

    def test_basis_points(self):
        result = normalize_unit("200 bps", None)
        assert result["parse_ok"] is True
        assert result["normalized_value"] == pytest.approx(200.0)
        assert result["normalized_unit"] == "bps"

    def test_cross_currency_not_merged(self):
        """
        A fact with value in INR and unit labelled USD must be flagged as a
        cross-currency conflict — NOT silently converted or merged.
        """
        result = normalize_unit("100 crore INR", "USD")
        assert result["currency_conflict"] is True, (
            "Cross-currency INR/USD fact was NOT flagged — this would cause "
            "incorrect cross-currency merges in Phase 4."
        )

    def test_inr_facts_stay_inr(self):
        """Two INR facts should NOT trigger a currency conflict."""
        result = normalize_unit("50 crore", "INR")
        assert result["currency_conflict"] is False

    def test_fy_period_normalization(self):
        result = normalize_unit("FY2024-25", None)
        assert result["parse_ok"] is True
        assert result["period"] == "FY"
        assert result["vintage"] == "FY2024-25" or "2024" in result["vintage"]

    def test_quarter_period_normalization(self):
        result = normalize_unit("Q3 FY24", None)
        assert result["parse_ok"] is True
        assert result["period"] == "Q3"
        assert "2024" in result["vintage"]

    def test_same_currency_helper(self):
        assert same_currency("INR", "INR") is True
        assert same_currency("USD", "INR") is False
        assert same_currency(None, None) is True

    def test_unparseable_value_returns_parse_ok_false(self):
        result = normalize_unit("unknown gibberish", None)
        assert result["parse_ok"] is False


# ---------------------------------------------------------------------------
# 4. End-to-end pipeline
# ---------------------------------------------------------------------------

class TestCanonicalizationPipeline:

    def _make_embedder(self):
        """
        Embedder for pipeline tests with four concepts:
          - "Acme" and "Acme Corp" → nearly identical (high similarity)
          - "profit_margin" and "net_profit_margin" → nearly identical
          - "SomeOtherEntity" and "headcount" → orthogonal to all above
        """
        return FakeEmbedder({
            "Acme Corp":          [1.0, 0.0, 0.0, 0.0],
            "Acme":               [0.99, 0.02, 0.0, 0.0],
            "SomeOtherEntity":    [0.0, 0.0, 1.0, 0.0],
            "profit_margin":      [0.0, 1.0, 0.0, 0.0],
            "net_profit_margin":  [0.0, 0.97, 0.02, 0.0],
            "headcount":          [0.0, 0.0, 0.0, 1.0],
            "some_attr":          [0.5, 0.5, 0.0, 0.0],
        })

    def test_entity_resolved_in_pipeline(self, tmp_path):
        """Entity near-duplicates are merged by the pipeline."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        fact1 = _make_fact(canonical_entity="Acme Corp", raw_attribute="profit_margin")
        fact2 = _make_fact(canonical_entity="Acme",      raw_attribute="net_profit_margin")

        results = canonicalize_facts(
            [fact1, fact2], embedder,
            entity_store=store_e, attribute_store=store_a,
        )

        assert results[0].canonical_entity == "Acme Corp"
        assert results[1].canonical_entity == "Acme Corp", (
            "Second fact's entity 'Acme' should have resolved to 'Acme Corp'"
        )

    def test_attribute_snapped_in_pipeline(self, tmp_path):
        """Attribute near-duplicates are snapped by the pipeline."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        fact1 = _make_fact(canonical_entity="Acme Corp", raw_attribute="profit_margin")
        fact2 = _make_fact(canonical_entity="Acme Corp", raw_attribute="net_profit_margin")

        results = canonicalize_facts(
            [fact1, fact2], embedder,
            entity_store=store_e, attribute_store=store_a,
        )

        assert results[0].raw_attribute == "profit_margin"
        assert results[1].raw_attribute == "profit_margin", (
            "net_profit_margin should snap to profit_margin"
        )

    def test_original_attribute_preserved_in_context(self, tmp_path):
        """The raw attribute phrasing must be preserved in context."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        fact1 = _make_fact(canonical_entity="Acme Corp", raw_attribute="profit_margin")
        fact2 = _make_fact(canonical_entity="Acme Corp", raw_attribute="net_profit_margin")

        results = canonicalize_facts(
            [fact1, fact2], embedder,
            entity_store=store_e, attribute_store=store_a,
        )
        assert results[1].context.get("original_attribute") == "net_profit_margin"

    def test_cross_currency_not_merged_in_pipeline(self, tmp_path):
        """Cross-currency facts must be flagged, not silently merged."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        inr_fact = _make_fact(
            canonical_entity="Acme Corp",
            raw_attribute="profit_margin",
            value="100 crore INR",
            unit="USD",  # deliberate mismatch → conflict
        )
        results = canonicalize_facts(
            [inr_fact], embedder,
            entity_store=store_e, attribute_store=store_a,
        )
        assert results[0].context.get("currency_conflict") is True

    def test_content_hash_recomputed(self, tmp_path):
        """After canonicalization the content_hash must reflect canonical values."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        fact1 = _make_fact(canonical_entity="Acme Corp", raw_attribute="profit_margin", value="10%")
        fact2 = _make_fact(canonical_entity="Acme",      raw_attribute="net_profit_margin", value="10%")

        results = canonicalize_facts(
            [fact1, fact2], embedder,
            entity_store=store_e, attribute_store=store_a,
        )
        # After canonicalization, both facts have same entity+attribute+value
        # so their content_hashes must match (enabling Phase 4 dedup)
        assert results[0].content_hash == results[1].content_hash, (
            "Canonicalized near-duplicate facts should have identical content_hashes"
        )

    def test_input_facts_not_mutated(self, tmp_path):
        """The pipeline must return new objects; inputs must be unchanged."""
        embedder = self._make_embedder()
        store_e = tmp_path / "entities.json"
        store_a = tmp_path / "attrs.json"

        original_entity = "Acme"
        fact = _make_fact(canonical_entity=original_entity, raw_attribute="headcount")
        # Pre-populate registry with "Acme Corp" so "Acme" will be resolved
        from canonicalization.entity_resolver import EntityRegistry
        reg = EntityRegistry(store_path=store_e, threshold=ENTITY_MATCH_THRESHOLD)
        reg._canonical_names = ["Acme Corp"]
        reg.save()

        results = canonicalize_facts(
            [fact], embedder,
            entity_store=store_e, attribute_store=store_a,
        )
        assert fact.canonical_entity == original_entity, "Input fact was mutated!"
