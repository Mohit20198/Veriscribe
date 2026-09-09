"""
canonicalization/pipeline.py
------------------------------
Orchestrate entity resolution, attribute snapping, and unit normalization
for a list of Facts.

The pipeline:
1. Resolve each fact's ``canonical_entity`` via embedding similarity.
2. Snap each fact's ``raw_attribute`` to the canonical taxonomy; preserve
   the original phrasing in ``context["original_attribute"]``.
3. Normalize ``value`` + ``unit`` using the deterministic unit normalizer;
   store the result in ``context["normalized"]``.
4. Recompute ``content_hash`` so downstream deduplication (Phase 4) keys
   off canonical, normalized values rather than raw LLM text.

The pipeline is stateful across facts in the same batch (entity/attribute
registries are shared so cross-document near-duplicates get merged).  It
is re-entrant across batches via the JSON persistence store.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, List, Optional

from extraction.models import Fact
from canonicalization.entity_resolver import EntityRegistry, ENTITY_MATCH_THRESHOLD
from canonicalization.attribute_taxonomy import AttributeTaxonomy, ATTRIBUTE_MATCH_THRESHOLD
from canonicalization.units import normalize_unit

log = logging.getLogger(__name__)

_DEFAULT_ENTITY_STORE    = Path("output") / "canonical_entities.json"
_DEFAULT_ATTRIBUTE_STORE = Path("output") / "canonical_attributes.json"


def canonicalize_facts(
    facts: List[Fact],
    embedder: Any,
    entity_store: Optional[Path] = None,
    attribute_store: Optional[Path] = None,
    entity_threshold: float = ENTITY_MATCH_THRESHOLD,
    attribute_threshold: float = ATTRIBUTE_MATCH_THRESHOLD,
) -> List[Fact]:
    """
    Apply the full canonicalization pipeline to a list of Facts.

    Returns new Fact objects (the input list is not mutated).

    Parameters
    ----------
    facts:
        List of Facts from Phase 2 extraction.
    embedder:
        Object with ``encode(texts: List[str]) -> np.ndarray``.
        For production: ``SentenceTransformer("all-MiniLM-L6-v2")``.
        For tests: any mock satisfying the same signature.
    entity_store:
        Path to entity registry JSON.  Defaults to output/canonical_entities.json.
    attribute_store:
        Path to attribute taxonomy JSON.  Defaults to output/canonical_attributes.json.
    entity_threshold:
        Cosine similarity threshold for entity matching.
    attribute_threshold:
        Cosine similarity threshold for attribute snapping.
    """
    entity_registry = EntityRegistry(
        store_path=entity_store or _DEFAULT_ENTITY_STORE,
        threshold=entity_threshold,
    )
    attr_taxonomy = AttributeTaxonomy(
        store_path=attribute_store or _DEFAULT_ATTRIBUTE_STORE,
        threshold=attribute_threshold,
    )

    canonicalized: List[Fact] = []

    for fact in facts:
        # Work on a deep copy so we never mutate the caller's objects
        f = fact.model_copy(deep=True)

        # ------------------------------------------------------------------
        # Step 1: Entity resolution
        # ------------------------------------------------------------------
        canonical_entity, entity_score = entity_registry.resolve(
            f.canonical_entity, embedder
        )
        if canonical_entity != f.canonical_entity:
            log.info(
                "Entity %r → %r (score=%.3f) for fact %s",
                f.canonical_entity, canonical_entity, entity_score, f.fact_id,
            )
        f.canonical_entity = canonical_entity

        # ------------------------------------------------------------------
        # Step 2: Attribute snapping
        # ------------------------------------------------------------------
        canonical_attr, original_attr = attr_taxonomy.snap(f.raw_attribute, embedder)

        # Preserve original phrasing in context — never discard what the LLM said
        if "original_attribute" not in f.context:
            f.context["original_attribute"] = original_attr

        if canonical_attr != f.raw_attribute:
            log.info(
                "Attribute %r → %r for fact %s",
                f.raw_attribute, canonical_attr, f.fact_id,
            )
        f.raw_attribute = canonical_attr

        # ------------------------------------------------------------------
        # Step 3: Unit normalization
        # ------------------------------------------------------------------
        norm = normalize_unit(f.value, f.unit)
        f.context["normalized"] = norm

        # If cross-currency, flag explicitly so Phase 4 does NOT merge these
        if norm.get("currency_conflict"):
            f.context["currency_conflict"] = True
            log.warning(
                "Cross-currency conflict in fact %s: value=%r, unit=%r",
                f.fact_id, f.value, f.unit,
            )

        # ------------------------------------------------------------------
        # Step 4: Recompute content_hash on canonical/normalized values
        # ------------------------------------------------------------------
        f.content_hash = f.compute_hash()

        canonicalized.append(f)

    return canonicalized
