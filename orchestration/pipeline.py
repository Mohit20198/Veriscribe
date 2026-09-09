"""
orchestration/pipeline.py
--------------------------
Single-document ingestion entry point for Veriscribe.

Wires Phases 1-4 into one persistent, incremental, idempotent call:

    ingest_document(pdf_path, document_id, store, vector_store, client, embedder)
    -> IngestResult

Flow
----
0. Idempotency check — if document_id already has facts in SQLite, skip
   unless force=True.
1. CorroborationClusters rebuilt from store (union-find reflects ALL
   prior corroborations, not just this session).
2. Phase 1 — parse_document() -> List[Chunk]
3. Phase 2 — extract_document_facts() -> List[Fact]  (LLM extraction)
4. Phase 3 — canonicalize_facts() -> List[Fact]      (entity/attr/unit)
5. Phase 4 — for each canonical fact: compare_new_fact()
   - try_deterministic_reconcile() fires first (zero LLM calls if match)
   - falls back to judge_pair() only when deterministic path returns None
6. Returns IngestResult summary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestion.pipeline import parse_document
from extraction.pipeline import extract_document_facts
from canonicalization.pipeline import canonicalize_facts
from comparison.pipeline import compare_new_fact
from comparison.storage import SQLiteStore
from comparison.vector_store import VectorStore
from comparison.clustering import CorroborationClusters
from comparison.models import RelationshipType

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: str
    skipped: bool = False
    fact_count: int = 0
    chunk_count: int = 0
    relationship_counts: Dict[str, int] = field(default_factory=dict)
    deterministic_matches: int = 0
    error_count: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.skipped:
            return (
                f"[SKIPPED] {self.document_id} already ingested "
                f"(use force=True to re-process)."
            )
        rel_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.relationship_counts.items())
        )
        return (
            f"[DONE] {self.document_id} | "
            f"{self.chunk_count} chunks -> {self.fact_count} facts | "
            f"relationships: {rel_str or 'none'} "
            f"({self.deterministic_matches} deterministic) | "
            f"{self.error_count} errors | "
            f"{self.elapsed_seconds:.1f}s"
        )


def ingest_document(
    pdf_path: str,
    document_id: str,
    store: SQLiteStore,
    vector_store: VectorStore,
    client: Any,
    embedder: Any,
    force: bool = False,
    entity_store: Optional[Path] = None,
    attribute_store: Optional[Path] = None,
) -> IngestResult:
    """
    Full pipeline: PDF -> stored facts + relationships.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the source PDF.
    document_id:
        Unique, caller-supplied identifier.  Used for idempotency checks.
    store:
        SQLiteStore instance (shared across all ingestion calls).
    vector_store:
        VectorStore instance (shared across all ingestion calls).
    client:
        OpenAI-compatible HTTP client pointed at OpenRouter.
    embedder:
        SentenceTransformer (shared, not reloaded per call).
    force:
        If True, re-ingest even if document_id already has facts stored.
    entity_store / attribute_store:
        Optional overrides for JSON persistence paths (useful for tests).
    """
    t_start = time.perf_counter()
    result = IngestResult(document_id=document_id)

    # ------------------------------------------------------------------
    # 0. Idempotency check
    # ------------------------------------------------------------------
    existing = store.get_facts_by_document(document_id)
    if existing and not force:
        log.info(
            "Document %r already has %d facts -- skipping (use force=True to re-process).",
            document_id, len(existing)
        )
        result.skipped = True
        result.fact_count = len(existing)
        return result

    if existing and force:
        log.warning(
            "force=True: re-ingesting %r (already has %d facts). "
            "Existing records are NOT deleted.",
            document_id, len(existing)
        )

    # ------------------------------------------------------------------
    # 1. Rebuild CorroborationClusters from ALL stored corroborations
    #    BEFORE any compare_new_fact() call so union-find reflects the
    #    full global state, not just this run.
    # ------------------------------------------------------------------
    prior_corroborations = store.get_all_corroborations()
    clusters = CorroborationClusters(initial_pairs=prior_corroborations)
    log.info("CorroborationClusters seeded with %d prior pairs.", len(prior_corroborations))

    # ------------------------------------------------------------------
    # 2. Phase 1 -- Ingestion
    # ------------------------------------------------------------------
    log.info("Phase 1: parsing %s", pdf_path)
    import os
    try:
        chunks = parse_document(pdf_path, document_id)
        dev_keyword = os.environ.get("DEV_FILTER_KEYWORD")
        if dev_keyword:
            filtered = [c for c in chunks if c.text and dev_keyword.lower() in c.text.lower()]
            log.warning(f"DEV_FILTER_KEYWORD={dev_keyword} applied: keeping {len(filtered)} out of {len(chunks)} chunks")
            chunks = filtered
    except Exception as exc:
        msg = f"Phase 1 failed for {pdf_path}: {exc}"
        log.error(msg)
        result.errors.append(msg)
        result.error_count += 1
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    result.chunk_count = len(chunks)
    log.info("Phase 1 done: %d chunks.", result.chunk_count)

    # ------------------------------------------------------------------
    # 3. Phase 2 -- Extraction
    # ------------------------------------------------------------------
    log.info("Phase 2: extracting facts from %d chunks.", result.chunk_count)
    try:
        raw_facts = extract_document_facts(chunks, client)
    except Exception as exc:
        msg = f"Phase 2 failed: {exc}"
        log.error(msg)
        result.errors.append(msg)
        result.error_count += 1
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    log.info("Phase 2 done: %d raw facts.", len(raw_facts))

    if not raw_facts:
        log.warning("No facts extracted from %r -- nothing to store.", document_id)
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    # ------------------------------------------------------------------
    # 4. Phase 3 -- Canonicalization
    # ------------------------------------------------------------------
    log.info("Phase 3: canonicalizing %d facts.", len(raw_facts))
    try:
        canon_facts = canonicalize_facts(
            raw_facts,
            embedder,
            entity_store=entity_store,
            attribute_store=attribute_store,
        )
    except Exception as exc:
        msg = f"Phase 3 failed: {exc}"
        log.error(msg)
        result.errors.append(msg)
        result.error_count += 1
        result.elapsed_seconds = time.perf_counter() - t_start
        return result

    result.fact_count = len(canon_facts)
    log.info("Phase 3 done: %d canonical facts.", result.fact_count)

    # ------------------------------------------------------------------
    # 5. Phase 4 -- Comparison (per-fact, incremental)
    # ------------------------------------------------------------------
    log.info("Phase 4: comparing and storing %d facts.", result.fact_count)
    rel_counts: Dict[str, int] = {}
    det_matches = 0

    for fact in canon_facts:
        try:
            new_rels = compare_new_fact(
                fact=fact,
                store=store,
                vector_store=vector_store,
                client=client,
                embedder=embedder,
                clusters=clusters,
                entity_store=entity_store,
                attribute_store=attribute_store,
            )
            for rel in new_rels:
                key = rel.relationship_type.value
                rel_counts[key] = rel_counts.get(key, 0) + 1
                if rel.confidence == 1.0 and "within the" in rel.justification or "exact" in rel.justification.lower() or "identical" in rel.justification.lower():
                    det_matches += 1
        except Exception as exc:
            msg = f"Phase 4 error on fact {fact.fact_id}: {exc}"
            log.error(msg)
            result.errors.append(msg)
            result.error_count += 1

    result.relationship_counts = rel_counts
    result.deterministic_matches = det_matches
    result.elapsed_seconds = time.perf_counter() - t_start
    log.info("Phase 4 done. %s", result.summary())
    return result
