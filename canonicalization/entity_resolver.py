"""
canonicalization/entity_resolver.py
-------------------------------------
Resolve raw entity name strings to a growing canonical entity list using
cosine similarity over sentence embeddings.

Design notes:
- Embedder is injected, not instantiated here — caller owns the model lifecycle.
- The canonical entity list grows organically from whatever flows through;
  NO pre-seeded domain-specific names exist in this file.
- Persistence: the canonical list is saved to / loaded from a plain JSON file
  so it survives across runs without a database dependency.
- ENTITY_MATCH_THRESHOLD is a named constant. The value 0.85 is a starting
  point calibrated to tolerate abbreviation vs full-name variance (e.g.
  "Acme" vs "Acme Corp") while rejecting genuinely different entities. It
  SHOULD be tuned based on observed false-positive / false-negative rates in
  production, and may need to be lower for short, ambiguous strings.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable threshold — NOT a magic number.
# Increase to require tighter matches (fewer merges, more false-negatives).
# Decrease to merge more aggressively (fewer splits, more false-positives).
# ---------------------------------------------------------------------------
ENTITY_MATCH_THRESHOLD: float = 0.85

_DEFAULT_STORE = Path("output") / "canonical_entities.json"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EntityRegistry:
    """
    A mutable registry of canonical entity names backed by a JSON file.

    The registry is intentionally open-schema — it makes no assumptions about
    what domains or entity types will flow through it.

    Parameters
    ----------
    store_path:
        Path to the JSON persistence file.  Created on first write if absent.
    threshold:
        Cosine similarity above which a raw entity name is considered a match
        for an existing canonical name.
    """

    def __init__(
        self,
        store_path: os.PathLike = _DEFAULT_STORE,
        threshold: float = ENTITY_MATCH_THRESHOLD,
    ) -> None:
        self.store_path = Path(store_path)
        self.threshold = threshold
        # canonical_names: ordered list of canonical strings
        # embeddings: parallel list of np.ndarray (loaded lazily)
        self._canonical_names: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load canonical entity names from disk (embeddings recomputed on demand)."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                self._canonical_names = data.get("entities", [])
                log.debug("Loaded %d canonical entities from %s", len(self._canonical_names), self.store_path)
            except Exception as exc:
                log.warning("Could not load entity registry from %s: %s", self.store_path, exc)
                self._canonical_names = []
        self._embeddings = []  # will be populated on first embedder call

    def save(self) -> None:
        """Persist canonical entity names to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entities": self._canonical_names}
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.debug("Saved %d canonical entities to %s", len(self._canonical_names), self.store_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def resolve(
        self,
        raw_entity: str,
        embedder: Any,
    ) -> Tuple[str, float]:
        """
        Resolve a raw entity string to a canonical name.

        Parameters
        ----------
        raw_entity:
            The entity name as extracted by the LLM (may be noisy, abbreviated,
            or phrased differently from existing canonical names).
        embedder:
            Any object with an ``encode(texts: List[str]) -> np.ndarray`` method
            (sentence-transformers SentenceTransformer satisfies this interface,
            as does any mock that matches the same signature).

        Returns
        -------
        (canonical_name, similarity)
            If a match is found: the existing canonical name and the cosine
            similarity score.
            If no match: ``raw_entity`` is promoted to the canonical set
            (and persisted) and returned with similarity 1.0.
        """
        raw_entity = raw_entity.strip()
        if not raw_entity:
            return raw_entity, 0.0

        query_vec = embedder.encode([raw_entity])[0]

        # Lazily embed any canonical names that don't yet have embeddings
        # (e.g. loaded fresh from disk on a new run)
        if self._canonical_names and len(self._embeddings) < len(self._canonical_names):
            missing = self._canonical_names[len(self._embeddings):]
            new_vecs = embedder.encode(missing)
            self._embeddings.extend(new_vecs)

        # Find best match among existing canonical names
        best_score = -1.0
        best_idx = -1
        for i, vec in enumerate(self._embeddings):
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.threshold and best_idx >= 0:
            canonical = self._canonical_names[best_idx]
            log.debug(
                "Entity %r resolved to %r (similarity=%.3f)", raw_entity, canonical, best_score
            )
            return canonical, best_score

        # No match — promote raw_entity as a new canonical name
        log.debug("Entity %r promoted as new canonical (best_score=%.3f)", raw_entity, best_score)
        self._canonical_names.append(raw_entity)
        self._embeddings.append(query_vec)
        self.save()
        return raw_entity, 1.0

    @property
    def canonical_names(self) -> List[str]:
        return list(self._canonical_names)

    def __len__(self) -> int:
        return len(self._canonical_names)


# ---------------------------------------------------------------------------
# Module-level convenience function (used by pipeline.py)
# ---------------------------------------------------------------------------

def resolve_entity(
    raw_entity: str,
    known_entities: List[str],
    embedder: Any,
    threshold: float = ENTITY_MATCH_THRESHOLD,
) -> Tuple[str, float]:
    """
    Stateless convenience wrapper: resolve a raw entity name against an
    explicit list of known canonical names.

    Unlike ``EntityRegistry.resolve``, this does NOT persist anything — it is
    a pure function suitable for testing and for callers that manage their own
    canonical list externally.

    Parameters
    ----------
    raw_entity:
        The raw entity string to resolve.
    known_entities:
        List of currently canonical entity names.
    embedder:
        Object with ``encode(texts: List[str]) -> np.ndarray``.
    threshold:
        Cosine similarity cutoff.

    Returns
    -------
    (matched_or_raw, similarity)
    """
    raw_entity = raw_entity.strip()
    if not raw_entity:
        return raw_entity, 0.0

    if not known_entities:
        return raw_entity, 1.0

    all_texts = [raw_entity] + list(known_entities)
    vecs = embedder.encode(all_texts)
    query_vec = vecs[0]
    known_vecs = vecs[1:]

    best_score = -1.0
    best_idx = -1
    for i, vec in enumerate(known_vecs):
        score = _cosine(query_vec, vec)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= threshold and best_idx >= 0:
        return known_entities[best_idx], best_score

    return raw_entity, 1.0
