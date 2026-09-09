"""
canonicalization/attribute_taxonomy.py
----------------------------------------
Snap raw attribute strings to a growing canonical attribute taxonomy using
cosine similarity over sentence embeddings.

Design:
- Parallel structure to entity_resolver — same embedder interface, same
  persistence pattern, same threshold-as-named-constant discipline.
- Each canonical attribute stores its canonical form; the raw phrasing is
  preserved by the caller in context["original_attribute"].
- The taxonomy grows organically: no domain-specific starter values are
  pre-seeded here. The JSON store starts empty and expands as new attributes
  are encountered across documents.
- ATTRIBUTE_MATCH_THRESHOLD is deliberately slightly higher than the entity
  threshold (0.88 vs 0.85) because attribute names tend to be shorter and
  more ambiguous than entity names — a lower threshold here would cause
  semantically different attributes (e.g. "revenue" vs "net_revenue") to
  collapse incorrectly. Tune based on observed merge/split errors.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from canonicalization.entity_resolver import _cosine

import re
from rapidfuzz import fuzz

def _is_acronym_match(a: str, b: str) -> bool:
    """True if one is an acronym of the other, e.g. PAT == Profit After Tax."""
    # Determine which is the potential acronym (fully uppercase, length >= 2)
    # and which is the phrase
    if a.isupper() and len(a) >= 2 and not b.isupper():
        acronym, phrase = a, b
    elif b.isupper() and len(b) >= 2 and not a.isupper():
        acronym, phrase = b, a
    else:
        return False

    # Extract first letter of each word in the phrase
    words = re.findall(r'[a-zA-Z]+', phrase)
    if not words:
        return False
    phrase_acronym = "".join(w[0].upper() for w in words)
    return acronym == phrase_acronym

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable threshold — NOT a magic number.
# With multi-modal matching (lexical + acronym) handling the ambiguous
# short strings, semantic threshold can safely drop to 0.85 to match entities
# without incorrectly collapsing "Revenue" and "Profit".
# ---------------------------------------------------------------------------
ATTRIBUTE_MATCH_THRESHOLD: float = 0.85

_DEFAULT_STORE = Path("output") / "canonical_attributes.json"


class AttributeTaxonomy:
    """
    A mutable, persisted taxonomy of canonical attribute names.

    Attributes are stored as a flat set of canonical strings.  No hierarchy,
    no domain-specific structure — the taxonomy is purely data-driven.

    Parameters
    ----------
    store_path:
        Path to the JSON persistence file.  Created on first write if absent.
    threshold:
        Cosine similarity above which a raw attribute is snapped to an
        existing canonical attribute.
    """

    def __init__(
        self,
        store_path: os.PathLike = _DEFAULT_STORE,
        threshold: float = ATTRIBUTE_MATCH_THRESHOLD,
    ) -> None:
        self.store_path = Path(store_path)
        self.threshold = threshold
        self._canonical_attrs: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                self._canonical_attrs = data.get("attributes", [])
                log.debug(
                    "Loaded %d canonical attributes from %s",
                    len(self._canonical_attrs),
                    self.store_path,
                )
            except Exception as exc:
                log.warning(
                    "Could not load attribute taxonomy from %s: %s", self.store_path, exc
                )
                self._canonical_attrs = []
        self._embeddings = []

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"attributes": self._canonical_attrs}
        self.store_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.debug(
            "Saved %d canonical attributes to %s",
            len(self._canonical_attrs),
            self.store_path,
        )

# ------------------------------------------------------------------
# Core API
# ------------------------------------------------------------------

    def snap(
        self,
        raw_attribute: str,
        embedder: Any,
    ) -> Tuple[str, str]:
        raw_attribute = raw_attribute.strip()
        if not raw_attribute:
            return raw_attribute, raw_attribute

        # 1. Acronym or Lexical Match
        # Fast, deterministic path before leaning on embeddings
        for canonical in self._canonical_attrs:
            # Acronym check
            if _is_acronym_match(raw_attribute, canonical):
                log.debug("Attribute %r snapped to %r (ACRONYM)", raw_attribute, canonical)
                return canonical, raw_attribute
            
            a_lower = raw_attribute.lower()
            c_lower = canonical.lower()
            
            # Fuzzy match check
            if fuzz.token_set_ratio(a_lower, c_lower) > 90:
                log.debug("Attribute %r snapped to %r (LEXICAL_FUZZY)", raw_attribute, canonical)
                return canonical, raw_attribute
                
            # Prefix check for abbreviations (e.g. market_cap vs market_capitalization)
            # Must be a strict prefix of length >= 5 to avoid short generic words matching
            if min(len(a_lower), len(c_lower)) >= 5:
                if a_lower.startswith(c_lower) or c_lower.startswith(a_lower):
                    log.debug("Attribute %r snapped to %r (LEXICAL_PREFIX)", raw_attribute, canonical)
                    return canonical, raw_attribute

        # 2. Semantic Embedding Match
        query_vec = embedder.encode([raw_attribute])[0]

        # Lazily embed names loaded from disk
        if self._canonical_attrs and len(self._embeddings) < len(self._canonical_attrs):
            missing = self._canonical_attrs[len(self._embeddings):]
            new_vecs = embedder.encode(missing)
            self._embeddings.extend(new_vecs)

        best_score = -1.0
        best_idx = -1
        for i, vec in enumerate(self._embeddings):
            score = _cosine(query_vec, vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self.threshold and best_idx >= 0:
            canonical = self._canonical_attrs[best_idx]
            log.debug(
                "Attribute %r snapped to %r (SEMANTIC: %.3f)",
                raw_attribute, canonical, best_score,
            )
            return canonical, raw_attribute

        # Promote raw_attribute as a new canonical
        log.debug(
            "Attribute %r promoted as new canonical (best_score=%.3f)",
            raw_attribute, best_score,
        )
        self._canonical_attrs.append(raw_attribute)
        self._embeddings.append(query_vec)
        self.save()
        return raw_attribute, raw_attribute

    @property
    def canonical_attributes(self) -> List[str]:
        return list(self._canonical_attrs)

    def __len__(self) -> int:
        return len(self._canonical_attrs)


# ---------------------------------------------------------------------------
# Module-level stateless convenience function (used by tests and pipeline)
# ---------------------------------------------------------------------------

def snap_attribute(
    raw_attribute: str,
    known_attributes: List[str],
    embedder: Any,
    threshold: float = ATTRIBUTE_MATCH_THRESHOLD,
) -> Tuple[str, str]:
    """
    Stateless convenience wrapper: snap a raw attribute against an explicit list.

    Does NOT mutate or persist anything.  Suitable for testing and for callers
    that manage their own attribute list externally.

    Returns
    -------
    (canonical_attribute, raw_attribute_preserved)
    """
    raw_attribute = raw_attribute.strip()
    if not raw_attribute:
        return raw_attribute, raw_attribute

    if not known_attributes:
        return raw_attribute, raw_attribute

    # 1. Acronym or Lexical Match
    for canonical in known_attributes:
        if _is_acronym_match(raw_attribute, canonical):
            return canonical, raw_attribute
            
        a_lower = raw_attribute.lower()
        c_lower = canonical.lower()
        if fuzz.token_set_ratio(a_lower, c_lower) > 90:
            return canonical, raw_attribute
            
        if min(len(a_lower), len(c_lower)) >= 5:
            if a_lower.startswith(c_lower) or c_lower.startswith(a_lower):
                return canonical, raw_attribute

    # 2. Semantic Embedding Match
    all_texts = [raw_attribute] + list(known_attributes)
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
        return known_attributes[best_idx], raw_attribute

    return raw_attribute, raw_attribute
