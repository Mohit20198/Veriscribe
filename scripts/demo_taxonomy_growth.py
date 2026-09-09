"""
scripts/demo_taxonomy_growth.py
---------------------------------
Demonstrates the canonicalization taxonomy growing organically.
Feed 5 synthetic facts with varied entity/attribute phrasing — some are
near-duplicates, some are genuinely novel — and show which get snapped
vs promoted.

Run:
    python scripts/demo_taxonomy_growth.py

No API calls are made. Uses the real sentence-transformers model
(all-MiniLM-L6-v2) for realistic embeddings.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction.models import Fact
from canonicalization.entity_resolver import EntityRegistry, ENTITY_MATCH_THRESHOLD
from canonicalization.attribute_taxonomy import AttributeTaxonomy, ATTRIBUTE_MATCH_THRESHOLD
from canonicalization.pipeline import canonicalize_facts

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)


def make_fact(entity, attr, value, unit=None, doc_id="demo_doc"):
    return Fact(
        document_id=doc_id,
        statement=f"{entity} has {attr} of {value}.",
        canonical_entity=entity,
        raw_attribute=attr,
        value=value,
        unit=unit,
        confidence=0.9,
        source_chunk_id="chunk0",
        source_type="text",
        evidence_quote=value,
        evidence_bbox=(0.0, 0.0, 1.0, 0.1),
    )


def print_taxonomy(label, entities, attributes):
    print(f"\n{'-'*60}")
    print(f"  {label}")
    print(f"{'-'*60}")
    print(f"  Canonical entities  ({len(entities)}): {entities or ['(empty)']}")
    print(f"  Canonical attributes ({len(attributes)}): {attributes or ['(empty)']}")


def main():
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    # Use a temp dir so we don't pollute the real output/ taxonomy
    with tempfile.TemporaryDirectory() as tmp:
        entity_store    = Path(tmp) / "entities.json"
        attribute_store = Path(tmp) / "attributes.json"

        entity_registry = EntityRegistry(store_path=entity_store, threshold=ENTITY_MATCH_THRESHOLD)
        attr_taxonomy   = AttributeTaxonomy(store_path=attribute_store, threshold=ATTRIBUTE_MATCH_THRESHOLD)

        print_taxonomy("BEFORE (empty taxonomy)", entity_registry.canonical_names, attr_taxonomy.canonical_attributes)

        # Five synthetic facts:
        # 1. Novel entity + novel attribute                → both promoted
        # 2. Near-duplicate entity, same attribute        → entity snapped
        # 3. Same entity, near-duplicate attribute        → attribute snapped
        # 4. Completely different entity + attribute      → both promoted
        # 5. Repeat of #4's entity, slightly diff attr   → entity snapped, attr may snap or promote
        facts = [
            make_fact("Acme Corporation",  "quarterly_revenue",        "500 million",  "USD"),
            make_fact("Acme Corp",          "quarterly_revenue",        "500 million",  "USD"),
            make_fact("Acme Corporation",  "revenue_for_the_quarter",  "500 million",  "USD"),
            make_fact("BetaTech Limited",   "market_capitalization",    "2 billion",    "USD"),
            make_fact("BetaTech",           "market_cap",               "2 billion",    "USD"),
            # Adversarial pair: similar spelling/prefix, but distinct entities
            make_fact("General Electric",   "EBITDA",                   "10 billion",   "USD"),
            make_fact("General Motors",     "EBITDA",                   "15 billion",   "USD"),
        ]

        print("\nProcessing 7 synthetic facts through canonicalization pipeline...\n")

        results = canonicalize_facts(
            facts, embedder,
            entity_store=entity_store,
            attribute_store=attribute_store,
        )

        print(f"\n{'='*60}")
        print("  FACT-BY-FACT RESULTS")
        print(f"{'='*60}")
        for orig, canon in zip(facts, results):
            entity_changed = orig.canonical_entity != canon.canonical_entity
            attr_changed   = orig.raw_attribute    != canon.raw_attribute
            norm           = canon.context.get("normalized", {})
            print(f"\n  Fact: {orig.canonical_entity!r} / {orig.raw_attribute!r} / {orig.value!r}")
            print(f"    Entity : {orig.canonical_entity!r}  ->  {canon.canonical_entity!r}  {'[SNAPPED]' if entity_changed else '[promoted/unchanged]'}")
            print(f"    Attr   : {orig.raw_attribute!r}  ->  {canon.raw_attribute!r}  {'[SNAPPED]' if attr_changed else '[promoted/unchanged]'}")
            if norm.get("parse_ok"):
                print(f"    Value  : {orig.value!r}  ->  {norm['normalized_value']:.4g} {norm.get('normalized_unit', '')}")
            print(f"    Hash   : {canon.content_hash[:16]}...")

        # Reload registries from disk to show what was persisted
        entity_registry2 = EntityRegistry(store_path=entity_store, threshold=ENTITY_MATCH_THRESHOLD)
        attr_taxonomy2   = AttributeTaxonomy(store_path=attribute_store, threshold=ATTRIBUTE_MATCH_THRESHOLD)

        print_taxonomy(
            "AFTER (grown taxonomy)",
            entity_registry2.canonical_names,
            attr_taxonomy2.canonical_attributes,
        )

        # Check that near-duplicate facts produce the same content_hash
        print(f"\n{'-'*60}")
        print("  DEDUP CHECK: do near-duplicates produce identical content_hash?")
        print(f"{'-'*60}")
        hash_0 = results[0].content_hash
        hash_1 = results[1].content_hash
        hash_2 = results[2].content_hash
        print(f"  Fact 1 (Acme Corporation / quarterly_revenue):   {hash_0[:20]}...")
        print(f"  Fact 2 (Acme Corp / quarterly_revenue):           {hash_1[:20]}...")
        print(f"  Fact 3 (Acme Corporation / revenue_for_quarter):  {hash_2[:20]}...")
        if hash_0 == hash_1 == hash_2:
            print("  [PASS] All three hash identically — Phase 4 will deduplicate them correctly.")
        elif hash_0 == hash_1:
            print("  [PASS] Facts 1+2 (entity near-dup) hash identically.")
            if hash_0 == hash_2:
                print("  [PASS] Fact 3 (attribute near-dup) also matches.")
            else:
                print(f"  [INFO] Fact 3 has a different hash — attribute similarity was below threshold ({ATTRIBUTE_MATCH_THRESHOLD}).")
        else:
            print("  [FAIL] Unexpected: facts 1 and 2 do not hash identically. Check threshold tuning.")

        print()


if __name__ == "__main__":
    main()
