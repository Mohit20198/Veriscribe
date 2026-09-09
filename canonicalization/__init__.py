"""
canonicalization/
-----------------
Phase 3: Normalize entity names and snap attributes to a growing canonical
taxonomy so facts from different documents can be matched in Phase 4.

Modules:
  entity_resolver    – fuzzy entity-name deduplication via embeddings
  attribute_taxonomy – attribute-name snapping to a canonical set
  units              – deterministic value/unit normalization (no LLM)
  pipeline           – orchestrates all three for a List[Fact]
"""
