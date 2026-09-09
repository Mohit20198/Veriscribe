"""
extraction/models.py
--------------------
Pydantic models for Fact extraction.
"""

import hashlib
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class Fact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    statement: str
    canonical_entity: str
    raw_attribute: str
    value: str
    unit: Optional[str] = None
    temporal_scope: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    source_chunk_id: str
    source_type: Literal["text", "table", "chart"]
    evidence_quote: Optional[str] = None
    evidence_bbox: Tuple[float, float, float, float]
    evidence_context_window: Optional[str] = None
    content_hash: str = ""
    occurrences: List[Dict[str, Any]] = Field(default_factory=list)

    def compute_hash(self) -> str:
        """Compute the content hash used for intra-document deduplication."""
        # sha256(canonical_entity + raw_attribute + normalized value + temporal_scope)
        scope_str = str(sorted(self.temporal_scope.items())) if self.temporal_scope else ""
        norm_val = str(self.value).strip().lower()
        
        raw_str = f"{self.canonical_entity}|{self.raw_attribute}|{norm_val}|{scope_str}"
        return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
