from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

class DocumentResponse(BaseModel):
    document_id: str
    status: str

class DocumentSummary(BaseModel):
    document_id: str
    fact_count: int

class SystemStatusResponse(BaseModel):
    documents: List[DocumentSummary]
    total_facts: int
    relationships_by_type: Dict[str, int]

class JobStatusResponse(BaseModel):
    document_id: str
    status: str
    fact_count: Optional[int] = None
    relationship_counts: Optional[Dict[str, int]] = None
    elapsed_seconds: Optional[float] = None
    error_count: Optional[int] = None
    errors: Optional[List[str]] = None

class FactDetail(BaseModel):
    fact_id: str
    document_id: str
    statement: str
    canonical_entity: str
    raw_attribute: str
    value: str
    unit: Optional[str] = None
    temporal_scope: Dict[str, Any]
    confidence: float
    evidence_quote: Optional[str] = None
    evidence_bbox: Optional[Tuple[float, float, float, float]] = None
    evidence_context_window: Optional[str] = None
    source_type: str
    image_path: Optional[str] = None
    
class PaginatedFacts(BaseModel):
    total: int
    limit: int
    offset: int
    data: List[FactDetail]

class RelatedFactSummary(BaseModel):
    fact_id: str
    document_id: str
    statement: str

class RelationshipDetail(BaseModel):
    relationship_type: str
    confidence: float
    justification: str
    related_fact: RelatedFactSummary
