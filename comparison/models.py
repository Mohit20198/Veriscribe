"""
comparison/models.py
--------------------
Data models for fact relationships.
"""

from enum import Enum
import time
from typing import Literal
from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RECONCILED = "reconciled"
    AMBIGUOUS = "ambiguous"
    UNRELATED = "unrelated"


class Relationship(BaseModel):
    fact_id_a: str
    fact_id_b: str
    relationship_type: RelationshipType
    confidence: float
    justification: str
    created_at: float = Field(default_factory=time.time)


class JudgeResponse(BaseModel):
    relationship: Literal["corroborates", "contradicts", "reconciled", "ambiguous", "unrelated"]
    confidence: float
    justification: str
