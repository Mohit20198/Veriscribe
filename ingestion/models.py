"""
ingestion/models.py
-------------------
Pydantic data model for a parsed document chunk produced by the
Veriscribe ingestion pipeline.  No document-specific logic lives here.
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A single content unit extracted from a PDF page.

    Attributes
    ----------
    chunk_id:
        Unique identifier (UUID4 string) assigned at creation time.
    document_id:
        Caller-supplied identifier for the source document.
    page_number:
        1-based page index within the source PDF.
    chunk_type:
        Content category: ``"text"``, ``"table"``, or ``"chart_candidate"``.
    text:
        Raw textual content.  Empty string for ``chart_candidate`` chunks
        unless a nearby caption was found.
    bbox:
        Normalised bounding box ``(x0, y0, x1, y1)`` where all values are
        in the range ``[0, 1]`` relative to the page dimensions.
    reading_order_index:
        Zero-based position of this chunk in the corrected reading order
        for its page (column-aware, top-to-bottom within each column).
    table_data:
        Populated only for ``chunk_type == "table"``.  Outer list is rows;
        inner list is cell strings (None cells are converted to empty strings).
    image_path:
        Absolute or relative path to the cropped PNG saved for
        ``chunk_type == "chart_candidate"`` chunks.  ``None`` otherwise.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    page_number: int
    chunk_type: Literal["text", "table", "chart_candidate"]
    text: Optional[str] = None
    bbox: Tuple[float, float, float, float]
    reading_order_index: int
    table_data: Optional[List[List[str]]] = None
    image_path: Optional[str] = None

    model_config = ConfigDict(frozen=False)
