"""
extraction/extractor.py
-----------------------
Core extraction logic routing chunks to Anthropic LLM.
"""

import base64
import json
import logging
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from ingestion.models import Chunk
from extraction.models import Fact
from extraction.heuristics import is_likely_equation
from extraction.prompts import TEXT_EXTRACTION_PROMPT, TABLE_EXTRACTION_PROMPT, CHART_EXTRACTION_PROMPT

log = logging.getLogger(__name__)

# Two-tier model strategy — names kept as module-level constants so they can be
# audited and overridden without touching business logic.
# Charts need vision+reasoning → use the higher-quality model.
# Text/table extraction is routed to a cheaper model for cost efficiency.
EXTRACTION_MODEL_CHART = "anthropic/claude-sonnet-5"       # vision-capable, higher cost
EXTRACTION_MODEL_TEXT  = "deepseek/deepseek-v4-flash-vision-exp"  # cheap, fast

# Temporary schema to parse Anthropic JSON output before coercing to Fact
class ExtractedFact(BaseModel):
    statement: str
    canonical_entity: str
    raw_attribute: str
    value: str
    unit: str | None = None
    temporal_scope: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    confidence: float
    evidence_quote: str | None = None

class ExtractionResult(BaseModel):
    facts: List[ExtractedFact]


def extract_facts_from_chunk(chunk: Chunk, context_window: str, client: Any) -> List[Fact]:
    """Route chunk to the correct prompt and extract facts using the LLM."""
    
    if chunk.chunk_type == "text" and chunk.text and is_likely_equation(chunk.text):
        log.info(f"skipped: equation-like content for chunk {chunk.chunk_id}")
        return []

    # Prepare user message content based on chunk type
    content: List[Dict[str, Any]] = []
    
    if chunk.chunk_type == "text":
        system_prompt = TEXT_EXTRACTION_PROMPT
        user_text = f"CONTEXT WINDOW:\n{context_window}\n\nTARGET CHUNK:\n{chunk.text}"
        content.append({"type": "text", "text": user_text})
        
    elif chunk.chunk_type == "table":
        system_prompt = TABLE_EXTRACTION_PROMPT
        # For simplicity, convert table_data to a string representation
        table_str = ""
        if chunk.table_data:
            for row in chunk.table_data:
                table_str += " | ".join((cell or "") for cell in row) + "\n"
        caption = f"Caption: {chunk.text}\n\n" if chunk.text else ""
        user_text = f"{caption}TABLE DATA:\n{table_str}"
        content.append({"type": "text", "text": user_text})
        
    elif chunk.chunk_type == "chart_candidate":
        system_prompt = CHART_EXTRACTION_PROMPT
        if not chunk.image_path:
            log.warning(f"chart_candidate {chunk.chunk_id} missing image_path")
            return []
            
        try:
            with open(chunk.image_path, "rb") as f:
                img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode("utf-8")
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
        except Exception as e:
            log.error(f"Failed to read image {chunk.image_path}: {e}")
            return []
            
        if chunk.text:
            content.append({"type": "text", "text": f"Chart Caption: {chunk.text}"})
    else:
        log.warning(f"Unknown chunk type: {chunk.chunk_type}")
        return []

    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_extracted_facts",
                "description": "Submit the list of extracted facts.",
                "parameters": ExtractionResult.model_json_schema()
            }
        }
    ]

    # Two-Tier Model Strategy
    # Charts require high-quality vision reasoning, so we use Sonnet 5
    # Text and tables are routed to the much cheaper DeepSeek V4 model
    if chunk.chunk_type == "chart_candidate":
        target_model = EXTRACTION_MODEL_CHART
    else:
        # Default text/table model. Change to moonshotai/kimi-k2.6 if DeepSeek reliability drops.
        target_model = EXTRACTION_MODEL_TEXT

    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "schema": ExtractionResult.model_json_schema(),
                    "strict": True
                }
            },
            max_tokens=4000,
            temperature=0.0
        )
    except Exception as e:
        log.error(f"LLM API call failed for chunk {chunk.chunk_id}: {e}")
        return []

    # Parse output
    extracted_facts = []
    if response.choices and response.choices[0].message.content:
        try:
            args = json.loads(response.choices[0].message.content)
            result = ExtractionResult(**args)
            extracted_facts = result.facts
        except (ValidationError, json.JSONDecodeError) as e:
            log.error(f"LLM returned malformed JSON for chunk {chunk.chunk_id}: {e}")
            return []

    # Convert to proper Fact objects and validate evidence
    valid_facts = []
    source_type_map = {"text": "text", "table": "table", "chart_candidate": "chart"}
    
    for ef in extracted_facts:
        # Enforce evidence pointer
        if chunk.chunk_type in ("text", "table") and not ef.evidence_quote:
            log.warning(f"Dropped fact missing evidence_quote for chunk {chunk.chunk_id}")
            continue
            
        # Inject image path into context for chart facts so the API can serve it
        if chunk.chunk_type == "chart_candidate" and chunk.image_path:
            ef.context["image_path"] = str(chunk.image_path)
            
        fact = Fact(
            document_id=chunk.document_id,
            statement=ef.statement,
            canonical_entity=ef.canonical_entity,
            raw_attribute=ef.raw_attribute,
            value=ef.value,
            unit=ef.unit,
            temporal_scope=ef.temporal_scope,
            context=ef.context,
            confidence=ef.confidence,
            source_chunk_id=chunk.chunk_id,
            source_type=source_type_map.get(chunk.chunk_type, "text"),
            evidence_quote=ef.evidence_quote,
            evidence_bbox=chunk.bbox,
            evidence_context_window=context_window if chunk.chunk_type == "text" else None,
            occurrences=[{"page_number": chunk.page_number, "bbox": chunk.bbox}]
        )
        fact.content_hash = fact.compute_hash()
        valid_facts.append(fact)

    return valid_facts
