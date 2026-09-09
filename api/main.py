import os
import json
import uuid
from typing import Optional, List
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from api.models import (
    DocumentResponse, SystemStatusResponse, JobStatusResponse, DocumentSummary,
    FactDetail, RelationshipDetail, RelatedFactSummary, PaginatedFacts
)
from api import jobs
from orchestration.status import get_system_status
from comparison.storage import SQLiteStore
from comparison.vector_store import VectorStore
from orchestration.pipeline import ingest_document
from orchestration.cli import _build_client, _build_embedder, DB_PATH, CHROMA_PATH

app = FastAPI(title="Veriscribe API")

# Allow the Vite dev server and any localhost origin to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads dir exists
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def background_ingest(pdf_path: str, document_id: str):
    """Background task to run the ingestion pipeline."""
    try:
        store = SQLiteStore(DB_PATH)
        vector_store = VectorStore(CHROMA_PATH)
        client = _build_client()
        embedder = _build_embedder()

        result = ingest_document(
            pdf_path=pdf_path,
            document_id=document_id,
            store=store,
            vector_store=vector_store,
            client=client,
            embedder=embedder,
            force=False
        )
        jobs.update_job_success(document_id, result)
    except Exception as e:
        jobs.update_job_error(document_id, str(e))


@app.post("/documents", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not document_id:
        document_id = os.path.splitext(file.filename)[0]

    # Check if job already running
    if jobs.get_job(document_id) and jobs.get_job(document_id)["status"] == "processing":
        raise HTTPException(status_code=409, detail="Document is already processing.")

    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}_{uuid.uuid4().hex[:8]}.pdf")
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Init job and kick off background task
    jobs.create_job(document_id)
    background_tasks.add_task(background_ingest, file_path, document_id)

    return DocumentResponse(document_id=document_id, status="processing")


@app.get("/documents/{document_id}/status", response_model=JobStatusResponse)
def get_document_status(document_id: str):
    job = jobs.get_job(document_id)
    if not job:
        raise HTTPException(status_code=404, detail="Document job not found.")
    
    resp = JobStatusResponse(document_id=document_id, status=job["status"])
    if job["status"] == "done" and job["result"]:
        res = job["result"]
        resp.fact_count = res.fact_count
        resp.relationship_counts = res.relationship_counts
        resp.elapsed_seconds = res.elapsed_seconds
        resp.error_count = res.error_count
        resp.errors = res.errors
    elif job["status"] == "error":
        resp.errors = [job["error"]]
        
    return resp


@app.get("/documents", response_model=SystemStatusResponse)
def list_documents():
    store = SQLiteStore(DB_PATH)
    status_data = get_system_status(store)
    return SystemStatusResponse(
        documents=[DocumentSummary(**d) for d in status_data["documents"]],
        total_facts=status_data["total_facts"],
        relationships_by_type=status_data["relationships_by_type"]
    )


@app.get("/facts", response_model=PaginatedFacts)
def list_facts(
    document_id: Optional[str] = None,
    entity: Optional[str] = None,
    attribute: Optional[str] = None,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    store = SQLiteStore(DB_PATH)
    
    # We will fetch all facts from store for the document (or all docs if None)
    # Since SQLiteStore doesn't have an advanced filtering API, we filter in memory for prototype.
    if document_id:
        all_facts = store.get_facts_by_document(document_id)
    else:
        # If no doc id is provided, fetch across all docs
        doc_ids = store.get_document_ids()
        all_facts = []
        for did in doc_ids:
            all_facts.extend(store.get_facts_by_document(did))
            
    # Filter
    if entity:
        all_facts = [f for f in all_facts if f.canonical_entity == entity]
    if attribute:
        all_facts = [f for f in all_facts if f.raw_attribute == attribute]
        
    # Paginate
    total = len(all_facts)
    paginated = all_facts[offset:offset+limit]
    
    fact_details = []
    for f in paginated:
        fact_details.append(FactDetail(
            fact_id=f.fact_id,
            document_id=f.document_id,
            statement=f.statement,
            canonical_entity=f.canonical_entity,
            raw_attribute=f.raw_attribute,
            value=f.value,
            unit=f.unit,
            temporal_scope=f.temporal_scope,
            confidence=f.confidence,
            evidence_quote=f.evidence_quote,
            evidence_bbox=f.evidence_bbox,
            evidence_context_window=f.evidence_context_window,
            source_type=f.source_type,
            image_path=f.context.get("image_path")
        ))
        
    return PaginatedFacts(
        total=total,
        limit=limit,
        offset=offset,
        data=fact_details
    )


@app.get("/facts/{fact_id}", response_model=FactDetail)
def get_fact(fact_id: str):
    store = SQLiteStore(DB_PATH)
    f = store.get_fact(fact_id)
    if not f:
        raise HTTPException(status_code=404, detail="Fact not found.")
        
    return FactDetail(
        fact_id=f.fact_id,
        document_id=f.document_id,
        statement=f.statement,
        canonical_entity=f.canonical_entity,
        raw_attribute=f.raw_attribute,
        value=f.value,
        unit=f.unit,
        temporal_scope=f.temporal_scope,
        confidence=f.confidence,
        evidence_quote=f.evidence_quote,
        evidence_bbox=f.evidence_bbox,
        evidence_context_window=f.evidence_context_window,
        source_type=f.source_type,
        image_path=f.context.get("image_path")
    )


@app.get("/facts/{fact_id}/relationships", response_model=List[RelationshipDetail])
def get_fact_relationships(fact_id: str):
    store = SQLiteStore(DB_PATH)
    f = store.get_fact(fact_id)
    if not f:
        raise HTTPException(status_code=404, detail="Fact not found.")
        
    rels = store.get_relationships_for_fact(fact_id)
    
    details = []
    for rel in rels:
        # Determine the related fact ID
        related_id = rel.fact_id_b if rel.fact_id_a == fact_id else rel.fact_id_a
        related_fact = store.get_fact(related_id)
        if not related_fact:
            continue
            
        details.append(RelationshipDetail(
            relationship_type=rel.relationship_type.value,
            confidence=rel.confidence,
            justification=rel.justification,
            related_fact=RelatedFactSummary(
                fact_id=related_fact.fact_id,
                document_id=related_fact.document_id,
                statement=related_fact.statement
            )
        ))
        
    return details


@app.get("/facts/{fact_id}/evidence-image")
def get_fact_evidence_image(fact_id: str):
    store = SQLiteStore(DB_PATH)
    f = store.get_fact(fact_id)
    if not f:
        raise HTTPException(status_code=404, detail="Fact not found.")
        
    image_path = f.context.get("image_path")
    if not image_path:
        raise HTTPException(status_code=400, detail="This fact does not have an associated image.")
        
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
        
    return FileResponse(image_path)
