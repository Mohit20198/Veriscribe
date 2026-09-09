"""
api/jobs.py
-----------
Simple in-memory background job tracking for document ingestion.
Resets on server restart.
"""

from typing import Dict, Any, Optional

# Job state is stored in memory as document_id -> job info
# e.g., {"status": "processing" | "done" | "error", "result": IngestResult, "error": "msg"}
_jobs: Dict[str, Dict[str, Any]] = {}

def get_job(document_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(document_id)

def create_job(document_id: str) -> None:
    _jobs[document_id] = {
        "status": "processing",
        "result": None,
        "error": None
    }

def update_job_success(document_id: str, result: Any) -> None:
    if document_id in _jobs:
        _jobs[document_id]["status"] = "done"
        _jobs[document_id]["result"] = result

def update_job_error(document_id: str, error_msg: str) -> None:
    if document_id in _jobs:
        _jobs[document_id]["status"] = "error"
        _jobs[document_id]["error"] = error_msg
