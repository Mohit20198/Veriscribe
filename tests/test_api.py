import pytest
import os
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Create a dummy sqlite DB and vector store path for tests
TEST_DB_PATH = "test_api_veriscribe.db"

@pytest.fixture(autouse=True)
def patch_db_paths():
    with patch("api.main.DB_PATH", TEST_DB_PATH), \
         patch("api.main.CHROMA_PATH", "test_api_chroma"):
        yield

@pytest.fixture(scope="module")
def test_db():
    # Setup test DB
    import sqlite3
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            canonical_entity TEXT NOT NULL,
            raw_attribute TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            fact_id_a TEXT NOT NULL,
            fact_id_b TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            justification TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (fact_id_a, fact_id_b)
        )
    """)
    
    # Seed a fact
    fact_data = {
        "fact_id": "f1",
        "document_id": "doc1",
        "statement": "Rev 100",
        "canonical_entity": "Company",
        "raw_attribute": "Rev",
        "value": "100",
        "unit": "USD",
        "temporal_scope": {},
        "context": {"image_path": "fake.png"},
        "confidence": 1.0,
        "source_chunk_id": "c1",
        "source_type": "chart",
        "evidence_quote": "Rev 100",
        "evidence_bbox": (0,0,0,0),
        "evidence_context_window": "Rev 100",
        "content_hash": "hash1",
        "occurrences": []
    }
    
    fact_data2 = {
        "fact_id": "f2",
        "document_id": "doc2",
        "statement": "Rev 100",
        "canonical_entity": "Company",
        "raw_attribute": "Rev",
        "value": "100",
        "unit": "USD",
        "temporal_scope": {},
        "context": {},
        "confidence": 1.0,
        "source_chunk_id": "c2",
        "source_type": "text",
        "evidence_quote": "Rev 100",
        "evidence_bbox": (0,0,0,0),
        "evidence_context_window": "Rev 100",
        "content_hash": "hash2",
        "occurrences": []
    }
    
    cursor.execute("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)", 
                  ("f1", "doc1", "Company", "Rev", "hash1", json.dumps(fact_data)))
    cursor.execute("INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?)", 
                  ("f2", "doc2", "Company", "Rev", "hash2", json.dumps(fact_data2)))
                  
    cursor.execute("INSERT INTO relationships VALUES (?, ?, ?, ?, ?, ?)",
                  ("f1", "f2", "corroborates", 0.9, "Matches.", 123.0))
                  
    conn.commit()
    conn.close()
    
    yield
    
    try:
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
    except PermissionError:
        pass


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)

def test_upload_non_pdf(client):
    response = client.post("/documents", files={"file": ("test.txt", b"hello")})
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]

@patch("api.main.BackgroundTasks.add_task")
def test_upload_pdf(mock_add_task, client):
    response = client.post("/documents", data={"document_id": "test_doc"}, files={"file": ("test.pdf", b"pdf content")})
    assert response.status_code == 202
    assert response.json()["document_id"] == "test_doc"
    assert response.json()["status"] == "processing"
    
    mock_add_task.assert_called_once()
    
def test_get_status_transitions():
    from api import jobs
    from orchestration.pipeline import IngestResult
    jobs._jobs.clear()
    jobs.create_job("test_doc_2")
    
    from api.main import app
    with TestClient(app) as client:
        # Check processing
        resp = client.get("/documents/test_doc_2/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"
        
        # Simulate done
        res = IngestResult(document_id="test_doc_2", fact_count=10, relationship_counts={"corroborates": 2}, elapsed_seconds=5.0)
        jobs.update_job_success("test_doc_2", res)
        
        resp2 = client.get("/documents/test_doc_2/status")
        assert resp2.json()["status"] == "done"
        assert resp2.json()["fact_count"] == 10
        assert resp2.json()["relationship_counts"]["corroborates"] == 2

def test_get_documents(client, test_db):
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_facts"] == 2
    assert len(data["documents"]) == 2

def test_get_facts(client, test_db):
    resp = client.get("/facts?document_id=doc1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["fact_id"] == "f1"
    
def test_get_fact_detail(client, test_db):
    resp = client.get("/facts/f1")
    assert resp.status_code == 200
    assert resp.json()["fact_id"] == "f1"
    assert resp.json()["image_path"] == "fake.png"
    
def test_get_relationships(client, test_db):
    resp = client.get("/facts/f1/relationships")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["relationship_type"] == "corroborates"
    assert data[0]["related_fact"]["fact_id"] == "f2"

def test_404_fact(client, test_db):
    assert client.get("/facts/missing").status_code == 404
    assert client.get("/facts/missing/relationships").status_code == 404
    assert client.get("/facts/missing/evidence-image").status_code == 404
    assert client.get("/documents/missing/status").status_code == 404
