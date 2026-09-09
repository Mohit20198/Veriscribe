"""
comparison/vector_store.py
--------------------------
ChromaDB wrapper for fact vector retrieval.
"""

import os
from typing import List, Optional
import chromadb
from extraction.models import Fact


class VectorStore:
    def __init__(self, persist_directory: str = "output/chroma_db", client: Optional[chromadb.ClientAPI] = None):
        """
        Initialize the VectorStore.
        If a client is provided (e.g. EphemeralClient for testing), it will be used.
        Otherwise, a PersistentClient is created at `persist_directory`.
        """
        if client:
            self.client = client
        else:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            
        self.collection = self.client.get_or_create_collection(name="facts")

    def add_facts(self, facts: List[Fact], embedder) -> None:
        """Embed and store facts in Chroma."""
        if not facts:
            return

        documents = []
        metadatas = []
        ids = []

        for f in facts:
            documents.append(f"{f.canonical_entity} {f.raw_attribute} {f.statement}")
            metadatas.append({
                "fact_id": f.fact_id,
                "canonical_entity": f.canonical_entity,
                "raw_attribute": f.raw_attribute,
                "document_id": f.document_id,
            })
            ids.append(f.fact_id)

        # Encode using the provided sentence-transformer model
        embeddings = embedder.encode(documents)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        # Chroma handles upserts
        self.collection.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_similar(self, fact: Fact, embedder, top_k: int = 8, exclude_same_document: bool = True) -> List[str]:
        """
        Return a list of similar fact_ids.
        Retrieval only; source of truth for Fact data stays in SQLite.
        """
        if self.collection.count() == 0:
            return []

        query_text = f"{fact.canonical_entity} {fact.raw_attribute} {fact.statement}"
        query_embedding = embedder.encode([query_text])
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        where_clause = None
        if exclude_same_document:
            where_clause = {"document_id": {"$ne": fact.document_id}}

        # Ensure we don't ask for more results than exist in the collection
        n_results = min(top_k, self.collection.count())
        
        if n_results == 0:
            return []

        # We must filter out the exact fact if it somehow was indexed,
        # but typically this is called before adding the new fact.
        # Just to be safe, if exclude_same_document is false, we might hit ourselves.
        if not exclude_same_document:
            where_clause = {"fact_id": {"$ne": fact.fact_id}}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_clause
        )

        if not results["ids"] or not results["ids"][0]:
            return []
            
        return results["ids"][0]
