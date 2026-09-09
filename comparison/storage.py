"""
comparison/storage.py
---------------------
SQLite persistence for Facts and their Relationships.
"""

import sqlite3
import os
import logging
from typing import List, Optional, Tuple
from extraction.models import Fact
from comparison.models import Relationship, RelationshipType

log = logging.getLogger(__name__)

class SQLiteStore:
    def __init__(self, db_path: str = "output/veriscribe.db"):
        self.db_path = db_path
        # :memory: databases vanish when the connection closes, so we must
        # hold ONE persistent connection for the lifetime of the object.
        # File-backed databases use a new connection per call (safer for
        # multi-process access and WAL mode).
        self._conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the connection to use for this operation."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Facts table
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
            
            # Relationships table
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
            conn.commit()

    def save_fact(self, fact: Fact) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO facts 
                (fact_id, document_id, canonical_entity, raw_attribute, content_hash, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fact.fact_id,
                fact.document_id,
                fact.canonical_entity,
                fact.raw_attribute,
                fact.content_hash,
                fact.model_dump_json()
            ))
            conn.commit()

    def save_relationship(self, rel: Relationship) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            # Insert edge A->B
            cursor.execute("""
                INSERT OR REPLACE INTO relationships 
                (fact_id_a, fact_id_b, relationship_type, confidence, justification, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rel.fact_id_a,
                rel.fact_id_b,
                rel.relationship_type.value,
                rel.confidence,
                rel.justification,
                rel.created_at
            ))
            
            # Insert symmetric edge B->A for easy querying without complex joins
            cursor.execute("""
                INSERT OR REPLACE INTO relationships 
                (fact_id_a, fact_id_b, relationship_type, confidence, justification, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rel.fact_id_b,
                rel.fact_id_a,
                rel.relationship_type.value,
                rel.confidence,
                rel.justification,
                rel.created_at
            ))
            conn.commit()

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data_json FROM facts WHERE fact_id = ?", (fact_id,))
            row = cursor.fetchone()
            if row:
                return Fact.model_validate_json(row[0])
        return None

    def get_relationships_for_fact(self, fact_id: str) -> List[Relationship]:
        rels = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT fact_id_a, fact_id_b, relationship_type, confidence, justification, created_at
                FROM relationships WHERE fact_id_a = ?
            """, (fact_id,))
            for row in cursor.fetchall():
                rels.append(Relationship(
                    fact_id_a=row[0],
                    fact_id_b=row[1],
                    relationship_type=RelationshipType(row[2]),
                    confidence=row[3],
                    justification=row[4],
                    created_at=row[5]
                ))
        return rels

    def get_all_corroborations(self) -> List[Tuple[str, str]]:
        """Returns pairs of (fact_id_a, fact_id_b) that corroborate each other, to seed the Union-Find structure."""
        corroborations = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact_id_a, fact_id_b FROM relationships WHERE relationship_type = ?", 
                           (RelationshipType.CORROBORATES.value,))
            for row in cursor.fetchall():
                corroborations.append((row[0], row[1]))
        return corroborations

    def get_facts_by_document(self, document_id: str) -> List[Fact]:
        """Return all Facts stored for a given document_id."""
        facts = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data_json FROM facts WHERE document_id = ?", (document_id,)
            )
            for row in cursor.fetchall():
                try:
                    facts.append(Fact.model_validate_json(row[0]))
                except Exception as exc:
                    log.warning("Failed to deserialize fact from DB: %s", exc)
        return facts

    def get_document_ids(self) -> List[str]:
        """Return a sorted list of all distinct document_ids in the facts table."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT document_id FROM facts ORDER BY document_id")
            return [row[0] for row in cursor.fetchall()]

    def count_facts(self) -> int:
        """Total number of stored facts."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM facts")
            return cursor.fetchone()[0]

    def count_relationships(self) -> dict:
        """Count of relationships by type.  Returns a dict {type_str: count}."""
        with self._connect() as conn:
            cursor = conn.cursor()
            # Divide by 2 because save_relationship stores both A→B and B→A
            cursor.execute(
                "SELECT relationship_type, COUNT(*) FROM relationships GROUP BY relationship_type"
            )
            return {row[0]: row[1] // 2 for row in cursor.fetchall()}
