"""
orchestration/cli.py
---------------------
Command-line entry point for Veriscribe.

Usage
-----
    python -m orchestration.cli ingest <pdf_path> [--document-id ID] [--force]
    python -m orchestration.cli status

The `ingest` command is the canonical "how to run this" entry point —
it replaces ad-hoc scripts and gives a consistent, reproducible interface.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure UTF-8 output on Windows (handles Rs., etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import openai
from sentence_transformers import SentenceTransformer

from comparison.storage import SQLiteStore
from comparison.vector_store import VectorStore
from orchestration.pipeline import ingest_document

# ---------------------------------------------------------------------------
# Logging setup — INFO to stdout so progress is visible without a log file
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("veriscribe.cli")

# ---------------------------------------------------------------------------
# Shared singletons (loaded once per process)
# ---------------------------------------------------------------------------
DB_PATH     = "output/veriscribe.db"
CHROMA_PATH = "output/chroma_db"


def _build_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set. Add it to your .env file.")
        sys.exit(1)
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _build_embedder() -> SentenceTransformer:
    log.info("Loading sentence embedder (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a single PDF document into the knowledge store."""
    pdf_path = args.pdf_path
    document_id = args.document_id or os.path.splitext(os.path.basename(pdf_path))[0]

    log.info("=== INGEST: %s (document_id=%r) ===", pdf_path, document_id)

    store        = SQLiteStore(DB_PATH)
    vector_store = VectorStore(CHROMA_PATH)
    client       = _build_client()
    embedder     = _build_embedder()

    result = ingest_document(
        pdf_path=pdf_path,
        document_id=document_id,
        store=store,
        vector_store=vector_store,
        client=client,
        embedder=embedder,
        force=args.force,
    )

    print()
    print(result.summary())
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current state of the knowledge store."""
    store = SQLiteStore(DB_PATH)

    from orchestration.status import get_system_status
    status_data = get_system_status(store)

    print()
    print("=== Veriscribe Knowledge Store Status ===")
    print(f"  Database  : {os.path.abspath(DB_PATH)}")
    print(f"  Documents : {len(status_data['documents'])}")
    for doc in status_data['documents']:
        print(f"    - {doc['document_id']}  ({doc['fact_count']} facts)")
    print(f"  Total facts         : {status_data['total_facts']}")
    print(f"  Relationships by type:")
    
    rel_cnts = status_data['relationships_by_type']
    if rel_cnts:
        for rel_type, count in sorted(rel_cnts.items()):
            print(f"    {rel_type:<20} {count}")
    else:
        print("    (none)")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m orchestration.cli",
        description="Veriscribe — financial fact extraction and cross-document comparison.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a PDF document.")
    p_ingest.add_argument("pdf_path", help="Path to the PDF file.")
    p_ingest.add_argument(
        "--document-id",
        dest="document_id",
        default=None,
        help="Explicit document identifier (defaults to filename without extension).",
    )
    p_ingest.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-ingest even if this document_id already has stored facts.",
    )

    # status
    sub.add_parser("status", help="Show stored documents, facts, and relationships.")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
