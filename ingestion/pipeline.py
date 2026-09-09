"""
ingestion/pipeline.py
---------------------
Top-level orchestration for the Veriscribe PDF ingestion pipeline.

The single public function ``parse_document`` ties together:
1. Table extraction  (pdfplumber, per page)
2. Layout-aware text ordering with header/footer stripping (PyMuPDF)
3. Chart / figure candidate detection (PyMuPDF)

It returns a flat list of :class:`~ingestion.models.Chunk` objects covering
all pages, with ``reading_order_index`` values correct within each page.

No assumptions are made about any specific document — this function must
work correctly on any well-formed PDF passed to it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import fitz           # PyMuPDF
import pdfplumber

from ingestion.layout import detect_header_footer_zones, order_blocks
from ingestion.models import Chunk
from ingestion.tables import extract_tables
from ingestion.triage import extract_chart_candidates

log = logging.getLogger(__name__)


def parse_document(pdf_path: str, document_id: str) -> List[Chunk]:
    """Full ingestion pipeline: open the PDF and return all parsed chunks.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the source PDF file.
    document_id:
        Caller-supplied identifier for this document (used in chunk IDs,
        image crop filenames, etc.).

    Returns
    -------
    A list of :class:`~ingestion.models.Chunk` objects covering every page,
    sorted by (page_number, reading_order_index).

    Notes
    -----
    * Works on any PDF; no document-specific logic or hardcoded names.
    * Does not call any LLM.
    * Writes cropped PNG files for chart candidates to
      ``./output/chart_crops/``.
    """
    pdf_path = str(pdf_path)
    all_chunks: List[Chunk] = []

    # ------------------------------------------------------------------
    # Open with both backends
    # ------------------------------------------------------------------
    fitz_doc = fitz.open(pdf_path)
    plumber_doc = pdfplumber.open(pdf_path)

    try:
        n_pages = len(fitz_doc)
        log.info("Opened '%s' — %d page(s).", pdf_path, n_pages)

        # ------------------------------------------------------------------
        # Pass 1: global header/footer detection across ALL pages
        # ------------------------------------------------------------------
        all_pages = [fitz_doc[i] for i in range(n_pages)]
        header_bands, footer_bands = detect_header_footer_zones(all_pages)
        log.debug(
            "Header bands: %s  |  Footer bands: %s",
            header_bands,
            footer_bands,
        )

        # ------------------------------------------------------------------
        # Pass 2: per-page extraction
        # ------------------------------------------------------------------
        for page_idx in range(n_pages):
            page_number = page_idx + 1  # 1-based
            fitz_page = fitz_doc[page_idx]
            plumber_page = plumber_doc.pages[page_idx]

            pw, ph = fitz_page.rect.width, fitz_page.rect.height
            if pw == 0 or ph == 0:
                log.warning("Page %d has zero dimensions — skipping.", page_number)
                continue

            page_chunks: List[Chunk] = []

            # ---- (a) Tables -----------------------------------------------
            table_infos = extract_tables(plumber_page)
            table_norm_bboxes = []

            for tbl in table_infos:
                chunk = Chunk(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_type="table",
                    text=None,
                    bbox=tbl["bbox"],
                    reading_order_index=0,     # patched below
                    table_data=tbl["table_data"],
                )
                page_chunks.append(chunk)
                table_norm_bboxes.append(tbl["bbox"])

            # ---- (b) Text blocks ------------------------------------------
            text_block_infos = order_blocks(
                fitz_page,
                header_bands=header_bands,
                footer_bands=footer_bands,
                excluded_bboxes=table_norm_bboxes,
            )

            text_norm_bboxes = []
            for blk in text_block_infos:
                chunk = Chunk(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_type="text",
                    text=blk["text"],
                    bbox=blk["bbox"],
                    reading_order_index=0,     # patched below
                )
                page_chunks.append(chunk)
                text_norm_bboxes.append(blk["bbox"])

            # ---- (c) Chart candidates -------------------------------------
            covered_bboxes = table_norm_bboxes + text_norm_bboxes
            chart_infos = extract_chart_candidates(
                fitz_page,
                document_id=document_id,
                page_number=page_number,
                existing_norm_bboxes=covered_bboxes,
            )

            for info in chart_infos:
                chunk = Chunk(
                    document_id=document_id,
                    page_number=page_number,
                    chunk_type="chart_candidate",
                    text=info["text"],
                    bbox=info["bbox"],
                    reading_order_index=0,     # patched below
                    image_path=info["image_path"],
                )
                page_chunks.append(chunk)

            # ---- (d) Assign reading_order_index within this page ----------
            # Sort the combined page chunks:
            #   tables first (they are inline with text), then text, then charts
            # within each type, sort by top-y of bbox.
            # We actually sort all of them together by vertical position so
            # reading order reflects page layout naturally.
            page_chunks.sort(key=lambda c: (c.bbox[1], c.bbox[0]))
            for i, chunk in enumerate(page_chunks):
                chunk.reading_order_index = i

            all_chunks.extend(page_chunks)

    finally:
        fitz_doc.close()
        plumber_doc.close()

    log.info(
        "parse_document finished: %d total chunks from '%s'.",
        len(all_chunks),
        pdf_path,
    )
    return all_chunks
