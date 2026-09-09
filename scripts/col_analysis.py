"""scripts/col_analysis.py — column detection analysis on both PDFs."""
import sys
sys.path.insert(0, '.')
import fitz
from ingestion.layout import (
    _cluster_midpoints, _col_bands_from_centres,
    _band_overlap_fraction, detect_header_footer_zones,
    _MIN_COL_BLOCK_WIDTH, _MAX_COL_BLOCK_WIDTH
)
from ingestion.pipeline import parse_document

NL = '\n'

def analyse_pdf(pdf_path, pages_to_check):
    print(f'\n{"="*70}')
    print(f'PDF: {pdf_path}')
    print('='*70)
    doc = fitz.open(pdf_path)
    all_pages = [doc[i] for i in range(len(doc))]
    hbands, fbands = detect_header_footer_zones(all_pages)
    print(f'  Header bands: {hbands}')
    print(f'  Footer bands: {[(round(a,3),round(b,3)) for a,b in fbands]}')
    print(f'  MIN_COL_BLOCK_WIDTH filter: {_MIN_COL_BLOCK_WIDTH}')
    print()

    for pg_idx in pages_to_check:
        if pg_idx >= len(doc):
            break
        page = doc[pg_idx]
        pw, ph = page.rect.width, page.rect.height
        raw = page.get_text('blocks')
        blks = [b for b in raw if b[6] == 0 and b[4].strip()]
        if not blks:
            print(f'  page {pg_idx+1:2d}: (empty)')
            continue

        # Mirror exactly what order_blocks does:
        # use only wide blocks for computing column centres
        wide_mids = [
            (b[0] + b[2]) / 2 / pw
            for b in blks
            if _MIN_COL_BLOCK_WIDTH <= (b[2] - b[0]) / pw <= _MAX_COL_BLOCK_WIDTH
        ]
        x_mids_for_clustering = wide_mids if wide_mids else [
            (b[0] + b[2]) / 2 / pw for b in blks
        ]
        col_centres = _cluster_midpoints(x_mids_for_clustering, pw)
        col_bands = _col_bands_from_centres(col_centres)

        narrow_excluded = len(blks) - len(wide_mids)
        col_count, span_count = 0, 0
        span_examples = []
        for b in blks:
            x0n, x1n = b[0] / pw, b[2] / pw
            best_frac = max(
                _band_overlap_fraction(x0n, x1n, lo, hi)
                for lo, hi in col_bands
            )
            if best_frac >= 0.70:
                col_count += 1
            else:
                span_count += 1
                text = b[4].replace(NL, ' ').strip()[:55]
                span_examples.append(f'x:[{x0n:.3f}-{x1n:.3f}] w={x1n-x0n:.3f}: "{text}"')

        band_str = ', '.join(f'[{lo:.3f}-{hi:.3f}]' for lo, hi in col_bands)
        centre_str = ', '.join(f'{c:.3f}' for c in col_centres)
        narrow_note = f' (excl {narrow_excluded} narrow blk(s))' if narrow_excluded else ''
        print(f'  page {pg_idx+1:2d}: {len(col_centres)} col(s){narrow_note}  '
              f'centres=[{centre_str}]  bands=[{band_str}]')
        print(f'           {col_count} col-blocks  {span_count} spanning-blocks')
        for ex in span_examples[:2]:
            print(f'           SPAN {ex}')

    doc.close()


analyse_pdf('output/arxiv_paper.pdf',               range(0, 9))
analyse_pdf('output/attention_is_all_you_need.pdf', range(0, 7))
