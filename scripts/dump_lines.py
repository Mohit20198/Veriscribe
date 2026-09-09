"""Diagnostic: dump all word lines on page 2 to see what the Table 1 region looks like."""
import pdfplumber, itertools

def get_lines(page):
    words = page.extract_words(keep_blank_chars=False)
    keyfunc = lambda w: round((w["top"] + w["bottom"]) / 2)
    words_sorted = sorted(words, key=keyfunc)
    lines = []
    for _, group in itertools.groupby(words_sorted, key=keyfunc):
        wlist = list(group)
        text = " ".join(w["text"] for w in wlist)
        left_x = min(w["x0"] for w in wlist)
        top = min(w["top"] for w in wlist)
        bottom = max(w["bottom"] for w in wlist)
        right_x = max(w["x1"] for w in wlist)
        lines.append({"text": text, "x0": left_x, "top": top, "bottom": bottom, "x1": right_x})
    return lines

with pdfplumber.open("output/arxiv_paper.pdf") as pdf:
    page = pdf.pages[1]  # page 2 (0-indexed)
    print(f"Page size: {page.width} x {page.height}")
    lines = get_lines(page)
    for ln in lines:
        has_digit = any(c.isdigit() for c in ln["text"])
        is_short = len(ln["text"]) < 60
        print(f"  top={ln['top']:.1f} x0={ln['x0']:.1f} len={len(ln['text']):3d} digit={has_digit} short={is_short}  |  {ln['text'][:80]}")
