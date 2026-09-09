"""Debug candidate regions on page 2 - dump the cluster-line details."""
import pdfplumber
import itertools
from collections import defaultdict

def debug_page2():
    with pdfplumber.open("output/arxiv_paper.pdf") as pdf:
        page = pdf.pages[1]
        pw = page.width
        ph = page.height

        words = page.extract_words(keep_blank_chars=False)
        keyfunc = lambda w: round((w["top"] + w["bottom"]) / 2)
        words_sorted = sorted(words, key=keyfunc)

        _X_COLUMN_GAP = pw * 0.25
        cluster_lines = []
        for y_mid, group in itertools.groupby(words_sorted, key=keyfunc):
            line_words = sorted(list(group), key=lambda w: w["x0"])
            clusters = [[]]
            for w in line_words:
                if clusters[-1] and (w["x0"] - clusters[-1][-1]["x1"]) > _X_COLUMN_GAP:
                    clusters.append([])
                clusters[-1].append(w)
            for cl in clusters:
                if not cl:
                    continue
                text = " ".join(w["text"] for w in cl)
                x0 = min(w["x0"] for w in cl)
                x1 = max(w["x1"] for w in cl)
                top = min(w["top"] for w in cl)
                bottom = max(w["bottom"] for w in cl)
                col_anchor = round(x0 / 20) * 20
                has_digit = any(c.isdigit() for c in text)
                is_short = len(text) < 60
                tabular = is_short and has_digit
                cluster_lines.append({
                    "y_mid": y_mid, "x0": x0, "x1": x1, "top": top, "bottom": bottom,
                    "text": text, "col_anchor": col_anchor, "tabular": tabular
                })

        # Show lines in the table region (top ~430-580)
        print("Cluster lines in table region (top 400-600):")
        for cl in cluster_lines:
            if 400 < cl["top"] < 600:
                print(f"  y={cl['y_mid']:5.0f} col={cl['col_anchor']:5.0f} x0={cl['x0']:6.1f} "
                      f"tab={cl['tabular']} | {cl['text'][:60]}")

        # Show by col_anchor grouping for the table region
        print("\nBy col_anchor (table region):")
        by_col = defaultdict(list)
        for cl in cluster_lines:
            if 400 < cl["top"] < 600:
                by_col[cl["col_anchor"]].append(cl)
        for col_anchor in sorted(by_col.keys()):
            col_lines = sorted(by_col[col_anchor], key=lambda c: c["y_mid"])
            tabular_run = [c for c in col_lines if c["tabular"]]
            print(f"  col_anchor={col_anchor}: {len(col_lines)} lines, {len(tabular_run)} tabular")
            for cl in col_lines:
                print(f"    y={cl['y_mid']:5.0f} tab={cl['tabular']} | {cl['text'][:50]}")

if __name__ == "__main__":
    debug_page2()
