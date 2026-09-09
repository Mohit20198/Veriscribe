import pdfplumber

def test_page_4():
    with pdfplumber.open("output/arxiv_paper.pdf") as pdf:
        p4 = pdf.pages[3]
        
        # Test 1
        t1 = p4.find_tables({"vertical_strategy": "text", "horizontal_strategy": "lines"})
        print(f"Test 1 (text/lines): {len(t1)}")
        if t1: print(t1[0].bbox)

        # Test 2
        t2 = p4.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
        print(f"Test 2 (text/text): {len(t2)}")
        if t2: 
            for t in t2:
                print(f" bbox: {t.bbox}, cols: {len(t.extract()[0] or [])}")

        # Test 3
        t3 = p4.find_tables({
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_x_tolerance": 1,
            "join_x_tolerance": 1,
        })
        print(f"Test 3 (text/text tol 15): {len(t3)}")
        if t3: 
            for t in t3:
                print(f" bbox: {t.bbox}, cols: {len(t.extract()[0] or [])}")

if __name__ == "__main__":
    test_page_4()
