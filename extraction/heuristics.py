"""
extraction/heuristics.py
------------------------
Heuristics to filter out chunks that should not be sent to the LLM.
"""

import re


def is_likely_equation(text: str) -> bool:
    """Flag text chunks with high symbol-density or trailing equation numbers.
    
    Numeric facts are expected to be captured from prose restatements near 
    formulas, not by parsing the formulas themselves.
    """
    if not text:
        return False
        
    text_stripped = text.strip()
    
    # Check for trailing equation numbers like "(1)", "[2a]", etc.
    if re.search(r'[\(\[][0-9a-zA-Z]+[\)\]]$', text_stripped):
        # Only if the line isn't just a list item like "(1) First point..."
        # If the equation number is at the very end of a relatively short block or 
        # follows math symbols.
        if len(text_stripped) < 150:
            return True

    # Check symbol density
    math_symbols = set("=+-*/√∑∏∫αβγδθλμΣΠ()[]{}^_")
    symbol_count = sum(1 for c in text if c in math_symbols)
    
    if len(text) > 0 and (symbol_count / len(text)) > 0.15:
        return True
        
    return False
