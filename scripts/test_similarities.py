import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from canonicalization.attribute_taxonomy import AttributeTaxonomy, ATTRIBUTE_MATCH_THRESHOLD

def main():
    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    pairs = [
        # Should snap
        ("market_capitalization", "market_cap"),
        ("quarterly_revenue", "revenue_for_the_quarter"),
        ("profit after tax", "PAT"),
        ("Revenue from operations", "Total Revenue"),
        ("Total Income", "Total Revenue"),
        
        # Adversarial (should NOT merge!)
        ("Total Revenue", "Operating Expenses"),
        ("Gross Profit", "Net Income"),
        ("EBITDA", "Net Profit"),
        ("Revenue", "Profit"),
        
        # Entities (sanity check)
        ("General Electric", "General Motors")
    ]

    print("\n--- Multi-Modal Snapping Analysis ---\n")
    print(f"{'Base Attribute':<25} | {'Incoming Attribute':<25} | {'Result':<15}")
    print("-" * 75)
    
    # Enable debug logging for taxonomy to see the signal output
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    for base, incoming in pairs:
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "attrs.json"
            taxonomy = AttributeTaxonomy(store_path=store, threshold=ATTRIBUTE_MATCH_THRESHOLD)
            
            # Seed the base attribute
            taxonomy.snap(base, embedder)
            
            # Attempt to snap the incoming attribute
            canon, _ = taxonomy.snap(incoming, embedder)
            
            if canon == base:
                result = "[SNAPPED]"
            else:
                result = "[NEW]"
                
            print(f"{base:<25} | {incoming:<25} | {result:<15}")
            print("-" * 75)

if __name__ == "__main__":
    main()
