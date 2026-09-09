"""
Hardcoding audit — runs grep-based checks directly using Python's pathlib
without needing ripgrep installed.
"""
import sys
import re
from pathlib import Path

TARGET_DIRS = [
    "ingestion", "extraction", "canonicalization",
    "comparison", "orchestration", "api"
]

CHECKS = [
    (
        "Hardcoded API keys / secrets",
        re.compile(r'sk-[a-zA-Z0-9]{10}')
    ),
    (
        "Hardcoded model strings INLINE in function calls (not at module level)",
        # Only flag model= assignments inside function calls, not constant declarations
        re.compile(r'model\s*=\s*"(anthropic|deepseek|openai|google)/')
    ),
    (
        "Hardcoded local file paths (data/delhivery, data/india-macro)",
        re.compile(r'data/delhivery|data/india-macro|data\\delhivery')
    ),
]

ROOT = Path(".")

all_clean = True
for label, pattern in CHECKS:
    hits = []
    for d in TARGET_DIRS:
        for f in (ROOT / d).rglob("*.py"):
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"  {f}:{i}: {line.strip()}")
    print(f"\n{'='*60}")
    print(f"CHECK: {label}")
    print(f"{'='*60}")
    if hits:
        for h in hits[:20]:
            print(h)
        all_clean = False
    else:
        print("  [CLEAN — no hits]")

# Special: model routing — expect deepseek only as a named constant
print(f"\n{'='*60}")
print("CHECK: Model name references (deepseek expected only as EXTRACTION_MODEL constant)")
print(f"{'='*60}")
pat = re.compile(r'deepseek')
for d in ["extraction", "comparison", "orchestration"]:
    for f in (ROOT / d).rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line):
                print(f"  {f}:{i}: {line.strip()}")

print(f"\n{'='*60}")
if all_clean:
    print("AUDIT RESULT: CLEAN — no hardcoded secrets or paths found.")
else:
    print("AUDIT RESULT: ISSUES FOUND — see above.")
print(f"{'='*60}")
