import sys
import sqlite3
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
c = sqlite3.connect('output/veriscribe.db')
for r in c.execute("SELECT relationship_type, justification FROM relationships"):
    print(f"--- {r[0].upper()} ---")
    print(r[1])
    print()
