import sqlite3

c = sqlite3.connect('output/veriscribe.db')
c.execute("UPDATE relationships SET justification=substr(justification, 1, 413) WHERE relationship_type='contradicts'")
c.commit()
