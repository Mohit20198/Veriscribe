from comparison.storage import SQLiteStore
from comparison.deterministic_checks import try_deterministic_reconcile

store = SQLiteStore("output/veriscribe.db")
f_a = store.get_fact("755ad359-ec19-43fa-898f-596be784bd73")
f_b = store.get_fact("f3d28d30-20e3-4ae3-95de-f7dc7bbb5d07")

res = try_deterministic_reconcile(f_a, f_b)
print(res)
