"""Test normalize_unit handles Mn and Cr abbreviations correctly."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from canonicalization.units import normalize_unit

pairs = [
    ('1266', 'Mn'),
    ('1266', 'mn'),
    ('127',  'Cr'),
    ('127',  'cr'),
    ('127',  'crore'),
    ('1266', 'million'),
    ('42',   'crore'),
    ('0.00042', 'lakh crore'),
]

print(f"{'Input value':<20} {'Unit':<12} {'normalized_value':>20}  parse_ok")
print('-' * 65)
for val, unit in pairs:
    r = normalize_unit(val, unit)
    print(f"{val:<20} {unit:<12} {str(r['normalized_value']):>20}  {r['parse_ok']}")

print()
# Cross-check: 1266 Mn vs 127 Cr (the dry-run corroboration pair)
a = normalize_unit('1266', 'Mn')
b = normalize_unit('127',  'Cr')
pct_diff = abs(a['normalized_value'] - b['normalized_value']) / b['normalized_value'] * 100
print(f"1266 Mn -> {a['normalized_value']}")
print(f"127 Cr  -> {b['normalized_value']}")
print(f"Difference: {pct_diff:.3f}% (expected <1% for rounding)")
print('MATCH (within 1%)' if pct_diff < 1 else 'MISMATCH')
