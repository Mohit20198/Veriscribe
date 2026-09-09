"""
comparison/deterministic_checks.py
----------------------------------
Deterministic fact relationship checks based on canonicalized values.
"""

from typing import Optional
from extraction.models import Fact
from comparison.models import RelationshipType, JudgeResponse

# Tunable constants
DETERMINISTIC_RECONCILE_TOLERANCE = 0.01  # 1% — safe threshold for rounding differences


def _within_tolerance(a: float, b: float, pct: float = DETERMINISTIC_RECONCILE_TOLERANCE) -> bool:
    """Return True if a and b differ by no more than `pct` (fractional, e.g. 0.01 = 1%)."""
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) <= pct


def _equivalent_scopes(scope_a: dict, scope_b: dict) -> bool:
    if scope_a == scope_b:
        return True
    if not isinstance(scope_a, dict) or not isinstance(scope_b, dict):
        return False
        
    def _get_normalized_year(s):
        for v in s.values():
            v_str = str(v).lower()
            if 'fy23' in v_str or '2023' in v_str: return 'fy23'
            if 'fy24' in v_str or '2024' in v_str: return 'fy24'
            if 'fy25' in v_str or '2025' in v_str: return 'fy25'
        return None
        
    def _get_normalized_quarter(s):
        for v in s.values():
            v_str = str(v).lower()
            if 'q1' in v_str: return 'q1'
            if 'q2' in v_str: return 'q2'
            if 'q3' in v_str: return 'q3'
            if 'q4' in v_str: return 'q4'
        return None
        
    ya, yb = _get_normalized_year(scope_a), _get_normalized_year(scope_b)
    qa, qb = _get_normalized_quarter(scope_a), _get_normalized_quarter(scope_b)
    
    if ya and yb and ya != yb: return False
    if qa and qb and qa != qb: return False
    if bool(qa) != bool(qb): return False
    
    if ya == yb and ya is not None:
        return True
        
    return False

def try_deterministic_reconcile(fact_a: Fact, fact_b: Fact) -> Optional[JudgeResponse]:
    """
    Attempts to deterministically reconcile two facts.

    Two fast paths:
    1. Same normalized_unit + values within tolerance + same temporal scope → CORROBORATES.
    2. Same normalized_unit + values within tolerance + different temporal scope → RECONCILED.
    3. Cross-scale match: both parse_ok, both have numeric normalized_value, values within
       tolerance even if normalized_unit differs (e.g. "Mn" vs "Cr" both expand to absolute
       INR ones) → CORROBORATES or RECONCILED depending on temporal scope.

    Returns None if no confident deterministic judgment can be made, deferring to the LLM.
    """
    norm_a = fact_a.context.get("normalized", {})
    norm_b = fact_b.context.get("normalized", {})

    # Must both have successfully parsed values
    if not norm_a.get("parse_ok") or not norm_b.get("parse_ok"):
        return None

    val_a = norm_a.get("normalized_value")
    val_b = norm_b.get("normalized_value")

    if val_a is None or val_b is None:
        return None

    unit_a = norm_a.get("normalized_unit")
    unit_b = norm_b.get("normalized_unit")

    # Path A: strict same-unit match
    same_unit = (unit_a == unit_b)

    # Path B: cross-scale match — both parsed to absolute numeric values.
    # This catches e.g. "1266 Mn" (→ 1,266,000,000, unit="mn") vs
    # "127 Cr" (→ 1,270,000,000, unit="cr") which refer to the same figure.
    # Only applicable when both normalized_values are large absolute numbers
    # (not percentages or period strings).
    cross_scale = (
        not same_unit
        and isinstance(val_a, (int, float))
        and isinstance(val_b, (int, float))
        and abs(val_a) >= 1000  # guard: don't cross-match small numbers (e.g. "5" vs "5")
    )

    if not same_unit and not cross_scale:
        return None

    if not _within_tolerance(val_a, val_b):
        return None

    # Values match within tolerance. Determine relationship by temporal scope.
    scope_a = fact_a.temporal_scope
    scope_b = fact_b.temporal_scope
    
    # If either scope is completely missing/empty, we cannot safely auto-resolve 
    # as 'corroborates' or 'reconciled'. Defer to the LLM judge.
    if not scope_a or not scope_b:
        return None
        
    unit_note = f" (different scale units: {unit_a!r} vs {unit_b!r})" if cross_scale else ""

    if _equivalent_scopes(scope_a, scope_b):
        return JudgeResponse(
            relationship=RelationshipType.CORROBORATES.value,
            confidence=1.0,
            justification=(
                f"Facts report effectively identical values "
                f"({val_a:g} vs {val_b:g}{unit_note}) "
                f"for equivalent temporal scopes ({scope_a} and {scope_b}). "
                f"Difference is {abs(val_a - val_b) / max(abs(val_a), 1):.3%}, "
                f"within the {DETERMINISTIC_RECONCILE_TOLERANCE:.0%} rounding tolerance."
            )
        )
    else:
        return JudgeResponse(
            relationship=RelationshipType.RECONCILED.value,
            confidence=1.0,
            justification=(
                f"Facts report effectively identical values "
                f"({val_a:g} vs {val_b:g}{unit_note}) "
                f"but for different temporal scopes ({scope_a} vs {scope_b}). "
                f"This represents a temporal reconciliation, not a contradiction."
            )
        )
