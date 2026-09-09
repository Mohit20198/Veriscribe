"""
canonicalization/units.py
--------------------------
Deterministic, LLM-free unit normalization.

Design principles:
1. Scale normalization within a currency is safe (crore → million is exact).
2. Cross-currency conversion is explicitly OUT OF SCOPE — exchange rates are
   time-varying and currency-specific.  Facts in different currencies are
   flagged with a ``currency_conflict=True`` marker and left unconverted
   rather than silently merged at a possibly-stale rate.
3. Percentage ↔ basis points conversion is deterministic.
4. Date/period normalization is best-effort pattern matching — formats that
   don't match any known pattern are returned unchanged.
5. No domain-specific lookup tables or hard-coded constants beyond the unit
   mappings themselves.

All normalization is pure-function; no state, no persistence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Currency scale constants (within a single currency only)
# ---------------------------------------------------------------------------

# All scale factors are relative to "ones" (e.g. 1 crore = 10_000_000 ones)
_SCALE_MAP: Dict[str, float] = {
    # Indian numbering system
    "crore":        1e7,
    "cr":           1e7,
    "lakh crore":   1e12,
    "lakh_crore":   1e12,
    "lakh":         1e5,
    "lac":          1e5,
    # International numbering
    "million":      1e6,
    "mn":           1e6,
    "m":            1e6,
    "billion":      1e9,
    "bn":           1e9,
    "b":            1e9,
    "trillion":     1e12,
    "tn":           1e12,
    "thousand":     1e3,
    "k":            1e3,
    # Unscaled
    "":             1.0,
}

# Recognised currency prefixes/suffixes — used to detect and preserve currency type
_CURRENCY_SYMBOLS = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR",
    "$": "USD", "usd": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
    "¥": "JPY", "jpy": "JPY",
}


# ---------------------------------------------------------------------------
# Basis-point / percentage
# ---------------------------------------------------------------------------

_BPS_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:bps?|basis\s*points?)\s*$", re.I)
_PCT_PATTERN  = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$")


# ---------------------------------------------------------------------------
# Date / period normalization
# ---------------------------------------------------------------------------

# Common period tokens → normalized label
_PERIOD_TOKENS: Dict[str, str] = {
    "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4",
    "fy": "FY", "h1": "H1", "h2": "H2",
    "jan": "Jan", "feb": "Feb", "mar": "Mar", "apr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "aug": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec",
}

# FY2024, FY24, 2024-25, 2024/25 → {period: FY, vintage: "2024-25"}
_FY_LONG   = re.compile(r"\bFY\s*(\d{4})\s*[-/]\s*(\d{2,4})\b", re.I)
_FY_SHORT  = re.compile(r"\bFY\s*(\d{2,4})\b", re.I)
_Q_YEAR    = re.compile(r"\b(Q[1-4])\s*(?:FY)?\s*(\d{2,4})\b", re.I)
_YEAR_ONLY = re.compile(r"\b((?:19|20)\d{2})\b")


def _detect_currency(text: str) -> Optional[str]:
    """Return ISO currency code if a currency symbol/abbreviation is found."""
    t = text.strip().lower()
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in t:
            return code
    return None


def _parse_numeric_and_scale(value_str: str) -> Tuple[Optional[float], str, str]:
    """
    Parse a string like "42.5 crore" or "₹ 1,200 million" into
    (numeric_value, scale_key, cleaned_text_without_number_and_scale).

    Returns (None, "", value_str) if parsing fails.
    """
    # Remove commas (thousands separators)
    text = value_str.replace(",", "").strip()

    # Try to extract a leading or trailing number
    # Pattern: optional currency sign, number, optional scale
    pattern = re.compile(
        r"^"
        r"(?P<prefix>[₹$€£¥]?\s*)?"                      # optional currency prefix
        r"(?P<number>[+-]?\d+(?:\.\d+)?)"                  # numeric part
        r"\s*"
        r"(?P<scale>lakh\s+crore|lakh_crore|lakh|crore|lac|"
        r"million|billion|trillion|thousand|mn|bn|tn|cr|k|m|b)?"  # optional scale
        r"\s*"
        r"(?P<suffix>[₹$€£¥]?\s*(?:inr|usd|eur|gbp|jpy|rs\.?)?)?$",  # optional suffix
        re.I,
    )
    m = pattern.match(text)
    if not m:
        return None, "", value_str

    num_str = m.group("number")
    scale_str = (m.group("scale") or "").lower().replace(" ", "_")

    # Map "lakh_crore" → key in _SCALE_MAP
    if "lakh" in scale_str and "crore" in scale_str:
        scale_key = "lakh crore"
    else:
        scale_key = scale_str.replace("_", " ").strip()

    try:
        num = float(num_str)
    except ValueError:
        return None, "", value_str

    return num, scale_key, text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_unit(value: str, unit: Optional[str]) -> Dict[str, Any]:
    """
    Normalize a (value, unit) pair into a structured dict.

    Parameters
    ----------
    value:
        Raw value string from Fact.value, e.g. "42 crore", "3.5%", "200 bps".
    unit:
        Raw unit string from Fact.unit, e.g. "INR", "%", "bps", or None.

    Returns
    -------
    Dict with keys:
    - ``normalized_value``:  float or None if parsing failed
    - ``normalized_unit``:   canonical unit string
    - ``currency``:          ISO currency code or None
    - ``currency_conflict``: True if value and unit imply different currencies
    - ``period``:            normalized period string or None (e.g. "Q1")
    - ``vintage``:           normalized year string or None (e.g. "FY2024-25")
    - ``parse_ok``:          bool — False if normalization failed / not applicable
    """
    value = (value or "").strip()
    unit_str = (unit or "").strip()

    result: Dict[str, Any] = {
        "normalized_value": None,
        "normalized_unit": unit_str or None,
        "currency": None,
        "currency_conflict": False,
        "period": None,
        "vintage": None,
        "parse_ok": False,
    }

    # ----------------------------------------------------------------
    # 1. Basis points
    # ----------------------------------------------------------------
    m = _BPS_PATTERN.match(value)
    if m:
        result.update({
            "normalized_value": float(m.group(1)),
            "normalized_unit": "bps",
            "parse_ok": True,
        })
        return result

    # ----------------------------------------------------------------
    # 2. Percentage
    # ----------------------------------------------------------------
    m = _PCT_PATTERN.match(value)
    if m:
        result.update({
            "normalized_value": float(m.group(1)),
            "normalized_unit": "%",
            "parse_ok": True,
        })
        return result

    # ----------------------------------------------------------------
    # 3. Numeric + scale (currency amounts)
    # ----------------------------------------------------------------
    num, scale_key, _ = _parse_numeric_and_scale(value)
    if num is not None:
        # If value had no scale word, check if the unit string contains one
        if not scale_key and unit_str:
            unit_scale_pattern = re.compile(
                r"\b(lakh\s+crore|lakh_crore|lakh|crore|lac|"
                r"million|billion|trillion|thousand|mn|bn|tn|cr|k|m|b)\b",
                re.I
            )
            m_scale = unit_scale_pattern.search(unit_str)
            if m_scale:
                s = m_scale.group(1).lower()
                if "lakh" in s and "crore" in s:
                    scale_key = "lakh crore"
                else:
                    scale_key = s

        scale_factor = _SCALE_MAP.get(scale_key, 1.0)
        normalized_num = num * scale_factor

        # Currency detection — from value string and from unit field
        currency_from_value = _detect_currency(value)
        currency_from_unit  = _detect_currency(unit_str) if unit_str else None

        currency = currency_from_value or currency_from_unit
        conflict = (
            currency_from_value is not None
            and currency_from_unit is not None
            and currency_from_value != currency_from_unit
        )

        # Normalized unit: currency code if known, else the scale label
        if currency:
            norm_unit = currency
        elif scale_key:
            norm_unit = scale_key
        else:
            norm_unit = unit_str or ""

        result.update({
            "normalized_value": normalized_num,
            "normalized_unit": norm_unit or None,
            "currency": currency,
            "currency_conflict": conflict,
            "parse_ok": True,
        })
        return result

    # ----------------------------------------------------------------
    # 4. Date / period normalization
    # ----------------------------------------------------------------
    period_result = normalize_period(value)
    if period_result["parse_ok"]:
        result.update(period_result)
        return result

    # ----------------------------------------------------------------
    # 5. Fallback — return as-is, mark parse_ok=False
    # ----------------------------------------------------------------
    result["normalized_unit"] = unit_str or None
    return result


def normalize_period(text: str) -> Dict[str, Any]:
    """
    Try to extract a normalized {period, vintage} from a date/period string.

    Returns a dict compatible with normalize_unit's output shape.
    """
    result: Dict[str, Any] = {
        "normalized_value": None,
        "normalized_unit": "period",
        "currency": None,
        "currency_conflict": False,
        "period": None,
        "vintage": None,
        "parse_ok": False,
    }
    t = text.strip()

    # FY2024-25 / FY2024/25
    m = _FY_LONG.search(t)
    if m:
        y1, y2 = m.group(1), m.group(2)
        if len(y2) == 2:
            y2 = y1[:2] + y2
        result.update({"period": "FY", "vintage": f"{y1}-{y2}", "parse_ok": True})
        return result

    # Q1 FY24 / Q2 2024
    m = _Q_YEAR.search(t)
    if m:
        quarter = m.group(1).upper()
        year = m.group(2)
        if len(year) == 2:
            year = "20" + year
        result.update({"period": quarter, "vintage": f"FY{year}", "parse_ok": True})
        return result

    # FY24 / FY2024 (no sub-period)
    m = _FY_SHORT.search(t)
    if m:
        year = m.group(1)
        if len(year) == 2:
            year = "20" + year
        result.update({"period": "FY", "vintage": f"FY{year}", "parse_ok": True})
        return result

    # Bare year
    m = _YEAR_ONLY.search(t)
    if m:
        result.update({"period": "annual", "vintage": m.group(1), "parse_ok": True})
        return result

    return result


def same_currency(unit_a: Optional[str], unit_b: Optional[str]) -> bool:
    """Return True if both units refer to the same currency (or are both None)."""
    if unit_a is None and unit_b is None:
        return True
    ca = _detect_currency(unit_a or "")
    cb = _detect_currency(unit_b or "")
    if ca is None and cb is None:
        return unit_a == unit_b
    return ca == cb
