"""Number formatting shared by README templates, interview notes and deck text.

One implementation, so "12.4%" in the README and "12.4%" on slide 6 are the
same string produced from the same value in reports/metrics.json.

Copy into each project as ``src/<package>/fmt.py``.
"""

from __future__ import annotations

import math


def _check(v) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        raise ValueError("Tried to format a missing value; fix the metric, do not paper over it")
    return float(v)


def pct(v, d: int = 1) -> str:
    """0.1234 -> '12.3%'."""
    return f"{_check(v) * 100:.{d}f}%"


def pp(v, d: int = 1, signed: bool = True) -> str:
    """Difference of two rates, 0.021 -> '+2.1 pp'."""
    x = _check(v) * 100
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return f"{s} pp"


def num(v, d: int = 0) -> str:
    """12345.6 -> '12,346'."""
    return f"{_check(v):,.{d}f}"


def compact(v, d: int = 1) -> str:
    """1234567 -> '1.2M', 45600 -> '45.6k'."""
    x = _check(v)
    a = abs(x)
    if a >= 1e9:
        return f"{x / 1e9:.{d}f}B"
    if a >= 1e6:
        return f"{x / 1e6:.{d}f}M"
    if a >= 1e3:
        return f"{x / 1e3:.{d}f}k"
    return f"{x:.0f}"


def money(v, cur: str = "€", d: int = 0, short: bool = False) -> str:
    """Currency. short=True -> '€1.2M'. Negative -> '-€1.2M'."""
    x = _check(v)
    sign = "-" if x < 0 else ""
    body = compact(abs(x), 1) if short else f"{abs(x):,.{d}f}"
    return f"{sign}{cur}{body}"


def signed(v, d: int = 1) -> str:
    return f"{_check(v):+,.{d}f}"


def ratio(v, d: int = 1) -> str:
    """2.345 -> '2.3x'."""
    return f"{_check(v):.{d}f}x"


def dec(v, d: int = 2) -> str:
    """Plain decimal, e.g. an AUC: 0.7123 -> '0.71'."""
    return f"{_check(v):.{d}f}"


def usd(v, d: int = 1) -> str:
    """Northvale amounts are US dollars (Lending Club data). 12_681_234 -> '$12.7M'."""
    x = _check(v)
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e9:
        body = f"{a / 1e9:.{d}f}B"
    elif a >= 1e6:
        body = f"{a / 1e6:.{d}f}M"
    elif a >= 1e3:
        body = f"{a / 1e3:.{d}f}k"
    else:
        body = f"{a:,.0f}"
    return f"{sign}${body}"


def susd(v, d: int = 1) -> str:
    """Signed dollars for changes: '+$13.8M' / '-$1.1M'."""
    return ("+" if _check(v) >= 0 else "") + usd(v, d)


FILTERS = {
    "pct": pct,
    "pp": pp,
    "num": num,
    "compact": compact,
    "money": money,
    "signed": signed,
    "ratio": ratio,
    "dec": dec,
    "usd": usd,
    "susd": susd,
}
