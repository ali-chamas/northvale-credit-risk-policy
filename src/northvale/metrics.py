"""reports/metrics.json: the single source for every number in README, deck and notes.

Each analysis step writes its own top-level section. Values are raw (fractions,
not percentages); formatting happens only in fmt.py.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import config as C


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValueError("NaN/inf metric: fix the computation instead of writing it")
        return round(f, 10)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def load() -> dict:
    if C.METRICS.exists():
        return json.loads(C.METRICS.read_text())
    return {}


def update(section: str, values: dict) -> None:
    """Replace one top-level section of metrics.json."""
    m = load()
    m[section] = _clean(values)
    C.METRICS.parent.mkdir(parents=True, exist_ok=True)
    C.METRICS.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n")


def reset() -> None:
    if C.METRICS.exists():
        C.METRICS.unlink()
