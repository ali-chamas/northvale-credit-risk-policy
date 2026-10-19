"""Render README.md and INTERVIEW_NOTES.md from Jinja templates + reports/metrics.json.

Every number in those documents is a template expression such as
``{{ m.delivery.late_rate | pct }}``, so it always matches the latest run.
Undefined keys raise an error instead of rendering blank.

Usage (from the project root):
    python -m <package>.render_docs

Copy into each project as ``src/<package>/render_docs.py`` and adjust the
import of ``fmt`` if the package name differs.
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import fmt

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "docs" / "templates"
METRICS = ROOT / "reports" / "metrics.json"

HEADER = "<!-- Generated from docs/templates/{name}.j2 by `make docs`. Edit the template, not this file. -->\n"


class _Dot(dict):
    """dict with attribute access, so templates can write m.delivery.late_rate."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise KeyError(f"metrics.json has no key '{k}'") from e
        return _Dot(v) if isinstance(v, dict) else v


def main() -> None:
    metrics = _Dot(json.loads(METRICS.read_text()))
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(fmt.FILTERS)
    # Top-level templates render to the repo root; templates/docs/* render into docs/.
    for tpl in sorted(TEMPLATES.rglob("*.md.j2")):
        rel = tpl.relative_to(TEMPLATES).as_posix()
        name = rel[: -len(".j2")]
        text = env.get_template(rel).render(m=metrics)
        (ROOT / name).write_text(HEADER.format(name=name) + text)
        print(f"rendered {name}")


if __name__ == "__main__":
    main()
