"""Writing and honesty checks for README, interview notes, docs and deck text.

Fails (exit 1) on:
- banned filler words (Ali's list)
- a README without the fictional-case-study notice near the top
- unrendered template syntax or placeholder text (TODO, TBD, lorem, XX%)
Warns on softer AI-sounding phrasing and heavy em-dash use.

Usage (from the project root):
    python -m <package>.check_text README.md INTERVIEW_NOTES.md deck/deck_spec.json docs/*.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED = [
    r"\bdelv(e|es|ed|ing)\b",
    r"\bleverag(e|es|ed|ing)\b",
    r"\bunlock(s|ed|ing)?\b",
    r"\brobust(ly|ness)?\b",
    r"\bseamless(ly)?\b",
    r"in today'?s fast[- ]paced world",
    r"it'?s worth noting",
    r"\bgame[- ]changer\b",
    r"\bdive deep\b",
    r"\bdeep[- ]dive\b",
    r"\bharness(es|ed|ing)?\b",
    r"\blandscape\b",
    r"\btapestry\b",
]

SOFT = [
    r"\bcrucial\b",
    r"\bpivotal\b",
    r"\bcomprehensive\b",
    r"\bcutting[- ]edge\b",
    r"\bempower(s|ed|ing)?\b",
    r"\belevate(s|d)?\b",
    r"\bshowcas(e|es|ed|ing)\b",
    r"\bmeticulous(ly)?\b",
    r"\bintricate\b",
    r"\btestament\b",
    r"\brealm\b",
    r"\bembark\b",
    r"\bnavigat(e|ing) the\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin conclusion\b",
    r"\bkey takeaways?\b",
    r"\bactionable insights\b",
    r"\bdata-driven decisions\b",
]

PLACEHOLDERS = [r"\{\{", r"\{%", r"\bTODO\b", r"\bTBD\b", r"\bFIXME\b", r"lorem ipsum", r"\bXX+(\.X+)?%?"]

NOTICE = re.compile(r"fictional", re.I)


def scan(path: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        low = line.lower()
        for pat in BANNED:
            if re.search(pat, low):
                errors.append(f"{path}:{i}: banned phrase /{pat}/ -> {line.strip()[:100]}")
        for pat in PLACEHOLDERS:
            if re.search(pat, line, re.I if pat.startswith("lorem") else 0):
                errors.append(f"{path}:{i}: placeholder or template syntax /{pat}/ -> {line.strip()[:100]}")
        for pat in SOFT:
            if re.search(pat, low):
                warnings.append(f"{path}:{i}: soft filler /{pat}/ -> {line.strip()[:100]}")
    dashes = text.count("—")
    if dashes > 3:
        warnings.append(f"{path}: {dashes} em dashes; prefer a colon, comma or full stop")
    if path.name == "README.md":
        head = "\n".join(lines[:25])
        if not NOTICE.search(head):
            errors.append(f"{path}: the fictional case-study notice must appear in the first 25 lines")
    return errors, warnings


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv if Path(p).is_file()]
    if not paths:
        print("check_text: no files given")
        return 1
    all_err, all_warn = [], []
    for p in paths:
        e, w = scan(p)
        all_err += e
        all_warn += w
    for w in all_warn:
        print("WARN ", w)
    for e in all_err:
        print("ERROR", e)
    print(f"check_text: {len(paths)} files, {len(all_err)} errors, {len(all_warn)} warnings")
    return 1 if all_err else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
