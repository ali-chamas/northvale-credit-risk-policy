"""Shared chart style for the analytics portfolio.

Copy this file into each project as ``src/<package>/viz.py``. Every figure in a
notebook, README and deck goes through ``use_style`` and ``save`` so the three
case studies read as one body of work.

Rules baked in (see STANDARDS.md):
- one y-axis per chart, hairline grid on the value axis only
- emphasis = project accent on the mark that carries the message, grey for context
- categorical colours in fixed order (SERIES), never cycled past eight
- text is ink/grey, never the series colour
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Ink and chrome
INK = "#1a1a1a"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
CONTEXT = "#cfcdc4"  # de-emphasised marks (the "everything else" grey)

# Categorical palette, validated for colour-vision deficiency in this order.
# Use in order; with more than three series on a scatter, facet instead.
SERIES = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]

# Sequential (single hue, light -> dark) for heatmaps such as cohort tables.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# Diverging: blue <-> red with a neutral grey midpoint.
DIVERGING = ["#1c5cab", "#6da7ec", "#f0efec", "#ec8b8a", "#c22d2c"]

# Status colours mean good/bad only. Never use them as a series colour.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# One accent per project; it marks the data point the slide title talks about.
ACCENTS = {"vitrine": "#eb6834", "northvale": "#2a78d6", "pulse": "#6250d6"}

# Standard figure sizes in inches. They match the deck's image boxes exactly,
# so a figure is never rescaled on a slide (font sizes stay true).
SIZES = {
    "finding": (8.1, 4.8),  # left of the commentary panel on a finding slide
    "full": (12.1, 4.5),  # full-width chart slide
    "half": (5.85, 4.2),  # one of two charts side by side
    "square": (6.0, 6.0),  # README-only, e.g. a confusion matrix
}

DPI = 220

_accent = SERIES[0]


def use_style(project: str | None = None) -> str:
    """Apply the shared rcParams. Returns the project's accent colour."""
    global _accent
    _accent = ACCENTS.get(project or "", SERIES[0])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.titlecolor": INK,
            "axes.labelsize": 11,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "axes.facecolor": SURFACE,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "legend.labelcolor": INK_2,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "patch.linewidth": 0,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": None,  # keep the exact deck size; use fig.tight_layout or constrained layout
        }
    )
    return _accent


def accent() -> str:
    return _accent


def figure(size: str = "finding", **kwargs):
    """Create a figure at one of the standard sizes with constrained layout."""
    fig, ax = plt.subplots(figsize=SIZES[size], layout="constrained", **kwargs)
    return fig, ax


def titles(ax, title: str, subtitle: str | None = None) -> None:
    """Chart title says what is measured; the slide title carries the conclusion."""
    ax.set_title(title, loc="left", pad=26 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom", ha="left")


def note(fig, text: str) -> None:
    """Source / definition note in the bottom-left corner (reserves space so it never hits tick labels)."""
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(0, 0.05, 1, 0.95))
    fig.text(0.01, 0.01, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")


# Axis formatters -------------------------------------------------------------

def pct_axis(ax, axis: str = "y", decimals: int = 0, scale: float = 100.0) -> None:
    """Format an axis holding fractions (0.12) as percentages (12%)."""
    fmt = FuncFormatter(lambda v, _: f"{v * scale:.{decimals}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands_axis(ax, axis: str = "y", prefix: str = "", suffix: str = "") -> None:
    def _f(v, _):
        a = abs(v)
        if a >= 1e6:
            s = f"{v / 1e6:.1f}M".replace(".0M", "M")
        elif a >= 1e3:
            s = f"{v / 1e3:.0f}k"
        else:
            s = f"{v:.0f}"
        return f"{prefix}{s}{suffix}"

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(_f))


def emphasis_colors(labels, highlight) -> list[str]:
    """Accent for highlighted label(s), context grey for the rest."""
    hl = {highlight} if isinstance(highlight, str) else set(highlight)
    return [_accent if lab in hl else CONTEXT for lab in labels]


def label_bars(ax, bars, fmt=lambda v: f"{v:.0f}", inside: bool = False, color: str = INK_2, size: float = 10) -> None:
    """Direct-label bars selectively (pass only the bars that matter)."""
    for b in bars:
        v = b.get_height()
        x = b.get_x() + b.get_width() / 2
        y = v if not inside else v / 2
        ax.annotate(
            fmt(v),
            (x, y),
            xytext=(0, 4 if not inside else 0),
            textcoords="offset points",
            ha="center",
            va="bottom" if not inside else "center",
            fontsize=size,
            color=color,
        )


def label_hbars(ax, bars, fmt=lambda v: f"{v:.0f}", color: str = INK_2, size: float = 10) -> None:
    for b in bars:
        v = b.get_width()
        ax.annotate(
            fmt(v),
            (v, b.get_y() + b.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=size,
            color=color,
        )


def hbar_style(ax) -> None:
    """Horizontal bars: grid on x, no y spine."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["left"].set_visible(False)


def save(fig, name: str, out_dir: str | Path = "reports/figures") -> Path:
    """Save as PNG at deck resolution. Name without extension, snake_case."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
