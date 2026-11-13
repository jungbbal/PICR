"""Shared figure styling: readable label sizes.

The defaults produce 10 pt text, which is unreadable once a figure is scaled down
into a slide or viewed next to a 1024 px video panel. Import and call `apply()`
before creating any figure.
"""

from __future__ import annotations

BASE = 17


def apply(scale: float = 1.0) -> None:
    import matplotlib

    s = BASE * scale
    matplotlib.rcParams.update(
        {
            "font.size": s,
            "axes.titlesize": s * 1.15,
            "axes.labelsize": s * 1.05,
            "xtick.labelsize": s * 0.9,
            "ytick.labelsize": s * 0.9,
            "legend.fontsize": s * 0.9,
            "figure.titlesize": s * 1.3,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
        }
    )
