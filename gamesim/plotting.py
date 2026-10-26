"""Shared matplotlib style so every chart in the portfolio looks like one system."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind-safe categorical palette (Okabe–Ito inspired, slightly muted)
PALETTE = ["#2F6DB5", "#E07B39", "#3A9E6F", "#C2455B", "#7A5BA6", "#8C7A5B", "#4FA3C7", "#B8A13A"]
GREY = "#8A8F98"
INK = "#1F2430"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#C9CDD3",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#E6E8EB",
            "grid.linewidth": 0.8,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "legend.frameon": False,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


setup()
