# ABOUTME: Shared colours and matplotlib settings for the validation figures.
"""
Our replication is always blue, Ken French's always orange (a colour-blind-safe pair),
so every chart in the presentation reads the same way.
"""

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt  # noqa: E402

COLOR_MINE = "#2a78d6"
COLOR_FRENCH = "#eb6834"
COLOR_TEXT = "#52514e"
COLOR_GRID = "#e4e3df"
COLOR_INK = "#0b0b0b"
# Further categorical colours, used in fixed order when a chart needs more series.
COLOR_EXTRA = ["#1baf7a", "#eda100", "#6250d6"]


def apply_style():
    """Light background, recessive grid, no top/right spines."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": COLOR_GRID, "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT, "ytick.color": COLOR_TEXT, "text.color": COLOR_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLOR_GRID, "grid.linewidth": 0.8,
        "font.size": 10, "legend.frameon": False,
    })
