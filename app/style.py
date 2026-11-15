"""Shared figure style for the app: categorical palette (dataviz reference, slots 1-5 in fixed order)."""

from stegaudit.appcore import POLICIES

COLORS = dict(zip(POLICIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    """Recessive grid and axes."""
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
