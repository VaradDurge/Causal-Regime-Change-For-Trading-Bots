"""Shared matplotlib style for IEEE two-column figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = 3.5  # single column width, inches
FULL = 7.16  # double column width, inches

REGIME_COLORS = {
    "bull": "#009E73",
    "bear": "#E69F00",
    "calm": "#56B4E9",
    "crisis": "#D55E00",
}
SYSTEM_COLORS = {
    "conditioned": "#0072B2",
    "blind": "#CC79A7",
    "buy_hold": "#444444",
}


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )
