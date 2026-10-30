import os
from pathlib import Path

import matplotlib.pyplot as plt

# Sweeps the figures are made from (the sweep_name used for training; "default"
# is also the training default). All outputs of one run go flat into FIGURES_DIR.
SWEEP_NAME = os.environ.get("SWEEP_NAME", "default")
WB_SWEEP_NAME = os.environ.get("WB_SWEEP_NAME", "default")


def figures_dir(sweep: str = SWEEP_NAME) -> Path:
    return Path(os.environ.get("FIGURES_DIR", f"figures/{sweep}"))


FIGURES_DIR = figures_dir()
CACHE_DIR = Path("figures/.cache")
# File format of all figures: "pdf" (default) or "png".
FIG_FORMAT = os.environ.get("FIG_FORMAT", "pdf")



def _pt_to_inch(pt):
    return pt / 72


BASELINES = ["s-NLL", "m-NLL", "m-CRPS", "m-arCRPS", "m-KDE"]
BASELINES_WITH_AR = BASELINES  # backwads comp/
# TAB10 = plt.get_cmap("tab10").colors
BASELINE_COLORS = {
    "s-NLL": "tab:red",
    "m-NLL": "tab:blue",
    "m-KDE": "tab:green",
    "m-CRPS": "tab:orange",
    "m-arNLL": "tab:gray",
    "m-arCRPS": "tab:purple",
}

BASELINES_NAMES = {
    "s-NLL": r"\texttt{s-NLL}",
    "m-NLL": r"\texttt{m-NLL}",
    "m-KDE": r"\texttt{m-KDE}",
    "m-CRPS": r"\texttt{m-CRPS}",
    "m-arCRPS": r"\texttt{m-arCRPS}",
    "m-arNLL": r"\texttt{m-arNLL}",
}



CORL_SETTINGS = {
    "font.family": "serif",
    "font.size": 8,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.grid": False,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 7,
    "text.usetex": True,  # Set to True if you have LaTeX installed and want to use it for rendering text
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{bm}"
}

CORL_SETTINGS_FULL_WIDTH = {
    **CORL_SETTINGS,
    "figure.figsize": (_pt_to_inch(397), _pt_to_inch(144)),
}


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
