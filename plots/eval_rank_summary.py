#!/usr/bin/env python3
"""Create a stacked (vertical) summary visualization for eval sweep results.

Top panel:
  Average rank per method across all (env, metric) rows.
Bottom panel:
  Heatmap of average rank per metric and method.

Choose the baselines
in BASELINES below (comment out a line to drop it). Runs whose training
diverged are ranked last (DIVERGED_POLICY = "last") or left out ("exclude").

Usage:
    python plots/eval_rank_summary.py            # SWEEP_NAME from plots/settings.py
    python plots/eval_rank_summary.py <sweep_name>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

# Allow direct sibling import when run as:
#   python plots/plot_eval_summary_stacked.py ...
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from make_eval_latex_table import (
    DEFAULT_METRICS,
    collect_sweep_data,
    find_diverged_runs,
    metric_direction,
    metric_target,
)
from settings import (
    CORL_SETTINGS_FULL_WIDTH,
    FIG_FORMAT,
    SWEEP_NAME,
    figures_dir,
    set_size,
)


plt.rcParams.update(CORL_SETTINGS_FULL_WIDTH)

# ---------------------------------------------------------------------------
# Baselines. Comment out a line to drop it from the figure; ranks are computed
# among the baselines listed here only.
#   (label in stats.pkl, display name, color)
# ---------------------------------------------------------------------------
BASELINES = [
    # ("s-NLL",        "s-NLL (no HB)", "tab:pink"),
    ("s-NLL-HB",     "s-NLL",     "tab:red"),
    ("m-NLL",        "m-NLL",     "tab:blue"),
    ("m-CRPS",       "m-CRPS",    "tab:orange"),
    ("m-arNLL",       "m-arNLL",   "tab:gray"),
    ("m-arCRPS",      "m-arCRPS",  "tab:purple"),
    ("m-KDE",         "m-KDE",     "tab:green"),
]

DEFAULT_SWEEP = SWEEP_NAME
OUTPUT_TAG = ""  # appended to output file names, e.g. "_no_arnll"
DIVERGED_POLICY = "last"  # "last": diverged runs share the worst ranks; "exclude": skip them
# For baselines with diverged runs, draw the top-panel bar as a solid part
# (mean rank over converged runs only) plus a hatched part on top up to the
# mean rank over all runs (diverged ranked last). Only used with "last".
SPLIT_DIVERGED_BARS = True

LABEL_ALIASES = {raw: name for raw, name, _ in BASELINES}
BASELINE_COLORS = {name: color for _, name, color in BASELINES}
BASELINES_NAMES = {name: rf"\texttt{{{name}}}" for _, name, _ in BASELINES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked summary of eval sweep results."
    )
    parser.add_argument(
        "sweep_name",
        type=str,
        nargs="?",
        default=DEFAULT_SWEEP,
        help=f"Sweep name under results/eval/<sweep_name>/ (default: {DEFAULT_SWEEP})",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("results/eval"),
        help="Root eval directory (default: results/eval).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS.keys()),
        help="Metric keys to include.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Use *_mean or *_median values from stats records.",
    )
    parser.add_argument(
        "--width-pt",
        type=float,
        default=397.0,
        help=(
            "Reference template width in pt. Figure width is fixed to 0.5 * width_pt."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path (default: figures/<sweep_name>/rank_summary_stacked.pdf, see plots/settings.py)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    return parser.parse_args()


def _rank_with_ties(scores: List[float], atol: float = 1e-12) -> List[float]:
    """Return 1-based average ranks for a list of scores (lower is better)."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        j = pos + 1
        while j < n and np.isclose(scores[order[j]], scores[order[pos]], rtol=0.0, atol=atol):
            j += 1
        avg_rank = (pos + 1 + j) / 2.0
        for k in range(pos, j):
            ranks[order[k]] = avg_rank
        pos = j
    return ranks


def _metric_label(metric_key: str) -> str:
    return DEFAULT_METRICS.get(metric_key, metric_key.replace("_", " "))


def _canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def main() -> None:
    args = parse_args()
    sweep_dir = args.eval_root / args.sweep_name
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    metrics = list(args.metrics)
    envs, methods, values, _ = collect_sweep_data(
        sweep_dir=sweep_dir,
        metrics=metrics,
        aggregate=args.aggregate,
    )
    if not envs or not methods:
        raise RuntimeError(f"No eval data found under: {sweep_dir}")

    # Only the baselines listed in BASELINES, in that order.
    order = [raw for raw, _, _ in BASELINES]
    methods = sorted((m for m in methods if m in LABEL_ALIASES), key=order.index)
    diverged = find_diverged_runs(sweep_dir, args.sweep_name)
    print(f"Diverged runs ({DIVERGED_POLICY}): {diverged}")

    method_ranks: Dict[str, List[float]] = {m: [] for m in methods}
    # Ranks from environments where the method converged.
    converged_ranks: Dict[str, List[float]] = {m: [] for m in methods}
    metric_method_ranks: Dict[str, Dict[str, List[float]]] = {
        mk: {m: [] for m in methods} for mk in metrics
    }

    for env in envs:
        for metric in metrics:
            active_methods = []
            scores = []
            for method in methods:
                # if method not in BASELINE_COLORS:
                #     continue
                if method in diverged.get(env, set()):
                    if DIVERGED_POLICY == "last":
                        active_methods.append(method)
                        scores.append(np.inf)
                    continue
                center = values.get(env, {}).get(method, {}).get(metric, {}).get("center", np.nan)
                if not np.isfinite(center):
                    continue
                direction = metric_direction(metric)
                if direction == "target":
                    tgt = metric_target(metric)
                    assert tgt is not None
                    score = abs(float(center) - tgt)
                else:
                    score = float(center)
                active_methods.append(method)
                scores.append(score)

            if len(active_methods) == 0:
                continue

            ranks = _rank_with_ties(scores)
            for method, rk in zip(active_methods, ranks):
                method_ranks[method].append(rk)
                if method not in diverged.get(env, set()):
                    converged_ranks[method].append(rk)
                metric_method_ranks[metric][method].append(rk)

    mean_rank = {
        m: float(np.mean(method_ranks[m])) if len(method_ranks[m]) > 0 else np.nan for m in methods
    }
    methods_sorted = sorted(methods, key=lambda m: (np.inf if np.isnan(mean_rank[m]) else mean_rank[m]))
    x = np.arange(len(methods_sorted))
    y = np.array([mean_rank[m] for m in methods_sorted], dtype=float)

    heat = np.full((len(metrics), len(methods_sorted)), np.nan, dtype=float)
    for ri, metric in enumerate(metrics):
        for ci, method in enumerate(methods_sorted):
            vals = metric_method_ranks[metric][method]
            if len(vals) > 0:
                heat[ri, ci] = float(np.mean(vals))

    # Width explicitly fixed to 0.5 * width_pt.
    fig_w, fig_h = set_size(args.width_pt, fraction=0.5, subplots=(2, 1))
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(fig_w, 0.8*fig_h),
        gridspec_kw={"height_ratios": [1., 1.4], "hspace": 0.1},
    )

    bar_colors = [BASELINE_COLORS.get(_canonical_label(m), "tab:blue") for m in methods_sorted]
    has_diverged = [
        any(m in diverged.get(env, set()) for env in envs) for m in methods_sorted
    ]
    split = SPLIT_DIVERGED_BARS and DIVERGED_POLICY == "last"
    y_converged = np.array(
        [
            float(np.mean(converged_ranks[m])) if (split and div and converged_ranks[m]) else y_all
            for m, div, y_all in zip(methods_sorted, has_diverged, y)
        ],
        dtype=float,
    )
    bars = ax0.bar(x, y_converged, color=bar_colors, alpha=0.85, width=0.75)
    if split and any(has_diverged):
        for xi, yc, ya, col, div in zip(x, y_converged, y, bar_colors, has_diverged):
            if div and ya > yc:
                ax0.bar(
                    xi, ya - yc, bottom=yc, width=0.75, color=col, alpha=0.3,
                    hatch="////", edgecolor="white", linewidth=0.0,
                )
                ax0.text(xi, ya + 0.1, "all", ha="center", va="bottom", fontsize=5)
                ax0.text(xi, yc - 0.12, "conv.", ha="center", va="top", fontsize=3.5, color="white")
        print("Mean rank, converged runs only:", {m: round(float(v), 2) for m, v in zip(methods_sorted, y_converged)})
    ax0.set_xticks([])
    y_top = float(np.nanmax(y)) + 0.6
    ax0.set_ylim(0.0, y_top)
    ax0.set_yticks(np.arange(0, int(np.floor(y_top)) + 1))
    # ax0.set_xticklabels(
    #     # [BASELINES_NAMES.get(_canonical_label(m), m) for m in methods_sorted],
    #     ["" for m in methods_sorted],
    #     rotation=30,
    #     ha="right",
    #     fontsize=7,
    # )
    ax0.set_ylabel("Overall Rank", fontsize=7)
    # ax0.set_title("Overall Ranking")
    ax0.grid(axis="y", alpha=0.25)

    ax0.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3.5,
        width=0.8,
        bottom=False,
        left=True,
        top=False,
        right=False,
        labelsize=6,
    )
    
    if sns is not None:
        sns.despine(ax=ax0, offset=3, trim=True, bottom=True, right=False)

    ax0.spines["bottom"].set_visible(False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    print(methods_sorted)
    for b, method in zip(bars, methods_sorted):
        y_pos = 0.55 * b.get_height()
        method_label = BASELINES_NAMES.get(_canonical_label(method), method)
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            0.2, #y_pos,
            method_label,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=90,
            color="black",
        )

    im = ax1.imshow(
        heat,
        aspect="auto",
        cmap="PiYG_r",
        vmin=1.0,
        vmax=max(2.0, len(methods)),
        interpolation="nearest",
    )
    ax1.set_xticks(np.arange(len(methods_sorted)))
    # Smaller labels and cell numbers when many baselines are shown.
    many = len(methods_sorted) > 6
    ax1.set_xticklabels(
        [BASELINES_NAMES.get(_canonical_label(m), m) for m in methods_sorted],
        rotation=45 if many else 30,
        ha="right",
        rotation_mode="anchor" if many else "default",
        fontsize=6 if many else 7,
    )
    for tick_label, method in zip(ax1.get_xticklabels(), methods_sorted):
        tick_label.set_color("black")
        tick_label.set_bbox(
            dict(
                boxstyle="round,pad=0.2",
                facecolor=BASELINE_COLORS.get(_canonical_label(method), "white"),
                edgecolor="none",
                alpha=0.2,
            )
        )
    ax1.set_yticks(np.arange(len(metrics)))
    ax1.set_yticklabels([_metric_label(mk) for mk in metrics], fontsize=7)
    # ax1.set_title("Mean Rank by Metric")
    ax1.tick_params(axis="both", length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks(np.arange(-0.5, len(methods_sorted), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax1.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
    ax1.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.05, pad=0.02)
    cbar.set_label("Average Rank", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            if np.isfinite(heat[r, c]):
                ax1.text(
                    c,
                    r,
                    f"${heat[r, c]:.2f}$",
                    ha="center",
                    va="center",
                    fontsize=6 if many else 7,
                    color="black" if heat[r, c] > (len(methods) + 1) / 2 else "black",
                )

    fig.tight_layout()

    output = args.output
    if output is None:
        output = figures_dir(args.sweep_name) / f"rank_summary_stacked{OUTPUT_TAG}.{FIG_FORMAT}"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    print(f"Wrote stacked summary figure to: {output}")
    print(f"Figure width: {fig_w:.3f} in  (0.5 * {args.width_pt} pt)")


if __name__ == "__main__":
    main()
