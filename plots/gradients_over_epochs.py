#!/usr/bin/env python3
"""Plot gradient norms and parameter updates over epochs from local CSV logs.

Optimization diagnostics of the joint backbone/posterior training. All objectives use
global gradient clipping at norm 1; this script shows how large the gradients
are before clipping and how much the representation and the posterior still move.
Reads from each run's training_metrics.csv:
  Gradients/preclip_{backbone,last_layer}_norm_p95   95th percentile over the
      batches of an epoch of the pre-clipping gradient norm per parameter group
  Parameters/backbone_relative_update, Posterior/W_mean_relative_update,
  Posterior/covariance_parameter_relative_update     epoch-to-epoch relative
      change ||theta_t - theta_{t-1}|| / ||theta_{t-1}||

Choose the baselines in BASELINES and the environments in ENVIRONMENTS below
(comment out a line to drop it). Runs whose training diverged (last 25% of the
loss history non-finite or > 1e6) are left out and named in the panel.

Outputs (in figures/<sweep>/optimization_statistics/, see plots/settings.py):
  grad_norms_<env>.pdf     backbone | last-layer gradient norm, per environment (off, MAKE_SINGLE)
  grad_norms_grid.pdf      backbone gradient norm, all environments in one grid
  grad_updates_<env>.pdf   backbone | posterior mean | covariance relative update (off, MAKE_SINGLE)
  grad_updates_summary.pdf the three relative updates (columns), median and IQR over
                           the non-chaotic and the chaotic systems (rows)
  grad_updates_<q>_grid.pdf  one relative-update grid per quantity (backbone,
                             posterior_mean, posterior_cov)
  grad_summary.tex         final-epoch medians over environments (booktabs table)

Usage:
    PATH=/Library/TeX/texbin:$PATH python plots/gradients_over_epochs.py
"""

from __future__ import annotations

import csv
import functools
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import CORL_SETTINGS, FIG_FORMAT, FIGURES_DIR, SWEEP_NAME, set_size  # noqa: E402

plt.rcParams.update(CORL_SETTINGS)
plt.rcParams.update({"text.usetex": True})


# ---------------------------------------------------------------------------
# Baselines. Comment out a line to drop it; the order is the legend order.
#   (training loss = model folder name, display name, color)
# ---------------------------------------------------------------------------
BASELINES = [
    # ("single_step_elbo",               "s-NLL (no HB)", "tab:pink"),
    ("single_step_elbo_hbatch",        "s-NLL",     "tab:red"),
    ("multi_step_elbo",                "m-NLL",     "tab:blue"),
    ("multi_step_factored_crps",       "m-CRPS",    "tab:orange"),
    ("multi_step_ar_elbo",             "m-arNLL",   "tab:gray"),
    ("multi_step_ar_crps",             "m-arCRPS",  "tab:purple"),
    # ("multi_step_kde",                 "m-KDE",     "tab:green"),
]

# Environments. Each gets its own figures; all of them go into the grids and the table.
ENVIRONMENTS = [
    "variable_speed_limit_cycle",
    "spiral",
    "swimmer",
    "bifurcating_limit_cycle",
    "glycolytic_oscillator",
    "dysts_lorenz",
    "dysts_rossler",
    "dysts_thomas",
    "dysts_chen",
    "dysts_chua",
    "dysts_dadras",
    "dysts_aizawa",
    "dysts_rucklidge",
    "dysts_sprott_a",
    "dysts_sprott_b",
    "dysts_nose_hoover",
    "dysts_lorenz84",
    "dysts_rabinovich_fabrikant",
]

SWEEP = SWEEP_NAME
RESULTS_ROOT = Path("results")
TRAIN_SEED = 42
OUT_DIR = FIGURES_DIR / "optimization_statistics"
OUTPUT_TAG = ""  # appended to output file names, e.g. "_no_arnll"
MAKE_SINGLE = False  # per-environment figures; the grids are the overview
MAKE_GRID = True
# One update grid per quantity (backbone, posterior mean, posterior cov.).
MAKE_UPDATES_GRID = True
MAKE_UPDATES_SUMMARY = True

DIVERGENCE_TAIL_FRACTION = 0.25
DIVERGENCE_THRESHOLD = 1e6

CLIP_THRESHOLD = 1.0  # global gradient-norm clipping used by all objectives
# Epoch 0 holds the untrained-model transient (pre-clip norms up to 1e11 for
# m-arCRPS on the chaotic systems), which squashes the log axis; 0 shows it.
FIRST_EPOCH = 1
# Panels: (metric in training_metrics.csv, panel title).
NORM_PANELS = [
    ("Gradients/preclip_backbone_norm_p95", "Backbone"),
    ("Gradients/preclip_last_layer_norm_p95", "Last layer"),
]
UPDATE_PANELS = [
    ("Parameters/backbone_relative_update", "Backbone"),
    ("Posterior/W_mean_relative_update", "Posterior mean"),
    ("Posterior/covariance_parameter_relative_update", "Posterior cov."),
]
# Line style per update panel in the update grid (one panel per environment).
UPDATE_STYLES = ["-", "--", ":"]
DPI = 200
# ---------------------------------------------------------------------------


SHORT_ENV_NAMES = {
    "variable_speed_limit_cycle": "VSLC",
    "bifurcating_limit_cycle": "Bifurcating LC",
    "glycolytic_oscillator": "Glycolytic",
    "dysts_rabinovich_fabrikant": "Rabinovich-F.",
}


def _prettify_env(env: str) -> str:
    if env in SHORT_ENV_NAMES:
        return SHORT_ENV_NAMES[env]
    if env.startswith("dysts_"):
        env = env[len("dysts_"):]
    return env.replace("_", " ").title()


def _run_dir(env: str, loss: str) -> Path:
    return RESULTS_ROOT / SWEEP / "vbll_delta" / env / loss / str(TRAIN_SEED)


def _loss_history_diverged(losses) -> bool:
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return False
    tail = arr[int((1.0 - DIVERGENCE_TAIL_FRACTION) * arr.size):]
    return bool(np.any(~np.isfinite(tail)) or np.any(np.abs(tail) > DIVERGENCE_THRESHOLD))


@functools.lru_cache(maxsize=None)
def _diverged(env: str, loss: str) -> bool:
    path = _run_dir(env, loss) / "vbll_model.pkl"
    if not path.exists():
        return False
    with path.open("rb") as handle:
        saved = pickle.load(handle)
    return _loss_history_diverged(saved.get("train_losses", [])) or _loss_history_diverged(
        saved.get("val_losses", [])
    )


@functools.lru_cache(maxsize=None)
def _load_metrics(env: str, loss: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]] | None:
    """Return {metric: (epochs, values)} for the plotted metrics of one run."""
    path = _run_dir(env, loss) / "training_metrics.csv"
    if not path.exists():
        return None
    wanted = {metric for metric, _ in NORM_PANELS + UPDATE_PANELS}
    series: Dict[str, Dict[int, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] in wanted:
                series.setdefault(row["metric"], {})[int(row["epoch"])] = float(row["value"])
    out = {}
    for metric, by_epoch in series.items():
        epochs = np.array([e for e in sorted(by_epoch) if e >= FIRST_EPOCH])
        values = np.array([by_epoch[e] for e in epochs])
        # Non-finite or zero values cannot be drawn on a log axis.
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        out[metric] = (epochs, values)
    return out


def _draw_metric(ax, env: str, metric: str, linestyle: str = "-") -> List[str]:
    """Draw one metric for all baselines; return the diverged baselines."""
    skipped: List[str] = []
    for loss, name, color in BASELINES:
        if _diverged(env, loss):
            skipped.append(name)
            continue
        data = _load_metrics(env, loss)
        if data is None or metric not in data:
            continue
        epochs, values = data[metric]
        ax.plot(epochs, values, linewidth=0.6, color=color, linestyle=linestyle,
                label=rf"\texttt{{{name}}}")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="both", which="major", labelsize=5, pad=1, length=2)
    ax.tick_params(axis="y", which="minor", length=1)
    return skipped


def _clip_line(ax) -> None:
    ax.axhline(CLIP_THRESHOLD, color="black", linestyle="--", linewidth=0.7, zorder=0)


def _legend_handles(clip: bool) -> List[plt.Line2D]:
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.2, label=rf"\texttt{{{name}}}")
        for _, name, color in BASELINES
    ]
    if clip:
        handles.append(plt.Line2D([0], [0], color="black", linewidth=0.7, linestyle="--",
                                  label="clip norm"))
    return handles


def _diverged_note(skipped: List[str]) -> str:
    return ", ".join(rf"\texttt{{{s}}}" for s in skipped) + " diverged (not shown)"


def _plot_panels(env: str, panels, ylabel: str, clip: bool, width_fraction: float, stem: str) -> Path:
    x_fig, _ = set_size(397 * width_fraction, fraction=1.0)
    y_fig = 1.45
    fig, axes = plt.subplots(1, len(panels), figsize=(x_fig, y_fig), sharey=True)
    skipped: List[str] = []
    for ax, (metric, title) in zip(axes, panels):
        skipped = _draw_metric(ax, env, metric)
        if clip:
            _clip_line(ax)
        ax.set_title(title, fontsize=6, pad=2)
        ax.set_xlabel("Epoch", fontsize=6, labelpad=1)
    axes[0].set_ylabel(ylabel, fontsize=6, labelpad=1)
    handles = _legend_handles(clip)
    # One legend row at full width; two rows on the narrower norm figure.
    ncol = len(handles) if len(panels) > 2 else math.ceil(len(handles) / 2)
    n_rows = math.ceil(len(handles) / ncol)
    fig.legend(handles=handles, loc="upper center", ncol=ncol, fontsize=5.5,
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.4,
               handletextpad=0.4, columnspacing=0.8, labelspacing=0.2)
    top = 1.0 - (0.115 * n_rows + 0.03) / y_fig  # below the legend (inches -> fraction)
    if skipped:
        fig.text(0.5, top, _diverged_note(skipped), ha="center", va="top", fontsize=5, color="0.35")
        top -= 0.1 / y_fig
    fig.tight_layout(rect=(0, 0, 1, top), w_pad=0.6)
    out = OUT_DIR / f"{stem}_{env}{OUTPUT_TAG}.{FIG_FORMAT}"
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)
    return out


def plot_norms_single(env: str) -> Path:
    return _plot_panels(env, NORM_PANELS, r"Grad.\ norm (p95)", clip=True,
                        width_fraction=0.5, stem="grad_norms")


def plot_updates_single(env: str) -> Path:
    return _plot_panels(env, UPDATE_PANELS, "Relative update", clip=False,
                        width_fraction=0.75, stem="grad_updates")


def _plot_grid(envs: List[str], metrics, ylabel: str, clip: bool, stem: str) -> Path:
    """Grid over environments; several metrics per panel use UPDATE_STYLES.

    Diverged runs are left out and named in the top right of their panel.
    """
    ncols = 6
    nrows = math.ceil(len(envs) / ncols)
    x_fig, _ = set_size(397, fraction=1.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(x_fig, 1.15 * nrows), squeeze=False)
    for ax, env in zip(axes.flat, envs):
        skipped: List[str] = []
        for (metric, _), style in zip(metrics, UPDATE_STYLES):
            skipped = _draw_metric(ax, env, metric, linestyle=style)
        if clip:
            _clip_line(ax)
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=4))
        ax.set_title(_prettify_env(env), fontsize=6, pad=2)
        if skipped:
            # Log-axis headroom so the note does not sit on the curves.
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top * (top / bottom) ** 0.45)
            note = "\n".join(rf"\texttt{{{name}}}" + "\ndiverged" for name in skipped)
            ax.text(0.97, 0.96, note, transform=ax.transAxes, ha="right", va="top",
                    multialignment="right", linespacing=1.1, fontsize=4.5, color="0.35")
    for ax in axes.flat[len(envs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch", fontsize=6, labelpad=1)
    axes[nrows // 2, 0].set_ylabel(ylabel, fontsize=6, labelpad=1)
    handles = _legend_handles(clip)
    if len(metrics) > 1:
        handles += [
            plt.Line2D([0], [0], color="0.35", linewidth=0.9, linestyle=style, label=title.lower())
            for (_, title), style in zip(metrics, UPDATE_STYLES)
        ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=6,
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.4)
    out = OUT_DIR / f"{stem}_grid{OUTPUT_TAG}.{FIG_FORMAT}"
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)
    return out


def plot_norms_grid(envs: List[str]) -> Path:
    return _plot_grid(envs, NORM_PANELS[:1], r"Backbone grad.\ norm (p95)", clip=True, stem="grad_norms")


def plot_updates_grids(envs: List[str]) -> List[Path]:
    stems = ["backbone", "posterior_mean", "posterior_cov"]
    return [
        _plot_grid(envs, [panel], f"{panel[1]} relative update", clip=False, stem=f"grad_updates_{stem}")
        for panel, stem in zip(UPDATE_PANELS, stems)
    ]


def plot_updates_summary(envs: List[str]) -> Path:
    """Relative updates in one figure: median (line) and IQR (band) over systems."""
    groups = [
        ("Non-chaotic systems", [e for e in envs if not e.startswith("dysts_")]),
        ("Chaotic systems", [e for e in envs if e.startswith("dysts_")]),
    ]
    groups = [(title, group) for title, group in groups if group]
    x_fig, _ = set_size(397, fraction=1.0)
    fig, axes = plt.subplots(len(groups), len(UPDATE_PANELS), figsize=(x_fig, 1.2 * len(groups) + 0.35),
                             sharex=True, squeeze=False)
    for row, (group_title, group) in enumerate(groups):
        for col, (metric, panel_title) in enumerate(UPDATE_PANELS):
            ax = axes[row, col]
            missing: List[str] = []
            for loss, name, color in BASELINES:
                curves = []
                for env in group:
                    data = None if _diverged(env, loss) else _load_metrics(env, loss)
                    if data is not None and metric in data:
                        curves.append(data[metric])
                if not curves:
                    missing.append(name)
                    continue
                n = min(len(epochs) for epochs, _ in curves)
                values = np.stack([v[:n] for _, v in curves])
                q25, q50, q75 = np.nanpercentile(values, [25, 50, 75], axis=0)
                epochs = curves[0][0][:n]
                ax.fill_between(epochs, q25, q75, color=color, alpha=0.15, linewidth=0)
                ax.plot(epochs, q50, color=color, linewidth=0.8)
            ax.set_yscale("log")
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.tick_params(axis="both", which="major", labelsize=5, pad=1, length=2)
            ax.tick_params(axis="y", which="minor", length=1)
            if row == 0:
                ax.set_title(panel_title, fontsize=6, pad=2)
            if col == 0:
                ax.set_ylabel(f"{group_title}\nrelative update", fontsize=6, labelpad=1)
            if row == len(groups) - 1:
                ax.set_xlabel("Epoch", fontsize=6, labelpad=1)
            if missing:
                note = "\n".join(rf"\texttt{{{name}}}" + "\ndiverged" for name in missing)
                ax.text(0.97, 0.96, note, transform=ax.transAxes, ha="right", va="top",
                        multialignment="right", linespacing=1.1, fontsize=4.5, color="0.35")
    fig.legend(handles=_legend_handles(clip=False), loc="upper center", ncol=len(BASELINES),
               fontsize=6, bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.6,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.12 / fig.get_figheight()), h_pad=0.5, w_pad=0.6)
    out = OUT_DIR / f"grad_updates_summary{OUTPUT_TAG}.{FIG_FORMAT}"
    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)
    return out


def _final(env: str, loss: str, metric: str) -> float:
    data = _load_metrics(env, loss)
    if data is None or metric not in data:
        return float("nan")
    return float(data[metric][1][-1])


def write_summary(envs: List[str]) -> Path:
    """Median over non-diverged runs of the final-epoch values, one row per baseline."""
    metrics = [metric for metric, _ in NORM_PANELS + UPDATE_PANELS]
    rows = []
    for loss, name, _ in BASELINES:
        runs = [env for env in envs if _load_metrics(env, loss) is not None]
        ok = [env for env in runs if not _diverged(env, loss)]
        medians = [np.nanmedian([_final(env, loss, m) for env in ok]) if ok else float("nan")
                   for m in metrics]
        rows.append((name, len(ok), len(runs), medians))

    def fmt(x: float, scale: float = 1.0) -> str:
        if not np.isfinite(x):
            return "--"
        x = x / scale
        return f"{x:.1f}" if x >= 10 else f"{x:.2f}"

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Grad.\ norm p95 (pre-clip)} & \multicolumn{3}{c}{Relative update ($\times 10^{-3}$)} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-7}",
        r"Loss & Finite runs & Backbone & Last layer & Backbone & Post.\ mean & Post.\ cov. \\",
        r"\midrule",
    ]
    for name, n_ok, n_runs, medians in rows:
        cells = [rf"\texttt{{{name}}}", f"{n_ok}/{n_runs}"]
        cells += [fmt(x) for x in medians[:len(NORM_PANELS)]]
        cells += [fmt(x, 1e-3) for x in medians[len(NORM_PANELS):]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table = "\n".join(lines) + "\n"
    out = OUT_DIR / f"grad_summary{OUTPUT_TAG}.tex"
    out.write_text(table)

    print(f"\nFinal-epoch medians over non-diverged runs (clip norm {CLIP_THRESHOLD:g}):")
    print(f"{'loss':10s} {'finite':>7s} {'bb p95':>9s} {'ll p95':>9s} "
          f"{'bb upd':>9s} {'W upd':>9s} {'cov upd':>9s}")
    for name, n_ok, n_runs, medians in rows:
        print(f"{name:10s} {f'{n_ok}/{n_runs}':>7s} " + " ".join(f"{x:9.4g}" for x in medians))
    return out


def main() -> None:
    print(f"Writing to {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    envs = [env for env in ENVIRONMENTS if (RESULTS_ROOT / SWEEP / "vbll_delta" / env).exists()]
    for env in envs if MAKE_SINGLE else []:
        print(f"Saved plot: {plot_norms_single(env)}")
        print(f"Saved plot: {plot_updates_single(env)}")
    if MAKE_GRID and envs:
        print(f"Saved grid: {plot_norms_grid(envs)}")
    if MAKE_UPDATES_SUMMARY and envs:
        print(f"Saved plot: {plot_updates_summary(envs)}")
    if MAKE_UPDATES_GRID and envs:
        for out in plot_updates_grids(envs):
            print(f"Saved grid: {out}")
    if envs:
        print(f"Saved table: {write_summary(envs)}")


if __name__ == "__main__":
    main()
