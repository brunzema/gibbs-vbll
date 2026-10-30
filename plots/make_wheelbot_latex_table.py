#!/usr/bin/env python3
"""Generate the Booktabs LaTeX results table for the Wheelbot experiment.

Reads per-run outputs from
    results/wheelbot_dynamics/<sweep>/wheelbot_racing/<run folder>/<seed>/
Choose the baselines in BASELINES below (comment out a line to drop a row).

Each cell is median^q75_q25. Bold marks the significantly best method per
column (one-sided paired Wilcoxon signed-rank test against every other row).
Cells with |value| > DIVERGED_ABS_THRESHOLD are shown as DIVERGED_MARKER.

Sources per run:
- one-step: one_step_metrics_sweep_eval.json (per-sample arrays, written by the
  runs/eval_wheelbot.py); falls back to the scalar test means in
  one_step_test_metrics.json (no quartiles, no bolding).
- multi-step: multistep_xy_metrics.json, "horizon_metrics"/<HORIZON>.
- training time: training_logs.json, training/train_time_sec.

Usage:
    python plots/make_wheelbot_latex_table.py            # WB_SWEEP_NAME from plots/settings.py
    python plots/make_wheelbot_latex_table.py <sweep_name>
"""

from __future__ import annotations
import sys

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
from scipy.stats import wilcoxon

from settings import WB_SWEEP_NAME, figures_dir


# ---------------------------------------------------------------------------
# Baselines. Comment out a line to drop it; the order here is the row order.
# A \cmidrule separates consecutive rows of different groups.
#   (run folder = <loss config name>_<H>, display name, group, citation)
# ---------------------------------------------------------------------------
BASELINES = [
    # ("single_step_elbo_1",                 "s-NLL (no HB)", "single", ""),
    ("single_step_elbo_hbatch_1",           "s-NLL",        "single", r"~\citep{harrison2024variational,brunzema2026vision}"),
    ("multi_step_elbo_40",                  "m-NLL",        "multi",  ""),
    ("multi_step_factored_crps_40",         "m-CRPS",       "multi",  ""),
    ("multi_step_kde_40",                   "m-KDE",        "ar",     r"~\citep{ivanovic2023expanding}"),
    ("multi_step_ar_elbo_40",               "m-arNLL",      "ar",     ""),
    ("multi_step_ar_crps_40",               "m-arCRPS",     "ar",     ""),
]

DEFAULT_SWEEP = WB_SWEEP_NAME
RESULTS_ROOT = Path("results/wheelbot_dynamics")
ENV_NAME = "wheelbot_racing"
TRAIN_SEED = 42
HORIZON = 100  # multi-step metrics are reported at this rollout horizon
OUTPUT_TAG = ""  # appended to the output file name, e.g. "_no_arnll"
# The label keeps the paper's existing \ref{tab:wheelbot_training_wheelbot_all_losses_v3_booktabs}.
LABEL = "tab:wheelbot_training_wheelbot_all_losses_v3_booktabs"
CAPTION = (
    r"\textit{Quantitative results on Wheelbot data.} "
    rf"Multi-step metrics are reported at {HORIZON} steps." "\n"
    r"Each cell reports $\mathrm{median}^{\mathrm{q75}}_{\mathrm{q25}}$." "\n"
    r"Bold values are significantly best (paired Wilcoxon signed-rank test, $\alpha=0.05$)."
)
SIGNIFICANCE_ALPHA = 0.05
DIVERGED_ABS_THRESHOLD = 1000.0
DIVERGED_MARKER = r"\textit{diverged}"
MISSING_MARKER = "--"
PRECISION = 3
SCI_LOW, SCI_HIGH = 1e-3, 1e4  # scientific notation outside [SCI_LOW, SCI_HIGH)

# (key in our stats dict, source key, header)
ONE_STEP_METRICS = [
    ("mse", "mse_norm", r"MSE $\downarrow$"),
    ("nll", "nll_norm", r"NLL $\downarrow$"),
    ("crps", "crps_norm", r"CRPS $\downarrow$"),
]
MULTI_STEP_METRICS = [
    ("final_l2", "final_l2", r"Final L2 $\downarrow$"),
    ("cumul_l2", "cum_l2", r"Cumul. L2 $\downarrow$"),
    ("dtw", "dtw", r"DTW $\downarrow$"),
    ("energy_dist", "energy_dist", r"Energy dist. $\downarrow$"),
]
METRIC_KEYS = [k for k, _, _ in ONE_STEP_METRICS + MULTI_STEP_METRICS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the Booktabs LaTeX table of the Wheelbot results."
    )
    parser.add_argument(
        "sweep_name",
        type=str,
        nargs="?",
        default=DEFAULT_SWEEP,
        help=f"Sweep name under {RESULTS_ROOT}/ (default: {DEFAULT_SWEEP})",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Root of the Wheelbot sweeps (default: {RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file path (default: figures/<SWEEP_NAME>/table_wheelbot.tex, see plots/settings.py).",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _stats(payload: Any) -> Dict[str, Any]:
    """{median, q25, q75, values} from a stats dict or a scalar (median only)."""
    if not isinstance(payload, dict):
        return {"median": _finite_or_none(payload), "q25": None, "q75": None, "values": None}
    values = None
    for key in ("values_by_chunk", "values_by_sample", "values"):
        if isinstance(payload.get(key), list):
            values = np.array([np.nan if v is None else v for v in payload[key]], dtype=float)
            break
    return {
        "median": _finite_or_none(payload.get("median", payload.get("mean"))),
        "q25": _finite_or_none(payload.get("q25")),
        "q75": _finite_or_none(payload.get("q75")),
        "values": values,
    }


def load_run(run_dir: Path) -> Dict[str, Any]:
    logs = _load_json(run_dir / "training_logs.json")
    one_step_eval = _load_json(run_dir / "one_step_metrics_sweep_eval.json").get("metrics")
    one_step_eval = one_step_eval if isinstance(one_step_eval, dict) else {}
    one_step_scalar = _load_json(run_dir / "one_step_test_metrics.json")
    horizon_metrics = _load_json(run_dir / "multistep_xy_metrics.json").get("horizon_metrics", {})
    multi_step = horizon_metrics.get(str(HORIZON), {}) if isinstance(horizon_metrics, dict) else {}

    stats: Dict[str, Dict[str, Any]] = {}
    for key, src, _ in ONE_STEP_METRICS:
        stats[key] = _stats(one_step_eval[src] if src in one_step_eval else one_step_scalar.get(src))
    for key, src, _ in MULTI_STEP_METRICS:
        stats[key] = _stats(multi_step.get(src))

    sources = []
    if not one_step_eval:
        sources.append("one-step: scalar test means only (no one_step_metrics_sweep_eval.json)")
    if not multi_step:
        sources.append(f"multi-step: no multistep_xy_metrics.json with horizon {HORIZON}")
    return {
        "stats": stats,
        "train_time_sec": _finite_or_none(logs.get("training", {}).get("train_time_sec")),
        "warnings": sources,
    }


def _num(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if abs(value) >= SCI_HIGH or 0 < abs(value) < SCI_LOW:
        return f"{value:.{PRECISION}e}"
    return f"{value:.{PRECISION}f}"


def _is_diverged(stats: Dict[str, Any]) -> bool:
    return any(
        stats[k] is not None and abs(stats[k]) > DIVERGED_ABS_THRESHOLD
        for k in ("median", "q25", "q75")
    )


def format_cell(stats: Dict[str, Any], bold: bool) -> str:
    if _is_diverged(stats):
        return DIVERGED_MARKER
    median = _num(stats["median"])
    if median is None:
        return MISSING_MARKER
    q25, q75 = _num(stats["q25"]), _num(stats["q75"])
    if q25 is None or q75 is None:
        cell = rf"$\num{{{median}}}$"
    else:
        cell = rf"$\num{{{median}}}^{{\num{{{q75}}}}}_{{\num{{{q25}}}}}$"
    if bold:
        cell = r"{\boldmath " + re.sub(r"\\num\{([^{}]+)\}", r"\\mathbf{\1}", cell) + "}"
    return cell


def _significantly_less(cand: np.ndarray, other: np.ndarray) -> bool:
    """One-sided paired Wilcoxon: is `cand` significantly lower than `other`?"""
    if cand.shape != other.shape:
        return False
    valid = np.isfinite(cand) & np.isfinite(other)
    diffs = other[valid] - cand[valid]
    if not np.any(diffs != 0.0):
        return False
    try:
        _, pval = wilcoxon(diffs, alternative="greater", zero_method="wilcox", correction=False)
    except ValueError:
        return False
    return bool(np.isfinite(pval) and pval < SIGNIFICANCE_ALPHA)


def significantly_best(column: List[Dict[str, Any]]) -> List[bool]:
    """Mark the row that beats every other row (all must have per-sample values)."""
    mask = [False] * len(column)
    if len(column) < 2 or any(s["values"] is None for s in column):
        return mask
    for i, cand in enumerate(column):
        if _is_diverged(cand):
            continue
        mask[i] = all(
            _significantly_less(cand["values"], other["values"])
            for j, other in enumerate(column)
            if j != i
        )
    return mask


def build_latex_table(rows: List[Dict[str, Any]]) -> str:
    ncols = 1 + len(METRIC_KEYS) + 1
    bold = {k: significantly_best([r["stats"][k] for r in rows]) for k in METRIC_KEYS}
    n_one, n_multi = len(ONE_STEP_METRICS), len(MULTI_STEP_METRICS)
    headers = [h for _, _, h in ONE_STEP_METRICS + MULTI_STEP_METRICS]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{CAPTION}}}",
        rf"\label{{{LABEL}}}",
        r"\sisetup{group-separator={,},group-minimum-digits=4,detect-weight=true,detect-inline-weight=math}",
        r"\footnotesize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * (ncols - 1)}}}",
        r"\toprule",
        rf" & \multicolumn{{{n_one}}}{{c}}{{One-step metrics}} & "
        rf"\multicolumn{{{n_multi}}}{{c}}{{Multi-step metrics}} & "
        r"\multicolumn{1}{c}{Training\footnotemark} \\",
        rf"\cmidrule(lr){{2-{1 + n_one}}}\cmidrule(lr){{{2 + n_one}-{1 + n_one + n_multi}}}"
        rf"\cmidrule(lr){{{ncols}-{ncols}}}",
        r"\textbf{Method} & " + " & ".join(headers) + r" & Time [min] \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i > 0 and row["group"] != rows[i - 1]["group"]:
            lines.append(rf"\cmidrule(lr){{1-{ncols}}}")
        cells = [rf"\texttt{{{row['name']}}}{row['cite']}"]
        cells += [format_cell(row["stats"][k], bold[k][i]) for k in METRIC_KEYS]
        minutes = _num(None if row["train_time_sec"] is None else row["train_time_sec"] / 60.0)
        cells.append(MISSING_MARKER if minutes is None else rf"$\num{{{minutes}}}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\vspace{-1em}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    env_dir = args.results_root / args.sweep_name / ENV_NAME
    if not env_dir.exists():
        raise FileNotFoundError(f"Sweep env directory not found: {env_dir}")

    rows = []
    for folder, name, group, cite in BASELINES:
        run_dir = env_dir / folder / str(TRAIN_SEED)
        if not (run_dir / "training_logs.json").exists():
            print(f"WARNING: skipping {name}: no training_logs.json in {run_dir}")
            continue
        run = load_run(run_dir)
        for warning in run["warnings"]:
            print(f"WARNING: {name}: {warning}")
        rows.append({"name": name, "group": group, "cite": cite, "run_dir": run_dir, **run})
    if not rows:
        raise RuntimeError(f"No runs found under: {env_dir}")

    output_path = args.output or figures_dir() / f"table_wheelbot{OUTPUT_TAG}.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "% Auto-generated by plots/make_wheelbot_latex_table.py\n"
        "% Requires: \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{siunitx}\n"
        f"% Generated at: {dt.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "% Source run directories:\n"
        + "".join(f"%   {r['name']}: {r['run_dir']}\n" for r in rows)
        + "\n"
    )
    output_path.write_text(header + build_latex_table(rows), encoding="utf-8")

    print(f"Wrote table to: {output_path}")
    print(f"Sweep: {args.sweep_name}")
    print(f"Methods ({len(rows)}): {[r['name'] for r in rows]}")


if __name__ == "__main__":
    main()
