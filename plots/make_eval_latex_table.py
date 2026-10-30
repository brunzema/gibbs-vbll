#!/usr/bin/env python3
"""Generate a Booktabs LaTeX table from results/eval/<sweep>/.../stats.pkl.

Choose the baselines in
BASELINES below (comment out a line to drop a column). Runs whose training
diverged (see DIVERGENCE_* below) are shown as DIVERGED_MARKER.

Usage:
    python plots/make_eval_latex_table.py            # SWEEP_NAME from plots/settings.py
    python plots/make_eval_latex_table.py <sweep_name>

The produced table has:
- Methods as columns
- Environments as grouped rows (multirow)
- Metrics as sub-rows

Best method per (environment, metric) row is highlighted in bold.
"""

from __future__ import annotations
import sys

import argparse
import datetime as dt
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np

from settings import SWEEP_NAME, figures_dir


# ---------------------------------------------------------------------------
# Baselines. Comment out a line to drop it; the order here is the column order.
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
# Caption and label base; " (Part i/n)" and "_part{i}" are appended per table.
# The label keeps the paper's existing \ref{tab:eval_new_v1_test_booktabs_part*}.
CAPTION = r"\textit{Results on the benchmarking of Sec.~\ref{sec:results-benchmark}.}"
LABEL = "tab:eval_new_v1_test_booktabs"
OUTPUT_TAG = ""  # appended to output file names, e.g. "_no_arnll"

# A run is diverged if the last DIVERGENCE_TAIL_FRACTION of its train/val loss
# history is non-finite or exceeds DIVERGENCE_THRESHOLD. Diverged cells are
# shown as DIVERGED_MARKER and never bolded.
RESULTS_ROOT = Path("results")
TRAIN_SEED = 42
DIVERGENCE_TAIL_FRACTION = 0.25
DIVERGENCE_THRESHOLD = 1e6
DIVERGED_MARKER = r"\textit{div.}"

LABEL_ALIASES = {raw: name for raw, name, _ in BASELINES}
BASELINE_COLORS = {name: color for _, name, color in BASELINES}
BASELINES_NAMES = {name: rf"\texttt{{{name}}}" for _, name, _ in BASELINES}
PREFERRED_METHOD_ORDER = [name for _, name, _ in BASELINES]

PREFERRED_ENV_ORDER = [
    "spiral",
    "swimmer",
    "variable_speed_limit_cycle",
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

DEFAULT_METRICS = OrderedDict(
    [
        ("final_l2", "Final L2"),
        ("cumul_l2", "Cumul. L2"),
        ("dtw", "DTW"),
        ("mmd", "MMD$^2$"),
        ("energy_dist", "Energy Dist."),
        ("terminal_mmd", "Term. MMD$^2$"),
        ("terminal_energy", "Term. Energy"),
        ("coverage_50", "Coverage@50%"),
        ("coverage_70", "Coverage@70%"),
        ("coverage_90", "Coverage@90%"),
        ("coverage_95", "Coverage@95%"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Booktabs LaTeX table from eval sweep stats."
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
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .tex file path "
            "(default: figures/<sweep_name>/table_booktabs.tex, see plots/settings.py)."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS.keys()),
        help=(
            "Metric keys to include, in order. "
            "Defaults to: " + ", ".join(DEFAULT_METRICS.keys())
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Method columns to include, in order. Accepts raw stats labels "
            "(e.g. m-NLL, m-CRPS) or display/canonical labels "
            "(e.g. m-NLL, m-CRPS). By default, all known methods are shown "
            "in PREFERRED_METHOD_ORDER."
        ),
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="Use *_mean or *_median values from stats records.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimals for numeric formatting.",
    )
    parser.add_argument(
        "--sci-high",
        type=float,
        default=1e4,
        help="Use scientific notation when abs(value) >= this threshold.",
    )
    parser.add_argument(
        "--sci-low",
        type=float,
        default=1e-3,
        help="Use scientific notation when 0 < abs(value) < this threshold.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="Table caption (optional).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Table label (optional).",
    )
    parser.add_argument(
        "--missing-marker",
        type=str,
        default="--",
        help="String for missing values.",
    )
    parser.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Disable bolding the best method per row.",
    )
    parser.add_argument(
        "--longtable",
        action="store_true",
        help="Generate a longtable version instead of a floating table*.",
    )
    parser.add_argument(
        "--max-envs-per-booktabs-table",
        type=int,
        default=5,
        help=(
            "Maximum number of environments per Booktabs table before splitting "
            "into multiple consecutive tables (default: 5). Ignored with --longtable."
        ),
    )
    return parser.parse_args()


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _canonical_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def prettify_env_name(env_name: str) -> str:
    if env_name.startswith("dysts_"):
        rest = env_name[len("dysts_") :]
        return rest.replace("_", " ").title()
    return env_name.replace("_", " ").title()


def env_sort_key(env_name: str) -> Tuple[int, str]:
    if env_name in PREFERRED_ENV_ORDER:
        return (PREFERRED_ENV_ORDER.index(env_name), env_name)
    return (len(PREFERRED_ENV_ORDER) + 1, env_name)


def method_sort_key(method_name: str) -> Tuple[int, str]:
    canonical = _canonical_label(method_name)
    if canonical in PREFERRED_METHOD_ORDER:
        return (PREFERRED_METHOD_ORDER.index(canonical), canonical)
    return (len(PREFERRED_METHOD_ORDER) + 1, canonical)


def order_methods(
    discovered_methods: List[str],
    requested_methods: Iterable[str] | None,
) -> List[str]:
    """Return raw discovered method labels in requested/canonical display order."""
    known_methods = [
        method
        for method in sorted(discovered_methods, key=method_sort_key)
        if method in LABEL_ALIASES
    ]
    if requested_methods is None:
        return known_methods

    by_raw = {method: method for method in known_methods}
    by_canonical: Dict[str, List[str]] = {}
    for method in known_methods:
        by_canonical.setdefault(_canonical_label(method), []).append(method)

    ordered: List[str] = []
    missing: List[str] = []
    for requested in requested_methods:
        if requested in by_raw:
            method = by_raw[requested]
        else:
            matches = by_canonical.get(_canonical_label(requested), [])
            method = matches[0] if matches else None

        if method is None:
            missing.append(requested)
            continue
        if method not in ordered:
            ordered.append(method)

    if missing:
        available = ", ".join(
            f"{method} ({_canonical_label(method)})" for method in known_methods
        )
        raise ValueError(
            "Requested methods not found: "
            + ", ".join(missing)
            + f". Available methods: {available}"
        )
    return ordered


def method_latex_label(method_name: str) -> str:
    canonical = _canonical_label(method_name)
    if canonical in BASELINES_NAMES:
        return BASELINES_NAMES[canonical]
    return rf"\texttt{{{latex_escape(canonical)}}}"


def metric_target(metric_key: str) -> float | None:
    if not metric_key.startswith("coverage_"):
        return None
    suffix = metric_key.rsplit("_", maxsplit=1)[-1]
    if suffix.isdigit():
        return int(suffix) / 100.0
    return None


def metric_direction(metric_key: str) -> str:
    # Errors/distances: lower is better.
    # Coverage metrics: target nominal coverage (e.g., 0.90).
    if metric_target(metric_key) is not None:
        return "target"
    return "down"


def metric_with_arrow(metric_key: str, metric_label: str) -> str:
    metric_label = metric_label.replace("%", r"\%")
    direction = metric_direction(metric_key)
    if direction == "down":
        return f"{metric_label} " + r"$\downarrow$"
    target = metric_target(metric_key)
    if target is not None:
        return f"{metric_label} " # + rf"($\approx {target:.2f}$)"
    return f"{metric_label}"


def find_latest_stats_for_env(env_dir: Path) -> Path | None:
    candidates = [p / "stats.pkl" for p in env_dir.iterdir() if p.is_dir()]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_stats_records(stats_path: Path) -> List[dict]:
    with open(stats_path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {stats_path}, got {type(data).__name__}")
    return data


def _loss_history_diverged(losses) -> bool:
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return False
    tail = arr[int((1.0 - DIVERGENCE_TAIL_FRACTION) * arr.size):]
    return bool(np.any(~np.isfinite(tail)) or np.any(np.abs(tail) > DIVERGENCE_THRESHOLD))


def find_diverged_runs(
    sweep_dir: Path,
    sweep_name: str,
    results_root: Path = RESULTS_ROOT,
    seed: int = TRAIN_SEED,
) -> Dict[str, set]:
    """Return {env: {eval label}} for runs whose training diverged.

    Looks up each evaluated model's training history at
    <results_root>/<sweep_name>/vbll_delta/<env>/<training_loss>/<seed>/vbll_model.pkl.
    """
    diverged: Dict[str, set] = {}
    for env_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        stats_path = find_latest_stats_for_env(env_dir)
        if stats_path is None:
            continue
        for rec in load_stats_records(stats_path):
            loss = rec.get("training_loss")
            model_path = results_root / sweep_name / "vbll_delta" / env_dir.name / str(loss) / str(seed) / "vbll_model.pkl"
            if not model_path.exists():
                continue
            with open(model_path, "rb") as handle:
                saved = pickle.load(handle)
            if _loss_history_diverged(saved.get("train_losses", [])) or _loss_history_diverged(
                saved.get("val_losses", [])
            ):
                diverged.setdefault(env_dir.name, set()).add(str(rec.get("label", loss)))
    return diverged


def extract_metric_stats(record: dict, metric: str, aggregate: str) -> Dict[str, float]:
    center = np.nan
    q05 = np.nan
    q95 = np.nan

    agg_key = f"{metric}_{aggregate}"
    q05_key = f"{metric}_q05"
    q95_key = f"{metric}_q95"

    if agg_key in record:
        center = float(record[agg_key])
    if q05_key in record:
        q05 = float(record[q05_key])
    if q95_key in record:
        q95 = float(record[q95_key])

    if metric in record:
        arr = np.asarray(record[metric], dtype=float)
        finite_arr = arr[np.isfinite(arr)]
        if finite_arr.size == 0:
            return {
                "center": center,
                "q05": q05,
                "q95": q95,
            }
        if not np.isfinite(center):
            if aggregate == "mean":
                center = float(np.mean(finite_arr))
            else:
                center = float(np.median(finite_arr))
        if not np.isfinite(q05):
            q05 = float(np.percentile(finite_arr, 5))
        if not np.isfinite(q95):
            q95 = float(np.percentile(finite_arr, 95))

    return {
        "center": center,
        "q05": q05,
        "q95": q95,
    }


def _needs_scientific(value: float, sci_low: float, sci_high: float) -> bool:
    if not np.isfinite(value):
        return False
    av = abs(value)
    return av > 0.0 and (av >= sci_high or av < sci_low)


def _format_scaled_pm(
    center: float,
    upper: float,
    lower: float,
    exp: int,
    precision: int,
) -> str:
    scale = 10.0 ** exp
    cs = center / scale
    us = upper / scale
    ls = lower / scale
    return (
        rf"$(\num{{{cs:.{precision}f}}}^{{+\num{{{us:.{precision}f}}}}}_{{-\num{{{ls:.{precision}f}}}}})"
        rf"\times10^{{{exp}}}$"
    )


def _format_scaled_scalar(value: float, exp: int, precision: int) -> str:
    scale = 10.0 ** exp
    vs = value / scale
    return rf"\num{{{vs:.{precision}f}e{exp}}}"


def fmt_value(
    stats: Dict[str, float],
    precision: int,
    missing_marker: str,
    sci_low: float,
    sci_high: float,
) -> str:
    center = float(stats.get("center", np.nan))
    q05 = float(stats.get("q05", np.nan))
    q95 = float(stats.get("q95", np.nan))

    if not np.isfinite(center):
        return missing_marker

    # Avoid displaying negative zero from floating point noise.
    eps = 0.5 * (10 ** (-precision))
    if abs(center) < eps:
        center = 0.0
    center_txt = f"{center:.{precision}f}"

    if np.isfinite(q05) and np.isfinite(q95):
        lower = abs(center - q05)
        upper = abs(q95 - center)
        if abs(lower) < eps:
            lower = 0.0
        if abs(upper) < eps:
            upper = 0.0
        scale_ref = max(abs(center), abs(upper), abs(lower))
        if _needs_scientific(scale_ref, sci_low=sci_low, sci_high=sci_high):
            exp = int(np.floor(np.log10(scale_ref)))
            return _format_scaled_pm(
                center=center,
                upper=upper,
                lower=lower,
                exp=exp,
                precision=precision,
            )
        lower_txt = f"{lower:.{precision}f}"
        upper_txt = f"{upper:.{precision}f}"
        return (
            rf"$\num{{{center_txt}}}^{{+\num{{{upper_txt}}}}}_{{-\num{{{lower_txt}}}}}$"
        )
    if _needs_scientific(center, sci_low=sci_low, sci_high=sci_high):
        exp = int(np.floor(np.log10(abs(center))))
        return _format_scaled_scalar(value=center, exp=exp, precision=precision)
    return rf"\num{{{center_txt}}}"


def bold_latex_cell(cell: str) -> str:
    """Bold a LaTeX cell, handling math-mode values correctly."""
    stripped = cell.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        # Force bold numerals for best-cells by replacing \num{...} with \mathbf{...}
        # in math mode; this is robust across siunitx versions/configs.
        bold_math_cell = re.sub(r"\\num\{([^{}]+)\}", r"\\mathbf{\1}", cell)
        return r"{\boldmath " + bold_math_cell + "}"
    return rf"\textbf{{{cell}}}"


def wrap_words(text: str, max_chars_per_line: int = 14) -> List[str]:
    words = text.split()
    if not words:
        return [text]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= max_chars_per_line:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def format_rotated_env_cell(env_name: str) -> str:
    pretty = prettify_env_name(env_name)
    wrapped = wrap_words(pretty, max_chars_per_line=14)
    escaped_lines = [latex_escape(x) for x in wrapped]
    stacked = r"\\ ".join(escaped_lines)
    return rf"\rotatebox[origin=c]{{90}}{{\shortstack[c]{{\strut {stacked} \strut}}}}"


def metric_group(metric_key: str) -> str:
    if metric_key.startswith("coverage_"):
        return "coverage"
    if metric_key in {"final_l2", "cumul_l2", "dtw"}:
        return "trajectory"
    return "distributional"


def count_metric_group_breaks(metrics: List[str]) -> int:
    count = 0
    for i in range(len(metrics) - 1):
        if metric_group(metrics[i]) != metric_group(metrics[i + 1]):
            count += 1
    return count


def collect_sweep_data(
    sweep_dir: Path,
    metrics: Iterable[str],
    aggregate: str,
) -> Tuple[
    List[str],
    List[str],
    Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    Dict[str, Path],
]:
    env_dirs = sorted([p for p in sweep_dir.iterdir() if p.is_dir()], key=lambda p: env_sort_key(p.name))
    env_names: List[str] = []
    methods_seen: List[str] = []
    values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    stats_sources: Dict[str, Path] = {}

    for env_dir in env_dirs:
        latest_stats = find_latest_stats_for_env(env_dir)
        if latest_stats is None:
            continue

        records = load_stats_records(latest_stats)
        env_name = env_dir.name
        env_names.append(env_name)
        stats_sources[env_name] = latest_stats
        values[env_name] = {}

        for rec in records:
            method = str(rec.get("label", rec.get("training_loss", "unknown")))
            if method not in methods_seen:
                methods_seen.append(method)
            values[env_name][method] = {
                metric: extract_metric_stats(rec, metric, aggregate) for metric in metrics
            }

    methods = sorted(methods_seen, key=method_sort_key)
    return env_names, methods, values, stats_sources


def build_latex_table(
    sweep_name: str,
    env_names: List[str],
    methods: List[str],
    metrics: List[str],
    values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    precision: int,
    missing_marker: str,
    sci_low: float,
    sci_high: float,
    bold_best: bool,
    caption: str | None,
    label: str | None,
    use_longtable: bool,
    max_envs_per_booktabs_table: int,
    diverged: Dict[str, set] | None = None,
) -> str:
    diverged = diverged or {}
    if (
        not use_longtable
        and max_envs_per_booktabs_table > 0
        and len(env_names) > max_envs_per_booktabs_table
    ):
        chunks = [
            env_names[i : i + max_envs_per_booktabs_table]
            for i in range(0, len(env_names), max_envs_per_booktabs_table)
        ]
        total_parts = len(chunks)
        split_tables: List[str] = []
        for part_idx, env_chunk in enumerate(chunks, start=1):
            part_caption = caption
            part_label = label
            if part_caption is not None:
                part_caption = f"{part_caption} (Part {part_idx}/{total_parts})"
            if part_label is not None:
                part_label = f"{part_label}_part{part_idx}"
            split_tables.append(
                build_latex_table(
                    sweep_name=sweep_name,
                    env_names=env_chunk,
                    methods=methods,
                    metrics=metrics,
                    values=values,
                    precision=precision,
                    missing_marker=missing_marker,
                    sci_low=sci_low,
                    sci_high=sci_high,
                    bold_best=bold_best,
                    caption=part_caption,
                    label=part_label,
                    use_longtable=False,
                    max_envs_per_booktabs_table=max_envs_per_booktabs_table,
                    diverged=diverged,
                )
            )
        return "\n".join(split_tables)

    metric_labels = [DEFAULT_METRICS.get(m, m.replace("_", " ")) for m in metrics]
    ncols = 2 + len(methods)
    header_methods = " & ".join(method_latex_label(m) for m in methods)
    header_line = rf"Environment & Metric & {header_methods} \\"

    if use_longtable:
        # Build a fixed-width multipage table that fits textwidth exactly.
        # This avoids overflow while keeping page breaks.
        env_w = 0.07
        metric_w = 0.19
        method_total_w = 1.0 - env_w - metric_w
        if len(methods) == 0:
            raise ValueError("Expected at least one method column.")
        method_w = method_total_w / len(methods)

        def _w(frac: float) -> str:
            return f"{frac:.6f}\\textwidth"

        method_cols = "".join(
            rf">{{\centering\arraybackslash}}p{{{_w(method_w)}}}" for _ in methods
        )
        colspec = (
            "@{}"
            + rf">{{\centering\arraybackslash}}p{{{_w(env_w)}}}"
            + rf">{{\raggedright\arraybackslash}}p{{{_w(metric_w)}}}"
            + method_cols
            + "@{}"
        )
    else:
        colspec = "cl" + ("c" * len(methods))

    lines: List[str] = []
    if use_longtable:
        lines.append(
            r"\sisetup{group-separator={,},group-minimum-digits=4,detect-weight=true,detect-inline-weight=math}"
        )
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{0pt}")
        lines.append(rf"\begin{{longtable}}{{{colspec}}}")
        lines.append(rf"\caption{{{caption}}}\label{{{label}}}\\")
        lines.append(r"\toprule")
        lines.append(header_line)
        lines.append(r"\midrule")
        lines.append(r"\endfirsthead")
        lines.append(r"\toprule")
        lines.append(header_line)
        lines.append(r"\midrule")
        lines.append(r"\endhead")
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{{ncols}}}{{r}}{{\emph{{Continued on next page}}}} \\"
        )
        lines.append(r"\midrule")
        lines.append(r"\endfoot")
        lines.append(r"\bottomrule")
        lines.append(r"\endlastfoot")
    else:
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{label}}}")
        lines.append(
            r"\sisetup{group-separator={,},group-minimum-digits=4,detect-weight=true,detect-inline-weight=math}"
        )
        lines.append(r"\footnotesize")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(rf"\begin{{tabular}}{{{colspec}}}")
        lines.append(r"\toprule")
        lines.append(header_line)
        lines.append(r"\midrule")

    env_row_span = len(metrics) + count_metric_group_breaks(metrics)

    for env_idx, env_name in enumerate(env_names):
        rotated_env = format_rotated_env_cell(env_name)

        for metric_idx, (metric_key, metric_label) in enumerate(zip(metrics, metric_labels)):
            row_stats = []
            row_diverged = []
            for method in methods:
                stats = values.get(env_name, {}).get(method, {}).get(
                    metric_key,
                    {"center": np.nan, "q05": np.nan, "q95": np.nan},
                )
                row_stats.append(stats)
                row_diverged.append(method in diverged.get(env_name, set()))

            row_centers = [
                np.nan if div else float(s.get("center", np.nan))
                for s, div in zip(row_stats, row_diverged)
            ]
            finite_vals = [v for v in row_centers if np.isfinite(v)]
            best_mask = [False] * len(row_centers)
            if finite_vals:
                direction = metric_direction(metric_key)
                if direction == "target":
                    target = metric_target(metric_key)
                    assert target is not None
                    best_score = min(abs(v - target) for v in finite_vals)
                    best_mask = [
                        bool(
                            np.isfinite(v)
                            and np.isclose(abs(v - target), best_score, rtol=1e-12, atol=1e-12)
                        )
                        for v in row_centers
                    ]
                else:
                    best_val = min(finite_vals)
                    best_mask = [
                        bool(np.isfinite(v) and np.isclose(v, best_val, rtol=1e-12, atol=1e-12))
                        for v in row_centers
                    ]

            value_cells = []
            for i, stats in enumerate(row_stats):
                if row_diverged[i]:
                    value_cells.append(DIVERGED_MARKER)
                    continue
                cell = fmt_value(
                    stats,
                    precision=precision,
                    missing_marker=missing_marker,
                    sci_low=sci_low,
                    sci_high=sci_high,
                )
                if bold_best and best_mask[i] and cell != missing_marker:
                    cell = bold_latex_cell(cell)
                value_cells.append(cell)

            metric_cell = metric_with_arrow(metric_key, metric_label)
            row_values_text = " & ".join(value_cells)

            if metric_idx == 0:
                lines.append(
                    rf"\multirow[c]{{{env_row_span}}}{{*}}{{{rotated_env}}} & "
                    rf"{metric_cell} & {row_values_text} \\"
                )
            else:
                lines.append(rf" & {metric_cell} & {row_values_text} \\")

            if metric_idx < len(metrics) - 1:
                this_group = metric_group(metric_key)
                next_group = metric_group(metrics[metric_idx + 1])
                if this_group != next_group:
                    lines.append(rf"\cmidrule(lr){{2-{ncols}}}")

        if env_idx != len(env_names) - 1:
            lines.append(r"\midrule")

    if use_longtable:
        lines.append(r"\end{longtable}")
    else:
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"}")
        lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    sweep_dir = args.eval_root / args.sweep_name
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    output_path = args.output
    if output_path is None:
        suffix = "longtable" if args.longtable else "booktabs"
        output_path = figures_dir(args.sweep_name) / f"table_{suffix}{OUTPUT_TAG}.tex"

    metrics = list(args.metrics)
    env_names, methods, values, stats_sources = collect_sweep_data(
        sweep_dir=sweep_dir,
        metrics=metrics,
        aggregate=args.aggregate,
    )

    if not env_names:
        raise RuntimeError(f"No stats.pkl files found under: {sweep_dir}")
    if not methods:
        raise RuntimeError(f"No method records found under: {sweep_dir}")

    # Keep raw method labels for value lookup, but order/filter by canonical display labels.
    methods = order_methods(methods, args.methods)
    diverged = find_diverged_runs(sweep_dir, args.sweep_name)

    if args.caption is None and CAPTION:
        caption = CAPTION
    elif args.caption is None:
        caption = (
            f"Evaluation results for sweep \\texttt{{{latex_escape(args.sweep_name)}}}. "
            "Best value per (environment, metric) row is bold."
        )
    else:
        caption = args.caption
    if args.label is None and LABEL:
        label = LABEL
    elif args.label is None:
        label = (
            f"tab:eval_{args.sweep_name}_longtable"
            if args.longtable
            else f"tab:eval_{args.sweep_name}_booktabs"
        )
    else:
        label = args.label

    table_text = build_latex_table(
        sweep_name=args.sweep_name,
        env_names=env_names,
        methods=methods,
        metrics=metrics,
        values=values,
        precision=args.precision,
        missing_marker=args.missing_marker,
        sci_low=args.sci_low,
        sci_high=args.sci_high,
        bold_best=(not args.no_bold_best),
        caption=caption,
        label=label,
        use_longtable=args.longtable,
        max_envs_per_booktabs_table=args.max_envs_per_booktabs_table,
        diverged=diverged,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_lines = "\n".join(
        f"%   {env}: {stats_sources[env]}" for env in env_names if env in stats_sources
    )
    header = (
        "% Auto-generated by plots/make_eval_latex_table.py\n"
        "% Requires: \\usepackage{booktabs}, \\usepackage{multirow}, \\usepackage{graphicx}, \\usepackage{siunitx}"
        + (", \\usepackage{longtable}\n" if args.longtable else "\n")
        + ("% (Longtable mode also uses \\usepackage{array})\n" if args.longtable else "")
        + f"% Generated at: {generated_at}\n"
        "% Source stats.pkl files:\n"
        f"{source_lines}\n"
        + "% Diverged runs (shown as div.): "
        + ("; ".join(f"{env}: {', '.join(sorted(m))}" for env, m in sorted(diverged.items())) or "none")
        + "\n\n"
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write(table_text)

    print(f"Wrote table to: {output_path}")
    print(f"Environments: {len(env_names)}")
    print(f"Methods: {len(methods)} -> {methods}")
    print(f"Metrics: {len(metrics)} -> {metrics}")
    print(f"Diverged runs: {sum(len(m) for m in diverged.values())} -> {diverged}")


if __name__ == "__main__":
    main()
