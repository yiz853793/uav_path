#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/experiment/data_plotting.py

Draw paper-ready comparison figures from reproducibility result CSV files.

Input expected:
  <in_root>/**/reproducibility_cases.csv

Typical files are produced by:
  experiments/run_suite.py
  experiments/run_ablation_suite.py
  experiments/run_reproducibility.py

This script intentionally reads reproducibility_cases.csv directly, because it
contains RRT*, PRM, MOEA/D representative solutions, runtime, MTOE generation,
archive size and stop reason in one place.

Examples:
  # Draw all available figures under outputs/_figures
  python -m experiments.plot_results --in_root outputs --out_dir outputs/_figures

  # Draw only method comparison figures
  python -m experiments.plot_results --in_root outputs/suite_hillcity024_s0to19_r5/full --kind method

  # Draw only ablation figures
  python -m experiments.plot_results --in_root outputs/ablation_hillcity024_s0to19_r5 --kind ablation
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Safe for servers/headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Basic parsing helpers
# -----------------------------

def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def to_float(value, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return default
    if s.lower() in {"nan", "none", "null"}:
        return default
    if s.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if s.lower() in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(s)
    except Exception:
        return default


def to_int(value, default: int = 0) -> int:
    x = to_float(value, math.nan)
    if math.isfinite(x):
        return int(x)
    return default


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(x) for x in values if _is_finite(x)]


def mean(values: Iterable[float]) -> float:
    xs = finite_values(values)
    return float(np.mean(xs)) if xs else math.nan


def std(values: Iterable[float]) -> float:
    xs = finite_values(values)
    return float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0 if len(xs) == 1 else math.nan


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "unknown"


def infer_context(csv_path: Path, in_root: Path) -> Tuple[str, str, str]:
    """
    Infer (experiment, variant, terrain_tag) from a result path.

    Expected:
      <in_root>/<experiment>/<variant>/<terrain_tag>/reproducibility_cases.csv
    but also works when in_root is deeper.
    """
    terrain_tag = csv_path.parent.name
    variant = csv_path.parent.parent.name if csv_path.parent.parent != csv_path.parent else "default"
    experiment = csv_path.parent.parent.parent.name if csv_path.parent.parent.parent != csv_path.parent.parent else "experiment"

    # If user passed a deeper root, avoid useless names like ".".
    try:
        rel = csv_path.relative_to(in_root)
        parts = rel.parts
        if len(parts) >= 4:
            experiment, variant, terrain_tag = parts[0], parts[1], parts[2]
        elif len(parts) >= 3:
            variant, terrain_tag = parts[0], parts[1]
            experiment = in_root.name
        elif len(parts) >= 2:
            terrain_tag = parts[0]
            variant = in_root.name
            experiment = in_root.parent.name
    except Exception:
        pass
    return experiment, variant, terrain_tag


@dataclass
class CaseRow:
    experiment: str
    variant: str
    terrain_tag: str
    csv_path: str
    terrain_seed: int
    planner_seed: int
    invalid_case: int
    profile: str

    rrt_found: int
    rrt_ms: float
    rrt_f1: float
    rrt_f2: float
    rrt_f3: float

    prm_found: int
    prm_ms: float
    prm_f1: float
    prm_f2: float
    prm_f3: float

    moead_only: int
    moead_ms: float
    moead_archive_size: int
    moead_stop_reason: str
    moead_n_gen: float

    min_f1_f1: float
    min_f1_f2: float
    min_f1_f3: float

    min_f2_f1: float
    min_f2_f2: float
    min_f2_f3: float

    min_f3_f1: float
    min_f3_f2: float
    min_f3_f3: float

    compromise_f1: float
    compromise_f2: float
    compromise_f3: float


@dataclass
class MethodRecord:
    experiment: str
    variant: str
    terrain_tag: str
    terrain_seed: int
    planner_seed: int
    method: str
    found: int
    runtime_s: float
    f1: float
    f2: float
    f3: float


def read_cases_csv(csv_path: Path, in_root: Path) -> List[CaseRow]:
    experiment, variant, terrain_tag = infer_context(csv_path, in_root)
    rows: List[CaseRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(CaseRow(
                experiment=experiment,
                variant=variant,
                terrain_tag=r.get("size_tag") or terrain_tag,
                csv_path=str(csv_path),
                terrain_seed=to_int(r.get("terrain_seed"), -1),
                planner_seed=to_int(r.get("planner_seed"), -1),
                invalid_case=to_int(r.get("invalid_case"), 0),
                profile=r.get("profile", ""),

                rrt_found=to_int(r.get("rrt_found"), 0),
                rrt_ms=to_float(r.get("rrt_ms")),
                rrt_f1=to_float(r.get("rrt_f1")),
                rrt_f2=to_float(r.get("rrt_f2")),
                rrt_f3=to_float(r.get("rrt_f3")),

                prm_found=to_int(r.get("prm_found"), 0),
                prm_ms=to_float(r.get("prm_ms")),
                prm_f1=to_float(r.get("prm_f1")),
                prm_f2=to_float(r.get("prm_f2")),
                prm_f3=to_float(r.get("prm_f3")),

                moead_only=to_int(r.get("moead_only"), 0),
                moead_ms=to_float(r.get("moead_ms")),
                moead_archive_size=to_int(r.get("moead_archive_size"), 0),
                moead_stop_reason=(r.get("moead_stop_reason") or "").strip() or "unknown",
                moead_n_gen=to_float(r.get("moead_n_gen")),

                min_f1_f1=to_float(r.get("min_f1_f1")),
                min_f1_f2=to_float(r.get("min_f1_f2")),
                min_f1_f3=to_float(r.get("min_f1_f3")),

                min_f2_f1=to_float(r.get("min_f2_f1")),
                min_f2_f2=to_float(r.get("min_f2_f2")),
                min_f2_f3=to_float(r.get("min_f2_f3")),

                min_f3_f1=to_float(r.get("min_f3_f1")),
                min_f3_f2=to_float(r.get("min_f3_f2")),
                min_f3_f3=to_float(r.get("min_f3_f3")),

                compromise_f1=to_float(r.get("compromise_f1")),
                compromise_f2=to_float(r.get("compromise_f2")),
                compromise_f3=to_float(r.get("compromise_f3")),
            ))
    return rows


def discover_cases(in_root: Path) -> List[CaseRow]:
    csvs = sorted(in_root.rglob("reproducibility_cases.csv"))
    cases: List[CaseRow] = []
    for p in csvs:
        try:
            cases.extend(read_cases_csv(p, in_root))
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
    return cases


def to_method_records(cases: Sequence[CaseRow], exclude_invalid: bool = True) -> List[MethodRecord]:
    out: List[MethodRecord] = []
    for r in cases:
        if exclude_invalid and r.invalid_case:
            continue

        # Sampling baselines.
        if r.rrt_found and _is_finite(r.rrt_f1):
            out.append(MethodRecord(r.experiment, r.variant, r.terrain_tag, r.terrain_seed, r.planner_seed,
                                    "RRT*", 1, r.rrt_ms / 1000.0, r.rrt_f1, r.rrt_f2, r.rrt_f3))
        if r.prm_found and _is_finite(r.prm_f1):
            out.append(MethodRecord(r.experiment, r.variant, r.terrain_tag, r.terrain_seed, r.planner_seed,
                                    "PRM", 1, r.prm_ms / 1000.0, r.prm_f1, r.prm_f2, r.prm_f3))

        # MOEA/D representatives.
        reps = [
            ("MOEA/D-min_f1", r.min_f1_f1, r.min_f1_f2, r.min_f1_f3),
            ("MOEA/D-min_f2", r.min_f2_f1, r.min_f2_f2, r.min_f2_f3),
            ("MOEA/D-min_f3", r.min_f3_f1, r.min_f3_f2, r.min_f3_f3),
            ("MOEA/D-compromise", r.compromise_f1, r.compromise_f2, r.compromise_f3),
        ]
        for name, f1, f2, f3 in reps:
            found = int(r.moead_archive_size > 0 and _is_finite(f1) and _is_finite(f2) and _is_finite(f3))
            if found:
                out.append(MethodRecord(r.experiment, r.variant, r.terrain_tag, r.terrain_seed, r.planner_seed,
                                        name, 1, r.moead_ms / 1000.0, f1, f2, f3))
    return out


# -----------------------------
# Plot helpers
# -----------------------------

def save_boxplot(
    groups: Dict[str, List[float]],
    title: str,
    ylabel: str,
    out_file: Path,
    dpi: int,
    rotate_labels: bool = True,
) -> Optional[Path]:
    labels = [k for k, v in groups.items() if finite_values(v)]
    data = [finite_values(groups[k]) for k in labels]
    if len(labels) < 1:
        return None

    ensure_dir(out_file.parent)
    fig_width = max(7.0, 0.7 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if rotate_labels:
        ax.tick_params(axis="x", labelrotation=28)
    fig.subplots_adjust(bottom=0.25, left=0.12, right=0.98, top=0.88)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file


def save_barplot(
    values: Dict[str, float],
    title: str,
    ylabel: str,
    out_file: Path,
    dpi: int,
    ylim: Optional[Tuple[float, float]] = None,
    rotate_labels: bool = True,
) -> Optional[Path]:
    labels = list(values.keys())
    ys = [values[k] for k in labels]
    if not labels:
        return None

    ensure_dir(out_file.parent)
    fig_width = max(7.0, 0.7 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.bar(labels, ys)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if rotate_labels:
        ax.tick_params(axis="x", labelrotation=28)
    fig.subplots_adjust(bottom=0.25, left=0.12, right=0.98, top=0.88)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file


def save_stacked_stop_reason(counts_by_variant: Dict[str, Counter], title: str, out_file: Path, dpi: int) -> Optional[Path]:
    variants = list(counts_by_variant.keys())
    reasons = sorted({r for c in counts_by_variant.values() for r in c.keys()})
    if not variants or not reasons:
        return None

    ensure_dir(out_file.parent)
    fig_width = max(7.0, 0.7 * len(variants) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    bottom = np.zeros(len(variants), dtype=float)
    x = np.arange(len(variants))
    for reason in reasons:
        vals = np.array([counts_by_variant[v].get(reason, 0) for v in variants], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=reason)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=28, ha="right")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.subplots_adjust(bottom=0.25, left=0.12, right=0.98, top=0.88)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    return out_file


def write_summary_csv(cases: Sequence[CaseRow], records: Sequence[MethodRecord], out_file: Path) -> Path:
    ensure_dir(out_file.parent)
    groups: Dict[Tuple[str, str, str, str], List[MethodRecord]] = defaultdict(list)
    for r in records:
        groups[(r.experiment, r.variant, r.terrain_tag, r.method)].append(r)

    with out_file.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment", "variant", "terrain_tag", "method",
            "n", "runtime_s_mean", "runtime_s_std",
            "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(groups.keys()):
            rs = groups[key]
            w.writerow({
                "experiment": key[0],
                "variant": key[1],
                "terrain_tag": key[2],
                "method": key[3],
                "n": len(rs),
                "runtime_s_mean": f"{mean(x.runtime_s for x in rs):.6g}",
                "runtime_s_std": f"{std(x.runtime_s for x in rs):.6g}",
                "f1_mean": f"{mean(x.f1 for x in rs):.6g}",
                "f1_std": f"{std(x.f1 for x in rs):.6g}",
                "f2_mean": f"{mean(x.f2 for x in rs):.6g}",
                "f2_std": f"{std(x.f2 for x in rs):.6g}",
                "f3_mean": f"{mean(x.f3 for x in rs):.6g}",
                "f3_std": f"{std(x.f3 for x in rs):.6g}",
            })
    return out_file


# -----------------------------
# Figure generation
# -----------------------------

def plot_method_comparisons(records: Sequence[MethodRecord], out_dir: Path, dpi: int) -> List[Path]:
    written: List[Path] = []
    by_ctx: Dict[Tuple[str, str, str], List[MethodRecord]] = defaultdict(list)
    for r in records:
        by_ctx[(r.experiment, r.variant, r.terrain_tag)].append(r)

    preferred_order = ["RRT*", "PRM", "MOEA/D-min_f1", "MOEA/D-min_f2", "MOEA/D-min_f3", "MOEA/D-compromise"]
    metric_specs = [
        ("f1", "Path length f1"),
        ("f2", "Threat cost f2"),
        ("f3", "Energy-like cost f3"),
    ]

    for (exp, var, terrain), rs in sorted(by_ctx.items()):
        methods = sorted({r.method for r in rs}, key=lambda m: preferred_order.index(m) if m in preferred_order else 999)
        if len(methods) < 2:
            continue
        ctx_dir = out_dir / "method_compare" / safe_name(exp) / safe_name(var) / safe_name(terrain)

        for metric, ylabel in metric_specs:
            groups = {m: [getattr(r, metric) for r in rs if r.method == m] for m in methods}
            fn = ctx_dir / f"{metric}_boxplot.png"
            out = save_boxplot(groups, f"{terrain} / {var}: {ylabel}", ylabel, fn, dpi)
            if out:
                written.append(out)

        groups_rt = {m: [r.runtime_s for r in rs if r.method == m] for m in methods}
        out = save_boxplot(groups_rt, f"{terrain} / {var}: runtime", "runtime / s", ctx_dir / "runtime_boxplot.png", dpi)
        if out:
            written.append(out)

        success = {}
        # Since records only contain found solutions, success is better computed by case count
        # for MOEA/D reps vs baseline methods. Here we report observed valid sample count.
        for m in methods:
            success[m] = len([r for r in rs if r.method == m])
        out = save_barplot(success, f"{terrain} / {var}: valid solution count", "valid solution count", ctx_dir / "valid_count.png", dpi)
        if out:
            written.append(out)

    return written


def plot_ablation(cases: Sequence[CaseRow], out_dir: Path, dpi: int) -> List[Path]:
    written: List[Path] = []
    by_exp_terrain: Dict[Tuple[str, str], List[CaseRow]] = defaultdict(list)
    for r in cases:
        if r.invalid_case:
            continue
        by_exp_terrain[(r.experiment, r.terrain_tag)].append(r)

    # Only useful when there are at least two variants.
    metric_specs = [
        ("min_f1_f1", "best path length from min_f1", "f1"),
        ("min_f2_f2", "best threat cost from min_f2", "f2"),
        ("min_f3_f3", "best energy-like cost from min_f3", "f3"),
        ("compromise_f1", "compromise path length", "f1"),
        ("compromise_f2", "compromise threat cost", "f2"),
        ("compromise_f3", "compromise energy-like cost", "f3"),
        ("moead_ms", "MOEA/D runtime", "runtime / ms"),
        ("moead_n_gen", "MOEA/D stop generation", "generation"),
        ("moead_archive_size", "MOEA/D archive size", "archive size"),
    ]

    for (exp, terrain), rows in sorted(by_exp_terrain.items()):
        variants = sorted({r.variant for r in rows})
        if len(variants) < 2:
            continue

        ctx_dir = out_dir / "ablation" / safe_name(exp) / safe_name(terrain)

        for attr, title, ylabel in metric_specs:
            groups = {v: [getattr(r, attr) for r in rows if r.variant == v] for v in variants}
            # Convert runtime ms to seconds for prettier axis.
            if attr == "moead_ms":
                groups = {k: [x / 1000.0 for x in vals] for k, vals in groups.items()}
                ylabel = "runtime / s"
            out = save_boxplot(groups, f"{terrain}: {title}", ylabel, ctx_dir / f"{attr}_boxplot.png", dpi)
            if out:
                written.append(out)

        # Success/valid archive rate.
        total_by_v = Counter(r.variant for r in rows)
        succ_by_v = Counter(r.variant for r in rows if r.moead_archive_size > 0 and _is_finite(r.compromise_f1))
        rates = {v: (succ_by_v[v] / total_by_v[v] * 100.0 if total_by_v[v] else 0.0) for v in variants}
        out = save_barplot(rates, f"{terrain}: MOEA/D archive success rate", "success rate / %", ctx_dir / "archive_success_rate.png", dpi, ylim=(0, 105))
        if out:
            written.append(out)

        # Stop reason stacked bar.
        counts_by_v = {v: Counter(r.moead_stop_reason for r in rows if r.variant == v) for v in variants}
        out = save_stacked_stop_reason(counts_by_v, f"{terrain}: MOEA/D stop reasons", ctx_dir / "stop_reason_counts.png", dpi)
        if out:
            written.append(out)

    return written


def plot_suite_variant(cases: Sequence[CaseRow], out_dir: Path, dpi: int) -> List[Path]:
    """
    Compare MOEA/D behavior across suite variants such as baseline vs full.
    This complements method_compare, which compares RRT*/PRM/MOEA reps inside one variant.
    """
    written: List[Path] = []
    rows = [r for r in cases if not r.invalid_case]
    by_exp_terrain: Dict[Tuple[str, str], List[CaseRow]] = defaultdict(list)
    for r in rows:
        by_exp_terrain[(r.experiment, r.terrain_tag)].append(r)

    for (exp, terrain), rs in sorted(by_exp_terrain.items()):
        variants = sorted({r.variant for r in rs})
        if len(variants) < 2:
            continue
        ctx_dir = out_dir / "suite_variant" / safe_name(exp) / safe_name(terrain)
        specs = [
            ("compromise_f1", "MOEA/D compromise f1", "f1"),
            ("compromise_f2", "MOEA/D compromise f2", "f2"),
            ("compromise_f3", "MOEA/D compromise f3", "f3"),
            ("moead_ms", "MOEA/D runtime", "runtime / ms"),
            ("moead_n_gen", "MOEA/D stop generation", "generation"),
            ("moead_archive_size", "MOEA/D archive size", "archive size"),
        ]
        for attr, title, ylabel in specs:
            groups = {v: [getattr(r, attr) for r in rs if r.variant == v] for v in variants}
            if attr == "moead_ms":
                groups = {k: [x / 1000.0 for x in vals] for k, vals in groups.items()}
                ylabel = "runtime / s"
            out = save_boxplot(groups, f"{terrain}: {title}", ylabel, ctx_dir / f"{attr}_by_variant.png", dpi)
            if out:
                written.append(out)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw comparison figures from reproducibility_cases.csv.")
    ap.add_argument("--in_root", type=str, default="outputs", help="Root directory containing result CSV files.")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir. Default: <in_root>/_figures")
    ap.add_argument("--kind", choices=["auto", "method", "ablation", "suite_variant"], default="auto",
                    help="Which figure group to draw.")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no_summary", action="store_true", help="Do not write plot_summary.csv.")
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_root / "_figures"
    ensure_dir(out_dir)

    cases = discover_cases(in_root)
    if not cases:
        raise SystemExit(f"[plot_results] No reproducibility_cases.csv found under: {in_root}")

    records = to_method_records(cases, exclude_invalid=True)

    written: List[Path] = []
    if args.kind in {"auto", "method"}:
        written.extend(plot_method_comparisons(records, out_dir, dpi=args.dpi))
    if args.kind in {"auto", "ablation"}:
        written.extend(plot_ablation(cases, out_dir, dpi=args.dpi))
    if args.kind in {"auto", "suite_variant"}:
        written.extend(plot_suite_variant(cases, out_dir, dpi=args.dpi))

    if not args.no_summary:
        summary_path = write_summary_csv(cases, records, out_dir / "plot_summary.csv")
        written.append(summary_path)

    print("[plot_results] done")
    print("  in_root          :", in_root)
    print("  cases            :", len(cases))
    print("  method records   :", len(records))
    print("  output dir       :", out_dir)
    print("  files written    :", len(written))
    for p in written[:30]:
        print("  wrote            :", p)
    if len(written) > 30:
        print(f"  ... and {len(written) - 30} more")


if __name__ == "__main__":
    main()
