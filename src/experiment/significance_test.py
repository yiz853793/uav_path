from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s or "").split(",") if x.strip()]


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    return v if math.isfinite(v) else math.nan


def _is_found(row: Dict[str, Any]) -> bool:
    v = str(row.get("found", "")).strip().lower()
    return v in {"1", "true", "yes", "y"}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return math.nan
    vals = sorted(xs)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _rankdata(values: Sequence[float]) -> Tuple[List[float], List[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_sizes: List[int] = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_rank_sum(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided") -> Dict[str, float]:
    """Wilcoxon rank-sum test with tie correction and normal approximation.

    ``x`` is the tested method sample and ``y`` is the baseline sample.
    """
    x = [float(v) for v in x if math.isfinite(float(v))]
    y = [float(v) for v in y if math.isfinite(float(v))]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return {"u": math.nan, "z": math.nan, "p_value": math.nan}

    combined = x + y
    ranks, tie_sizes = _rankdata(combined)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mean_u = n1 * n2 / 2.0
    tie_sum = sum(t ** 3 - t for t in tie_sizes)
    if n <= 1:
        return {"u": u1, "z": math.nan, "p_value": math.nan}
    var_u = n1 * n2 / 12.0 * ((n + 1) - tie_sum / (n * (n - 1)))
    if var_u <= 0.0:
        p = 1.0 if abs(u1 - mean_u) < 1e-12 else 0.0
        return {"u": u1, "z": 0.0, "p_value": p}

    z = (u1 - mean_u) / math.sqrt(var_u)
    alternative = str(alternative or "two-sided").lower()
    if alternative == "less":
        p = _normal_cdf(z)
    elif alternative == "greater":
        p = 1.0 - _normal_cdf(z)
    else:
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"u": u1, "z": z, "p_value": max(0.0, min(1.0, p))}


def read_results_long(path: str, exclude_unfound: bool = True) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: List[Dict[str, Any]] = []
    for row in rows:
        if exclude_unfound and not _is_found(row):
            continue
        out.append(row)
    return out


def collect_values(rows: Iterable[Dict[str, Any]], metric: str) -> List[float]:
    vals = [_to_float(row.get(metric)) for row in rows]
    return [v for v in vals if math.isfinite(v)]


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_wilcoxon_rows(
    rows: List[Dict[str, Any]],
    *,
    baselines: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    group_by: Sequence[str],
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    baselines = [str(x) for x in baselines if str(x)]
    methods = [str(x) for x in methods if str(x)]
    metrics = [str(x) for x in metrics if str(x)] or ["f1", "f2", "f3"]
    group_by = [str(x) for x in group_by if str(x)]
    if not methods:
        all_methods = sorted({row.get("method", "") for row in rows if row.get("method")})
        methods = [m for m in all_methods if m not in set(baselines)]

    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in group_by)
        grouped[key].append(row)

    out_rows: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda kv: kv[0]):
        group_values = {col: val for col, val in zip(group_by, key)}
        by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in items:
            by_method[str(row.get("method", ""))].append(row)
        for baseline in baselines:
            for method in methods:
                if method == baseline:
                    continue
                for metric in metrics:
                    x = collect_values(by_method.get(method, []), metric)
                    y = collect_values(by_method.get(baseline, []), metric)
                    stat = wilcoxon_rank_sum(x, y, alternative=alternative)
                    p = stat["p_value"]
                    out_rows.append({
                        **group_values,
                        "baseline": baseline,
                        "method": method,
                        "metric": metric,
                        "alternative": alternative,
                        "n_method": len(x),
                        "n_baseline": len(y),
                        "method_mean": _mean(x),
                        "baseline_mean": _mean(y),
                        "method_median": _median(x),
                        "baseline_median": _median(y),
                        "median_delta_baseline_minus_method": (_median(y) - _median(x)) if x and y else math.nan,
                        "u": stat["u"],
                        "z": stat["z"],
                        "p_value": p,
                        "alpha": float(alpha),
                        "significant": int(math.isfinite(p) and p < float(alpha)),
                    })
    return out_rows


def write_wilcoxon_csv(
    *,
    in_csv: str,
    out_csv: str,
    baselines: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    group_by: Sequence[str],
    alternative: str = "two-sided",
    alpha: float = 0.05,
    include_unfound: bool = False,
) -> List[Dict[str, Any]]:
    rows = read_results_long(in_csv, exclude_unfound=not include_unfound)
    out_rows = build_wilcoxon_rows(
        rows,
        baselines=baselines,
        methods=methods,
        metrics=metrics,
        group_by=group_by,
        alternative=alternative,
        alpha=alpha,
    )
    write_csv(out_csv, out_rows)
    return out_rows


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run Wilcoxon rank-sum tests from results_long.csv.")
    ap.add_argument("--in_csv", required=True, help="Path to results_long.csv produced by collect_results.")
    ap.add_argument("--out_csv", default="", help="Output CSV path. Default: beside input as wilcoxon_rank_sum.csv")
    ap.add_argument("--baselines", default="PRM,NSGA-III-compromise", help="Comma-separated baseline method names.")
    ap.add_argument("--methods", default="MOEAD_compromise", help="Comma-separated tested method names; empty means all non-baseline methods.")
    ap.add_argument("--metrics", default="f1,f2,f3", help="Comma-separated metrics to test.")
    ap.add_argument("--group_by", default="size_tag,variant", help="Comma-separated grouping columns; use empty string for all rows together.")
    ap.add_argument("--alternative", choices=["two-sided", "less", "greater"], default="two-sided")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--include_unfound", action="store_true", help="Include rows with found=0. Default excludes them.")
    args = ap.parse_args(argv)

    baselines = _parse_csv_list(args.baselines)
    methods = _parse_csv_list(args.methods)
    metrics = _parse_csv_list(args.metrics) or ["f1", "f2", "f3"]
    group_by = _parse_csv_list(args.group_by)

    out_csv = args.out_csv.strip() or os.path.join(os.path.dirname(args.in_csv), "wilcoxon_rank_sum.csv")
    out_rows = write_wilcoxon_csv(
        in_csv=args.in_csv,
        out_csv=out_csv,
        baselines=baselines,
        methods=methods,
        metrics=metrics,
        group_by=group_by,
        alternative=args.alternative,
        alpha=float(args.alpha),
        include_unfound=bool(args.include_unfound),
    )
    print("[wilcoxon] wrote:", out_csv)
    print("[wilcoxon] rows :", len(out_rows))


if __name__ == "__main__":
    main()
