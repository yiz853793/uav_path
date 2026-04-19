#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/collect_results.py

Collect benchmark outputs produced by experiments/run_benchmark.py.

Scans for: <in_root>/**/metrics_*.json

Outputs:
  1) results_wide.csv     : one row per case (includes RRT* + MOEA/D reps)
  2) results_long.csv     : one row per (case, method_variant)
  3) summary_grouped.csv  : grouped stats by user-specified group keys

Key features:
- Filters: --size, --inflate, --terrain_seed, --planner_seed, --seed
- Grouping: --group_by supports multiple keys, incl. moead_cfg (MOEA/D parameter signature)
- Summary uses only found==1 for mean/std; success_rate uses all rows in group.

Example:
  python experiments/collect_results.py --in_root outputs --size S M --group_by size,method,moead_cfg --exclude_invalid
"""

import os
import json
import csv
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# Helpers
# -------------------------

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def to_float_or_inf(x: Any) -> float:
    if x is None:
        return float("inf")
    if is_number(x):
        return float(x)
    try:
        return float(x)
    except Exception:
        return float("inf")

def to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    try:
        return int(x)
    except Exception:
        return None

def obj3_from_json(obj: Any) -> Tuple[float, float, float]:
    if obj is None:
        return (float("inf"), float("inf"), float("inf"))
    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return (to_float_or_inf(obj[0]), to_float_or_inf(obj[1]), to_float_or_inf(obj[2]))
    return (float("inf"), float("inf"), float("inf"))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return (m, math.sqrt(v))

def finite_vals(vals: List[float]) -> List[float]:
    return [v for v in vals if is_number(v) and math.isfinite(v)]

def fmt_float(x: float) -> str:
    if x is None:
        return ""
    if not is_number(x):
        return str(x)
    if math.isinf(x):
        return "inf"
    if math.isnan(x):
        return "nan"
    return f"{x:.6g}"

def safe_get(d: Any, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def normalize_tag(tag: str) -> str:
    tag = (tag or "").strip()
    return tag.upper()

def parse_csv_list_arg(s: str) -> List[str]:
    # "a,b,c" -> ["a","b","c"]
    items = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            items.append(x)
    return items


# -------------------------
# MOEA/D cfg extraction
# -------------------------

# You can extend this list to match your project naming.
MOEAD_PARAM_CANDIDATES = [
    # common names
    "gen", "n_gen", "max_gen", "generations",
    "pop", "pop_size", "population", "N",
    "T", "K", "neighbors", "neighbor_size",
    "eval_step", "sample_step", "moead_eval_step",
    "weight_num", "n_weights",
    "F", "mutation_F", "CR", "crossover_CR",
    "seed", "moead_seed",
    # sometimes used
    "archive_size_limit", "archive_cap",
    "ref_point", "ideal_point",
]

# Common places where args/config may exist in your metrics json
MOEAD_CONFIG_PATHS = [
    ["moead", "config"],
    ["moead", "params"],
    ["moead", "args"],
    ["moead_args"],
    ["moead_config"],
    ["args"],        # overall args dumped by run script
    ["cmd_args"],    # if you store cli args
    ["config"],
]

def extract_moead_params(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to extract a compact set of MOEA/D parameters from metrics json.
    Returns dict of selected params (only those found).
    """
    # gather candidate dicts
    candidates: List[Dict[str, Any]] = []
    for path in MOEAD_CONFIG_PATHS:
        d = safe_get(metrics, path, None)
        if isinstance(d, dict):
            candidates.append(d)

    # also: if moead block itself has direct fields
    moead = metrics.get("moead")
    if isinstance(moead, dict):
        candidates.append(moead)

    # merge: later dicts override earlier ones (prefer more specific)
    merged: Dict[str, Any] = {}
    for d in candidates:
        for k, v in d.items():
            merged[k] = v

    out: Dict[str, Any] = {}
    for k in MOEAD_PARAM_CANDIDATES:
        if k in merged:
            out[k] = merged[k]

    # normalize some known ones
    # unify "gen" name
    if "gen" not in out:
        for kk in ["n_gen", "max_gen", "generations"]:
            if kk in out:
                out["gen"] = out[kk]
                break
    if "pop" not in out:
        for kk in ["pop_size", "population", "N"]:
            if kk in out:
                out["pop"] = out[kk]
                break

    # keep only a stable subset if present
    stable_keys = ["gen", "pop", "T", "K", "eval_step", "weight_num", "F", "CR"]
    compact: Dict[str, Any] = {}
    for k in stable_keys:
        if k in out:
            compact[k] = out[k]
    # If nothing found, keep empty -> signature "unknown"
    return compact

def moead_cfg_signature(moead_params: Dict[str, Any]) -> str:
    """
    Turn moead_params into a stable signature string for grouping.
    """
    if not moead_params:
        return "moead_cfg=unknown"
    # sort by key for stability
    parts = []
    for k in sorted(moead_params.keys()):
        v = moead_params[k]
        if is_number(v):
            if isinstance(v, float) and math.isfinite(v):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={str(v)}")
    return "moead_cfg:" + ",".join(parts)


# -------------------------
# Data model
# -------------------------

@dataclass
class CaseRowWide:
    # identifiers
    metrics_path: str
    terrain_file: str
    terrain_seed: Optional[int]
    planner_seed: Optional[int]
    size_tag: str
    H: int
    W: int
    inflate: int
    invalid_case: int  # 0/1

    # extracted moead cfg signature
    moead_cfg: str

    # RRT*
    rrt_found: int
    rrt_ms: float
    rrt_f1: float
    rrt_f2: float
    rrt_f3: float
    rrt_feasible: int
    rrt_violation: float

    # MOEA/D core
    moead_archive_size: int
    moead_ms: float

    # MOEA/D reps
    moead_min_f1_f1: float
    moead_min_f1_f2: float
    moead_min_f1_f3: float

    moead_min_f2_f1: float
    moead_min_f2_f2: float
    moead_min_f2_f3: float

    moead_min_f3_f1: float
    moead_min_f3_f2: float
    moead_min_f3_f3: float

    moead_comp_f1: float
    moead_comp_f2: float
    moead_comp_f3: float


@dataclass
class CaseRowLong:
    metrics_path: str
    terrain_file: str
    terrain_seed: Optional[int]
    planner_seed: Optional[int]
    size_tag: str
    H: int
    W: int
    inflate: int
    invalid_case: int
    moead_cfg: str

    method: str          # "RRT*" / "MOEAD_min_f1" / ...
    found: int
    runtime_ms: float
    f1: float
    f2: float
    f3: float


# -------------------------
# Parsing one metrics json
# -------------------------

def parse_one_metrics(path: str) -> Tuple[CaseRowWide, List[CaseRowLong]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terrain_file = str(data.get("terrain_file", ""))
    terrain_seed = to_int_or_none(data.get("terrain_seed", None))
    planner_seed = to_int_or_none(data.get("planner_seed", None))
    inflate = int(to_float_or_inf(data.get("inflate", 0)))
    invalid_case = 1 if bool(data.get("invalid_case", False)) else 0

    map_size = data.get("map_size", {}) if isinstance(data.get("map_size", {}), dict) else {}
    H = int(to_float_or_inf(map_size.get("H", -1)))
    W = int(to_float_or_inf(map_size.get("W", -1)))
    size_tag = normalize_tag(str(map_size.get("tag", "")))

    # MOEA/D cfg signature
    moead_params = extract_moead_params(data)
    moead_cfg = moead_cfg_signature(moead_params)

    # ---- RRT ----
    rrt = data.get("rrt", {}) if isinstance(data.get("rrt", {}), dict) else {}
    rrt_found = 1 if bool(rrt.get("found", False)) else 0
    rrt_ms = to_float_or_inf(rrt.get("runtime_ms", 0.0))
    rrt_obj = obj3_from_json(rrt.get("obj", None))
    rrt_feasible = 1 if bool(rrt.get("feasible", False)) else 0
    rrt_violation = to_float_or_inf(rrt.get("violation", None))

    # ---- MOEAD ----
    moead = data.get("moead", {}) if isinstance(data.get("moead", {}), dict) else {}
    moead_ms = to_float_or_inf(moead.get("runtime_ms", 0.0))
    moead_archive_size = int(to_float_or_inf(moead.get("archive_size", 0)))

    reps = moead.get("representatives", None)
    if not isinstance(reps, dict):
        reps = {}

    rep_min_f1 = obj3_from_json(reps.get("min_f1"))
    rep_min_f2 = obj3_from_json(reps.get("min_f2"))
    rep_min_f3 = obj3_from_json(reps.get("min_f3"))
    rep_comp = obj3_from_json(reps.get("compromise"))

    wide = CaseRowWide(
        metrics_path=path,
        terrain_file=terrain_file,
        terrain_seed=terrain_seed,
        planner_seed=planner_seed,
        size_tag=size_tag,
        H=H,
        W=W,
        inflate=inflate,
        invalid_case=invalid_case,

        moead_cfg=moead_cfg,

        rrt_found=rrt_found,
        rrt_ms=rrt_ms,
        rrt_f1=rrt_obj[0],
        rrt_f2=rrt_obj[1],
        rrt_f3=rrt_obj[2],
        rrt_feasible=rrt_feasible,
        rrt_violation=rrt_violation,

        moead_archive_size=moead_archive_size,
        moead_ms=moead_ms,

        moead_min_f1_f1=rep_min_f1[0],
        moead_min_f1_f2=rep_min_f1[1],
        moead_min_f1_f3=rep_min_f1[2],

        moead_min_f2_f1=rep_min_f2[0],
        moead_min_f2_f2=rep_min_f2[1],
        moead_min_f2_f3=rep_min_f2[2],

        moead_min_f3_f1=rep_min_f3[0],
        moead_min_f3_f2=rep_min_f3[1],
        moead_min_f3_f3=rep_min_f3[2],

        moead_comp_f1=rep_comp[0],
        moead_comp_f2=rep_comp[1],
        moead_comp_f3=rep_comp[2],
    )

    long_rows: List[CaseRowLong] = []

    # RRT*
    long_rows.append(CaseRowLong(
        metrics_path=path,
        terrain_file=terrain_file,
        terrain_seed=terrain_seed,
        planner_seed=planner_seed,
        size_tag=size_tag,
        H=H,
        W=W,
        inflate=inflate,
        invalid_case=invalid_case,
        moead_cfg=moead_cfg,

        method="RRT*",
        found=rrt_found,
        runtime_ms=rrt_ms,
        f1=wide.rrt_f1,
        f2=wide.rrt_f2,
        f3=wide.rrt_f3,
    ))

    # MOEAD reps found flags
    def moead_found_from_obj(o: Tuple[float, float, float]) -> int:
        if moead_archive_size <= 0:
            return 0
        if all(math.isinf(x) for x in o):
            return 0
        return 1

    for name, o in [
        ("MOEAD_min_f1", rep_min_f1),
        ("MOEAD_min_f2", rep_min_f2),
        ("MOEAD_min_f3", rep_min_f3),
        ("MOEAD_compromise", rep_comp),
    ]:
        long_rows.append(CaseRowLong(
            metrics_path=path,
            terrain_file=terrain_file,
            terrain_seed=terrain_seed,
            planner_seed=planner_seed,
            size_tag=size_tag,
            H=H,
            W=W,
            inflate=inflate,
            invalid_case=invalid_case,
            moead_cfg=moead_cfg,

            method=name,
            found=moead_found_from_obj(o),
            runtime_ms=moead_ms,
            f1=o[0], f2=o[1], f3=o[2],
        ))

    return wide, long_rows


# -------------------------
# Filters
# -------------------------

def pass_filters(w: CaseRowWide,
                 size_allow: Optional[List[str]],
                 inflate_allow: Optional[List[int]],
                 terrain_seed_allow: Optional[List[int]],
                 planner_seed_allow: Optional[List[int]],
                 seed_allow_any: Optional[List[int]]) -> bool:
    # size
    if size_allow is not None and w.size_tag not in size_allow:
        return False
    # inflate
    if inflate_allow is not None and w.inflate not in inflate_allow:
        return False
    # terrain_seed / planner_seed
    if terrain_seed_allow is not None:
        if w.terrain_seed is None or w.terrain_seed not in terrain_seed_allow:
            return False
    if planner_seed_allow is not None:
        if w.planner_seed is None or w.planner_seed not in planner_seed_allow:
            return False
    # generic seed: match either terrain_seed or planner_seed
    if seed_allow_any is not None:
        ok = False
        if w.terrain_seed is not None and w.terrain_seed in seed_allow_any:
            ok = True
        if w.planner_seed is not None and w.planner_seed in seed_allow_any:
            ok = True
        if not ok:
            return False
    return True


# -------------------------
# CSV writers
# -------------------------

def write_wide_csv(rows: List[CaseRowWide], out_csv: str):
    ensure_dir(os.path.dirname(out_csv) or ".")
    fields = list(CaseRowWide.__annotations__.keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = {k: getattr(r, k) for k in fields}
            for k, v in list(d.items()):
                if isinstance(v, float):
                    d[k] = fmt_float(v)
            w.writerow(d)

def write_long_csv(rows: List[CaseRowLong], out_csv: str):
    ensure_dir(os.path.dirname(out_csv) or ".")
    fields = list(CaseRowLong.__annotations__.keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = {k: getattr(r, k) for k in fields}
            for k, v in list(d.items()):
                if isinstance(v, float):
                    d[k] = fmt_float(v)
            w.writerow(d)

def make_group_key(r: CaseRowLong, group_keys: List[str]) -> Tuple[str, ...]:
    parts = []
    for k in group_keys:
        if k == "size":
            parts.append(r.size_tag)
        elif k == "method":
            parts.append(r.method)
        elif k == "inflate":
            parts.append(str(r.inflate))
        elif k == "moead_cfg":
            parts.append(r.moead_cfg)
        elif k == "H":
            parts.append(str(r.H))
        elif k == "W":
            parts.append(str(r.W))
        elif k == "terrain_seed":
            parts.append(str(r.terrain_seed) if r.terrain_seed is not None else "None")
        elif k == "planner_seed":
            parts.append(str(r.planner_seed) if r.planner_seed is not None else "None")
        else:
            # unknown key: still keep placeholder to avoid crash
            parts.append("UNKNOWN_KEY:" + k)
    return tuple(parts)

def write_summary_grouped(long_rows: List[CaseRowLong],
                          out_csv: str,
                          group_keys: List[str],
                          exclude_invalid: bool):
    """
    Group by user-specified keys. Summary columns:
      <group_keys...>,
      n_cases, success, success_rate,
      runtime_mean, runtime_std,
      f1_mean, f1_std, f2_mean, f2_std, f3_mean, f3_std
    """
    ensure_dir(os.path.dirname(out_csv) or ".")

    # group
    groups: Dict[Tuple[str, ...], List[CaseRowLong]] = {}
    for r in long_rows:
        if exclude_invalid and r.invalid_case:
            continue
        key = make_group_key(r, group_keys)
        groups.setdefault(key, []).append(r)

    fields = group_keys + [
        "n_cases", "success", "success_rate",
        "runtime_mean", "runtime_std",
        "f1_mean", "f1_std",
        "f2_mean", "f2_std",
        "f3_mean", "f3_std",
    ]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        # deterministic ordering
        for key in sorted(groups.keys()):
            rs = groups[key]
            n_cases = len(rs)
            succ = sum(1 for x in rs if x.found)

            ok = [x for x in rs if x.found]
            runtimes = finite_vals([x.runtime_ms for x in ok])
            f1s = finite_vals([x.f1 for x in ok])
            f2s = finite_vals([x.f2 for x in ok])
            f3s = finite_vals([x.f3 for x in ok])

            rt_m, rt_s = mean_std(runtimes)
            f1_m, f1_s = mean_std(f1s)
            f2_m, f2_s = mean_std(f2s)
            f3_m, f3_s = mean_std(f3s)

            row = {}
            for i, kname in enumerate(group_keys):
                row[kname] = key[i]

            row.update({
                "n_cases": n_cases,
                "success": succ,
                "success_rate": f"{(succ / n_cases * 100.0):.2f}" if n_cases > 0 else "nan",
                "runtime_mean": fmt_float(rt_m),
                "runtime_std": fmt_float(rt_s),
                "f1_mean": fmt_float(f1_m),
                "f1_std": fmt_float(f1_s),
                "f2_mean": fmt_float(f2_m),
                "f2_std": fmt_float(f2_s),
                "f3_mean": fmt_float(f3_m),
                "f3_std": fmt_float(f3_s),
            })
            w.writerow(row)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="outputs",
                    help="Root folder containing benchmark outputs (default: outputs)")
    ap.add_argument("--pattern", type=str, default="**/metrics_*.json",
                    help="Glob pattern under in_root (default: **/metrics_*.json)")
    ap.add_argument("--out_dir", type=str, default="",
                    help="Output directory (default: <in_root>/_collected)")

    # filters
    ap.add_argument("--size", nargs="*", default=None,
                    help="Filter by size_tag, e.g. --size S M L")
    ap.add_argument("--inflate", nargs="*", default=None,
                    help="Filter by inflate values, e.g. --inflate 1 2")
    ap.add_argument("--terrain_seed", nargs="*", default=None,
                    help="Filter by terrain_seed, e.g. --terrain_seed 0 1 2")
    ap.add_argument("--planner_seed", nargs="*", default=None,
                    help="Filter by planner_seed, e.g. --planner_seed 0 1 2")
    ap.add_argument("--seed", nargs="*", default=None,
                    help="Filter by seed (match either terrain_seed or planner_seed)")

    # grouping
    ap.add_argument("--group_by", type=str, default="size,method",
                    help=("Comma-separated group keys. Supported: "
                          "size,method,inflate,moead_cfg,H,W,terrain_seed,planner_seed. "
                          "Example: --group_by size,method,moead_cfg"))
    ap.add_argument("--exclude_invalid", action="store_true",
                    help="Exclude invalid_case rows from summary stats.")

    args = ap.parse_args()

    in_root = args.in_root
    search_pat = os.path.join(in_root, args.pattern)
    out_dir = args.out_dir.strip() or os.path.join(in_root, "_collected")
    ensure_dir(out_dir)

    # normalize filters
    size_allow = None
    if args.size is not None and len(args.size) > 0:
        size_allow = [normalize_tag(x) for x in args.size]

    inflate_allow = None
    if args.inflate is not None and len(args.inflate) > 0:
        inflate_allow = []
        for x in args.inflate:
            try:
                inflate_allow.append(int(x))
            except Exception:
                pass
        if not inflate_allow:
            inflate_allow = None

    terrain_seed_allow = None
    if args.terrain_seed is not None and len(args.terrain_seed) > 0:
        terrain_seed_allow = [int(x) for x in args.terrain_seed]

    planner_seed_allow = None
    if args.planner_seed is not None and len(args.planner_seed) > 0:
        planner_seed_allow = [int(x) for x in args.planner_seed]

    seed_allow_any = None
    if args.seed is not None and len(args.seed) > 0:
        seed_allow_any = [int(x) for x in args.seed]

    group_keys = parse_csv_list_arg(args.group_by)
    if not group_keys:
        group_keys = ["size", "method"]

    # scan
    paths = sorted(glob.glob(search_pat, recursive=True))
    if not paths:
        raise SystemExit(f"[collect_results] No metrics json found. pattern={search_pat}")

    wide_rows: List[CaseRowWide] = []
    long_rows: List[CaseRowLong] = []
    bad = 0
    filtered_out = 0

    for p in paths:
        try:
            w, ls = parse_one_metrics(p)
            if not pass_filters(w, size_allow, inflate_allow, terrain_seed_allow, planner_seed_allow, seed_allow_any):
                filtered_out += 1
                continue
            wide_rows.append(w)
            long_rows.extend(ls)
        except Exception as e:
            bad += 1
<<<<<<< HEAD
            print(f"[warn] failed to parse: {p}\n  err={e}", flush=True)
=======
            print(f"[warn] failed to parse: {p}\n  err={e}")
>>>>>>> origin/feature/3d

    out_wide = os.path.join(out_dir, "results_wide.csv")
    out_long = os.path.join(out_dir, "results_long.csv")
    out_sum = os.path.join(out_dir, "summary_grouped.csv")

    write_wide_csv(wide_rows, out_wide)
    write_long_csv(long_rows, out_long)
    write_summary_grouped(long_rows, out_sum, group_keys=group_keys, exclude_invalid=args.exclude_invalid)

<<<<<<< HEAD
    print("[collect_results] done", flush=True)
    print("  metrics found     :", len(paths), flush=True)
    print("  parsed ok         :", len(wide_rows), flush=True)
    print("  filtered out      :", filtered_out, flush=True)
    print("  parsed failed     :", bad, flush=True)
    print("  wrote             :", out_wide, flush=True)
    print("  wrote             :", out_long, flush=True)
    print("  wrote             :", out_sum, flush=True)
    print("  group_by          :", ",".join(group_keys), flush=True)
    if size_allow is not None:
        print("  filter size       :", size_allow, flush=True)
    if inflate_allow is not None:
        print("  filter inflate    :", inflate_allow, flush=True)
=======
    print("[collect_results] done")
    print("  metrics found     :", len(paths))
    print("  parsed ok         :", len(wide_rows))
    print("  filtered out      :", filtered_out)
    print("  parsed failed     :", bad)
    print("  wrote             :", out_wide)
    print("  wrote             :", out_long)
    print("  wrote             :", out_sum)
    print("  group_by          :", ",".join(group_keys))
    if size_allow is not None:
        print("  filter size       :", size_allow)
    if inflate_allow is not None:
        print("  filter inflate    :", inflate_allow)
>>>>>>> origin/feature/3d


if __name__ == "__main__":
    main()