# experiments/run_benchmark.py

import os
import json
import time
import glob
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from src.models.evaluator import evaluate_path
from experiments.vis_data import build_vis_payload, save_vis_payload

def meters_to_cells(env, value_m: float) -> float:
    return float(value_m) / float(env.resolution)

def cells_to_meters(env, value_cells: float) -> float:
    return float(value_cells) * float(env.resolution)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: str, text: str):
    ensure_dir(os.path.dirname(log_path) or ".")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def map_size_name(H: int, W: int) -> str:
    # 你项目里的默认小地图：H=160,W=200
    # 这里按你的需求：中地图=小地图3倍，大地图=小地图10倍
    if (H, W) == (160, 200):
        return "S"
    if (H, W) == (160 * 3, 200 * 3):
        return "M"
    if (H, W) == (160 * 10, 200 * 10):
        return "L"
    return f"H{H}W{W}"


def astar_tag_from_args(args) -> str:
    # 文件名里追加 A* seeding 参数（除 benchmark_summary.log 外）
    return (
        f"_astarR{args.init_astar_ratio}"
        f"_M{args.init_astar_max_paths}"
        f"_P{args.init_astar_penalty_step}"
        f"_TW{args.init_astar_threat_weight}"
        f"_J{args.init_astar_jitter_sigma}"
    )


def moead_core_tag_from_args(args) -> str:
    # 文件名里追加 MOEA/D 核心参数：gen, pop, K, T
    return (
        f"_g{args.moead_min_gen}-{args.moead_max_gen}_pop{args.moead_pop}_K{args.K}_T{args.moead_T}"
    )


def default_start_goal_for_env(
    env: GridEnv,
    height: np.ndarray,
    z_offset_m: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    W, H = env.W, env.H
    res = float(env.resolution)

    # 固定默认起终点（命令语义按“米”理解）
    sx_m, sy_m = 25.0, 25.0
    gx_m, gy_m = float(max(25.0 + 50.0, (W - 60) * res)), float(max(25.0 + 50.0, (H - 70) * res))

    sx = meters_to_cells(env, sx_m)
    sy = meters_to_cells(env, sy_m)
    gx = meters_to_cells(env, gx_m)
    gy = meters_to_cells(env, gy_m)

    sx_i, sy_i = int(round(sx)), int(round(sy))
    gx_i, gy_i = int(round(gx)), int(round(gy))

    sx_i = int(np.clip(sx_i, 0, W - 1))
    sy_i = int(np.clip(sy_i, 0, H - 1))
    gx_i = int(np.clip(gx_i, 0, W - 1))
    gy_i = int(np.clip(gy_i, 0, H - 1))

    sz = float(height[sy_i, sx_i]) + float(z_offset_m)
    gz = float(height[gy_i, gx_i]) + float(z_offset_m)

    return (
        np.array([float(sx_i), float(sy_i), sz], dtype=np.float32),
        np.array([float(gx_i), float(gy_i), gz], dtype=np.float32),
    )


def select_representatives(arch_items, weights=(1.0, 1.0, 1.0)):
    objs = np.array([it.er.obj for it in arch_items], dtype=float)  # [N,3]
    idx_f1 = int(np.argmin(objs[:, 0]))
    idx_f2 = int(np.argmin(objs[:, 1]))
    idx_f3 = int(np.argmin(objs[:, 2]))

    # min-max normalize for compromise selection
    mn = objs.min(axis=0)
    mx = objs.max(axis=0)
    denom = np.where((mx - mn) > 1e-12, (mx - mn), 1.0)
    norm = (objs - mn) / denom

    w = np.array(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    score = norm @ w
    idx_comp = int(np.argmin(score))

    return {
        "min_f1": idx_f1,
        "min_f2": idx_f2,
        "min_f3": idx_f3,
        "compromise": idx_comp,
        "objs": objs,
        "score": score,
        "norm": norm,
        "weights": w,
    }



def _compact_mtoe_stop_info(stop_info):
    if not isinstance(stop_info, dict):
        return None
    keep = [
        "gamma", "variance", "window_std", "stat", "critical_stat",
        "p_support", "tol_fun", "confidence", "mean_guard",
        "legacy_stop_without_mean_guard", "mtoe", "window_min",
        "window_max", "window_mean", "probe",
    ]
    return {k: stop_info.get(k) for k in keep if k in stop_info}


def split_moead_log(log):
    """Split MOEA/D log into metric-safe summary and debug-only payload."""
    log = log if isinstance(log, dict) else {}
    if not log:
        return {}, {}

    metric_keys = {
        "n_gen",
        "configured_n_gen",
        "requested_n_gen",
        "pop",
        "K",
        "T",
        "n_eval",
        "archive_size",
        "max_gen",
        "moead_min_gen",
        "mtoe_enabled",
        "mtoe_mode",
        "mtoe_tol_fun",
        "mtoe_confidence",
        "stop_reason",
        "mtoe_window",
        "ideal_point",
    }
    metric_log = {k: log[k] for k in metric_keys if k in log}
    debug_log = {k: v for k, v in log.items() if k not in metric_keys}
    return metric_log, debug_log




def write_mtoe_debug_log(log_path: str, *, terrain_file: str = None, terrain_seed: int = None, planner_seed: int = None, debug_payload=None):
    """Write debug-only MTOE/MOEA-D payload to a dedicated .log file."""
    if not log_path or not debug_payload:
        return
    lines = []
    if terrain_file is not None:
        lines.append(f"terrain_file: {terrain_file}")
    if terrain_seed is not None:
        lines.append(f"terrain_seed: {terrain_seed}")
    if planner_seed is not None:
        lines.append(f"planner_seed: {planner_seed}")
    lines.append("")
    if isinstance(debug_payload, dict):
        for k, v in debug_payload.items():
            lines.append(f"[{k}]")
            if isinstance(v, (list, dict)):
                lines.append(json.dumps(v, ensure_ascii=False, indent=2))
            else:
                lines.append(str(v))
            lines.append("")
    else:
        lines.append(str(debug_payload))
        lines.append("")
    Path(log_path).write_text("\n".join(lines).rstrip() + "\n", encoding='utf-8')

def build_moead_metrics_block(args, log, runtime_ms: float, archive_size: int):
    metric_log, debug_log = split_moead_log(log)
    stop_reason = metric_log.get("stop_reason")
    if stop_reason is None:
        stop_reason = "not_run" if not metric_log and float(runtime_ms) <= 0.0 else None
    block = {
        "n_gen": int(metric_log.get("n_gen", args.moead_max_gen)),
        "configured_n_gen": int(metric_log.get("configured_n_gen", args.moead_max_gen)),
        "requested_n_gen": int(metric_log.get("requested_n_gen", args.moead_max_gen)),
        "moead_min_gen": int(args.moead_min_gen),
        "moead_max_gen": int(args.moead_max_gen),
        "stop_reason": stop_reason,
        "mtoe_tol_fun": float(args.mtoe_tol_fun),
        "mtoe_confidence": float(args.mtoe_confidence),
        "mtoe": {
            "enabled": True,
            "mode": metric_log.get("mtoe_mode", "best_so_far_delta"),
            "window": int(metric_log.get("mtoe_window", 10)),
            "tol_fun": float(args.mtoe_tol_fun),
            "confidence": float(args.mtoe_confidence),
        },
        "pop": int(args.moead_pop),
        "K": int(args.K),
        "T": int(args.moead_T),
        "init_astar_ratio": float(args.init_astar_ratio),
        "init_astar_threat_weight": float(args.init_astar_threat_weight),
        "init_astar_jitter_sigma": float(args.init_astar_jitter_sigma),
        "init_astar_max_paths": int(args.init_astar_max_paths),
        "init_astar_penalty_step": float(args.init_astar_penalty_step),
        "init_stratified_ratio": float(args.init_stratified_ratio),
        "init_stratified_lateral_frac": float(args.init_stratified_lateral_frac),
        "init_stratified_n_bands": int(args.init_stratified_n_bands),
        "init_stratified_progress_jitter": float(args.init_stratified_progress_jitter),
        "init_global_random_ratio": float(args.init_global_random_ratio),
        "weight_extreme_bias": float(args.weight_extreme_bias),
        "utility_update_interval": int(args.utility_update_interval),
        "utility_use_archive_density": int(args.utility_use_archive_density),
        "log_flush_every": int(args.log_flush_every),
        "runtime_ms": float(runtime_ms),
        "archive_size": int(archive_size),
    }
    return block, debug_log


def summarize_results(all_rows: list, log_path: str, args):
    if not all_rows:
        append_log(log_path, f"[{now_str()}] Benchmark finished but no valid rows.")
        return

    # NOTE: some terrains can be invalid tasks (start/goal inside no-fly zone or out of bounds).
    # We keep them in rows for traceability, but exclude them from success-rate and runtime stats.
    invalid_rows = [r for r in all_rows if r.get("invalid_case", False)]
    valid_rows = [r for r in all_rows if not r.get("invalid_case", False)]

    n_total = len(all_rows)
    n_invalid = len(invalid_rows)
    n_valid = len(valid_rows)

    # valid-only success
    rrt_found = sum(1 for r in valid_rows if r["rrt_found"])
    prm_found = sum(1 for r in valid_rows if r.get("prm_found", False))
    moead_has_arch = sum(1 for r in valid_rows if r["moead_archive_size"] > 0)

    # RRT* / PRM objective stats (only for cases with a found path)
    rrt_obj_rows = [r for r in valid_rows if r.get("rrt_found") and r.get("rrt_obj") is not None]
    rrt_obj_stats = ""
    if rrt_obj_rows:
        A = np.array([r["rrt_obj"] for r in rrt_obj_rows], dtype=float)  # [M,3]
        m = A.mean(axis=0)
        s = A.std(axis=0)
        feas_cnt = int(sum(1 for r in rrt_obj_rows if r.get("rrt_feasible", False)))
        denom2 = max(1, len(rrt_obj_rows))
        rrt_obj_stats = (
            f"RRT*: objective mean±std over found paths (f1,f2,f3)=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f}) ± "
            f"({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}); feasible {feas_cnt}/{len(rrt_obj_rows)} ({feas_cnt/denom2*100:.2f}%)"
        )

    prm_obj_rows = [r for r in valid_rows if r.get("prm_found") and r.get("prm_obj") is not None]
    prm_obj_stats = ""
    if prm_obj_rows:
        A = np.array([r["prm_obj"] for r in prm_obj_rows], dtype=float)
        m = A.mean(axis=0)
        s = A.std(axis=0)
        feas_cnt = int(sum(1 for r in prm_obj_rows if r.get("prm_feasible", False)))
        denom2 = max(1, len(prm_obj_rows))
        prm_obj_stats = (
            f"PRM: objective mean±std over found paths (f1,f2,f3)=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f}) ± "
            f"({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}); feasible {feas_cnt}/{len(prm_obj_rows)} ({feas_cnt/denom2*100:.2f}%)"
        )

    rrt_ms = np.array([r["rrt_ms"] for r in valid_rows], dtype=float) if n_valid > 0 else np.array([0.0])
    prm_ms = np.array([r.get("prm_ms", 0.0) for r in valid_rows], dtype=float) if n_valid > 0 else np.array([0.0])
    moead_ms = np.array([r["moead_ms"] for r in valid_rows], dtype=float) if n_valid > 0 else np.array([0.0])
    arch_sz = np.array([r["moead_archive_size"] for r in valid_rows], dtype=float) if n_valid > 0 else np.array([0.0])

    def mean_std(x):
        return float(np.mean(x)), float(np.std(x))

    rrt_mean, rrt_std = mean_std(rrt_ms)
    prm_mean, prm_std = mean_std(prm_ms)
    moead_mean, moead_std = mean_std(moead_ms)
    arch_mean, arch_std = mean_std(arch_sz)

    moead_gen_vals = np.array([float(r.get("moead_n_gen", 0.0)) for r in valid_rows], dtype=float) if n_valid > 0 else np.array([0.0])
    moead_gen_mean, moead_gen_std = mean_std(moead_gen_vals)
    stop_reason_counter = {}
    for r in valid_rows:
        key = str(r.get("moead_stop_reason") or "unknown")
        stop_reason_counter[key] = stop_reason_counter.get(key, 0) + 1

    reps_rows = [r for r in valid_rows if r.get("rep_objs") is not None]
    rep_stats = ""
    if reps_rows:
        def stack_rep(name):
            return np.array([r["rep_objs"][name] for r in reps_rows], dtype=float)  # [M,3]

        for key in ["min_f1", "min_f2", "min_f3", "compromise"]:
            A = stack_rep(key)
            m = A.mean(axis=0)
            s = A.std(axis=0)
            rep_stats += (
                f"  - {key}: "
                f"mean(f1,f2,f3)=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f}), "
                f"std=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f})\n"
            )

    text = []
    text.append("=" * 80)
    text.append(f"[{now_str()}] Benchmark Summary")
    text.append(f"Terrain count: {n_total}")
    text.append(f"Valid cases: {n_valid}")
    text.append(f"Invalid cases (start/goal in no-fly or OOB): {n_invalid}")

    # Map size summary (H,W)
    size_counter = {}  # (H,W,tag) -> count
    for r in all_rows:
        H = r.get("H")
        W = r.get("W")
        tag = r.get("size_tag")
        if H is None or W is None:
            continue
        if tag is None:
            tag = map_size_name(int(H), int(W))
        key = (int(H), int(W), str(tag))
        size_counter[key] = size_counter.get(key, 0) + 1

    if size_counter:
        items = sorted(size_counter.items(), key=lambda kv: (-kv[1], kv[0][2], kv[0][0], kv[0][1]))
        if len(items) == 1:
            (H, W, tag), cnt = items[0]
            text.append(f"Map size: H={H}, W={W} (tag={tag})")
        else:
            text.append("Map sizes in this benchmark:")
            for (H, W, tag), cnt in items:
                text.append(f"  - {tag}: H={H}, W={W}  count={cnt}")
        text.append("")
    text.append("-" * 80)
    text.append("Experiment Parameters:")
    text.append(f"  inflate = {args.inflate} m")
    text.append(f"  rrt_iter = {args.rrt_iter}")
    text.append("  PRM:")
    text.append(f"    prm_samples      = {args.prm_samples}")
    text.append(f"    prm_k            = {args.prm_k}")
    text.append(f"    prm_max_edge_len = {args.prm_max_edge_len} m")
    text.append(f"    prm_threat_weight= {args.prm_threat_weight}")
    text.append("")
    text.append("  MOEA/D core:")
    text.append(f"    min_gen = {args.moead_min_gen}")
    text.append(f"    max_gen = {args.moead_max_gen}")
    text.append(f"    mtoe_tol_fun = {args.mtoe_tol_fun}")
    text.append(f"    mtoe_confidence = {args.mtoe_confidence}")
    text.append(f"    pop   = {args.moead_pop}")
    text.append(f"    K     = {args.K}")
    text.append(f"    T     = {args.moead_T}")
    text.append("")
    text.append("  A* seeding:")
    text.append(f"    init_astar_ratio         = {args.init_astar_ratio}")
    text.append(f"    init_astar_max_paths     = {args.init_astar_max_paths}")
    text.append(f"    init_astar_penalty_step  = {args.init_astar_penalty_step}")
    text.append(f"    init_stratified_ratio    = {args.init_stratified_ratio}")
    text.append(f"    init_strat_lateral_frac  = {args.init_stratified_lateral_frac}")
    text.append(f"    init_strat_n_bands       = {args.init_stratified_n_bands}")
    text.append(f"    init_strat_prog_jitter   = {args.init_stratified_progress_jitter}")
    text.append(f"    init_global_random_ratio = {args.init_global_random_ratio}")
    text.append(f"    weight_extreme_bias      = {args.weight_extreme_bias}")
    text.append(f"    init_astar_threat_weight = {args.init_astar_threat_weight}")
    text.append(f"    init_astar_jitter_sigma  = {args.init_astar_jitter_sigma}")
    text.append(f"    archive_size_limit      = {args.archive_size}")
    text.append(f"    archive_soft_limit      = {args.archive_soft_limit}")
    text.append(f"    archive_grid_bins       = {args.archive_grid_bins}")
    text.append(f"    archive_keep_extremes   = {int(args.archive_keep_extremes)}")
    text.append(f"    active_subproblem_ratio = {args.active_subproblem_ratio}")
    text.append("-" * 80)
    text.append(f"Start={args.start if args.start is not None else 'AUTO'}, Goal={args.goal if args.goal is not None else 'AUTO'}")
    denom = n_valid if n_valid > 0 else 1
    text.append(f"RRT*: found {rrt_found}/{n_valid} ({rrt_found/denom*100:.2f}%), runtime_ms mean±std = {rrt_mean:.1f} ± {rrt_std:.1f}  (valid-only)")
    if rrt_obj_stats:
        text.append(rrt_obj_stats)
    text.append(f"PRM: found {prm_found}/{n_valid} ({prm_found/denom*100:.2f}%), runtime_ms mean±std = {prm_mean:.1f} ± {prm_std:.1f}  (valid-only)")
    if prm_obj_stats:
        text.append(prm_obj_stats)

    prm_valid_stats = [r.get("prm_stats") for r in valid_rows if isinstance(r.get("prm_stats"), dict) and r.get("prm_stats")]
    if prm_valid_stats:
        def _avg(key):
            vals = [float(s.get(key, 0.0)) for s in prm_valid_stats]
            return float(np.mean(vals)), float(np.std(vals))
        sd_m, sd_s = _avg("start_degree")
        gd_m, gd_s = _avg("goal_degree")
        sc_m, sc_s = _avg("start_component_size")
        lc_m, lc_s = _avg("largest_component_size")
        conn = sum(1 for s in prm_valid_stats if bool(s.get("start_goal_connected", False)))
        text.append(
            f"PRM connectivity: start_degree mean±std = {sd_m:.2f} ± {sd_s:.2f}, "
            f"goal_degree mean±std = {gd_m:.2f} ± {gd_s:.2f}, "
            f"start_component_size mean±std = {sc_m:.1f} ± {sc_s:.1f}, "
            f"largest_component_size mean±std = {lc_m:.1f} ± {lc_s:.1f}, "
            f"start-goal connected {conn}/{len(prm_valid_stats)} ({conn/max(1,len(prm_valid_stats))*100:.2f}%)"
        )
    text.append(f"MOEA/D: archive>0 {moead_has_arch}/{n_valid} ({moead_has_arch/denom*100:.2f}%), runtime_ms mean±std = {moead_mean:.1f} ± {moead_std:.1f}  (valid-only)")
    text.append(f"MOEA/D: actual_n_gen mean±std = {moead_gen_mean:.1f} ± {moead_gen_std:.1f}")
    text.append(f"MOEA/D: archive_size mean±std = {arch_mean:.2f} ± {arch_std:.2f}")
    if stop_reason_counter:
        stop_items = ", ".join(f"{k}={v}" for k, v in sorted(stop_reason_counter.items()))
        text.append(f"MOEA/D stop reasons: {stop_items}")

    if rep_stats:
        text.append("Representatives objective statistics (only terrains with archive>0):")
        text.append(rep_stats.rstrip())

    hard_by_arch0 = [r for r in valid_rows if r["moead_archive_size"] == 0]
    if hard_by_arch0:
        show = hard_by_arch0[:10]
        text.append(f"Hard cases (MOEA/D archive_size=0) count={len(hard_by_arch0)} (show up to 10):")
        for r in show:
            st = r.get('size_tag', '?')
            text.append(f"  - [{st}] {r['terrain_base']}  (rrt_found={r['rrt_found']}, moead_ms={r['moead_ms']:.1f})")

    slow_moead = sorted(valid_rows, key=lambda x: -x["moead_ms"])[:5]
    text.append("Top-5 slowest MOEA/D cases:")
    for r in slow_moead:
        st = r.get('size_tag', '?')
        text.append(f"  - [{st}] {r['terrain_base']}  moead_ms={r['moead_ms']:.1f}  archive={r['moead_archive_size']}")

    append_log(log_path, "\n".join(text))


def _maybe_parse_xy_list(v: Optional[list]) -> Optional[list]:
    if v is None:
        return None
    if isinstance(v, list) and len(v) == 2:
        return [float(v[0]), float(v[1])]
    return None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--terrain_dir", type=str, default="terrains", help="directory containing *.npz terrains")
    ap.add_argument("--out_root", type=str, default="outputs")
    ap.add_argument("--terrain_type", type=str, default="mountain", choices=["mountain", "city", "hill_city"], help="terrain type when --glob is not provided")
    ap.add_argument("--city_density", type=float, default=0.24, help="city density tag when terrain_type=city/hill_city and --glob is not provided")
    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"], help="mountain size when terrain_type=mountain and --glob is not provided")

    ap.add_argument("--seed_from", type=int, default=0)
    ap.add_argument("--seed_to", type=int, default=50, help="exclusive; used if no --glob")
    ap.add_argument("--glob", type=str, default="", help="optional terrain glob, e.g. 'terrains/*/mountain_seed*.npz'")

    ap.add_argument("--planner_seed", type=int, default=0)
    ap.add_argument("--planner_seeds", type=int, nargs="+", default=None, help="optional list of planner seeds; if provided, overrides --planner_seed")
    ap.add_argument("--num_terrains", type=int, default=None, help="optional alias for seed_to-seed_from when not using --glob")
    ap.add_argument("--inflate", type=float, default=5.0, help="inflate obstacle radius in meters")

    ap.add_argument("--rrt_iter", type=int, default=4000)
    ap.add_argument("--prm_samples", type=int, default=1200)
    ap.add_argument("--prm_k", type=int, default=12)
    ap.add_argument("--prm_max_edge_len", type=float, default=30.0)
    ap.add_argument("--prm_threat_weight", type=float, default=0.0)

    ap.add_argument("--moead_min_gen", type=int, default=20)
    ap.add_argument("--moead_max_gen", type=int, default=None)
    ap.add_argument("--mtoe_tol_fun", type=float, default=1e-5)
    ap.add_argument("--mtoe_confidence", type=float, default=0.99)
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)

    # MOEA/D init seeding params (A* 多样化初始解控制)
    ap.add_argument("--moead_T", type=int, default=10, help="MOEA/D neighborhood size T")
    ap.add_argument("--init_astar_ratio", type=float, default=0.25, help="fraction of population initialized from A* seeding")
    ap.add_argument("--init_astar_threat_weight", type=float, default=0.0, help="A* cost threat weight for seeding (0=ignore threat)")
    ap.add_argument("--init_astar_jitter_sigma", type=float, default=1.5, help="std of Gaussian jitter for A* path points")
    ap.add_argument("--init_astar_max_paths", type=int, default=5, help="max number of diverse A* backbone paths to generate")
    ap.add_argument("--init_astar_penalty_step", type=float, default=2.5, help="penalty added on visited cells after each A* to encourage diversity")
    ap.add_argument("--init_stratified_ratio", type=float, default=0.60, help="fraction of population initialized by stratified corridor sampling")
    ap.add_argument("--init_stratified_lateral_frac", type=float, default=0.30, help="half-width of stratified corridor as a fraction of start-goal distance")
    ap.add_argument("--init_stratified_n_bands", type=int, default=5, help="number of lateral bands for stratified sampling")
    ap.add_argument("--init_stratified_progress_jitter", type=float, default=0.08, help="jitter on normalized progress of each control point")
    ap.add_argument("--init_global_random_ratio", type=float, default=0.15, help="small exploratory tail kept as fully random/noisy individuals")
    ap.add_argument("--weight_extreme_bias", type=float, default=0.20, help="share of MOEA/D weights reserved near single-objective extremes")

    # Extreme-direction resource allocation + directional local search
    ap.add_argument("--extreme_offspring_ratio", type=float, default=0.20, help="extra offspring ratio allocated adaptively to extreme directions")
    ap.add_argument("--extreme_potential_window", type=int, default=20, help="window size for estimating per-objective improvement potential")
    ap.add_argument("--extreme_min_extra_per_obj", type=int, default=1, help="minimum extra offspring reserved for each objective when extra budget is active")
    ap.add_argument("--extreme_max_frac_per_obj", type=float, default=0.60, help="maximum fraction of extra budget assigned to any single objective")
    ap.add_argument("--local_search_interval", type=int, default=10, help="run directional local search every N generations (0 to disable)")
    ap.add_argument("--local_search_elite_k", type=int, default=3, help="top-k archive elites per objective used for directional local search")
    ap.add_argument("--local_search_attempts_per_obj", type=int, default=2, help="directional local-search attempts per objective each trigger")
    ap.add_argument("--archive_size", type=int, default=0, help="MOEA/D archive upper bound (>0 uses hard cap; <=0 falls back to archive_soft_limit)")
    ap.add_argument("--archive_soft_limit", type=int, default=320, help="soft archive cap used when archive_size<=0; improves Pareto spread")
    ap.add_argument("--archive_grid_bins", type=int, default=0, help="objective-space grid bins for diversity-aware archive truncation (0=auto)")
    ap.add_argument("--archive_keep_extremes", type=int, default=1, help="protect objective extremes during archive truncation (1/0)")
    ap.add_argument("--active_subproblem_ratio", type=float, default=1.0, help="fraction of subproblems activated each generation (0,1]")
    ap.add_argument("--utility_update_interval", type=int, default=3, help="update MOEA/D utility every N generations")
    ap.add_argument("--utility_use_archive_density", type=int, default=0, help="whether to mix archive density into utility update (1/0)")
    ap.add_argument("--log_flush_every", type=int, default=10, help="flush MOEA/D debug log every N generations")

    # --- MOEA/D debug + performance knobs ---
    ap.add_argument("--moead_debug", action="store_true", help="write MOEA/D debug log to out_dir/moead_debug.log and MTOE stats to mtoe_debug_*.log")
    ap.add_argument("--moead_debug_every", type=int, default=1, help="log every N generations (default: 1)")
    ap.add_argument("--moead_debug_level", type=int, default=2, help="1=coarse, 2=timing breakdown, 3=collision profiling")
    ap.add_argument(
        "--moead_eval_step",
        type=float,
        default=2.5,
        help="sample step for evaluation (collision + threat), in meters. Larger => much faster but less precise.",
    )
    ap.add_argument(
        "--moead_smooth_step",
        type=float,
        default=2.5,
        help="sample step for shortcut-smooth collision checks, in meters. Larger => faster.",
    )

    # start/goal：允许 AUTO（不传即自动按地图尺寸生成）
    ap.add_argument("--start", type=float, nargs=2, default=None, help="start point in meters: x_m y_m")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal point in meters: x_m y_m")

    ap.add_argument("--outdir", type=str, default=None, help="alias of --out_root")
    ap.add_argument("--summary_log", type=str, default="", help="append overall summary to this log file (default: out_root/benchmark_summary.log)")
    ap.add_argument(
        "--start_goal_z_offset",
        type=float,
        default=15.0,
        help="default start/goal altitude offset above local ground, in meters",
    )
    args = ap.parse_args()
    if args.moead_max_gen is None:
        args.moead_max_gen = 80

    if args.num_terrains is not None and not args.glob.strip():
        args.seed_to = int(args.seed_from) + int(max(0, args.num_terrains))
    if args.outdir is not None and str(args.outdir).strip():
        args.out_root = str(args.outdir).strip()
    planner_seeds = list(args.planner_seeds) if args.planner_seeds else [int(args.planner_seed)]

    if args.glob.strip():
        terrain_files = sorted(glob.glob(args.glob))
    else:
        if args.terrain_type in ("city", "hill_city"):
            terrain_subdir = os.path.join(args.terrain_dir, f"{args.terrain_type}_{args.city_density:.2f}")
            prefix = "city" if args.terrain_type == "city" else "hill_city"
            terrain_files = [
                os.path.join(terrain_subdir, f"{prefix}_seed{seed:04d}.npz")
                for seed in range(args.seed_from, args.seed_to)
            ]
        else:
            size_dir = {"small": "S", "medium": "M", "large": "L"}[args.size]
            terrain_subdir = os.path.join(args.terrain_dir, size_dir)
            terrain_files = [
                os.path.join(terrain_subdir, f"mountain_seed{seed:04d}.npz")
                for seed in range(args.seed_from, args.seed_to)
            ]

    if not terrain_files:
        raise RuntimeError("No terrain files found.")

    ensure_dir(args.out_root)

    summary_log = args.summary_log.strip()
    if not summary_log:
        summary_log = os.path.join(args.out_root, "benchmark_summary.log")

    astar_tag = astar_tag_from_args(args)
    moead_core_tag = moead_core_tag_from_args(args)

    all_rows = []

    for planner_seed in planner_seeds:
        for tp in terrain_files:
            if not os.path.exists(tp):
                print("[skip missing]", tp)
                continue
    
            base = os.path.basename(tp)
    
            terrain_seed = None
            if "seed" in base:
                try:
                    terrain_seed = int(base.split("seed")[-1].split(".")[0])
                except Exception:
                    terrain_seed = None
    
            env, height, meta = GridEnv.load_npz(tp)
            inflate_cells = int(np.ceil(args.inflate / env.resolution)) if args.inflate > 0 else 0
            if inflate_cells > 0:
                env = env.inflate_obstacles(inflate_cells)
    
            if isinstance(meta, dict):
                size_tag = str(meta.get("map_size", {}).get("tag", map_size_name(env.H, env.W)))
            else:
                size_tag = map_size_name(env.H, env.W)
    
            # 输出目录：按尺寸分桶，避免不同尺寸的同 seed 覆盖
            if terrain_seed is None:
                out_dir = os.path.join(args.out_root, size_tag, os.path.splitext(base)[0])
            else:
                out_dir = os.path.join(args.out_root, size_tag, f"seed{terrain_seed:04d}")
            ensure_dir(out_dir)
    
            occupancy = getattr(env, "occupancy", None)
    
            # start/goal
            if args.start is None or args.goal is None:
                start, goal = default_start_goal_for_env(
                    env,
                    height,
                    z_offset_m=args.start_goal_z_offset,
                )
            else:
                sx_m, sy_m = float(args.start[0]), float(args.start[1])
                gx_m, gy_m = float(args.goal[0]), float(args.goal[1])
    
                sx = meters_to_cells(env, sx_m)
                sy = meters_to_cells(env, sy_m)
                gx = meters_to_cells(env, gx_m)
                gy = meters_to_cells(env, gy_m)
    
                sx_i, sy_i = int(round(sx)), int(round(sy))
                gx_i, gy_i = int(round(gx)), int(round(gy))
    
                sx_i = int(np.clip(sx_i, 0, env.W - 1))
                sy_i = int(np.clip(sy_i, 0, env.H - 1))
                gx_i = int(np.clip(gx_i, 0, env.W - 1))
                gy_i = int(np.clip(gy_i, 0, env.H - 1))
    
                sz = float(height[sy_i, sx_i]) + args.start_goal_z_offset
                gz = float(height[gy_i, gx_i]) + args.start_goal_z_offset
    
                start = np.array([float(sx_i), float(sy_i), sz], dtype=np.float32)
                goal = np.array([float(gx_i), float(gy_i), gz], dtype=np.float32)
    
            # ---- invalid case handling ----
            # 起降点不再直接因为 occupancy 判 invalid；这里只检查越界和是否低于地表
            invalid_start = False
            invalid_goal = False
            reason = []
    
            sx, sy = int(round(float(start[0]))), int(round(float(start[1])))
            gx, gy = int(round(float(goal[0]))), int(round(float(goal[1])))
    
            if sx < 0 or sx >= env.W or sy < 0 or sy >= env.H:
                invalid_start = True
                reason.append("start_oob")
            else:
                if float(start[2]) < float(height[sy, sx]):
                    invalid_start = True
                    reason.append("start_below_ground")
    
            if gx < 0 or gx >= env.W or gy < 0 or gy >= env.H:
                invalid_goal = True
                reason.append("goal_oob")
            else:
                if float(goal[2]) < float(height[gy, gx]):
                    invalid_goal = True
                    reason.append("goal_below_ground")
    
            invalid_case = bool(invalid_start or invalid_goal)
            if invalid_case:
                # still write a metrics json for traceability, but skip planning.
                base_noext = base.replace(".npz", "")
                file_tag = f"{size_tag}{moead_core_tag}{astar_tag}"
                out_json = os.path.join(out_dir, f"metrics_{base_noext}_pseed{planner_seed:04d}_{file_tag}.json")
                moead_metric_block, moead_debug_payload = build_moead_metrics_block(
                    args=args,
                    log=None,
                    runtime_ms=0.0,
                    archive_size=0,
                )
                data = {
                    "terrain_file": tp,
                    "terrain_seed": terrain_seed,
                    "planner_seed": planner_seed,
                    "inflate": args.inflate,
                    "map_size": {"H": int(env.H), "W": int(env.W), "tag": size_tag},
                    "start": start.tolist(),
                    "goal": goal.tolist(),
                    "invalid_case": True,
                    "invalid": {
                        "start_invalid": bool(invalid_start),
                        "goal_invalid": bool(invalid_goal),
                        "reason": reason,
                    },
                    "rrt": {
                        "iter": args.rrt_iter,
                        "runtime_ms": 0.0,
                        "found": False,
                        "path_len": None,
                        "obj": None,
                        "feasible": False,
                        "violation": None,
                        "detail": None,
                    },
                    "prm": {
                        "samples": args.prm_samples,
                        "k": args.prm_k,
                        "max_edge_len": args.prm_max_edge_len,
                        "threat_weight": args.prm_threat_weight,
                        "runtime_ms": 0.0,
                        "found": False,
                        "path_len": None,
                        "obj": None,
                        "feasible": False,
                        "violation": None,
                        "detail": None,
                    },
                    "moead": moead_metric_block,
                    "meta": meta,
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
    
                all_rows.append({
                    "terrain_base": base,
                    "terrain_seed": terrain_seed,
                    "H": int(env.H),
                    "W": int(env.W),
                    "size_tag": size_tag,
                    "invalid_case": True,
                    "invalid_reason": reason,
                    "rrt_found": False,
                    "rrt_ms": 0.0,
                    "rrt_obj": [float("inf"), float("inf"), float("inf")],
                    "rrt_feasible": False,
                    "prm_found": False,
                    "prm_ms": 0.0,
                    "prm_obj": [float("inf"), float("inf"), float("inf")],
                    "prm_feasible": False,
                    "moead_ms": 0.0,
                    "moead_archive_size": 0,
                    "moead_stop_reason": "not_run",
                    "moead_n_gen": 0,
                    "moead_mtoe_last": None,
                    "rep_objs": None,
                })
                print("[skip-invalid]", base, "reason=", ",".join(reason), "->", out_dir)
                continue
    
            eval_step_cells = args.moead_eval_step / env.resolution
            smooth_step_cells = args.moead_smooth_step / env.resolution
    
            # ---- RRT* ----
            t0 = time.time()
            path_rrt, _ = rrt_star(env, start, goal, n_iter=args.rrt_iter, seed=planner_seed)
            rrt_ms = (time.time() - t0) * 1000.0
    
            # RRT* objectives (match MOEA/D evaluator):
            # - if no path: set to +INF for easy downstream stats / plotting
            # - if path exists: evaluate (also record feasibility + violation for debugging)
            if path_rrt is not None:
                rrt_er = evaluate_path(env, path_rrt, sample_step=eval_step_cells)
                rrt_obj = [float(rrt_er.obj[0]), float(rrt_er.obj[1]), float(rrt_er.obj[2])]
                rrt_feasible = bool(rrt_er.feasible)
                rrt_violation = float(rrt_er.violation)
                rrt_detail = dict(rrt_er.detail)
            else:
                rrt_er = None
                rrt_obj = [float("inf"), float("inf"), float("inf")]
                rrt_feasible = False
                rrt_violation = float("inf")
                rrt_detail = None
    
            # ---- PRM ----
            t0b = time.time()
            path_prm, prm_graph = prm(
                env, start, goal,
                n_samples=args.prm_samples,
                k=args.prm_k,
                max_edge_len=args.prm_max_edge_len,
                threat_weight=args.prm_threat_weight,
                seed=planner_seed,
            )
            prm_ms = (time.time() - t0b) * 1000.0
    
            if path_prm is not None:
                prm_er = evaluate_path(env, path_prm, sample_step=eval_step_cells)
                prm_obj = [float(prm_er.obj[0]), float(prm_er.obj[1]), float(prm_er.obj[2])]
                prm_feasible = bool(prm_er.feasible)
                prm_violation = float(prm_er.violation)
                prm_detail = dict(prm_er.detail)
            else:
                path_prm = None
                prm_er = None
                prm_obj = [float("inf"), float("inf"), float("inf")]
                prm_feasible = False
                prm_violation = float("inf")
                prm_detail = None
    
            prm_stats = getattr(prm_graph, "stats", {}) if prm_graph is not None else {}
    
            # ---- MOEA/D ----
            t1 = time.time()
            moead_debug_log = os.path.join(out_dir, "moead_debug.log") if args.moead_debug else None
            _, arch, log = moead(
                env, start, goal,
                n_gen=args.moead_max_gen,
                moead_min_gen=args.moead_min_gen,
                moead_max_gen=args.moead_max_gen,
                mtoe_tol_fun=args.mtoe_tol_fun,
                mtoe_confidence=args.mtoe_confidence,
                pop=args.moead_pop,
                K=args.K,
                T=args.moead_T,
                seed=planner_seed,
                eval_sample_step=eval_step_cells,
                smooth_collision_step=smooth_step_cells,
                debug_log_path=moead_debug_log,
                debug_every=args.moead_debug_every,
                debug_level=args.moead_debug_level,
                init_astar_ratio=args.init_astar_ratio,
                init_astar_threat_weight=args.init_astar_threat_weight,
                init_astar_jitter_sigma=args.init_astar_jitter_sigma,
                init_astar_max_paths=args.init_astar_max_paths,
                init_astar_penalty_step=args.init_astar_penalty_step,
                init_stratified_ratio=args.init_stratified_ratio,
                init_stratified_lateral_frac=args.init_stratified_lateral_frac,
                init_stratified_n_bands=args.init_stratified_n_bands,
                init_stratified_progress_jitter=args.init_stratified_progress_jitter,
                init_global_random_ratio=args.init_global_random_ratio,
                weight_extreme_bias=args.weight_extreme_bias,
                extreme_offspring_ratio=args.extreme_offspring_ratio,
                extreme_potential_window=args.extreme_potential_window,
                extreme_min_extra_per_obj=args.extreme_min_extra_per_obj,
                extreme_max_frac_per_obj=args.extreme_max_frac_per_obj,
                local_search_interval=args.local_search_interval,
                local_search_elite_k=args.local_search_elite_k,
                local_search_attempts_per_obj=args.local_search_attempts_per_obj,
                archive_size=args.archive_size,
                archive_soft_limit=args.archive_soft_limit,
                archive_grid_bins=args.archive_grid_bins,
                archive_keep_extremes=bool(args.archive_keep_extremes),
                active_subproblem_ratio=args.active_subproblem_ratio,
            )
            moead_ms = (time.time() - t1) * 1000.0
    
            reps = None
            pareto_objs = None
            if len(arch.items) > 0:
                reps = select_representatives(arch.items, weights=(1.0, 1.0, 1.0))
                pareto_objs = reps["objs"]
    
            # ===== 保存可视化关键数据（不在 benchmark 阶段画图） =====
            base_noext = base.replace(".npz", "")
            file_tag = f"{size_tag}{moead_core_tag}{astar_tag}"
            vis_json = os.path.join(out_dir, f"visdata_{base_noext}_pseed{planner_seed:04d}_{file_tag}.json")
            title = f"Paths | {base_noext} size={size_tag} gen={args.moead_min_gen}-{args.moead_max_gen} pop={args.moead_pop} K={args.K} planner_seed={planner_seed}"
            vis_payload = build_vis_payload(
                terrain_file=tp,
                terrain_seed=terrain_seed,
                planner_seed=planner_seed,
                size_tag=size_tag,
                inflate=args.inflate,
                map_hw=(env.H, env.W),
                start=start,
                goal=goal,
                path_rrt=path_rrt,
                path_prm=path_prm,
                arch=arch,
                reps=reps,
                title=title,
                meta=meta,
                extra={
                    "planner_args": {
                        "rrt_iter": args.rrt_iter,
                        "prm_samples": args.prm_samples,
                        "prm_k": args.prm_k,
                        "prm_max_edge_len": args.prm_max_edge_len,
                        "prm_threat_weight": args.prm_threat_weight,
                        "moead_max_gen": args.moead_max_gen,
                        "moead_min_gen": args.moead_min_gen,
                        "moead_max_gen": args.moead_max_gen,
                        "mtoe_tol_fun": args.mtoe_tol_fun,
                        "mtoe_confidence": args.mtoe_confidence,
                        "moead_pop": args.moead_pop,
                        "moead_T": args.moead_T,
                        "K": args.K,
                        "init_astar_ratio": args.init_astar_ratio,
                        "init_astar_threat_weight": args.init_astar_threat_weight,
                        "init_astar_jitter_sigma": args.init_astar_jitter_sigma,
                        "init_astar_max_paths": args.init_astar_max_paths,
                        "init_astar_penalty_step": args.init_astar_penalty_step,
                    "init_stratified_ratio": args.init_stratified_ratio,
                    "init_stratified_lateral_frac": args.init_stratified_lateral_frac,
                    "init_stratified_n_bands": args.init_stratified_n_bands,
                    "init_stratified_progress_jitter": args.init_stratified_progress_jitter,
                    "init_global_random_ratio": args.init_global_random_ratio,
                    "weight_extreme_bias": args.weight_extreme_bias,
                        "init_stratified_ratio": args.init_stratified_ratio,
                        "init_stratified_lateral_frac": args.init_stratified_lateral_frac,
                        "init_stratified_n_bands": args.init_stratified_n_bands,
                        "init_stratified_progress_jitter": args.init_stratified_progress_jitter,
                        "init_global_random_ratio": args.init_global_random_ratio,
                        "weight_extreme_bias": args.weight_extreme_bias,
                    },
                    "runtime_ms": {
                        "rrt": float(rrt_ms),
                        "prm": float(prm_ms),
                        "moead": float(moead_ms),
                    },
                },
            )
            save_vis_payload(vis_json, vis_payload)
    
            out_json = os.path.join(out_dir, f"metrics_{base_noext}_pseed{planner_seed:04d}_{file_tag}.json")
    
            moead_metric_block, moead_debug_payload = build_moead_metrics_block(
                args=args,
                log=log,
                runtime_ms=moead_ms,
                archive_size=len(arch.items),
            )
            data = {
                "terrain_file": tp,
                "terrain_seed": terrain_seed,
                "planner_seed": planner_seed,
                "inflate": args.inflate,
                "map_size": {"H": int(env.H), "W": int(env.W), "tag": size_tag},
                "start": start.tolist(),
                "goal": goal.tolist(),
                "invalid_case": False,
                "invalid": {"start_invalid": False, "goal_invalid": False, "reason": []},
                "rrt": {
                    "iter": args.rrt_iter,
                    "runtime_ms": rrt_ms,
                    "found": path_rrt is not None,
                    "path_len": env.path_length(path_rrt) if path_rrt is not None else None,
                    # objectives aligned with MOEA/D: [f1=length, f2=threat, f3=energy approx]
                    # NOTE: json does not strictly support INF, so we use null when not found.
                    "obj": rrt_obj if path_rrt is not None else None,
                    "feasible": rrt_feasible if path_rrt is not None else False,
                    "violation": rrt_violation if path_rrt is not None else None,
                    "detail": rrt_detail if path_rrt is not None else None,
                },
                "prm": {
                    "samples": args.prm_samples,
                    "k": args.prm_k,
                    "max_edge_len": args.prm_max_edge_len,
                    "threat_weight": args.prm_threat_weight,
                    "runtime_ms": prm_ms,
                    "found": path_prm is not None,
                    "path_len": env.path_length(path_prm) if path_prm is not None else None,
                    "obj": prm_obj if path_prm is not None else None,
                    "feasible": prm_feasible if path_prm is not None else False,
                    "violation": prm_violation if path_prm is not None else None,
                    "detail": prm_detail if path_prm is not None else None,
                    "stats": prm_stats,
                },
                "moead": moead_metric_block,
                "meta": meta,
            }
    
            rep_objs = None
            if reps is not None:
                def obj_of(name):
                    idx = int(reps[name])
                    return [float(x) for x in arch.items[idx].er.obj]
    
                rep_objs = {
                    "min_f1": obj_of("min_f1"),
                    "min_f2": obj_of("min_f2"),
                    "min_f3": obj_of("min_f3"),
                    "compromise": obj_of("compromise"),
                }
                data["moead"]["representatives"] = rep_objs
    
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if args.moead_debug:
                debug_log = os.path.join(
                    out_dir,
                    os.path.basename(out_json).replace("metrics_", "mtoe_debug_").replace(".json", ".log")
                )
                write_mtoe_debug_log(
                    debug_log,
                    terrain_file=tp,
                    terrain_seed=terrain_seed,
                    planner_seed=args.planner_seed,
                    debug_payload=moead_debug_payload,
                )
    
            all_rows.append({
                "terrain_base": base,
                "terrain_seed": terrain_seed,
                "H": int(env.H),
                "W": int(env.W),
                "size_tag": size_tag,
                "invalid_case": False,
                "rrt_found": path_rrt is not None,
                "rrt_ms": float(rrt_ms),
                "rrt_obj": rrt_obj,
                "rrt_feasible": bool(rrt_feasible),
                "prm_found": path_prm is not None,
                "prm_ms": float(prm_ms),
                "prm_obj": prm_obj,
                "prm_feasible": bool(prm_feasible),
                "moead_ms": float(moead_ms),
                "moead_archive_size": int(len(arch.items)),
                "moead_stop_reason": log.get("stop_reason"),
                "moead_n_gen": int(log.get("n_gen", args.moead_max_gen)),
                "rep_objs": rep_objs,
            })
    
            print("[done]", base, "->", out_dir)
    
    summarize_results(all_rows, summary_log, args)

    print("Benchmark finished.")
    print("Summary appended to:", summary_log)


if __name__ == "__main__":
    main()
