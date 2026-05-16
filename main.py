# main.py

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from src.algorithms.nsga3 import nsga3
from src.viz.plot import plot_env
from src.experiment.path_planning import write_mtoe_debug_log


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def split_moead_log(log):
    log = log if isinstance(log, dict) else {}
    if not log:
        return {}, {}
    metric_keys = {
        "n_gen", "configured_n_gen", "requested_n_gen", "pop", "K", "T", "n_eval",
        "archive_size", "max_gen", "moead_min_gen", "mtoe_enabled", "mtoe_mode",
        "mtoe_tol_fun", "mtoe_confidence", "stop_reason", "mtoe_window", "ideal_point",
    }
    metric_log = {k: log[k] for k in metric_keys if k in log}
    debug_log = {k: v for k, v in log.items() if k not in metric_keys}
    return metric_log, debug_log


def map_size_to_hw(size: str):
    size = (size or "").lower()
    if size in ("s", "small"):
        return 160, 200
    if size in ("m", "medium"):
        # 中地图：小地图的 3 倍
        return 160 * 3, 200 * 3
    if size in ("l", "large"):
        # 大地图：小地图的 10 倍
        return 160 * 10, 200 * 10
    raise ValueError(f"Unknown size: {size} (use small/medium/large)")


def map_size_name(H: int, W: int) -> str:
    if (H, W) == (160, 200):
        return "S"
    if (H, W) == (160 * 3, 200 * 3):
        return "M"
    if (H, W) == (160 * 10, 200 * 10):
        return "L"
    return f"H{H}W{W}"


def default_start_goal(H: int, W: int, env: GridEnv | None = None):
    sx, sy = 5.0, 5.0
    gx, gy = float(max(0, W - 60)), float(max(0, H - 70))
    gx = max(gx, sx + 10.0)
    gy = max(gy, sy + 10.0)
    if env is None:
        return np.array([sx, sy], dtype=np.float32), np.array([gx, gy], dtype=np.float32)
    global_cruise = float(np.percentile(env.height, 85) + env.min_clearance + 6.0)
    sz = max(env.altitude_from_ground(sx, sy), global_cruise)
    gz = max(env.altitude_from_ground(gx, gy), global_cruise)
    return np.array([sx, sy, sz], dtype=np.float32), np.array([gx, gy, gz], dtype=np.float32)


def select_representatives(arch_items, weights=(1.0, 1.0, 1.0)):
    objs = np.array([it.er.obj for it in arch_items], dtype=float)  # [N,3]
    assert objs.ndim == 2 and objs.shape[1] == 3

    idx_f1 = int(np.argmin(objs[:, 0]))
    idx_f2 = int(np.argmin(objs[:, 1]))
    idx_f3 = int(np.argmin(objs[:, 2]))

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
    }


def astar_tag(init_astar_ratio, init_astar_max_paths, init_astar_penalty_step, init_astar_threat_weight, init_astar_jitter_sigma):
    return (
        f"_astarR{init_astar_ratio}"
        f"_M{init_astar_max_paths}"
        f"_P{init_astar_penalty_step}"
        f"_TW{init_astar_threat_weight}"
        f"_J{init_astar_jitter_sigma}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--terrain_seed", type=int, default=35)
    ap.add_argument("--planner_seed", type=int, default=0)

    ap.add_argument("--terrain_type", type=str, default="mountain", choices=["mountain", "city", "hill_city"])
    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"], help="mountain only")
    ap.add_argument("--terrace_levels", type=int, default=28)
    ap.add_argument("--city_density", type=float, default=0.24)
    ap.add_argument("--hill_scale", type=float, default=26.0)
    ap.add_argument("--inflate", type=int, default=1)

    # RRT*
    ap.add_argument("--rrt_iter", type=int, default=50000)

    # PRM
    ap.add_argument("--prm_samples", type=int, default=1200)
    ap.add_argument("--prm_k", type=int, default=12)
    ap.add_argument("--prm_max_edge_len", type=float, default=30.0)
    ap.add_argument("--prm_threat_weight", type=float, default=0.0)

    # MOEA/D core
    ap.add_argument("--moead_min_gen", type=int, default=20)
    ap.add_argument("--moead_max_gen", type=int, default=None)
    ap.add_argument("--mtoe_tol_fun", type=float, default=1e-5)
    ap.add_argument("--mtoe_confidence", type=float, default=0.995)
    ap.add_argument("--mtoe_window", type=int, default=10, help="MTOE early-stop rolling window size")
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--moead_T", type=int, default=10)
    ap.add_argument("--nsga3_max_gen", type=int, default=None)
    ap.add_argument("--nsga3_pop", type=int, default=None)
    ap.add_argument("--nsga3_ref_dirs", type=int, default=0)
    ap.add_argument("--nsga3_crossover_prob", type=float, default=0.90)
    ap.add_argument("--nsga3_mutation_prob", type=float, default=0.25)
    ap.add_argument("--nsga3_mutation_sigma", type=float, default=2.5)
    ap.add_argument("--skip_nsga3", action="store_true")

    # A* seeding（大图可增大 init_astar_max_expansions 以免 A* 未找到路径）
    ap.add_argument("--init_astar_ratio", type=float, default=0.25)
    ap.add_argument("--init_astar_threat_weight", type=float, default=0.0)
    ap.add_argument("--init_astar_jitter_sigma", type=float, default=1.5)
    ap.add_argument("--init_astar_max_paths", type=int, default=5)
    ap.add_argument("--init_astar_penalty_step", type=float, default=2.5)
    ap.add_argument("--init_astar_max_expansions", type=int, default=None, help="A* 扩展上限，默认按地图面积自动")
    ap.add_argument("--init_stratified_ratio", type=float, default=0.60)
    ap.add_argument("--init_stratified_lateral_frac", type=float, default=0.30)
    ap.add_argument("--init_stratified_n_bands", type=int, default=5)
    ap.add_argument("--init_stratified_progress_jitter", type=float, default=0.08)
    ap.add_argument("--init_global_random_ratio", type=float, default=0.15)
    ap.add_argument("--weight_extreme_bias", type=float, default=0.20)
    ap.add_argument("--extreme_offspring_ratio", type=float, default=0.20)
    ap.add_argument("--extreme_potential_window", type=int, default=20)
    ap.add_argument("--extreme_min_extra_per_obj", type=int, default=1)
    ap.add_argument("--extreme_max_frac_per_obj", type=float, default=0.70)
    ap.add_argument("--local_search_interval", type=int, default=10)
    ap.add_argument("--local_search_elite_k", type=int, default=3)
    ap.add_argument("--local_search_attempts_per_obj", type=int, default=2)
    ap.add_argument("--max_turn_deg", type=float, default=90.0, help="hard turn-angle limit in degrees")
    ap.add_argument("--soft_turn_deg", type=float, default=60.0, help="soft preferred turn-angle limit in degrees")
    ap.add_argument("--max_pitch_deg", type=float, default=35.0, help="hard pitch-angle limit in degrees")
    ap.add_argument("--soft_pitch_deg", type=float, default=25.0, help="soft preferred pitch-angle limit in degrees")
    ap.add_argument("--desired_clearance_margin", type=float, default=2.0, help="extra preferred clearance above min_clearance")
    ap.add_argument("--tau_soft", type=float, default=25.0, help="maximum allowed weighted soft-constraint violation")
    ap.add_argument("--archive_size", type=int, default=0, help="MOEA/D archive upper bound (>0 uses hard cap; <=0 falls back to archive_soft_limit)")
    ap.add_argument("--archive_soft_limit", type=int, default=320, help="soft archive cap used when archive_size<=0; improves Pareto spread")
    ap.add_argument("--archive_grid_bins", type=int, default=0, help="objective-space grid bins for diversity-aware archive truncation (0=auto)")
    ap.add_argument("--archive_keep_extremes", type=int, default=1, help="protect objective extremes during archive truncation (1/0)")
    ap.add_argument("--active_subproblem_ratio", type=float, default=1.0, help="fraction of subproblems activated each generation (0,1]")
    ap.add_argument("--utility_update_interval", type=int, default=3, help="update MOEA/D utility every N generations")
    ap.add_argument("--utility_use_archive_density", type=int, default=0, help="whether to mix archive density into utility update (1/0)")
    ap.add_argument("--log_flush_every", type=int, default=10, help="flush MOEA/D debug log every N generations")

    # start/goal：默认 AUTO（不传则按尺寸自动设）
    ap.add_argument("--start", type=float, nargs="+", default=None)
    ap.add_argument("--goal", type=float, nargs="+", default=None)

    ap.add_argument("--terrain_dir", type=str, default="terrains", help="where to save generated terrain npz")
    ap.add_argument("--out_root", type=str, default="outputs", help="where to save figures/logs")

    # Debug：MOEA/D 每代日志写入文件，并可实时输出到控制台（与 run_benchmark 一致）
    ap.add_argument("--debug", action="store_true", help="开启 MOEA/D 调试日志（写文件并实时打屏）")
    ap.add_argument("--debug_log", type=str, default=None, help="调试日志路径，默认在 --debug 时为 out_dir/mtoe_debug.log")
    ap.add_argument("--debug_every", type=int, default=1, help="每多少代打印一条日志")
    ap.add_argument("--debug_level", type=int, default=2, choices=[1, 2, 3], help="1=简要 2=含耗时 3=含碰撞统计")
    ap.add_argument("--no_debug_console", action="store_true", help="开启 --debug 时仅写文件、不实时打屏")

    args = ap.parse_args()
    if args.moead_max_gen is None:
        args.moead_max_gen = 80
    if args.nsga3_max_gen is None:
        args.nsga3_max_gen = args.moead_max_gen
    if args.nsga3_pop is None:
        args.nsga3_pop = args.moead_pop
    if int(args.nsga3_ref_dirs) <= 0:
        args.nsga3_ref_dirs = int(args.nsga3_pop)

    if args.terrain_type in ("city", "hill_city"):
        H, W = 1600, 2000
        size_tag = f"{args.terrain_type}_{args.city_density:.2f}"
    else:
        H, W = map_size_to_hw(args.size)
        size_tag = map_size_name(H, W)

    start = goal = None

    atag = astar_tag(
        args.init_astar_ratio,
        args.init_astar_max_paths,
        args.init_astar_penalty_step,
        args.init_astar_threat_weight,
        args.init_astar_jitter_sigma,
    )

    # 输出目录：按尺寸分桶
    out_dir = os.path.join(args.out_root, size_tag, f"seed{args.terrain_seed:04d}")
    ensure_dir(out_dir)
    ensure_dir(args.terrain_dir)

    # -----------------------------
    # Build / Save Terrain
    # -----------------------------
    if args.terrain_type == "city":
        env, height, meta = GridEnv.city_map(H=H, W=W, seed=args.terrain_seed, city_density=args.city_density)
    elif args.terrain_type == "hill_city":
        env, height, meta = GridEnv.hill_city_map(seed=args.terrain_seed, city_density=args.city_density, hill_scale=args.hill_scale)
    else:
        env, height, meta = GridEnv.mountain_map(H=H, W=W, seed=args.terrain_seed, terrace_levels=args.terrace_levels)
    env = env.inflate_obstacles(radius_cells=args.inflate)
    if args.start is None or args.goal is None:
        start, goal = default_start_goal(H, W, env)
    else:
        start = np.array(args.start, dtype=np.float32)
        goal = np.array(args.goal, dtype=np.float32)
        if start.shape[0] == 2:
            start = np.array([start[0], start[1], env.altitude_from_ground(start[0], start[1])], dtype=np.float32)
        if goal.shape[0] == 2:
            goal = np.array([goal[0], goal[1], env.altitude_from_ground(goal[0], goal[1])], dtype=np.float32)
        start = env.clamp_point(start)
        goal = env.clamp_point(goal)

    # terrain 保存到 terrains/{size_tag}/mountain_seedXXXX.npz
    terrain_subdir = os.path.join(args.terrain_dir, size_tag)
    ensure_dir(terrain_subdir)
    terrain_name = (
        f"city_seed{args.terrain_seed:04d}.npz" if args.terrain_type == "city" else
        f"hill_city_seed{args.terrain_seed:04d}.npz" if args.terrain_type == "hill_city" else
        f"mountain_seed{args.terrain_seed:04d}.npz"
    )
    terrain_path = os.path.join(terrain_subdir, terrain_name)
    env.save_npz(terrain_path, height=height, meta={**meta, "terrain_type": args.terrain_type, "city_density": (args.city_density if args.terrain_type in ("city", "hill_city") else None), "hill_scale": (args.hill_scale if args.terrain_type == "hill_city" else None), "map_size": {"H": H, "W": W, "tag": size_tag}})
    print("saved terrain:", terrain_path, flush=True)

    # -----------------------------
    # Plan（大图建议更多 rrt_iter，以利 RRT* 连通）
    # -----------------------------
    rrt_iter = args.rrt_iter
    if size_tag == "L" and rrt_iter == 50000:
        rrt_iter = 200_000
    elif size_tag == "M" and rrt_iter == 50000:
        rrt_iter = 100_000
    path_rrt, _ = rrt_star(env, start, goal, n_iter=rrt_iter, seed=args.planner_seed)
    path_prm, prm_graph = prm(
        env, start, goal,
        n_samples=args.prm_samples,
        k=args.prm_k,
        max_edge_len=args.prm_max_edge_len,
        threat_weight=args.prm_threat_weight,
        seed=args.planner_seed,
    )
    if prm_graph is not None and getattr(prm_graph, "stats", None):
        print("PRM stats:", prm_graph.stats, flush=True)

    debug_log_path = None
    if args.debug or (args.debug_log is not None and args.debug_log.strip()):
        debug_log_path = (args.debug_log or "").strip() or os.path.join(out_dir, "mtoe_debug.log")
    # 与 run_benchmark 一致：开启 debug 时默认实时打屏，--no_debug_console 则仅写文件
    debug_console = bool(debug_log_path) and not args.no_debug_console

    _, arch, log = moead(
        env, start, goal,
        n_gen=args.moead_max_gen, pop=args.moead_pop, K=args.K, T=args.moead_T, seed=args.planner_seed,
        max_turn_deg=args.max_turn_deg,
        soft_turn_deg=args.soft_turn_deg,
        max_pitch_deg=args.max_pitch_deg,
        soft_pitch_deg=args.soft_pitch_deg,
        desired_clearance_margin=args.desired_clearance_margin,
        tau_soft=args.tau_soft,
        moead_min_gen=args.moead_min_gen,
        moead_max_gen=args.moead_max_gen,
        mtoe_tol_fun=args.mtoe_tol_fun,
        mtoe_confidence=args.mtoe_confidence,
        mtoe_window=args.mtoe_window,
        init_astar_ratio=args.init_astar_ratio,
        init_astar_threat_weight=args.init_astar_threat_weight,
        init_astar_jitter_sigma=args.init_astar_jitter_sigma,
        init_astar_max_paths=args.init_astar_max_paths,
        init_astar_penalty_step=args.init_astar_penalty_step,
        init_astar_max_expansions=args.init_astar_max_expansions,
        init_stratified_ratio=getattr(args, "init_stratified_ratio", 0.60),
        init_stratified_lateral_frac=getattr(args, "init_stratified_lateral_frac", 0.30),
        init_stratified_n_bands=getattr(args, "init_stratified_n_bands", 5),
        init_stratified_progress_jitter=getattr(args, "init_stratified_progress_jitter", 0.08),
        init_global_random_ratio=getattr(args, "init_global_random_ratio", 0.15),
        weight_extreme_bias=getattr(args, "weight_extreme_bias", 0.20),
        extreme_offspring_ratio=getattr(args, "extreme_offspring_ratio", 0.20),
        extreme_potential_window=getattr(args, "extreme_potential_window", 20),
        extreme_min_extra_per_obj=getattr(args, "extreme_min_extra_per_obj", 1),
        extreme_max_frac_per_obj=getattr(args, "extreme_max_frac_per_obj", 0.70),
        local_search_interval=getattr(args, "local_search_interval", 10),
        local_search_elite_k=getattr(args, "local_search_elite_k", 3),
        local_search_attempts_per_obj=getattr(args, "local_search_attempts_per_obj", 2),
        archive_size=getattr(args, "archive_size", 0),
        archive_soft_limit=getattr(args, "archive_soft_limit", 320),
        archive_grid_bins=getattr(args, "archive_grid_bins", 0),
        archive_keep_extremes=bool(getattr(args, "archive_keep_extremes", 1)),
        active_subproblem_ratio=getattr(args, "active_subproblem_ratio", 1.0),
        utility_update_interval=getattr(args, "utility_update_interval", 3),
        utility_use_archive_density=bool(getattr(args, "utility_use_archive_density", 0)),
        log_flush_every=getattr(args, "log_flush_every", 10),
        debug_log_path=debug_log_path,
        debug_every=args.debug_every,
        debug_level=args.debug_level,
        debug_console=debug_console,
    )

    nsga3_arch = None
    nsga3_log = {"stop_reason": "skipped"}
    if not args.skip_nsga3:
        _, nsga3_arch, nsga3_log = nsga3(
            env,
            start,
            goal,
            n_gen=args.nsga3_max_gen,
            pop=args.nsga3_pop,
            K=args.K,
            seed=args.planner_seed,
            ref_dirs_count=args.nsga3_ref_dirs,
            crossover_prob=args.nsga3_crossover_prob,
            mutation_prob=args.nsga3_mutation_prob,
            mutation_sigma=args.nsga3_mutation_sigma,
            max_turn_deg=args.max_turn_deg,
            soft_turn_deg=args.soft_turn_deg,
            max_pitch_deg=args.max_pitch_deg,
            soft_pitch_deg=args.soft_pitch_deg,
            desired_clearance_margin=args.desired_clearance_margin,
            tau_soft=args.tau_soft,
            init_astar_ratio=args.init_astar_ratio,
            init_astar_threat_weight=args.init_astar_threat_weight,
            init_astar_jitter_sigma=args.init_astar_jitter_sigma,
            init_astar_max_paths=args.init_astar_max_paths,
            init_astar_penalty_step=args.init_astar_penalty_step,
            init_astar_max_expansions=args.init_astar_max_expansions,
            init_stratified_ratio=getattr(args, "init_stratified_ratio", 0.60),
            init_stratified_lateral_frac=getattr(args, "init_stratified_lateral_frac", 0.30),
            init_stratified_n_bands=getattr(args, "init_stratified_n_bands", 5),
            init_stratified_progress_jitter=getattr(args, "init_stratified_progress_jitter", 0.08),
            init_global_random_ratio=getattr(args, "init_global_random_ratio", 0.15),
            weight_extreme_bias=getattr(args, "weight_extreme_bias", 0.20),
            archive_size=getattr(args, "archive_size", 0),
            archive_soft_limit=getattr(args, "archive_soft_limit", 320),
            archive_grid_bins=getattr(args, "archive_grid_bins", 0),
            archive_keep_extremes=bool(getattr(args, "archive_keep_extremes", 1)),
        )

    # 保存 log（文件名带 A* 参数）；debug 细节单独放到 *_debug*.json
    metric_log, debug_log = split_moead_log(log)
    log_path = os.path.join(out_dir, f"log_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}{atag}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "terrain_seed": args.terrain_seed,
            "planner_seed": args.planner_seed,
            "map_size": {"H": H, "W": W, "tag": size_tag},
            "terrace_levels": args.terrace_levels,
            "inflate": args.inflate,
            "moead_min_gen": args.moead_min_gen,
            "moead_max_gen": args.moead_max_gen,
            "mtoe_tol_fun": args.mtoe_tol_fun,
            "mtoe_confidence": args.mtoe_confidence,
            "mtoe_window": args.mtoe_window,
            "archive_size": args.archive_size,
            "archive_soft_limit": args.archive_soft_limit,
            "archive_grid_bins": args.archive_grid_bins,
            "archive_keep_extremes": int(args.archive_keep_extremes),
            "max_turn_deg": args.max_turn_deg,
            "soft_turn_deg": args.soft_turn_deg,
            "max_pitch_deg": args.max_pitch_deg,
            "soft_pitch_deg": args.soft_pitch_deg,
            "desired_clearance_margin": args.desired_clearance_margin,
            "tau_soft": args.tau_soft,
            "utility_update_interval": args.utility_update_interval,
            "utility_use_archive_density": int(args.utility_use_archive_density),
            "log_flush_every": args.log_flush_every,
            "active_subproblem_ratio": args.active_subproblem_ratio,
            "start": start.tolist(),
            "goal": goal.tolist(),
            "rrt_iter": args.rrt_iter,
            "prm": {
                "samples": args.prm_samples,
                "k": args.prm_k,
                "max_edge_len": args.prm_max_edge_len,
                "threat_weight": args.prm_threat_weight,
                "found": path_prm is not None,
            },
            "moead": {
                "n_gen": args.moead_max_gen,
                "pop": args.moead_pop,
                "K": args.K,
                "T": args.moead_T,
                "init_astar_ratio": args.init_astar_ratio,
                "init_astar_threat_weight": args.init_astar_threat_weight,
                "init_astar_jitter_sigma": args.init_astar_jitter_sigma,
                "init_astar_max_paths": args.init_astar_max_paths,
                "init_astar_penalty_step": args.init_astar_penalty_step,
            },
            "nsga3": {
                "n_gen": args.nsga3_max_gen,
                "pop": args.nsga3_pop,
                "K": args.K,
                "ref_dirs": args.nsga3_ref_dirs,
                "crossover_prob": args.nsga3_crossover_prob,
                "mutation_prob": args.nsga3_mutation_prob,
                "mutation_sigma": args.nsga3_mutation_sigma,
                "stop_reason": nsga3_log.get("stop_reason") if isinstance(nsga3_log, dict) else None,
                "archive_size": len(nsga3_arch.items) if nsga3_arch is not None else 0,
            },
            "meta": meta,
            "log": metric_log,
            "archive_size": len(arch.items),
        }, f, ensure_ascii=False, indent=2)
    if debug_log:
        debug_log_path = os.path.join(out_dir, f"mtoe_debug_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}{atag}.log")
        write_mtoe_debug_log(
            debug_log_path,
            terrain_seed=args.terrain_seed,
            planner_seed=args.planner_seed,
            debug_payload=debug_log,
        )
        print("saved debug log:", debug_log_path, flush=True)
    print("saved log:", log_path, flush=True)

    # -----------------------------
    # Select representative MOEA/D paths
    # -----------------------------
    reps = None
    if len(arch.items) > 0:
        reps = select_representatives(arch.items, weights=(1.0, 1.0, 1.0))
        objs = reps["objs"]
        print("archive size:", len(arch.items), flush=True)
        print("objs min:", objs.min(axis=0), flush=True)
        print("objs max:", objs.max(axis=0), flush=True)

    nsga3_reps = None
    if nsga3_arch is not None and len(nsga3_arch.items) > 0:
        nsga3_reps = select_representatives(nsga3_arch.items, weights=(1.0, 1.0, 1.0))
        nsga3_objs = nsga3_reps["objs"]
        print("NSGA-III archive size:", len(nsga3_arch.items), flush=True)
        print("NSGA-III objs min:", nsga3_objs.min(axis=0), flush=True)
        print("NSGA-III objs max:", nsga3_objs.max(axis=0), flush=True)

    # -----------------------------
    # Figure 1: Paths
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_env(env, ax=ax)

    occ = env.occupancy.astype(np.float32)
    mask = (occ >= 0.5)
    overlay = np.zeros_like(occ, dtype=np.float32)
    overlay[mask] = 1.0
    ax.imshow(
        np.ma.masked_where(~mask, overlay),
        origin="lower",
        cmap="Reds",
        alpha=0.35,
        interpolation="nearest",
    )
    ax.contour(mask.astype(np.int32), levels=[0.5], colors="red", linewidths=1.0, origin="lower")

    ax.scatter([start[0]], [start[1]], marker="o", s=90, label="Start")
    ax.scatter([goal[0]],  [goal[1]],  marker="*", s=160, label="Goal")

    if path_rrt is not None:
        ax.plot(path_rrt[:, 0], path_rrt[:, 1], linewidth=2.2, label="RRT*")

    if path_prm is not None:
        ax.plot(path_prm[:, 0], path_prm[:, 1], linewidth=2.0, linestyle="--", label="PRM")

    if reps is not None:
        idx_map = {
            "MOEA/D min f1": reps["min_f1"],
            "MOEA/D min f2": reps["min_f2"],
            "MOEA/D min f3": reps["min_f3"],
            "MOEA/D compromise": reps["compromise"],
        }
        for label, idx in idx_map.items():
            path = arch.items[idx].x
            ax.plot(path[:, 0], path[:, 1], linewidth=2.0, label=label)

    if nsga3_reps is not None and nsga3_arch is not None:
        idx_map = {
            "NSGA-III min f1": nsga3_reps["min_f1"],
            "NSGA-III min f2": nsga3_reps["min_f2"],
            "NSGA-III min f3": nsga3_reps["min_f3"],
            "NSGA-III compromise": nsga3_reps["compromise"],
        }
        for label, idx in idx_map.items():
            path = nsga3_arch.items[idx].x
            ax.plot(path[:, 0], path[:, 1], linewidth=1.8, linestyle=":", label=label)

    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Paths | size={size_tag} seed={args.terrain_seed} pseed={args.planner_seed} (red=no-fly)")
    plt.tight_layout()

    path_png = os.path.join(out_dir, f"paths_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}_inf{args.inflate}_K{args.K}{atag}.png")
    plt.savefig(path_png, dpi=220)
    plt.close(fig)
    print("saved figure:", path_png, flush=True)

    # -----------------------------
    # Figure 2: Pareto 3D
    # -----------------------------
    if reps is not None or nsga3_reps is not None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        if reps is not None:
            objs = reps["objs"]
            ax2.scatter(objs[:, 0], objs[:, 1], objs[:, 2], s=12, label="MOEA/D")
        if nsga3_reps is not None:
            nsga3_objs = nsga3_reps["objs"]
            ax2.scatter(nsga3_objs[:, 0], nsga3_objs[:, 1], nsga3_objs[:, 2], s=12, marker="^", label="NSGA-III")

        ax2.set_xlabel("f1: length")
        ax2.set_ylabel("f2: threat")
        ax2.set_zlabel("f3: energy")
        ax2.set_title(f"Pareto 3D | size={size_tag} seed={args.terrain_seed} pseed={args.planner_seed}")
        ax2.legend(loc="best")

        plt.tight_layout()
        pareto_png = os.path.join(out_dir, f"pareto3d_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}_inf{args.inflate}_K{args.K}{atag}.png")
        plt.savefig(pareto_png, dpi=220)
        plt.close(fig2)
        print("saved figure:", pareto_png, flush=True)


if __name__ == "__main__":
    main()
