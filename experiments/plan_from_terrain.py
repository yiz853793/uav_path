import argparse
import os
import numpy as np

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from src.experiment.benchmark_shared import default_start_goal_for_env, meters_to_cells, split_moead_log
from src.experiment.vis_data import build_vis_payload, save_vis_payload



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", type=str, required=True, help="path to .npz terrain file")
    ap.add_argument("--planner", type=str, default="rrt", choices=["rrt", "prm", "moead", "all"])
    ap.add_argument("--inflate", type=float, default=5.0, help="inflate obstacles radius in meters")
    ap.add_argument("--seed", type=int, default=0, help="planner seed (NOT terrain seed)")
    ap.add_argument("--rrt_iter", type=int, default=4000)
    ap.add_argument("--prm_samples", type=int, default=1200)
    ap.add_argument("--prm_k", type=int, default=12)
    ap.add_argument("--prm_max_edge_len", type=float, default=30.0)
    ap.add_argument("--prm_threat_weight", type=float, default=0.0)
    ap.add_argument("--moead_min_gen", type=int, default=20)
    ap.add_argument("--moead_max_gen", type=int, default=None)
    ap.add_argument("--mtoe_tol_fun", type=float, default=1e-5)
    ap.add_argument("--mtoe_confidence", type=float, default=0.995)
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--moead_T", type=int, default=10)
    ap.add_argument("--out_json", type=str, default="", help="where to save visualization json")
    ap.add_argument("--init_astar_ratio", type=float, default=0.25)
    ap.add_argument("--init_astar_threat_weight", type=float, default=0.0)
    ap.add_argument("--init_astar_jitter_sigma", type=float, default=1.5)
    ap.add_argument("--init_astar_max_paths", type=int, default=5)
    ap.add_argument("--init_astar_penalty_step", type=float, default=2.5)
    ap.add_argument("--init_stratified_ratio", type=float, default=0.60)
    ap.add_argument("--init_stratified_lateral_frac", type=float, default=0.30)
    ap.add_argument("--init_stratified_n_bands", type=int, default=5)
    ap.add_argument("--init_stratified_progress_jitter", type=float, default=0.08)
    ap.add_argument("--init_global_random_ratio", type=float, default=0.15)
    ap.add_argument("--weight_extreme_bias", type=float, default=0.20)
    ap.add_argument("--extreme_offspring_ratio", type=float, default=0.20)
    ap.add_argument("--extreme_potential_window", type=int, default=20)
    ap.add_argument("--extreme_min_extra_per_obj", type=int, default=1)
    ap.add_argument("--extreme_max_frac_per_obj", type=float, default=0.60)
    ap.add_argument("--local_search_interval", type=int, default=10)
    ap.add_argument("--local_search_elite_k", type=int, default=3)
    ap.add_argument("--local_search_attempts_per_obj", type=int, default=2)
    ap.add_argument("--archive_size", type=int, default=0, help="MOEA/D archive upper bound (>0 uses hard cap; <=0 falls back to archive_soft_limit)")
    ap.add_argument("--archive_soft_limit", type=int, default=320, help="soft archive cap used when archive_size<=0; improves Pareto spread")
    ap.add_argument("--archive_grid_bins", type=int, default=0, help="objective-space grid bins for diversity-aware archive truncation (0=auto)")
    ap.add_argument("--archive_keep_extremes", type=int, default=1, help="protect objective extremes during archive truncation (1/0)")
    ap.add_argument("--active_subproblem_ratio", type=float, default=1.0, help="fraction of subproblems activated each generation (0,1]")
    ap.add_argument("--utility_update_interval", type=int, default=3, help="update MOEA/D utility every N generations")
    ap.add_argument("--utility_use_archive_density", type=int, default=0, help="whether to mix archive density into utility update (1/0)")
    ap.add_argument("--log_flush_every", type=int, default=10, help="flush MOEA/D debug log every N generations")
    ap.add_argument("--start", type=float, nargs=2, default=None, help="start point in meters: x_m y_m")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal point in meters: x_m y_m")
    ap.add_argument("--start_goal_z_offset", type=float, default=2.0, help="default start/goal altitude offset above local ground, in meters")
    args = ap.parse_args()
    if args.moead_max_gen is None:
        args.moead_max_gen = 80

    env, height, meta = GridEnv.load_npz(args.terrain)
    inflate_cells = int(np.ceil(args.inflate / env.resolution)) if args.inflate > 0 else 0
    if inflate_cells > 0:
        env = env.inflate_obstacles(inflate_cells)

    if height is None:
        height = env.height

    if args.start is None or args.goal is None:
        start, goal = default_start_goal_for_env(env, height, z_offset_m=args.start_goal_z_offset)
    else:
        sx = meters_to_cells(env, float(args.start[0]))
        sy = meters_to_cells(env, float(args.start[1]))
        gx = meters_to_cells(env, float(args.goal[0]))
        gy = meters_to_cells(env, float(args.goal[1]))
        sx_i, sy_i = int(np.clip(round(sx), 0, env.W - 1)), int(np.clip(round(sy), 0, env.H - 1))
        gx_i, gy_i = int(np.clip(round(gx), 0, env.W - 1)), int(np.clip(round(gy), 0, env.H - 1))
        sz = float(height[sy_i, sx_i]) + float(args.start_goal_z_offset)
        gz = float(height[gy_i, gx_i]) + float(args.start_goal_z_offset)
        start = np.array([float(sx_i), float(sy_i), sz], dtype=np.float32)
        goal = np.array([float(gx_i), float(gy_i), gz], dtype=np.float32)

    path_rrt = None
    path_prm = None
    arch = None
    reps = None
    log = None
    metric_log = None
    debug_log = None

    if args.planner in ("rrt", "all"):
        path_rrt, _ = rrt_star(env, start, goal, n_iter=args.rrt_iter, seed=args.seed)
        if path_rrt is None:
            print("RRT* failed to find a path.", flush=True)

    if args.planner in ("prm", "all"):
        path_prm, prm_graph = prm(
            env, start, goal,
            n_samples=args.prm_samples,
            k=args.prm_k,
            max_edge_len=args.prm_max_edge_len,
            threat_weight=args.prm_threat_weight,
            seed=args.seed,
        )
        if path_prm is None:
            print("PRM failed to find a path.", flush=True)
            if prm_graph is not None and getattr(prm_graph, "stats", None):
                print("PRM stats:", prm_graph.stats, flush=True)

    if args.planner in ("moead", "all"):
        _, arch, log = moead(
            env, start, goal,
            n_gen=args.moead_max_gen, pop=args.moead_pop, K=args.K, T=args.moead_T, seed=args.seed,
            moead_min_gen=args.moead_min_gen,
            moead_max_gen=args.moead_max_gen,
            mtoe_tol_fun=args.mtoe_tol_fun,
            mtoe_confidence=args.mtoe_confidence,
            init_stratified_ratio=getattr(args, "init_stratified_ratio", 0.60),
            init_stratified_lateral_frac=getattr(args, "init_stratified_lateral_frac", 0.30),
            init_stratified_n_bands=getattr(args, "init_stratified_n_bands", 5),
            init_stratified_progress_jitter=getattr(args, "init_stratified_progress_jitter", 0.08),
            init_global_random_ratio=getattr(args, "init_global_random_ratio", 0.15),
            weight_extreme_bias=getattr(args, "weight_extreme_bias", 0.20),
            extreme_offspring_ratio=getattr(args, "extreme_offspring_ratio", 0.20),
            extreme_potential_window=getattr(args, "extreme_potential_window", 20),
            extreme_min_extra_per_obj=getattr(args, "extreme_min_extra_per_obj", 1),
            extreme_max_frac_per_obj=getattr(args, "extreme_max_frac_per_obj", 0.60),
            local_search_interval=getattr(args, "local_search_interval", 10),
            local_search_elite_k=getattr(args, "local_search_elite_k", 3),
            local_search_attempts_per_obj=getattr(args, "local_search_attempts_per_obj", 2),
            archive_size=getattr(args, "archive_size", 0),
        archive_soft_limit=getattr(args, "archive_soft_limit", 320),
        archive_grid_bins=getattr(args, "archive_grid_bins", 0),
        archive_keep_extremes=bool(getattr(args, "archive_keep_extremes", 1)),
            active_subproblem_ratio=getattr(args, "active_subproblem_ratio", 1.0),
        )
        if arch is not None and len(arch.items) > 0:
            objs = np.array([it.er.obj for it in arch.items], dtype=float)
            idx_f1 = int(np.argmin(objs[:, 0]))
            idx_f2 = int(np.argmin(objs[:, 1]))
            idx_f3 = int(np.argmin(objs[:, 2]))
            mn = objs.min(axis=0)
            mx = objs.max(axis=0)
            denom = np.where((mx - mn) > 1e-12, (mx - mn), 1.0)
            norm = (objs - mn) / denom
            score = norm.mean(axis=1)
            reps = {
                "min_f1": idx_f1,
                "min_f2": idx_f2,
                "min_f3": idx_f3,
                "compromise": int(np.argmin(score)),
                "objs": objs,
                "score": score,
                "weights": np.array([1.0, 1.0, 1.0], dtype=float) / 3.0,
            }
        metric_log, debug_log = split_moead_log(log)
        print("log:", metric_log, flush=True)

    out_json = args.out_json.strip()
    if not out_json:
        stem = os.path.splitext(os.path.basename(args.terrain))[0]
        out_json = os.path.join(os.path.dirname(args.terrain), f"visdata_{stem}_planner-{args.planner}_pseed{args.seed:04d}.json")

    payload = build_vis_payload(
        terrain_file=args.terrain,
        terrain_seed=meta.get("seed") if isinstance(meta, dict) else None,
        planner_seed=args.seed,
        size_tag=str(meta.get("map_size", {}).get("tag", f"H{env.H}W{env.W}")) if isinstance(meta, dict) else f"H{env.H}W{env.W}",
        inflate=args.inflate,
        map_hw=(env.H, env.W),
        start=start,
        goal=goal,
        path_rrt=path_rrt,
        path_prm=path_prm,
        arch=arch,
        reps=reps,
        title=f"Plan from terrain | planner={args.planner} seed={args.seed}",
        meta=meta,
        extra={
            "planner": args.planner,
            "planner_log": metric_log if args.planner == "moead" or args.planner == "all" else log,
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
                "archive_size": args.archive_size,
                "archive_soft_limit": args.archive_soft_limit,
                "archive_grid_bins": args.archive_grid_bins,
                "archive_keep_extremes": int(args.archive_keep_extremes),
                "utility_update_interval": args.utility_update_interval,
                "utility_use_archive_density": int(args.utility_use_archive_density),
                "log_flush_every": args.log_flush_every,
                "active_subproblem_ratio": args.active_subproblem_ratio,
            },
        },
    )
    save_vis_payload(out_json, payload)
    print("saved vis json:", out_json, flush=True)


if __name__ == "__main__":
    main()
