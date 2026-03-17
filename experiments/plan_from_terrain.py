import argparse
import os
import numpy as np

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from experiments.vis_data import build_vis_payload, save_vis_payload

def meters_to_cells(env, value_m: float) -> float:
    return float(value_m) / float(env.resolution)

def cells_to_meters(env, value_cells: float) -> float:
    return float(value_cells) * float(env.resolution)

def default_start_goal_for_env(env: GridEnv, height: np.ndarray, z_offset_m: float = 2.0):
    res = float(env.resolution)
    sx_m, sy_m = 25.0, 25.0
    gx_m, gy_m = float(max(25.0 + 50.0, (env.W - 60) * res)), float(max(25.0 + 50.0, (env.H - 70) * res))

    sx = meters_to_cells(env, sx_m)
    sy = meters_to_cells(env, sy_m)
    gx = meters_to_cells(env, gx_m)
    gy = meters_to_cells(env, gy_m)

    sx_i, sy_i = int(np.clip(round(sx), 0, env.W - 1)), int(np.clip(round(sy), 0, env.H - 1))
    gx_i, gy_i = int(np.clip(round(gx), 0, env.W - 1)), int(np.clip(round(gy), 0, env.H - 1))

    sz = float(height[sy_i, sx_i]) + float(z_offset_m)
    gz = float(height[gy_i, gx_i]) + float(z_offset_m)
    return np.array([float(sx_i), float(sy_i), sz], dtype=np.float32), np.array([float(gx_i), float(gy_i), gz], dtype=np.float32)


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
    ap.add_argument("--moead_gen", type=int, default=80)
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--moead_T", type=int, default=10)
    ap.add_argument("--out_json", type=str, default="", help="where to save visualization json")
    ap.add_argument("--start", type=float, nargs=2, default=None, help="start point in meters: x_m y_m")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal point in meters: x_m y_m")
    ap.add_argument("--start_goal_z_offset", type=float, default=2.0, help="default start/goal altitude offset above local ground, in meters")
    args = ap.parse_args()

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

    if args.planner in ("rrt", "all"):
        path_rrt, _ = rrt_star(env, start, goal, n_iter=args.rrt_iter, seed=args.seed)
        if path_rrt is None:
            print("RRT* failed to find a path.")

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
            print("PRM failed to find a path.")
            if prm_graph is not None and getattr(prm_graph, "stats", None):
                print("PRM stats:", prm_graph.stats)

    if args.planner in ("moead", "all"):
        _, arch, log = moead(env, start, goal, n_gen=args.moead_gen, pop=args.moead_pop, K=args.K, T=args.moead_T, seed=args.seed)
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
        print("log:", log)

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
            "planner_log": log,
            "planner_args": {
                "rrt_iter": args.rrt_iter,
                "prm_samples": args.prm_samples,
                "prm_k": args.prm_k,
                "prm_max_edge_len": args.prm_max_edge_len,
                "prm_threat_weight": args.prm_threat_weight,
                "moead_gen": args.moead_gen,
                "moead_pop": args.moead_pop,
                "moead_T": args.moead_T,
                "K": args.K,
            },
        },
    )
    save_vis_payload(out_json, payload)
    print("saved vis json:", out_json)


if __name__ == "__main__":
    main()
