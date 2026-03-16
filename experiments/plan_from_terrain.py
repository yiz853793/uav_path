import argparse
import os
import numpy as np

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from experiments.vis_data import build_vis_payload, save_vis_payload


def default_start_goal_for_env(env: GridEnv, height: np.ndarray, z_offset: float = 5.0):
    sx, sy = 5.0, 5.0
    gx, gy = float(max(0, env.W - 60)), float(max(0, env.H - 70))
    gx = max(gx, sx + 10.0)
    gy = max(gy, sy + 10.0)
    sz = max(float(height[int(round(sy)), int(round(sx))]) + z_offset, env.altitude_from_ground(sx, sy))
    gz = max(float(height[int(round(gy)), int(round(gx))]) + z_offset, env.altitude_from_ground(gx, gy))
    return np.array([sx, sy, sz], dtype=np.float32), np.array([gx, gy, gz], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", type=str, required=True, help="path to .npz terrain file")
    ap.add_argument("--planner", type=str, default="rrt", choices=["rrt", "prm", "moead", "all"])
    ap.add_argument("--inflate", type=int, default=1, help="inflate obstacles radius in cells")
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
    args = ap.parse_args()

    env, height, meta = GridEnv.load_npz(args.terrain)
    if args.inflate > 0:
        env = env.inflate_obstacles(args.inflate)

    if height is None:
        height = env.height
    start, goal = default_start_goal_for_env(env, height, z_offset=5.0)

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
        path_prm, _ = prm(
            env, start, goal,
            n_samples=args.prm_samples,
            k=args.prm_k,
            max_edge_len=args.prm_max_edge_len,
            threat_weight=args.prm_threat_weight,
            seed=args.seed,
        )
        if path_prm is None:
            print("PRM failed to find a path.")

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
