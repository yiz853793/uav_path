# main.py

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.moead import moead
from src.viz.plot import plot_env


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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


def default_start_goal(H: int, W: int):
    # 继承你小地图默认：start=(5,5), goal≈(W-60, H-70)
    sx, sy = 5.0, 5.0
    gx, gy = float(max(0, W - 60)), float(max(0, H - 70))
    gx = max(gx, sx + 10.0)
    gy = max(gy, sy + 10.0)
    return np.array([sx, sy], dtype=np.float32), np.array([gx, gy], dtype=np.float32)


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

    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"])
    ap.add_argument("--H", type=int, default=0, help="override height (0 means use preset by --size)")
    ap.add_argument("--W", type=int, default=0, help="override width (0 means use preset by --size)")

    ap.add_argument("--terrace_levels", type=int, default=28)
    ap.add_argument("--inflate", type=int, default=1)

    # RRT*
    ap.add_argument("--rrt_iter", type=int, default=50000)

    # MOEA/D core
    ap.add_argument("--moead_gen", type=int, default=80)
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--moead_T", type=int, default=10)

    # A* seeding（大图可增大 init_astar_max_expansions 以免 A* 未找到路径）
    ap.add_argument("--init_astar_ratio", type=float, default=0.25)
    ap.add_argument("--init_astar_threat_weight", type=float, default=0.0)
    ap.add_argument("--init_astar_jitter_sigma", type=float, default=1.5)
    ap.add_argument("--init_astar_max_paths", type=int, default=5)
    ap.add_argument("--init_astar_penalty_step", type=float, default=2.5)
    ap.add_argument("--init_astar_max_expansions", type=int, default=None, help="A* 扩展上限，默认按地图面积自动")

    # start/goal：默认 AUTO（不传则按尺寸自动设）
    ap.add_argument("--start", type=float, nargs=2, default=None)
    ap.add_argument("--goal", type=float, nargs=2, default=None)

    ap.add_argument("--terrain_dir", type=str, default="terrains", help="where to save generated terrain npz")
    ap.add_argument("--out_root", type=str, default="outputs", help="where to save figures/logs")

    # Debug：MOEA/D 每代日志写入文件，并可实时输出到控制台（与 run_benchmark 一致）
    ap.add_argument("--debug", action="store_true", help="开启 MOEA/D 调试日志（写文件并实时打屏）")
    ap.add_argument("--debug_log", type=str, default=None, help="调试日志路径，默认在 --debug 时为 out_dir/moead_debug.log")
    ap.add_argument("--debug_every", type=int, default=1, help="每多少代打印一条日志")
    ap.add_argument("--debug_level", type=int, default=2, choices=[1, 2, 3], help="1=简要 2=含耗时 3=含碰撞统计")
    ap.add_argument("--no_debug_console", action="store_true", help="开启 --debug 时仅写文件、不实时打屏")

    args = ap.parse_args()

    if args.H > 0 and args.W > 0:
        H, W = args.H, args.W
    else:
        H, W = map_size_to_hw(args.size)
    size_tag = map_size_name(H, W)

    if args.start is None or args.goal is None:
        start, goal = default_start_goal(H, W)
    else:
        start = np.array(args.start, dtype=np.float32)
        goal = np.array(args.goal, dtype=np.float32)

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
    env, height, meta = GridEnv.mountain_map(
        H=H, W=W,
        seed=args.terrain_seed,
        terrace_levels=args.terrace_levels,
    )
    env = env.inflate_obstacles(radius_cells=args.inflate)

    # terrain 保存到 terrains/{size_tag}/mountain_seedXXXX.npz
    terrain_subdir = os.path.join(args.terrain_dir, size_tag)
    ensure_dir(terrain_subdir)
    terrain_path = os.path.join(terrain_subdir, f"mountain_seed{args.terrain_seed:04d}.npz")
    env.save_npz(terrain_path, height=height, meta={**meta, "map_size": {"H": H, "W": W, "tag": size_tag}})
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

    debug_log_path = None
    if args.debug or (args.debug_log is not None and args.debug_log.strip()):
        debug_log_path = (args.debug_log or "").strip() or os.path.join(out_dir, "moead_debug.log")
    # 与 run_benchmark 一致：开启 debug 时默认实时打屏，--no_debug_console 则仅写文件
    debug_console = bool(debug_log_path) and not args.no_debug_console

    _, arch, log = moead(
        env, start, goal,
        n_gen=args.moead_gen, pop=args.moead_pop, K=args.K, T=args.moead_T, seed=args.planner_seed,
        init_astar_ratio=args.init_astar_ratio,
        init_astar_threat_weight=args.init_astar_threat_weight,
        init_astar_jitter_sigma=args.init_astar_jitter_sigma,
        init_astar_max_paths=args.init_astar_max_paths,
        init_astar_penalty_step=args.init_astar_penalty_step,
        init_astar_max_expansions=args.init_astar_max_expansions,
        debug_log_path=debug_log_path,
        debug_every=args.debug_every,
        debug_level=args.debug_level,
        debug_console=debug_console,
    )

    # 保存 log（文件名带 A* 参数）
    log_path = os.path.join(out_dir, f"log_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}{atag}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "terrain_seed": args.terrain_seed,
            "planner_seed": args.planner_seed,
            "map_size": {"H": H, "W": W, "tag": size_tag},
            "terrace_levels": args.terrace_levels,
            "inflate": args.inflate,
            "start": start.tolist(),
            "goal": goal.tolist(),
            "rrt_iter": args.rrt_iter,
            "moead": {
                "n_gen": args.moead_gen,
                "pop": args.moead_pop,
                "K": args.K,
                "T": args.moead_T,
                "init_astar_ratio": args.init_astar_ratio,
                "init_astar_threat_weight": args.init_astar_threat_weight,
                "init_astar_jitter_sigma": args.init_astar_jitter_sigma,
                "init_astar_max_paths": args.init_astar_max_paths,
                "init_astar_penalty_step": args.init_astar_penalty_step,
            },
            "meta": meta,
            "log": log,
            "archive_size": len(arch.items),
        }, f, ensure_ascii=False, indent=2)
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
    if reps is not None:
        objs = reps["objs"]
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(objs[:, 0], objs[:, 1], objs[:, 2], s=12)

        ax2.set_xlabel("f1: length")
        ax2.set_ylabel("f2: threat")
        ax2.set_zlabel("f3: energy")
        ax2.set_title(f"Pareto 3D | size={size_tag} seed={args.terrain_seed} pseed={args.planner_seed}")

        plt.tight_layout()
        pareto_png = os.path.join(out_dir, f"pareto3d_seed{args.terrain_seed:04d}_pseed{args.planner_seed:04d}_{size_tag}_inf{args.inflate}_K{args.K}{atag}.png")
        plt.savefig(pareto_png, dpi=220)
        plt.close(fig2)
        print("saved figure:", pareto_png, flush=True)


if __name__ == "__main__":
    main()
