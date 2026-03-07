# experiments/plan_from_terrain.py

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.moead import moead
from src.viz.plot import plot_env, plot_path, plot_pareto

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", type=str, required=True, help="path to .npz terrain file")
    ap.add_argument("--planner", type=str, default="rrt", choices=["rrt", "moead"])
    ap.add_argument("--inflate", type=int, default=1, help="inflate obstacles radius in cells")
    ap.add_argument("--seed", type=int, default=0, help="planner seed (NOT terrain seed)")
    ap.add_argument("--rrt_iter", type=int, default=4000)
    ap.add_argument("--moead_gen", type=int, default=80)
    ap.add_argument("--moead_pop", type=int, default=60)
    ap.add_argument("--K", type=int, default=30)
    args = ap.parse_args()

    env, height, meta = GridEnv.load_npz(args.terrain)
    if args.inflate > 0:
        env = env.inflate_obstacles(args.inflate)

    # 默认起终点（你可以改成从 meta 读，或命令行传入）
    start = np.array([5.0, 5.0], dtype=np.float32)
    goal  = np.array([env.W - 20.0, env.H - 30.0], dtype=np.float32)

    fig, ax = plt.subplots()
    plot_env(env, ax=ax)

    if args.planner == "rrt":
        path, _ = rrt_star(env, start, goal, n_iter=args.rrt_iter, seed=args.seed)
        if path is None:
            print("RRT* failed to find a path.")
        else:
            plot_path(path, ax=ax, label="RRT*")
        ax.scatter([start[0]],[start[1]], marker="o", s=80)
        ax.scatter([goal[0]],[goal[1]], marker="*", s=140)
        plt.show()

    else:
        pop_inds, arch, log = moead(env, start, goal, n_gen=args.moead_gen, pop=args.moead_pop, K=args.K, seed=args.seed)
        # 画若干条档案路径
        ax.scatter([start[0]],[start[1]], marker="o", s=80)
        ax.scatter([goal[0]],[goal[1]], marker="*", s=140)
        for it in arch.items[:8]:
            plot_path(it.x, ax=ax)
        plt.show()

        # Pareto 投影
        if len(arch.items) > 0:
            objs = np.array([it.er.obj for it in arch.items], dtype=float)
            fig2, ax2 = plt.subplots()
            plot_pareto(objs, ax=ax2)
            plt.show()
        print("log:", log)

if __name__ == "__main__":
    main()
