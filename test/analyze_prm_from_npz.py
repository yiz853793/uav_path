from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.prm import prm
from src.env.grid_env import GridEnv


def ensure_dir(p: str | os.PathLike) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def default_start_goal(env: GridEnv) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = 5.0, 5.0
    gx, gy = float(max(0, env.W - 60)), float(max(0, env.H - 70))
    gx = max(gx, sx + 10.0)
    gy = max(gy, sy + 10.0)
    global_cruise = float(np.percentile(env.height, 90) + env.min_clearance + 10.0)
    sz = max(env.min_safe_altitude_at(sx, sy), global_cruise)
    gz = max(env.min_safe_altitude_at(gx, gy), global_cruise)
    start = np.array([sx, sy, sz], dtype=np.float32)
    goal = np.array([gx, gy, gz], dtype=np.float32)
    return env.clamp_point(start), env.clamp_point(goal)


def parse_point(vals: list[float] | None, env: GridEnv) -> np.ndarray | None:
    if vals is None:
        return None
    arr = np.array(vals, dtype=np.float32)
    if arr.shape[0] == 2:
        arr = np.array([arr[0], arr[1], env.min_safe_altitude_at(arr[0], arr[1]) + 2.0], dtype=np.float32)
    return env.clamp_point(arr)


def extract_edges(adj: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    edges = []
    weights = []
    for u, nbrs in enumerate(adj):
        for v, w in nbrs:
            if v > u:
                edges.append((u, v))
                weights.append(float(w))
    if not edges:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return np.asarray(edges, dtype=np.int32), np.asarray(weights, dtype=np.float32)


def compute_components(adj: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    comp_id = np.full(n, -1, dtype=np.int32)
    sizes = []
    cid = 0
    for root in range(n):
        if comp_id[root] != -1:
            continue
        stack = [root]
        comp_id[root] = cid
        count = 0
        while stack:
            u = stack.pop()
            count += 1
            for v, _ in adj[u]:
                if comp_id[v] == -1:
                    comp_id[v] = cid
                    stack.append(v)
        sizes.append(count)
        cid += 1
    return comp_id, np.asarray(sizes, dtype=np.int32)


def build_node_roles(n: int) -> np.ndarray:
    roles = np.zeros(n, dtype=np.uint8)
    if n > 0:
        roles[0] = 1
    if n > 1:
        roles[1] = 2
    return roles


def sample_edges_for_plot(edges: np.ndarray, max_draw_edges: int, rng: np.random.Generator) -> np.ndarray:
    if len(edges) <= max_draw_edges:
        return edges
    idx = rng.choice(len(edges), size=max_draw_edges, replace=False)
    return edges[np.sort(idx)]


def top_component_ids(component_sizes: np.ndarray, k: int = 6) -> list[int]:
    if component_sizes.size == 0:
        return []
    order = np.argsort(component_sizes)[::-1]
    return [int(i) for i in order[: min(k, len(order))]]


def make_visualization(
    env: GridEnv,
    points_xyz: np.ndarray,
    edges: np.ndarray,
    comp_id: np.ndarray,
    comp_sizes: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    path: np.ndarray | None,
    out_png: str,
    max_draw_edges: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    draw_edges = sample_edges_for_plot(edges, max_draw_edges=max_draw_edges, rng=rng)
    top_ids = top_component_ids(comp_sizes, k=6)
    occ = env.occupancy.astype(np.uint8)
    pts_xy = points_xyz[:, :2]
    z = points_xyz[:, 2] if points_xyz.shape[1] >= 3 else np.zeros(len(points_xyz), dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), dpi=180)

    ax = axes[0]
    ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest")
    if len(draw_edges) > 0:
        segs = np.stack([pts_xy[draw_edges[:, 0]], pts_xy[draw_edges[:, 1]]], axis=1)
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], linewidth=0.2, alpha=0.15, color="tab:blue")
    sc = ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=1.2, c=z, cmap="viridis", alpha=0.35)
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], linewidth=1.8, color="yellow", alpha=0.95)
    ax.scatter([start[0]], [start[1]], s=64, c="lime", marker="*", edgecolors="black", linewidths=0.8, label="start")
    ax.scatter([goal[0]], [goal[1]], s=64, c="red", marker="*", edgecolors="black", linewidths=0.8, label="goal")
    ax.set_title("3D PRM nodes/edges over occupancy")
    ax.legend(loc="upper right")
    ax.set_xlim(0, env.W - 1)
    ax.set_ylim(0, env.H - 1)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="node z")

    ax = axes[1]
    ax.imshow(occ, cmap="gray_r", origin="lower", interpolation="nearest")
    default_color = np.array([0.6, 0.6, 0.6, 0.12])
    palette = np.array([
        [0.121, 0.466, 0.705, 0.78],
        [1.000, 0.498, 0.054, 0.78],
        [0.172, 0.627, 0.172, 0.78],
        [0.839, 0.153, 0.157, 0.78],
        [0.580, 0.404, 0.741, 0.78],
        [0.549, 0.337, 0.294, 0.78],
    ])
    colors = np.repeat(default_color[None, :], len(points_xyz), axis=0)
    for rank, cid in enumerate(top_ids):
        colors[comp_id == cid] = palette[rank % len(palette)]
    ax.scatter(pts_xy[:, 0], pts_xy[:, 1], s=2.0, c=colors)
    if path is not None and len(path) > 1:
        ax.plot(path[:, 0], path[:, 1], linewidth=1.4, color="yellow", alpha=0.9)
    ax.scatter([start[0]], [start[1]], s=70, c="lime", marker="*", edgecolors="black", linewidths=0.8)
    ax.scatter([goal[0]], [goal[1]], s=70, c="red", marker="*", edgecolors="black", linewidths=0.8)
    ax.set_title("Connected components (top components highlighted)")
    ax.set_xlim(0, env.W - 1)
    ax.set_ylim(0, env.H - 1)

    ax = axes[2]
    ax.hist(z, bins=60)
    ax.axvline(float(start[2]), linestyle="--", linewidth=1.2)
    ax.axvline(float(goal[2]), linestyle="--", linewidth=1.2)
    ax.set_title("Node altitude histogram")
    ax.set_xlabel("z")
    ax.set_ylabel("count")

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a terrain npz, run PRM once, save debug dump and visualization.")
    ap.add_argument("--terrain_npz", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="prm_debug")
    ap.add_argument("--start", type=float, nargs="+", default=None)
    ap.add_argument("--goal", type=float, nargs="+", default=None)
    ap.add_argument("--inflate", type=int, default=0)
    ap.add_argument("--prm_samples", type=int, default=12000)
    ap.add_argument("--prm_k", type=int, default=48)
    ap.add_argument("--prm_max_edge_len", type=float, default=250.0)
    ap.add_argument("--prm_threat_weight", type=float, default=0.0)
    ap.add_argument("--collision_step", type=float, default=0.5)
    ap.add_argument("--sample_clearance", type=int, default=0)
    ap.add_argument("--clearance_margin", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_smooth", action="store_true")
    ap.add_argument("--smooth_n_try", type=int, default=80)
    ap.add_argument("--save_occ_height", action="store_true")
    ap.add_argument("--max_draw_edges", type=int, default=25000)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    env, _, meta = GridEnv.load_npz(args.terrain_npz)
    if args.inflate > 0:
        env = env.inflate_obstacles(radius_cells=args.inflate)

    start = parse_point(args.start, env)
    goal = parse_point(args.goal, env)
    if start is None or goal is None:
        start, goal = default_start_goal(env)

    path, graph = prm(
        env,
        start,
        goal,
        n_samples=args.prm_samples,
        k=args.prm_k,
        max_edge_len=args.prm_max_edge_len,
        collision_step=args.collision_step,
        threat_weight=args.prm_threat_weight,
        sample_clearance=args.sample_clearance,
        smooth=not args.no_smooth,
        smooth_n_try=args.smooth_n_try,
        seed=args.seed,
        clearance_margin=args.clearance_margin,
    )

    points_xyz = np.asarray(graph.points, dtype=np.float32)
    edges, edge_costs = extract_edges(graph.adj)
    comp_id, comp_sizes = compute_components(graph.adj)
    node_roles = build_node_roles(len(points_xyz))

    start_cid = int(comp_id[0]) if len(comp_id) > 0 else -1
    goal_cid = int(comp_id[1]) if len(comp_id) > 1 else -1
    start_comp_size = int(comp_sizes[start_cid]) if 0 <= start_cid < len(comp_sizes) else 0
    goal_comp_size = int(comp_sizes[goal_cid]) if 0 <= goal_cid < len(comp_sizes) else 0

    stem = Path(args.terrain_npz).stem
    tag = f"{stem}_s{args.seed}_n{args.prm_samples}_k{args.prm_k}_e{int(args.prm_max_edge_len)}"
    out_npz = Path(args.out_dir) / f"prm_debug_{tag}.npz"
    out_json = Path(args.out_dir) / f"prm_debug_{tag}.json"
    out_png = Path(args.out_dir) / f"prm_debug_{tag}.png"

    payload = {
        "terrain_npz": str(args.terrain_npz),
        "meta": meta,
        "seed": int(args.seed),
        "found": bool(path is not None),
        "n_nodes": int(len(points_xyz)),
        "n_edges": int(len(edges)),
        "n_components": int(len(comp_sizes)),
        "start_component_id": int(start_cid),
        "goal_component_id": int(goal_cid),
        "start_component_size": int(start_comp_size),
        "goal_component_size": int(goal_comp_size),
        "start_goal_connected": bool(start_cid >= 0 and start_cid == goal_cid),
        "path_nodes": int(0 if path is None else len(path)),
        "graph_stats": getattr(graph, "stats", {}),
    }

    save_dict = {
        "points_xyz": points_xyz.astype(np.float32),
        "points_xy": points_xyz[:, :2].astype(np.float32),
        "edges": edges.astype(np.int32),
        "edge_costs": edge_costs.astype(np.float32),
        "component_id": comp_id.astype(np.int32),
        "component_sizes": comp_sizes.astype(np.int32),
        "node_roles": node_roles.astype(np.uint8),
        "start": np.asarray(start, dtype=np.float32),
        "goal": np.asarray(goal, dtype=np.float32),
        "path": np.zeros((0, 3), dtype=np.float32) if path is None else np.asarray(path, dtype=np.float32),
        "stats_json": np.array([json.dumps(payload, ensure_ascii=False)], dtype=object),
    }
    if args.save_occ_height:
        save_dict["occupancy"] = env.occupancy.astype(np.uint8)
        save_dict["height"] = env.height.astype(np.float32)
        save_dict["threat"] = env.threat.astype(np.float32)

    np.savez_compressed(out_npz, **save_dict)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    make_visualization(
        env=env,
        points_xyz=points_xyz,
        edges=edges,
        comp_id=comp_id,
        comp_sizes=comp_sizes,
        start=np.asarray(start, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        path=None if path is None else np.asarray(path, dtype=np.float32),
        out_png=str(out_png),
        max_draw_edges=int(args.max_draw_edges),
        seed=int(args.seed),
    )

    print(f"saved npz : {out_npz}")
    print(f"saved json: {out_json}")
    print(f"saved png : {out_png}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
