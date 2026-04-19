from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.collision import segment_collision
from ..env.grid_env import GridEnv


@dataclass
class Node:
    p: np.ndarray
    parent: int
    cost: float


def _metric_dists(env: GridEnv, pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = pts - p[None, :]
    d = d.astype(np.float32, copy=False)
    d[:, 0] *= float(env.resolution)
    d[:, 1] *= float(env.resolution)
    return np.linalg.norm(d, axis=1)


def rrt_star(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_iter: int = 3000,
    step_len: float = 6.0,
    goal_sample_rate: float = 0.05,
    near_radius: float = 12.0,
    collision_step: float = 0.5,
    seed: int = 0,
):
    """Slightly optimized RRT* baseline with vectorized nearest/near queries."""
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    dim = int(start.shape[0])
    rng = np.random.default_rng(seed)

    max_nodes = int(n_iter) + 2
    pts = np.empty((max_nodes, dim), dtype=np.float32)
    parent = np.full(max_nodes, -1, dtype=np.int32)
    cost = np.full(max_nodes, np.inf, dtype=np.float32)
    pts[0] = start
    cost[0] = 0.0
    n_nodes = 1
    goal_idx: int | None = None

    step_len_m = float(step_len) * float(env.resolution)
    near_radius_m = float(near_radius) * float(env.resolution)

    def sample() -> np.ndarray:
        if float(rng.random()) < float(goal_sample_rate):
            return goal.copy()
        if dim == 2:
            return np.array([rng.uniform(0, env.W - 1), rng.uniform(0, env.H - 1)], dtype=np.float32)
        x = float(rng.uniform(0, env.W - 1))
        y = float(rng.uniform(0, env.H - 1))
        z_lo = float(env.min_safe_altitude_at(x, y))
        z = float(rng.uniform(z_lo, env.z_max))
        return np.array([x, y, z], dtype=np.float32)

    def steer(p_from: np.ndarray, p_to: np.ndarray, dist_m: float) -> np.ndarray:
        if dist_m < 1e-9:
            return p_from.copy()
        if dist_m <= step_len_m:
            q = p_to.copy()
        else:
            q = p_from + (p_to - p_from) * (step_len_m / max(1e-9, dist_m))
        return env.clamp_point(q) if dim == 3 else q.astype(np.float32)

    for _ in range(int(n_iter)):
        p_rand = sample()
        d_all = _metric_dists(env, pts[:n_nodes], p_rand)
        i_near = int(np.argmin(d_all))
        p_from = pts[i_near]
        p_new = steer(p_from, p_rand, float(d_all[i_near]))

        if dim == 2:
            if env.is_occupied(int(round(float(p_new[0]))), int(round(float(p_new[1])))):
                continue
        else:
            if not env.is_free_point(p_new):
                continue
        if segment_collision(env, p_from, p_new, step=collision_step):
            continue

        d_new = _metric_dists(env, pts[:n_nodes], p_new)
        cand_idx = np.flatnonzero(d_new <= near_radius_m).astype(np.int32)
        if cand_idx.size == 0:
            cand_idx = np.asarray([i_near], dtype=np.int32)
        elif i_near not in cand_idx:
            cand_idx = np.concatenate([cand_idx, np.asarray([i_near], dtype=np.int32)])

        best_parent = i_near
        best_cost = float(cost[i_near]) + float(d_new[i_near])
        for j in cand_idx.tolist():
            if j == i_near:
                continue
            if segment_collision(env, pts[j], p_new, step=collision_step):
                continue
            c = float(cost[j]) + float(d_new[j])
            if c < best_cost:
                best_cost = c
                best_parent = j

        new_idx = n_nodes
        pts[new_idx] = p_new
        parent[new_idx] = int(best_parent)
        cost[new_idx] = float(best_cost)
        n_nodes += 1

        for j in cand_idx.tolist():
            if j == best_parent:
                continue
            if segment_collision(env, p_new, pts[j], step=collision_step):
                continue
            c_through = float(cost[new_idx]) + float(d_new[j])
            if c_through + 1e-12 < float(cost[j]):
                parent[j] = int(new_idx)
                cost[j] = float(c_through)

        d_goal = float(np.linalg.norm(env.metric_delta(goal - p_new)))
        if d_goal < step_len_m and (not segment_collision(env, p_new, goal, step=collision_step)):
            pts[n_nodes] = goal.copy()
            parent[n_nodes] = int(new_idx)
            cost[n_nodes] = float(cost[new_idx]) + d_goal
            goal_idx = n_nodes
            n_nodes += 1
            break

    nodes = [Node(p=pts[i].copy(), parent=int(parent[i]), cost=float(cost[i])) for i in range(n_nodes)]
    if goal_idx is None:
        return None, nodes
    path = []
    cur = int(goal_idx)
    while cur != -1:
        path.append(pts[cur].copy())
        cur = int(parent[cur])
    path = np.vstack(path[::-1]).astype(np.float32)
    return path, nodes
