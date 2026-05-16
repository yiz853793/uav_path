from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.collision import sampled_points_array, segment_collision
from ..env.grid_env import GridEnv
from ..models.path import shortcut_smooth


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


def _edge_cost(
    env: GridEnv,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    threat_weight: float,
    collision_step: float,
    length_weight: float = 1.0,
    energy_weight: float = 0.0,
    energy_climb_weight: float = 2.0,
) -> float:
    base = env.metric_distance(p0, p1)
    cost = float(length_weight) * float(base)
    if float(energy_weight) != 0.0:
        climb = 0.0
        if np.asarray(p0).shape[0] >= 3 and np.asarray(p1).shape[0] >= 3:
            climb = max(0.0, float(p1[2]) - float(p0[2]))
        edge_energy = float(base) + float(energy_climb_weight) * float(climb)
        cost += float(energy_weight) * edge_energy
    if float(threat_weight) == 0.0:
        return float(cost)
    pts = sampled_points_array(
        np.asarray(p0, dtype=np.float32),
        np.asarray(p1, dtype=np.float32),
        step=collision_step,
        xy_resolution=env.resolution,
    )
    ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
    iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
    threat = env.threat[iy, ix].astype(np.float64, copy=False)
    if pts.shape[1] >= 3:
        ground = env.height[iy, ix]
        clearance = np.maximum(0.0, pts[:, 2] - ground)
        threat = threat * np.exp(-0.06 * clearance)
    return float(cost + float(threat_weight) * float(np.mean(threat, dtype=np.float64)))


def rrt_star(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_iter: int = 3000,
    step_len: float = 6.0,
    goal_sample_rate: float = 0.05,
    near_radius: float = 12.0,
    collision_step: float = 0.5,
    threat_weight: float = 0.0,
    length_weight: float = 1.0,
    energy_weight: float = 0.0,
    energy_climb_weight: float = 2.0,
    clearance_margin: float = 0.0,
    smooth: bool = True,
    smooth_n_try: int = 80,
    seed: int = 0,
):
    """Slightly optimized RRT* baseline with vectorized nearest/near queries."""
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    dim = int(start.shape[0])
    rng = np.random.default_rng(seed)

    max_nodes = int(n_iter) + 1
    pts = np.empty((max_nodes, dim), dtype=np.float32)
    parent = np.full(max_nodes, -1, dtype=np.int32)
    cost = np.full(max_nodes, np.inf, dtype=np.float64)
    pts[0] = start
    cost[0] = 0.0
    n_nodes = 1
    goal_candidates: list[int] = []

    step_len_m = float(step_len) * float(env.resolution)
    near_radius_m = float(near_radius) * float(env.resolution)
    clearance = float(env.min_clearance) + max(0.0, float(clearance_margin)) if dim == 3 else None

    def sample() -> np.ndarray:
        if float(rng.random()) < float(goal_sample_rate):
            return goal.copy()
        if dim == 2:
            return np.array([rng.uniform(0, env.W - 1), rng.uniform(0, env.H - 1)], dtype=np.float32)
        x = float(rng.uniform(0, env.W - 1))
        y = float(rng.uniform(0, env.H - 1))
        z_lo = float(env.min_safe_altitude_at(x, y, clearance=clearance))
        z = float(rng.uniform(z_lo, env.z_max))
        return np.array([x, y, z], dtype=np.float32)

    def steer(p_from: np.ndarray, p_to: np.ndarray, dist_m: float) -> np.ndarray:
        if dist_m < 1e-9:
            return p_from.copy()
        if dist_m <= step_len_m:
            q = p_to.copy()
        else:
            q = p_from + (p_to - p_from) * (step_len_m / max(1e-9, dist_m))
        return env.clamp_point(q, clearance=clearance) if dim == 3 else q.astype(np.float32)

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
            if not env.is_free_point(p_new, clearance=clearance):
                continue
        if segment_collision(env, p_from, p_new, step=collision_step, clearance=clearance):
            continue

        d_new = _metric_dists(env, pts[:n_nodes], p_new)
        cand_idx = np.flatnonzero(d_new <= near_radius_m).astype(np.int32)
        if cand_idx.size == 0:
            cand_idx = np.asarray([i_near], dtype=np.int32)
        elif i_near not in cand_idx:
            cand_idx = np.concatenate([cand_idx, np.asarray([i_near], dtype=np.int32)])

        best_parent = i_near
        best_cost = float(cost[i_near]) + _edge_cost(
            env,
            pts[i_near],
            p_new,
            threat_weight=threat_weight,
            collision_step=collision_step,
            length_weight=length_weight,
            energy_weight=energy_weight,
            energy_climb_weight=energy_climb_weight,
        )
        for j in cand_idx.tolist():
            if j == i_near:
                continue
            if segment_collision(env, pts[j], p_new, step=collision_step, clearance=clearance):
                continue
            c = float(cost[j]) + _edge_cost(
                env,
                pts[j],
                p_new,
                threat_weight=threat_weight,
                collision_step=collision_step,
                length_weight=length_weight,
                energy_weight=energy_weight,
                energy_climb_weight=energy_climb_weight,
            )
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
            if segment_collision(env, p_new, pts[j], step=collision_step, clearance=clearance):
                continue
            c_through = float(cost[new_idx]) + _edge_cost(
                env,
                p_new,
                pts[j],
                threat_weight=threat_weight,
                collision_step=collision_step,
                length_weight=length_weight,
                energy_weight=energy_weight,
                energy_climb_weight=energy_climb_weight,
            )
            if c_through + 1e-12 < float(cost[j]):
                parent[j] = int(new_idx)
                cost[j] = float(c_through)

        d_goal = float(np.linalg.norm(env.metric_delta(goal - p_new)))
        if d_goal < step_len_m and (not segment_collision(env, p_new, goal, step=collision_step, clearance=clearance)):
            goal_candidates.append(int(new_idx))

    nodes = [Node(p=pts[i].copy(), parent=int(parent[i]), cost=float(cost[i])) for i in range(n_nodes)]
    if not goal_candidates:
        return None, nodes
    goal_idx = min(
        goal_candidates,
        key=lambda i: float(cost[i]) + _edge_cost(
            env,
            pts[i],
            goal,
            threat_weight=threat_weight,
            collision_step=collision_step,
            length_weight=length_weight,
            energy_weight=energy_weight,
            energy_climb_weight=energy_climb_weight,
        ),
    )
    path = []
    cur = int(goal_idx)
    while cur != -1:
        path.append(pts[cur].copy())
        cur = int(parent[cur])
    path = np.vstack(path[::-1] + [goal.copy()]).astype(np.float32)
    if smooth and len(path) >= 3:
        path = shortcut_smooth(
            path,
            n_try=int(smooth_n_try),
            rng=rng,
            collision_fn=lambda a, b: segment_collision(
                env,
                np.asarray(a, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                step=float(collision_step),
                clearance=clearance,
            ),
        )
        path[0] = start
        path[-1] = goal
    return path, nodes
