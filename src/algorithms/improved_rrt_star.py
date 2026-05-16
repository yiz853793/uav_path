from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..env.collision import sampled_points_array, segment_collision
from ..env.grid_env import GridEnv
from ..models.path import shortcut_smooth


@dataclass
class ImprovedRRTStats:
    iterations: int
    nodes: int
    first_solution_iter: int | None
    improvements: int
    best_cost: float
    best_length: float
    informed_samples: int
    corridor_samples: int


def _metric_dists(env: GridEnv, pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = pts - p[None, :]
    d = d.astype(np.float32, copy=False)
    d[:, 0] *= float(env.resolution)
    d[:, 1] *= float(env.resolution)
    return np.linalg.norm(d, axis=1)


def _to_metric(env: GridEnv, p: np.ndarray) -> np.ndarray:
    return env.metric_point(np.asarray(p, dtype=np.float32))


def _from_metric(env: GridEnv, p: np.ndarray) -> np.ndarray:
    q = np.asarray(p, dtype=np.float32).copy()
    q[0] /= float(env.resolution)
    q[1] /= float(env.resolution)
    return q


def _rotation_from_x_axis(direction: np.ndarray) -> np.ndarray:
    dim = int(direction.size)
    e1 = np.zeros(dim, dtype=np.float32)
    e1[0] = 1.0
    a = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(a))
    if norm < 1e-9:
        return np.eye(dim, dtype=np.float32)
    a = a / norm
    v = e1 - a
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-6:
        return np.eye(dim, dtype=np.float32)
    v = v / v_norm
    return (np.eye(dim, dtype=np.float32) - 2.0 * np.outer(v, v)).astype(np.float32)


def _sample_unit_ball(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v[0] = 1.0
        n = 1.0
    radius = float(rng.random()) ** (1.0 / max(1, dim))
    return (v / n * radius).astype(np.float32)


def _edge_cost(
    env: GridEnv,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    threat_weight: float,
    collision_step: float,
) -> float:
    base = env.metric_distance(p0, p1)
    if float(threat_weight) == 0.0:
        return base
    pts = sampled_points_array(np.asarray(p0, dtype=np.float32), np.asarray(p1, dtype=np.float32), step=collision_step, xy_resolution=env.resolution)
    ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
    iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
    threat = env.threat[iy, ix].astype(np.float64, copy=False)
    if pts.shape[1] >= 3:
        ground = env.height[iy, ix]
        clearance = np.maximum(0.0, pts[:, 2] - ground)
        threat = threat * np.exp(-0.06 * clearance)
    return float(base + float(threat_weight) * float(np.mean(threat, dtype=np.float64)))


def _reconstruct_path(pts: np.ndarray, parent: np.ndarray, idx: int, goal: np.ndarray | None = None) -> np.ndarray:
    out = []
    cur = int(idx)
    while cur != -1:
        out.append(pts[cur].copy())
        cur = int(parent[cur])
    arr = np.vstack(out[::-1]).astype(np.float32)
    if goal is not None:
        arr = np.vstack([arr, np.asarray(goal, dtype=np.float32)])
    return arr.astype(np.float32)


def improved_rrt_star(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_iter: int = 4000,
    step_len: float = 8.0,
    min_step_len: float = 3.0,
    max_step_len: float = 16.0,
    goal_sample_rate: float = 0.08,
    corridor_sample_rate: float = 0.18,
    informed_sample_rate: float = 0.70,
    near_radius: float = 16.0,
    collision_step: float = 0.5,
    threat_weight: float = 0.0,
    clearance_margin: float = 2.0,
    smooth: bool = True,
    smooth_n_try: int = 120,
    seed: int = 0,
) -> tuple[np.ndarray | None, list[dict[str, Any]], ImprovedRRTStats]:
    """
    Improved RRT* baseline for 2.5D UAV planning.

    Compared with the plain baseline this variant keeps optimizing after the
    first feasible solution, biases pre-solution samples toward the start-goal
    corridor, switches to informed ellipsoid sampling after a solution is found,
    and applies shortcut smoothing to the final path.
    """
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

    clearance = float(env.min_clearance) + float(clearance_margin)
    base_step_m = float(step_len) * float(env.resolution)
    min_step_m = float(min_step_len) * float(env.resolution)
    max_step_m = float(max_step_len) * float(env.resolution)
    near_radius_m = float(near_radius) * float(env.resolution)
    threat_max = float(np.nanmax(env.threat)) if getattr(env, "threat", None) is not None else 0.0
    threat_max = max(threat_max, 1e-9)

    start_m = _to_metric(env, start)
    goal_m = _to_metric(env, goal)
    line_m = goal_m - start_m
    c_min = float(np.linalg.norm(line_m))
    center_m = (start_m + goal_m) * 0.5
    rot = _rotation_from_x_axis(line_m)

    best_parent: int | None = None
    best_cost = float("inf")
    best_length = float("inf")
    first_solution_iter: int | None = None
    improvements = 0
    informed_samples = 0
    corridor_samples = 0

    def sample_global() -> np.ndarray:
        if dim == 2:
            return np.array([rng.uniform(0, env.W - 1), rng.uniform(0, env.H - 1)], dtype=np.float32)
        x = float(rng.uniform(0, env.W - 1))
        y = float(rng.uniform(0, env.H - 1))
        z_lo = float(env.min_safe_altitude_at(x, y, clearance=clearance))
        z = float(rng.uniform(z_lo, env.z_max)) if env.z_max > z_lo else z_lo
        return np.array([x, y, z], dtype=np.float32)

    def sample_corridor() -> np.ndarray:
        nonlocal corridor_samples
        corridor_samples += 1
        t = float(rng.uniform(0.0, 1.0))
        p = start + (goal - start) * t
        xy_len = float(np.linalg.norm((goal[:2] - start[:2]) * float(env.resolution)))
        sigma_cells = max(4.0, min(60.0, 0.08 * xy_len / max(float(env.resolution), 1e-9)))
        if dim >= 2:
            dxy = goal[:2] - start[:2]
            dn = float(np.linalg.norm(dxy))
            if dn > 1e-9:
                normal = np.array([-dxy[1], dxy[0]], dtype=np.float32) / dn
                p[:2] += normal * float(rng.normal(0.0, sigma_cells))
                p[:2] += (dxy / dn) * float(rng.normal(0.0, sigma_cells * 0.25))
        if dim == 3:
            p[0] = float(np.clip(p[0], 0.0, env.W - 1.0))
            p[1] = float(np.clip(p[1], 0.0, env.H - 1.0))
            safe = float(env.min_safe_altitude_at(float(p[0]), float(p[1]), clearance=clearance))
            route_z = float((1.0 - t) * start[2] + t * goal[2])
            p[2] = float(np.clip(max(route_z, safe) + rng.uniform(0.0, 18.0), safe, env.z_max))
        return env.clamp_point(p, clearance=clearance) if dim == 3 else p.astype(np.float32)

    def sample_informed() -> np.ndarray | None:
        nonlocal informed_samples
        if not np.isfinite(best_length) or best_length <= c_min + 1e-6:
            return None
        informed_samples += 1
        radii = np.full(dim, np.sqrt(max(best_length * best_length - c_min * c_min, 0.0)) * 0.5, dtype=np.float32)
        radii[0] = best_length * 0.5
        q_m = center_m + rot @ (radii * _sample_unit_ball(rng, dim))
        q = _from_metric(env, q_m)
        if dim == 3:
            q = env.clamp_point(q, clearance=clearance)
        elif not (0.0 <= q[0] < env.W and 0.0 <= q[1] < env.H):
            return None
        return q.astype(np.float32)

    def sample() -> np.ndarray:
        if float(rng.random()) < float(goal_sample_rate):
            return goal.copy()
        if best_parent is not None and float(rng.random()) < float(informed_sample_rate):
            q = sample_informed()
            if q is not None:
                return q
        if float(rng.random()) < float(corridor_sample_rate):
            return sample_corridor()
        return sample_global()

    def adaptive_step(p_from: np.ndarray, p_to: np.ndarray) -> float:
        if dim < 2:
            return base_step_m
        threat = max(env.threat_at(float(p_to[0]), float(p_to[1])), env.threat_at(float(p_from[0]), float(p_from[1])))
        threat_norm = float(np.clip(threat / threat_max, 0.0, 1.0))
        step_m = base_step_m * (1.25 - 0.55 * threat_norm)
        if best_parent is not None:
            step_m *= 0.85
        return float(np.clip(step_m, min_step_m, max_step_m))

    def steer(p_from: np.ndarray, p_to: np.ndarray, dist_m: float) -> np.ndarray:
        if dist_m < 1e-9:
            return p_from.copy()
        step_m = adaptive_step(p_from, p_to)
        q = p_to.copy() if dist_m <= step_m else p_from + (p_to - p_from) * (step_m / max(1e-9, dist_m))
        return env.clamp_point(q, clearance=clearance) if dim == 3 else q.astype(np.float32)

    def point_free(q: np.ndarray) -> bool:
        if dim == 2:
            return not env.is_occupied(int(round(float(q[0]))), int(round(float(q[1]))))
        return env.is_free_point(q, clearance=clearance)

    for it in range(int(n_iter)):
        if n_nodes >= max_nodes:
            break
        p_rand = sample()
        d_all = _metric_dists(env, pts[:n_nodes], p_rand)
        i_near = int(np.argmin(d_all))
        p_new = steer(pts[i_near], p_rand, float(d_all[i_near]))

        if not point_free(p_new):
            continue
        if segment_collision(env, pts[i_near], p_new, step=collision_step, clearance=clearance if dim == 3 else None):
            continue

        d_new = _metric_dists(env, pts[:n_nodes], p_new)
        cand_idx = np.flatnonzero(d_new <= near_radius_m).astype(np.int32)
        if cand_idx.size == 0:
            cand_idx = np.asarray([i_near], dtype=np.int32)
        elif i_near not in cand_idx:
            cand_idx = np.concatenate([cand_idx, np.asarray([i_near], dtype=np.int32)])

        best_node_parent = i_near
        best_node_cost = float(cost[i_near]) + _edge_cost(env, pts[i_near], p_new, threat_weight=threat_weight, collision_step=collision_step)
        for j in cand_idx.tolist():
            if j == i_near:
                continue
            if segment_collision(env, pts[j], p_new, step=collision_step, clearance=clearance if dim == 3 else None):
                continue
            c = float(cost[j]) + _edge_cost(env, pts[j], p_new, threat_weight=threat_weight, collision_step=collision_step)
            if c < best_node_cost:
                best_node_cost = c
                best_node_parent = j

        new_idx = n_nodes
        pts[new_idx] = p_new
        parent[new_idx] = int(best_node_parent)
        cost[new_idx] = float(best_node_cost)
        n_nodes += 1

        for j in cand_idx.tolist():
            if j == best_node_parent:
                continue
            if segment_collision(env, p_new, pts[j], step=collision_step, clearance=clearance if dim == 3 else None):
                continue
            c_through = float(cost[new_idx]) + _edge_cost(env, p_new, pts[j], threat_weight=threat_weight, collision_step=collision_step)
            if c_through + 1e-12 < float(cost[j]):
                parent[j] = int(new_idx)
                cost[j] = float(c_through)

        d_goal = float(np.linalg.norm(env.metric_delta(goal - p_new)))
        if d_goal <= max_step_m and not segment_collision(env, p_new, goal, step=collision_step, clearance=clearance if dim == 3 else None):
            goal_cost = float(cost[new_idx]) + _edge_cost(env, p_new, goal, threat_weight=threat_weight, collision_step=collision_step)
            if goal_cost + 1e-12 < best_cost:
                candidate = _reconstruct_path(pts, parent, new_idx, goal=goal)
                candidate_length = env.path_length(candidate)
                best_parent = int(new_idx)
                best_cost = float(goal_cost)
                best_length = float(candidate_length)
                improvements += 1
                if first_solution_iter is None:
                    first_solution_iter = int(it + 1)

    nodes = [
        {"p": pts[i].copy(), "parent": int(parent[i]), "cost": float(cost[i])}
        for i in range(n_nodes)
    ]
    stats = ImprovedRRTStats(
        iterations=int(n_iter),
        nodes=int(n_nodes),
        first_solution_iter=first_solution_iter,
        improvements=int(improvements),
        best_cost=float(best_cost),
        best_length=float(best_length),
        informed_samples=int(informed_samples),
        corridor_samples=int(corridor_samples),
    )
    if best_parent is None:
        return None, nodes, stats

    path = _reconstruct_path(pts, parent, int(best_parent), goal=goal)
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
                clearance=clearance if dim == 3 else None,
            ),
        )
        path[0] = start
        path[-1] = goal
    return path.astype(np.float32), nodes, stats
