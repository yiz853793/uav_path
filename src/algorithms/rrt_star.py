from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import segment_collision


@dataclass
class Node:
    p: np.ndarray
    parent: int
    cost: float


def _dist(env: GridEnv, a: np.ndarray, b: np.ndarray) -> float:
    return env.metric_distance(a, b)


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
    """支持 2D/3D 的极简 RRT* baseline。"""
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    dim = int(start.shape[0])
    rng = np.random.default_rng(seed)
    nodes: list[Node] = [Node(p=start.copy(), parent=-1, cost=0.0)]

    def sample() -> np.ndarray:
        if rng.random() < goal_sample_rate:
            return goal.copy()
        if dim == 2:
            return np.array([rng.uniform(0, env.W - 1), rng.uniform(0, env.H - 1)], dtype=np.float32)
        x = rng.uniform(0, env.W - 1)
        y = rng.uniform(0, env.H - 1)
        z_lo = env.min_safe_altitude_at(x, y)
        z = rng.uniform(z_lo, env.z_max)
        return np.array([x, y, z], dtype=np.float32)

    def nearest(p: np.ndarray) -> int:
        return int(np.argmin([_dist(env, n.p, p) for n in nodes]))

    def steer(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        v = p_to - p_from
        d = env.metric_distance(p_from, p_to)
        if d < 1e-9:
            return p_from.copy()
        step_len_m = float(step_len) * env.resolution
        if d <= step_len_m:
            q = p_to.copy()
        else:
            q = p_from + v * (step_len_m / max(1e-9, d))
        return env.clamp_point(q) if dim == 3 else q.astype(np.float32)

    def near(p: np.ndarray) -> list[int]:
        near_radius_m = float(near_radius) * env.resolution
        return [i for i, n in enumerate(nodes) if _dist(env, n.p, p) <= near_radius_m]

    goal_idx = None
    for _ in range(int(n_iter)):
        p_rand = sample()
        i_near = nearest(p_rand)
        p_new = steer(nodes[i_near].p, p_rand)
        if (dim == 2 and env.is_occupied(int(round(p_new[0])), int(round(p_new[1])))) or (dim == 3 and (not env.is_free_point(p_new))):
            continue
        if segment_collision(env, nodes[i_near].p, p_new, step=collision_step):
            continue
        candidates = near(p_new)
        best_parent = i_near
        best_cost = nodes[i_near].cost + _dist(env, nodes[i_near].p, p_new)
        for j in candidates:
            if segment_collision(env, nodes[j].p, p_new, step=collision_step):
                continue
            c = nodes[j].cost + _dist(env, nodes[j].p, p_new)
            if c < best_cost:
                best_cost = c
                best_parent = j
        nodes.append(Node(p=p_new, parent=best_parent, cost=best_cost))
        new_idx = len(nodes) - 1
        for j in candidates:
            if j == best_parent:
                continue
            if segment_collision(env, nodes[new_idx].p, nodes[j].p, step=collision_step):
                continue
            c_through = nodes[new_idx].cost + _dist(env, nodes[new_idx].p, nodes[j].p)
            if c_through < nodes[j].cost:
                nodes[j].parent = new_idx
                nodes[j].cost = c_through
        if _dist(env, p_new, goal) < float(step_len) * env.resolution and (not segment_collision(env, p_new, goal, step=collision_step)):
            nodes.append(Node(p=goal.copy(), parent=new_idx, cost=nodes[new_idx].cost + _dist(env, p_new, goal)))
            goal_idx = len(nodes) - 1
            break
    if goal_idx is None:
        return None, nodes
    path = []
    cur = goal_idx
    while cur != -1:
        path.append(nodes[cur].p)
        cur = nodes[cur].parent
    path = np.vstack(path[::-1]).astype(np.float32)
    return path, nodes
