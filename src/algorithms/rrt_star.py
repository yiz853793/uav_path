# src/algorithms/rrt_star.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import segment_collision


@dataclass
class Node:
    p: np.ndarray
    parent: int
    cost: float  # cost-to-come (length)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


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
    """
    极简 RRT* baseline（以路径长度最优为主）。
    输出：path（找到则为 np.ndarray[N,2]，否则 None），以及 nodes（用于调试/可视化）。
    """
    rng = np.random.default_rng(seed)
    nodes: list[Node] = [Node(p=np.array(start, dtype=np.float32), parent=-1, cost=0.0)]

    def sample() -> np.ndarray:
        if rng.random() < goal_sample_rate:
            return np.array(goal, dtype=np.float32)
        return np.array([rng.uniform(0, env.W - 1), rng.uniform(0, env.H - 1)], dtype=np.float32)

    def nearest(p: np.ndarray) -> int:
        d = [_dist(n.p, p) for n in nodes]
        return int(np.argmin(d))

    def steer(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        v = p_to - p_from
        d = float(np.linalg.norm(v))
        if d < 1e-9:
            return p_from.copy()
        if d <= step_len:
            return p_to.copy()
        return p_from + v / d * step_len

    def near(p: np.ndarray) -> list[int]:
        idx = []
        for i, n in enumerate(nodes):
            if _dist(n.p, p) <= near_radius:
                idx.append(i)
        return idx

    goal_idx = None

    for _ in range(int(n_iter)):
        p_rand = sample()
        i_near = nearest(p_rand)
        p_new = steer(nodes[i_near].p, p_rand)

        if env.is_occupied(int(round(p_new[0])), int(round(p_new[1]))):
            continue
        if segment_collision(env, nodes[i_near].p, p_new, step=collision_step):
            continue

        # choose best parent
        candidates = near(p_new)
        best_parent = i_near
        best_cost = nodes[i_near].cost + _dist(nodes[i_near].p, p_new)

        for j in candidates:
            if segment_collision(env, nodes[j].p, p_new, step=collision_step):
                continue
            c = nodes[j].cost + _dist(nodes[j].p, p_new)
            if c < best_cost:
                best_cost = c
                best_parent = j

        nodes.append(Node(p=p_new, parent=best_parent, cost=best_cost))
        new_idx = len(nodes) - 1

        # rewire
        for j in candidates:
            if j == best_parent:
                continue
            if segment_collision(env, nodes[new_idx].p, nodes[j].p, step=collision_step):
                continue
            c_through = nodes[new_idx].cost + _dist(nodes[new_idx].p, nodes[j].p)
            if c_through < nodes[j].cost:
                nodes[j].parent = new_idx
                nodes[j].cost = c_through

        # try connect goal
        if _dist(p_new, goal) < step_len:
            if not segment_collision(env, p_new, goal, step=collision_step):
                nodes.append(
                    Node(p=np.array(goal, dtype=np.float32), parent=new_idx, cost=nodes[new_idx].cost + _dist(p_new, goal))
                )
                goal_idx = len(nodes) - 1
                break

    if goal_idx is None:
        return None, nodes

    # reconstruct
    path = []
    cur = goal_idx
    while cur != -1:
        path.append(nodes[cur].p)
        cur = nodes[cur].parent
    path = np.vstack(path[::-1]).astype(np.float32)

    return path, nodes
