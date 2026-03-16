from __future__ import annotations
import heapq
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import segment_collision, sampled_cells_array, sampled_points_array
from ..models.path import shortcut_smooth


@dataclass
class PRMGraph:
    points: np.ndarray
    adj: list[list[tuple[int, float]]]


def _dist(env: GridEnv, a: np.ndarray, b: np.ndarray) -> float:
    return env.metric_distance(a, b)


def _edge_cost(env: GridEnv, p0: np.ndarray, p1: np.ndarray, threat_weight: float, collision_step: float) -> float:
    base = _dist(env, p0, p1)
    if threat_weight == 0.0:
        return base
    if p0.shape[0] == 2:
        ix, iy = sampled_cells_array(p0, p1, step=collision_step, xy_resolution=env.resolution)
        ix = np.clip(ix, 0, env.W - 1)
        iy = np.clip(iy, 0, env.H - 1)
        threat_cost = float(env.threat[iy, ix].mean(dtype=np.float64))
    else:
        pts = sampled_points_array(p0, p1, step=collision_step, xy_resolution=env.resolution)
        ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
        clearance = np.maximum(0.0, pts[:, 2] - env.height[iy, ix])
        threat_cost = float(np.mean(env.threat[iy, ix] * np.exp(-0.06 * clearance), dtype=np.float64))
    return float(base + float(threat_weight) * threat_cost)


def _sample_free_points(env: GridEnv, start: np.ndarray, goal: np.ndarray, n_samples: int, rng: np.random.Generator, clearance: int = 0) -> np.ndarray:
    occ = env.occupancy
    if clearance > 0:
        occ = env.inflate_obstacles(clearance).occupancy
    free_y, free_x = np.where(~occ)
    if len(free_x) == 0:
        return np.vstack([start, goal]).astype(np.float32)
    n_pick = min(int(n_samples), int(len(free_x)))
    idx = rng.choice(len(free_x), size=n_pick, replace=False)
    xy = np.stack([free_x[idx], free_y[idx]], axis=1).astype(np.float32)
    xy[:, 0] += rng.uniform(-0.35, 0.35, size=n_pick).astype(np.float32)
    xy[:, 1] += rng.uniform(-0.35, 0.35, size=n_pick).astype(np.float32)
    xy[:, 0] = np.clip(xy[:, 0], 0.0, env.W - 1.0)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, env.H - 1.0)
    if start.shape[0] == 2:
        pts = xy
    else:
        ground = np.array([env.ground_height_at(x, y) for x, y in xy], dtype=np.float32)
        z_lo = ground + env.min_clearance
        z = rng.uniform(z_lo, np.full_like(z_lo, env.z_max)).astype(np.float32)
        pts = np.concatenate([xy, z[:, None]], axis=1)
    return np.vstack([start[None, :], goal[None, :], pts]).astype(np.float32)


def _build_roadmap(env: GridEnv, points: np.ndarray, k: int, max_edge_len: float, collision_step: float, threat_weight: float) -> PRMGraph:
    n = int(points.shape[0])
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if n <= 1:
        return PRMGraph(points=points, adj=adj)
    diff = points[:, None, :] - points[None, :, :]
    diff[..., 0] *= env.resolution
    diff[..., 1] *= env.resolution
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    k_eff = max(1, min(int(k), n - 1))
    for i in range(n):
        nbr_idx = np.argpartition(dist[i], kth=k_eff - 1)[:k_eff]
        for j in nbr_idx.tolist():
            if j <= i:
                continue
            dij = float(dist[i, j])
            if np.isfinite(max_edge_len) and dij > float(max_edge_len):
                continue
            if segment_collision(env, points[i], points[j], step=collision_step):
                continue
            w = _edge_cost(env, points[i], points[j], threat_weight=threat_weight, collision_step=collision_step)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return PRMGraph(points=points, adj=adj)


def _astar_graph(graph: PRMGraph, graph_env: GridEnv, start_idx: int = 0, goal_idx: int = 1) -> list[int] | None:
    pts = graph.points
    gscore = {start_idx: 0.0}
    parent: dict[int, int] = {}
    open_heap: list[tuple[float, float, int]] = []
    h0 = _dist(graph_env, pts[start_idx], pts[goal_idx])
    heapq.heappush(open_heap, (h0, 0.0, start_idx))
    while open_heap:
        f, g, u = heapq.heappop(open_heap)
        if g > gscore.get(u, float('inf')) + 1e-12:
            continue
        if u == goal_idx:
            idx_path = [u]
            while u in parent:
                u = parent[u]
                idx_path.append(u)
            idx_path.reverse()
            return idx_path
        for v, w in graph.adj[u]:
            ng = g + float(w)
            if ng < gscore.get(v, float('inf')):
                gscore[v] = ng
                parent[v] = u
                nf = ng + _dist(graph_env, pts[v], pts[goal_idx])
                heapq.heappush(open_heap, (nf, ng, v))
    return None


def prm(env: GridEnv, start: np.ndarray, goal: np.ndarray, n_samples: int = 800, k: int = 12, max_edge_len: float = 28.0, collision_step: float = 0.5, threat_weight: float = 0.0, sample_clearance: int = 0, smooth: bool = True, smooth_n_try: int = 80, seed: int = 0):
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if (not env.in_bounds(start)) or (not env.in_bounds(goal)) or (not env.is_free_point(start)) or (not env.is_free_point(goal)):
        return None, PRMGraph(points=np.vstack([start, goal]).astype(np.float32), adj=[[], []])
    rng = np.random.default_rng(seed)
    points = _sample_free_points(env, start, goal, n_samples=n_samples, rng=rng, clearance=sample_clearance)
    graph = _build_roadmap(env, points, k=k, max_edge_len=float(max_edge_len), collision_step=float(collision_step), threat_weight=float(threat_weight))
    idx_path = _astar_graph(graph, env, start_idx=0, goal_idx=1)
    if idx_path is None:
        return None, graph
    path = graph.points[np.asarray(idx_path, dtype=np.int32)].astype(np.float32)
    if smooth and len(path) >= 3:
        path = shortcut_smooth(path, n_try=int(smooth_n_try), rng=rng, collision_fn=lambda a, b: segment_collision(env, a, b, step=collision_step))
    return path, graph
