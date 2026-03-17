from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..env.collision import sampled_cells_array, segment_collision
from ..env.grid_env import GridEnv
from ..models.path import shortcut_smooth


@dataclass
class PRMGraph:
    points: np.ndarray  # [N,3]
    adj: list[list[tuple[int, float]]]
    stats: dict = field(default_factory=dict)


def _dist(env: GridEnv, a: np.ndarray, b: np.ndarray) -> float:
    return env.metric_distance(a, b)


def _point_is_free_3d(env: GridEnv, p: np.ndarray, clearance_margin: float = 0.0) -> bool:
    q = np.asarray(p, dtype=np.float32)
    if q.shape[0] != 3:
        raise ValueError("point must be 3D")
    return env.is_free_point(q, clearance=env.min_clearance + float(clearance_margin))


def _dedup_points(points_xyz: np.ndarray, snap_xy: float = 0.75, snap_z: float = 2.5) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.asarray(points_xyz, dtype=np.float32).reshape(0, 3)
    arr = np.asarray(points_xyz, dtype=np.float32)
    q = np.empty_like(arr, dtype=np.int32)
    q[:, 0] = np.round(arr[:, 0] / float(snap_xy)).astype(np.int32)
    q[:, 1] = np.round(arr[:, 1] / float(snap_xy)).astype(np.int32)
    q[:, 2] = np.round(arr[:, 2] / float(snap_z)).astype(np.int32)
    _, uniq_idx = np.unique(q, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return arr[uniq_idx].astype(np.float32)


def _global_bridge_altitude(env: GridEnv, clearance_margin: float) -> float:
    z = float(np.percentile(env.height, 97.0) + env.min_clearance + float(clearance_margin) + 18.0)
    return float(np.clip(z, env.z_min + 1.0, env.z_max - 1.0))


def _local_safe_altitude(env: GridEnv, x: float, y: float, clearance_margin: float) -> float:
    return float(np.clip(env.min_safe_altitude_at(x, y, clearance=env.min_clearance + float(clearance_margin)), env.z_min, env.z_max))


def _sample_z_given_xy(
    env: GridEnv,
    x: float,
    y: float,
    rng: np.random.Generator,
    clearance_margin: float,
    mode: str,
    bridge_alt: float,
) -> float:
    safe = _local_safe_altitude(env, x, y, clearance_margin)
    z_hi = env.z_max - 1.0
    if z_hi <= safe + 1e-3:
        return float(safe)

    if mode == "low":
        z = safe + float(rng.uniform(0.5, min(10.0, max(1.0, z_hi - safe))))
    elif mode == "high":
        low = max(safe + 0.5, bridge_alt - 12.0)
        high = min(z_hi, bridge_alt + 12.0)
        if high <= low:
            z = max(safe + 0.5, min(z_hi, bridge_alt))
        else:
            z = float(rng.uniform(low, high))
    else:  # mixed
        r = float(rng.random())
        if r < 0.35:
            z = safe + float(rng.uniform(0.5, min(10.0, max(1.0, z_hi - safe))))
        elif r < 0.75:
            z = safe + float(rng.uniform(8.0, min(28.0, max(9.0, z_hi - safe))))
        else:
            low = max(safe + 0.5, bridge_alt - 12.0)
            high = min(z_hi, bridge_alt + 12.0)
            if high <= low:
                z = max(safe + 0.5, min(z_hi, bridge_alt))
            else:
                z = float(rng.uniform(low, high))
    return float(np.clip(z, safe, z_hi))


def _sample_global_points_3d(
    env: GridEnv,
    n_samples: int,
    rng: np.random.Generator,
    clearance_margin: float,
) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = rng.uniform(0.0, env.W - 1.0, size=n_samples).astype(np.float32)
    ys = rng.uniform(0.0, env.H - 1.0, size=n_samples).astype(np.float32)
    bridge_alt = _global_bridge_altitude(env, clearance_margin)
    pts = np.zeros((n_samples, 3), dtype=np.float32)
    for i in range(n_samples):
        pts[i, 0] = xs[i]
        pts[i, 1] = ys[i]
        pts[i, 2] = _sample_z_given_xy(env, float(xs[i]), float(ys[i]), rng, clearance_margin, mode="mixed", bridge_alt=bridge_alt)
    return pts


def _sample_local_ball_3d(
    env: GridEnv,
    center_xyz: np.ndarray,
    radius_cells: float,
    n_samples: int,
    rng: np.random.Generator,
    clearance_margin: float,
    high_bias: float = 0.4,
) -> np.ndarray:
    if n_samples <= 0 or radius_cells <= 0.0:
        return np.zeros((0, 3), dtype=np.float32)
    c = np.asarray(center_xyz, dtype=np.float32)
    out = np.zeros((n_samples, 3), dtype=np.float32)
    bridge_alt = _global_bridge_altitude(env, clearance_margin)
    for i in range(n_samples):
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        r = float(radius_cells) * np.sqrt(float(rng.uniform(0.0, 1.0)))
        x = float(np.clip(c[0] + r * np.cos(theta), 0.0, env.W - 1.0))
        y = float(np.clip(c[1] + r * np.sin(theta), 0.0, env.H - 1.0))
        mode = "high" if float(rng.random()) < high_bias else "mixed"
        z = _sample_z_given_xy(env, x, y, rng, clearance_margin, mode=mode, bridge_alt=bridge_alt)
        out[i] = (x, y, z)
    return out


def _sample_corridor_3d(
    env: GridEnv,
    start_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    half_width_cells: float,
    n_samples: int,
    rng: np.random.Generator,
    clearance_margin: float,
    high_bias: float = 0.75,
) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    s = np.asarray(start_xyz, dtype=np.float32)
    g = np.asarray(goal_xyz, dtype=np.float32)
    dxy = g[:2] - s[:2]
    norm = float(np.linalg.norm(dxy))
    if norm < 1e-6:
        return _sample_local_ball_3d(env, s, max(4.0, half_width_cells), n_samples, rng, clearance_margin, high_bias=high_bias)
    tvec = dxy / norm
    nvec = np.array([-tvec[1], tvec[0]], dtype=np.float32)
    bridge_alt = max(_global_bridge_altitude(env, clearance_margin), float(max(s[2], g[2])) + 4.0)
    out = np.zeros((n_samples, 3), dtype=np.float32)
    for i in range(n_samples):
        t = float(rng.uniform(0.0, 1.0))
        lateral = float(rng.normal(0.0, max(1.0, half_width_cells / 2.4)))
        lateral = float(np.clip(lateral, -half_width_cells, half_width_cells))
        longitudinal = float(rng.normal(0.0, max(1.0, norm * 0.02)))
        xy = s[:2] + t * dxy + lateral * nvec + longitudinal * tvec
        x = float(np.clip(xy[0], 0.0, env.W - 1.0))
        y = float(np.clip(xy[1], 0.0, env.H - 1.0))
        mode = "high" if float(rng.random()) < high_bias else "mixed"
        z = _sample_z_given_xy(env, x, y, rng, clearance_margin, mode=mode, bridge_alt=bridge_alt)
        out[i] = (x, y, z)
    return out


def _assemble_points_3d(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    corridor_bias: float,
    local_bias: float,
    clearance_margin: float,
) -> np.ndarray:
    base = max(0, int(n_samples))
    corridor_n = int(round(base * float(corridor_bias)))
    local_n_total = int(round(base * float(local_bias)))
    global_n = max(0, base - corridor_n - local_n_total)
    local_each = local_n_total // 2

    parts = [start[None, :], goal[None, :]]
    if global_n > 0:
        parts.append(_sample_global_points_3d(env, global_n, rng, clearance_margin))
    if local_each > 0:
        radii = [20.0, 40.0, 80.0, 140.0]
        counts = [local_each // len(radii)] * len(radii)
        for i in range(local_each % len(radii)):
            counts[i] += 1
        for rad, cnt in zip(radii, counts):
            if cnt > 0:
                parts.append(_sample_local_ball_3d(env, start, rad, cnt, rng, clearance_margin, high_bias=0.45))
                parts.append(_sample_local_ball_3d(env, goal, rad, cnt, rng, clearance_margin, high_bias=0.45))
    if corridor_n > 0:
        line_len = float(np.linalg.norm(goal[:2] - start[:2]))
        widths = [max(18.0, line_len * 0.02), max(35.0, line_len * 0.04), max(60.0, line_len * 0.07)]
        counts = [corridor_n // len(widths)] * len(widths)
        for i in range(corridor_n % len(widths)):
            counts[i] += 1
        for w, cnt in zip(widths, counts):
            if cnt > 0:
                parts.append(_sample_corridor_3d(env, start, goal, w, cnt, rng, clearance_margin, high_bias=0.82))
    pts = np.vstack([p for p in parts if p is not None and len(p) > 0]).astype(np.float32)
    return _dedup_points(pts)


def _build_spatial_hash(points_xyz: np.ndarray, cell_size_cells: float) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    cell_size = max(1e-6, float(cell_size_cells))
    for i, p in enumerate(points_xyz):
        bx = int(np.floor(float(p[0]) / cell_size))
        by = int(np.floor(float(p[1]) / cell_size))
        buckets.setdefault((bx, by), []).append(i)
    return buckets


def _candidate_neighbors_from_hash(
    points_xyz: np.ndarray,
    i: int,
    buckets: dict[tuple[int, int], list[int]],
    cell_size_cells: float,
    search_rings: int,
) -> list[int]:
    p = points_xyz[i]
    bx = int(np.floor(float(p[0]) / cell_size_cells))
    by = int(np.floor(float(p[1]) / cell_size_cells))
    cand: list[int] = []
    for dx in range(-search_rings, search_rings + 1):
        for dy in range(-search_rings, search_rings + 1):
            cand.extend(buckets.get((bx + dx, by + dy), []))
    return cand


def _collect_component_info(graph: PRMGraph, start_idx: int = 0, goal_idx: int = 1) -> dict:
    n = len(graph.adj)
    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "start_degree": 0,
            "goal_degree": 0,
            "start_component_size": 0,
            "goal_component_size": 0,
            "largest_component_size": 0,
            "start_goal_connected": False,
        }
    seen = np.zeros(n, dtype=bool)
    comp_sizes = []
    start_comp = goal_comp = 0
    start_goal_connected = False
    for root in range(n):
        if seen[root]:
            continue
        q = deque([root])
        seen[root] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v, _ in graph.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        cset = set(comp)
        csize = len(comp)
        comp_sizes.append(csize)
        if start_idx in cset:
            start_comp = csize
            if goal_idx in cset:
                start_goal_connected = True
        if goal_idx in cset:
            goal_comp = csize
    n_edges = int(sum(len(x) for x in graph.adj) // 2)
    z_vals = graph.points[:, 2] if len(graph.points) > 0 and graph.points.shape[1] >= 3 else np.zeros(0, dtype=np.float32)
    return {
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "start_degree": int(len(graph.adj[start_idx])) if start_idx < n else 0,
        "goal_degree": int(len(graph.adj[goal_idx])) if goal_idx < n else 0,
        "start_component_size": int(start_comp),
        "goal_component_size": int(goal_comp),
        "largest_component_size": int(max(comp_sizes) if comp_sizes else 0),
        "start_goal_connected": bool(start_goal_connected),
        "z_min_nodes": float(np.min(z_vals)) if len(z_vals) else 0.0,
        "z_max_nodes": float(np.max(z_vals)) if len(z_vals) else 0.0,
        "z_mean_nodes": float(np.mean(z_vals)) if len(z_vals) else 0.0,
    }


def _edge_cost_3d(
    env: GridEnv,
    p0: np.ndarray,
    p1: np.ndarray,
    threat_weight: float,
    collision_step: float,
) -> float:
    base = _dist(env, p0, p1)
    if threat_weight == 0.0:
        return float(base)
    ix, iy = sampled_cells_array(p0[:2], p1[:2], step=collision_step, xy_resolution=env.resolution)
    ix = np.clip(ix, 0, env.W - 1)
    iy = np.clip(iy, 0, env.H - 1)
    threat_cost = float(env.threat[iy, ix].mean(dtype=np.float64))
    return float(base + float(threat_weight) * threat_cost)


def _build_roadmap_3d(
    env: GridEnv,
    points_xyz: np.ndarray,
    k: int,
    max_edge_len: float,
    collision_step: float,
    threat_weight: float,
    clearance_margin: float,
) -> PRMGraph:
    n = int(points_xyz.shape[0])
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    if n <= 1:
        return PRMGraph(points=points_xyz, adj=adj, stats={"n_nodes": n, "n_edges": 0})

    k_eff = max(1, min(int(k), n - 1))
    max_edge_len = float(max_edge_len)
    cell_size_cells = max(1.0, max_edge_len / max(float(env.resolution), 1e-6))
    buckets = _build_spatial_hash(points_xyz, cell_size_cells=cell_size_cells)
    max_search_rings = 10
    attempted_edges = 0
    collision_rejects = 0

    for i in range(n):
        cand: list[int] = []
        target_unique = max(k_eff * 10, 40)
        for rings in range(1, max_search_rings + 1):
            cand = _candidate_neighbors_from_hash(points_xyz, i, buckets, cell_size_cells, rings)
            uniq = len(set(c for c in cand if c != i))
            if uniq >= target_unique or rings == max_search_rings:
                break
        cand = [j for j in set(cand) if j != i]
        if not cand:
            continue

        pj = points_xyz[np.asarray(cand, dtype=np.int32)]
        dxyz = pj - points_xyz[i][None, :]
        dxyz[:, 0] *= env.resolution
        dxyz[:, 1] *= env.resolution
        dists = np.linalg.norm(dxyz, axis=1)
        mask = dists <= max_edge_len
        if not np.any(mask):
            continue
        cand_arr = np.asarray(cand, dtype=np.int32)[mask]
        dist_arr = dists[mask]
        order = np.argsort(dist_arr)
        cand_arr = cand_arr[order[:target_unique]]
        added = 0
        for j in cand_arr.tolist():
            if j <= i:
                continue
            attempted_edges += 1
            if segment_collision(env, points_xyz[i], points_xyz[j], step=collision_step, clearance=env.min_clearance + float(clearance_margin)):
                collision_rejects += 1
                continue
            w = _edge_cost_3d(env, points_xyz[i], points_xyz[j], threat_weight, collision_step)
            adj[i].append((j, float(w)))
            adj[j].append((i, float(w)))
            added += 1
            if added >= k_eff:
                break

    graph = PRMGraph(points=points_xyz, adj=adj)
    stats = _collect_component_info(graph)
    stats.update({
        "attempted_edges": int(attempted_edges),
        "collision_rejects": int(collision_rejects),
        "collision_reject_ratio": float(collision_rejects / max(1, attempted_edges)),
        "planner_mode": "3d_node_prm",
    })
    graph.stats = stats
    return graph


def _astar_graph(graph: PRMGraph, graph_env: GridEnv, start_idx: int = 0, goal_idx: int = 1) -> list[int] | None:
    pts = graph.points
    gscore = {start_idx: 0.0}
    parent: dict[int, int] = {}
    open_heap: list[tuple[float, float, int]] = []
    h0 = _dist(graph_env, pts[start_idx], pts[goal_idx])
    heapq.heappush(open_heap, (h0, 0.0, start_idx))
    while open_heap:
        f, g, u = heapq.heappop(open_heap)
        if g > gscore.get(u, float("inf")) + 1e-12:
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
            if ng < gscore.get(v, float("inf")):
                gscore[v] = ng
                parent[v] = u
                nf = ng + _dist(graph_env, pts[v], pts[goal_idx])
                heapq.heappush(open_heap, (nf, ng, v))
    return None


def prm(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_samples: int = 800,
    k: int = 12,
    max_edge_len: float = 28.0,
    collision_step: float = 0.5,
    threat_weight: float = 0.0,
    sample_clearance: int = 0,  # 兼容旧接口，当前 3D PRM 不使用该参数
    smooth: bool = True,
    smooth_n_try: int = 80,
    seed: int = 0,
    lift_to_3d_when_needed: bool = True,
    clearance_margin: float = 2.0,
):
    del sample_clearance
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if start.shape[0] == 2:
        start = np.array([start[0], start[1], _local_safe_altitude(env, float(start[0]), float(start[1]), clearance_margin) + 2.0], dtype=np.float32)
    if goal.shape[0] == 2:
        goal = np.array([goal[0], goal[1], _local_safe_altitude(env, float(goal[0]), float(goal[1]), clearance_margin) + 2.0], dtype=np.float32)
    start = env.clamp_point(start, clearance=env.min_clearance + float(clearance_margin))
    goal = env.clamp_point(goal, clearance=env.min_clearance + float(clearance_margin))
    empty_graph = PRMGraph(points=np.vstack([start, goal]).astype(np.float32), adj=[[], []], stats={})

    if (not env.in_bounds(start)) or (not env.in_bounds(goal)):
        empty_graph.stats = {"reason": "oob", "planner_mode": "3d_node_prm"}
        return None, empty_graph
    if (not _point_is_free_3d(env, start, clearance_margin=clearance_margin)) or (not _point_is_free_3d(env, goal, clearance_margin=clearance_margin)):
        empty_graph.stats = {"reason": "start_or_goal_not_free_3d", "planner_mode": "3d_node_prm"}
        return None, empty_graph

    rng = np.random.default_rng(seed)
    best_graph = empty_graph
    best_idx_path = None
    best_points = None

    retry_plans = [
        {"sample_mult": 1.00, "k_mult": 1.00, "edge_mult": 1.00, "corridor_bias": 0.28, "local_bias": 0.18},
        {"sample_mult": 1.18, "k_mult": 1.20, "edge_mult": 1.18, "corridor_bias": 0.38, "local_bias": 0.22},
        {"sample_mult": 1.35, "k_mult": 1.35, "edge_mult": 1.35, "corridor_bias": 0.48, "local_bias": 0.24},
    ]

    attempts_summary = []
    for attempt_id, plan in enumerate(retry_plans, start=1):
        ns = int(max(64, round(float(n_samples) * plan["sample_mult"])))
        kk = int(max(8, round(float(k) * plan["k_mult"])))
        edge = float(max_edge_len) * float(plan["edge_mult"])

        points = _assemble_points_3d(
            env,
            start,
            goal,
            n_samples=ns,
            rng=rng,
            corridor_bias=float(plan["corridor_bias"]),
            local_bias=float(plan["local_bias"]),
            clearance_margin=float(clearance_margin),
        )
        graph = _build_roadmap_3d(
            env,
            points,
            k=kk,
            max_edge_len=edge,
            collision_step=float(collision_step),
            threat_weight=float(threat_weight),
            clearance_margin=float(clearance_margin),
        )
        graph.stats.update({
            "attempt_id": int(attempt_id),
            "requested_samples": int(ns),
            "effective_points": int(len(points)),
            "k_used": int(kk),
            "max_edge_len_used": float(edge),
            "corridor_bias": float(plan["corridor_bias"]),
            "local_bias": float(plan["local_bias"]),
        })
        attempts_summary.append(dict(graph.stats))

        idx_path = _astar_graph(graph, env, start_idx=0, goal_idx=1)
        best_graph = graph
        best_idx_path = idx_path
        best_points = points
        if idx_path is not None:
            break

    best_graph.stats["attempts"] = attempts_summary
    best_graph.stats["n_attempts"] = len(attempts_summary)

    if best_idx_path is None or best_points is None:
        best_graph.stats["found"] = False
        return None, best_graph

    path_xyz = best_points[np.asarray(best_idx_path, dtype=np.int32)].astype(np.float32)
    if smooth and len(path_xyz) >= 3:
        path_xyz = shortcut_smooth(
            path_xyz,
            n_try=int(smooth_n_try),
            rng=rng,
            collision_fn=lambda a, b: segment_collision(
                env,
                np.asarray(a, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
                step=float(collision_step),
                clearance=env.min_clearance + float(clearance_margin),
            ),
        )
    path_xyz[0] = start
    path_xyz[-1] = goal
    best_graph.stats["found"] = True
    best_graph.stats["path_nodes"] = int(len(path_xyz))
    best_graph.stats["path_z_min"] = float(np.min(path_xyz[:, 2]))
    best_graph.stats["path_z_max"] = float(np.max(path_xyz[:, 2]))

    if lift_to_3d_when_needed:
        return path_xyz.astype(np.float32), best_graph
    return path_xyz[:, :2].astype(np.float32), best_graph
