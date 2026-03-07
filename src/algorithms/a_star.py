# src/algorithms/a_star.py

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.grid_env import GridEnv


@dataclass
class AStarResult:
    """A* search result."""

    path: Optional[np.ndarray]  # [N,2] float32 (x,y)
    n_expanded: int


def _in_bounds(env: GridEnv, x: int, y: int) -> bool:
    return 0 <= x < env.W and 0 <= y < env.H


def _nearest_free_cell(env: GridEnv, sx: int, sy: int, max_r: int = 20) -> Optional[Tuple[int, int]]:
    """If (sx,sy) is occupied, find the nearest free cell within a square radius.

    NOTE: This function can be hot in large-map A*. Use direct occupancy indexing to avoid Python method overhead.
    """
    occ = env.occupancy
    H, W = occ.shape

    if 0 <= sx < W and 0 <= sy < H and (not bool(occ[sy, sx])):
        return (sx, sy)

    for r in range(1, int(max_r) + 1):
        # perimeter scan (fast enough for our map sizes)
        for dx in range(-r, r + 1):
            for dy in (-r, r):
                x, y = sx + dx, sy + dy
                if 0 <= x < W and 0 <= y < H and (not bool(occ[y, x])):
                    return (x, y)
        for dy in range(-r + 1, r):
            for dx in (-r, r):
                x, y = sx + dx, sy + dy
                if 0 <= x < W and 0 <= y < H and (not bool(occ[y, x])):
                    return (x, y)
    return None


def _heuristic(x: int, y: int, gx: int, gy: int) -> float:
    # Euclidean heuristic is consistent for 8-neighborhood
    return float(((x - gx) ** 2 + (y - gy) ** 2) ** 0.5)


def astar(
    env: GridEnv,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    threat_weight: float = 0.0,
    penalty_map: Optional[np.ndarray] = None,
    allow_diagonal: bool = True,
    max_expansions: int = 300_000,
) -> AStarResult:
    """A* on the occupancy grid.

    Args:
        env: GridEnv
        start_xy, goal_xy: continuous (x,y). Internally rounded to int cells.
        threat_weight: extra cost term per cell: cost += threat_weight * threat[y,x]
        penalty_map: optional extra cost map (same shape as env.threat/occupancy),
          cost += penalty_map[y,x]. Used for generating diverse A* backbones.
        allow_diagonal: 8-connected if True else 4-connected
        max_expansions: safety cap
    """

    sx, sy = int(round(float(start_xy[0]))), int(round(float(start_xy[1])))
    gx, gy = int(round(float(goal_xy[0]))), int(round(float(goal_xy[1])))

    s2 = _nearest_free_cell(env, sx, sy)
    g2 = _nearest_free_cell(env, gx, gy)
    if s2 is None or g2 is None:
        return AStarResult(path=None, n_expanded=0)
    sx, sy = s2
    gx, gy = g2

    if (sx, sy) == (gx, gy):
        return AStarResult(path=np.array([[sx, sy]], dtype=np.float32), n_expanded=0)

    if allow_diagonal:
        moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, 2**0.5),
            (1, -1, 2**0.5),
            (-1, 1, 2**0.5),
            (-1, -1, 2**0.5),
        ]
    else:
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

    # Hot-path arrays: avoid repeated attribute lookups / method calls in the inner loop
    occ = env.occupancy
    thr = env.threat

    # (f, g, (x,y))
    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (_heuristic(sx, sy, gx, gy), 0.0, (sx, sy)))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    n_expanded = 0

    while open_heap:
        f, g, (x, y) = heapq.heappop(open_heap)

        # stale entry
        if g > gscore.get((x, y), float("inf")) + 1e-12:
            continue

        n_expanded += 1
        if n_expanded > int(max_expansions):
            break

        if (x, y) == (gx, gy):
            # reconstruct
            path_cells = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path_cells.append((x, y))
            path_cells.reverse()
            path = np.array(path_cells, dtype=np.float32)
            return AStarResult(path=path, n_expanded=n_expanded)

        for dx, dy, step_cost in moves:
            nx, ny = x + dx, y + dy
            if not _in_bounds(env, nx, ny):
                continue
            if bool(occ[ny, nx]):
                continue

            # discourage corner-cutting when diagonal
            if allow_diagonal and dx != 0 and dy != 0:
                if bool(occ[y, x + dx]) or bool(occ[y + dy, x]):
                    continue

            extra = 0.0
            if threat_weight != 0.0:
                extra += float(threat_weight) * float(thr[ny, nx])
            if penalty_map is not None:
                # penalty_map should be same shape as env.threat/occupancy (H,W)
                extra += float(penalty_map[ny, nx])

            ng = g + step_cost + extra
            if ng < gscore.get((nx, ny), float("inf")):
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = (x, y)
                nf = ng + _heuristic(nx, ny, gx, gy)
                heapq.heappush(open_heap, (nf, ng, (nx, ny)))

    return AStarResult(path=None, n_expanded=n_expanded)
