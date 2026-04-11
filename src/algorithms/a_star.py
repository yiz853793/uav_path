from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.grid_env import GridEnv


@dataclass
class AStarResult:
    path: Optional[np.ndarray]  # [N,2] float32 (x,y)
    n_expanded: int


def _nearest_free_cell(env: GridEnv, sx: int, sy: int, max_r: int = 20) -> Optional[Tuple[int, int]]:
    occ = env.occupancy
    H, W = occ.shape
    if 0 <= sx < W and 0 <= sy < H and (not bool(occ[sy, sx])):
        return (sx, sy)
    for r in range(1, int(max_r) + 1):
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


def _octile_heuristic(x: int, y: int, gx: int, gy: int) -> float:
    dx = abs(x - gx)
    dy = abs(y - gy)
    diag = min(dx, dy)
    straight = max(dx, dy) - diag
    return float(diag * np.sqrt(2.0) + straight)


def astar(
    env: GridEnv,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    *,
    threat_weight: float = 0.0,
    penalty_map: Optional[np.ndarray] = None,
    allow_diagonal: bool = True,
    max_expansions: int = 300_000,
    heuristic_weight: float = 1.0,
) -> AStarResult:
    """A* on the occupancy grid.

    ``heuristic_weight`` > 1.0 turns it into a greedier weighted-A*, which is
    useful when we only need a good initialization backbone quickly.
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
            (1, 1, float(np.sqrt(2.0))),
            (1, -1, float(np.sqrt(2.0))),
            (-1, 1, float(np.sqrt(2.0))),
            (-1, -1, float(np.sqrt(2.0))),
        ]
    else:
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

    occ = env.occupancy
    thr = env.threat
    h_weight = float(max(1.0, heuristic_weight))
    threat_w = float(threat_weight)

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (h_weight * _octile_heuristic(sx, sy, gx, gy), 0.0, (sx, sy)))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    gget = gscore.get
    n_expanded = 0

    while open_heap:
        _, g, (x, y) = heapq.heappop(open_heap)
        if g > gget((x, y), float('inf')) + 1e-12:
            continue

        n_expanded += 1
        if n_expanded > int(max_expansions):
            break
        if (x, y) == (gx, gy):
            path_cells = [(x, y)]
            while (x, y) in came_from:
                x, y = came_from[(x, y)]
                path_cells.append((x, y))
            path_cells.reverse()
            return AStarResult(path=np.asarray(path_cells, dtype=np.float32), n_expanded=n_expanded)

        for dx, dy, step_cost in moves:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= env.W or ny < 0 or ny >= env.H:
                continue
            if bool(occ[ny, nx]):
                continue
            if allow_diagonal and dx != 0 and dy != 0:
                if bool(occ[y, x + dx]) or bool(occ[y + dy, x]):
                    continue

            extra = 0.0
            if threat_w != 0.0:
                extra += threat_w * float(thr[ny, nx])
            if penalty_map is not None:
                extra += float(penalty_map[ny, nx])

            ng = g + float(step_cost) + extra
            if ng < gget((nx, ny), float('inf')):
                gscore[(nx, ny)] = ng
                came_from[(nx, ny)] = (x, y)
                nf = ng + h_weight * _octile_heuristic(nx, ny, gx, gy)
                heapq.heappush(open_heap, (nf, ng, (nx, ny)))

    return AStarResult(path=None, n_expanded=n_expanded)
