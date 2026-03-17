from __future__ import annotations
import numpy as np
from .grid_env import GridEnv


def sampled_points_array(p0: np.ndarray, p1: np.ndarray, step: float = 0.5, xy_resolution: float = 1.0) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    d = p1 - p0
    dm = d.copy()
    if dm.size >= 1:
        dm[0] *= float(xy_resolution)
    if dm.size >= 2:
        dm[1] *= float(xy_resolution)
    dist = float(np.linalg.norm(dm))
    if dist < 1e-9:
        return p0[None, :].copy()
    step_m = max(1e-6, float(step) * float(xy_resolution))
    n = max(1, int(np.ceil(dist / step_m)))
    t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[:, None]
    return (p0[None, :] + t * d[None, :]).astype(np.float32)


def sampled_cells(p0: np.ndarray, p1: np.ndarray, step: float = 0.5):
    ix, iy = sampled_cells_array(p0, p1, step=step)
    for x, y in zip(ix.tolist(), iy.tolist()):
        yield int(x), int(y)


def sampled_cells_array(p0: np.ndarray, p1: np.ndarray, step: float = 0.5, xy_resolution: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    pts = sampled_points_array(np.asarray(p0)[:2], np.asarray(p1)[:2], step=step, xy_resolution=xy_resolution)
    ix = np.rint(pts[:, 0]).astype(np.int32, copy=False)
    iy = np.rint(pts[:, 1]).astype(np.int32, copy=False)
    return ix, iy


def segment_collision(env: GridEnv, p0: np.ndarray, p1: np.ndarray, step: float = 0.5, clearance: float | None = None) -> bool:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    pts = sampled_points_array(p0, p1, step=step, xy_resolution=env.resolution)
    ix = np.rint(pts[:, 0]).astype(np.int32, copy=False)
    iy = np.rint(pts[:, 1]).astype(np.int32, copy=False)
    oob = (ix < 0) | (ix >= env.W) | (iy < 0) | (iy >= env.H)
    if bool(np.any(oob)):
        return True
    if pts.shape[1] < 3:
        if bool(np.any(env.occupancy[iy, ix])):
            return True
        return False
    z = pts[:, 2]
    ground = env.height[iy, ix]
    c = env.min_clearance if clearance is None else float(clearance)
    if bool(np.any(z < ground + c)):
        return True
    if bool(np.any(z < env.z_min)) or bool(np.any(z > env.z_max)):
        return True
    return False


def path_collision(env: GridEnv, path: np.ndarray, step: float = 0.5, clearance: float | None = None) -> bool:
    assert path.ndim == 2 and path.shape[1] in (2, 3)
    for i in range(1, len(path)):
        if segment_collision(env, path[i - 1], path[i], step=step, clearance=clearance):
            return True
    return False
