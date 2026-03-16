from __future__ import annotations
import numpy as np
from ..env.grid_env import GridEnv
from ..env.collision import sampled_cells_array, sampled_points_array


def objective_length(env: GridEnv, path: np.ndarray) -> float:
    return env.path_length(path)


def objective_threat(env: GridEnv, path: np.ndarray, sample_step: float = 0.5, altitude_decay: float = 0.06) -> float:
    total = 0.0
    for i in range(1, len(path)):
        if path.shape[1] == 2:
            ix, iy = sampled_cells_array(path[i - 1], path[i], step=sample_step, xy_resolution=env.resolution)
            ix2 = np.clip(ix, 0, env.W - 1)
            iy2 = np.clip(iy, 0, env.H - 1)
            total += float(env.threat[iy2, ix2].sum(dtype=np.float64))
        else:
            pts = sampled_points_array(path[i - 1], path[i], step=sample_step, xy_resolution=env.resolution)
            ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
            iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
            ground = env.height[iy, ix]
            clearance = np.maximum(0.0, pts[:, 2] - ground)
            weight = np.exp(-float(altitude_decay) * clearance)
            total += float(np.sum(env.threat[iy, ix] * weight, dtype=np.float64))
    return float(total)


def objective_smoothness(env: GridEnv, path: np.ndarray) -> float:
    if len(path) < 3:
        return 0.0
    mpath = env.metric_path(path)
    v1 = mpath[1:-1] - mpath[:-2]
    v2 = mpath[2:] - mpath[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 1e-9) & (n2 > 1e-9)
    if not np.any(ok):
        return 0.0
    c = np.einsum('ij,ij->i', v1[ok], v2[ok]) / (n1[ok] * n2[ok])
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    return float(np.sum(ang, dtype=np.float64))


def objective_energy_approx(env: GridEnv, path: np.ndarray, k_turn: float = 5.0, k_climb: float = 2.0) -> float:
    L = objective_length(env, path)
    S = objective_smoothness(env, path)
    climb = 0.0
    if path.shape[1] >= 3:
        dz = np.diff(path[:, 2])
        climb = float(np.sum(np.maximum(0.0, dz), dtype=np.float64))
    return float(L + float(k_turn) * S + float(k_climb) * climb)
