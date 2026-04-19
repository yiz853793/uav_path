from __future__ import annotations
import numpy as np
from ..env.grid_env import GridEnv
from ..env.collision import path_collision, sampled_points_array


def violation_bounds(env: GridEnv, path: np.ndarray) -> float:
    x = path[:, 0]
    y = path[:, 1]
    v = 0.0
    v += float(np.sum(np.maximum(0.0, -x), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, -y), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, x - (env.W - 1)), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, y - (env.H - 1)), dtype=np.float64))
    if path.shape[1] >= 3:
        z = path[:, 2]
        v += float(np.sum(np.maximum(0.0, env.z_min - z), dtype=np.float64))
        v += float(np.sum(np.maximum(0.0, z - env.z_max), dtype=np.float64))
    return float(v)


def violation_collision(env: GridEnv, path: np.ndarray, step: float = 0.5) -> float:
    return 1.0 if path_collision(env, path, step=step) else 0.0


def violation_max_turn(env: GridEnv, path: np.ndarray, max_turn_rad: float) -> float:
    if len(path) < 3:
        return 0.0
    proj = np.asarray(path[:, :2], dtype=np.float32).copy()
    proj[:, 0] *= env.resolution
    proj[:, 1] *= env.resolution
    v1 = proj[1:-1] - proj[:-2]
    v2 = proj[2:] - proj[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 1e-9) & (n2 > 1e-9)
    if not np.any(ok):
        return 0.0
    c = np.einsum('ij,ij->i', v1[ok], v2[ok]) / (n1[ok] * n2[ok])
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    over = np.maximum(0.0, ang - float(max_turn_rad))
    return float(np.sum(over, dtype=np.float64))


def violation_min_clearance(env: GridEnv, path: np.ndarray, sample_step: float = 0.5, min_clearance: float | None = None) -> float:
    if path.shape[1] < 3:
        return 0.0
    c = env.min_clearance if min_clearance is None else float(min_clearance)
    total = 0.0
    for i in range(1, len(path)):
        pts = sampled_points_array(path[i - 1], path[i], step=sample_step, xy_resolution=env.resolution)
        ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
        req = env.height[iy, ix] + c
        total += float(np.sum(np.maximum(0.0, req - pts[:, 2]), dtype=np.float64))
    return float(total)


def violation_max_pitch(env: GridEnv, path: np.ndarray, max_pitch_rad: float) -> float:
    if path.shape[1] < 3 or len(path) < 2:
        return 0.0
    proj = np.asarray(path[:, :2], dtype=np.float32).copy()
    proj[:, 0] *= env.resolution
    proj[:, 1] *= env.resolution
    dxy = np.linalg.norm(np.diff(proj, axis=0), axis=1)
    dz = np.abs(np.diff(path[:, 2]))
    pitch = np.arctan2(dz, np.maximum(1e-9, dxy))
    over = np.maximum(0.0, pitch - float(max_pitch_rad))
    return float(np.sum(over, dtype=np.float64))


def total_violation(env: GridEnv, path: np.ndarray, max_turn_rad: float, step: float = 0.5, max_pitch_rad: float | None = None) -> tuple[float, dict]:
    vb = violation_bounds(env, path)
    vc = violation_collision(env, path, step=step)
    vt = violation_max_turn(env, path, max_turn_rad=max_turn_rad)
    vcl = violation_min_clearance(env, path, sample_step=step)
    vp = 0.0 if max_pitch_rad is None else violation_max_pitch(env, path, max_pitch_rad=max_pitch_rad)
    total = vb + 1000.0 * vc + 10.0 * vt + 10.0 * vcl + 10.0 * vp
    detail = {'bounds': vb, 'collision': vc, 'turn': vt, 'clearance': vcl, 'pitch': vp, 'weighted_total': total}
    return float(total), detail
