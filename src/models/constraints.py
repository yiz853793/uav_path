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
    return 1.0 if path_collision(env, path, step=step, clearance=0.0) else 0.0


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


def split_violation(
    env: GridEnv,
    path: np.ndarray,
    hard_turn_rad: float,
    step: float = 0.5,
    hard_pitch_rad: float | None = None,
    *,
    soft_turn_rad: float | None = None,
    soft_pitch_rad: float | None = None,
    desired_clearance_margin: float = 2.0,
    tau_soft: float = 25.0,
) -> tuple[float, float, bool, dict]:
    vb = violation_bounds(env, path)
    vc = violation_collision(env, path, step=step)
    vt_hard = violation_max_turn(env, path, max_turn_rad=hard_turn_rad)
    vcl_hard = violation_min_clearance(env, path, sample_step=step, min_clearance=env.min_clearance)
    vp_hard = 0.0 if hard_pitch_rad is None else violation_max_pitch(env, path, max_pitch_rad=hard_pitch_rad)

    soft_turn_rad = hard_turn_rad if soft_turn_rad is None else min(float(soft_turn_rad), float(hard_turn_rad))
    vt_soft = violation_max_turn(env, path, max_turn_rad=soft_turn_rad)
    vcl_soft = violation_min_clearance(
        env,
        path,
        sample_step=step,
        min_clearance=env.min_clearance + max(0.0, float(desired_clearance_margin)),
    )
    if hard_pitch_rad is None:
        vp_soft = 0.0
    else:
        soft_pitch_rad = hard_pitch_rad if soft_pitch_rad is None else min(float(soft_pitch_rad), float(hard_pitch_rad))
        vp_soft = violation_max_pitch(env, path, max_pitch_rad=soft_pitch_rad)

    hard = float(vb + 1000.0 * vc + 10.0 * vcl_hard + 10.0 * vt_hard + 10.0 * vp_hard)
    soft = float(5.0 * vcl_soft + 10.0 * vt_soft + 10.0 * vp_soft)
    feasible = bool(hard < 1e-6 and soft <= float(tau_soft) + 1e-9)
    total = float(hard + max(0.0, soft - float(tau_soft)))
    detail = {
        'bounds': vb,
        'collision': vc,
        'turn': vt_hard,
        'clearance': vcl_hard,
        'pitch': vp_hard,
        'turn_soft': vt_soft,
        'clearance_soft': vcl_soft,
        'pitch_soft': vp_soft,
        'hard_violation': hard,
        'soft_violation': soft,
        'tau_soft': float(tau_soft),
        'weighted_total': total,
    }
    return hard, soft, feasible, detail


def total_violation(env: GridEnv, path: np.ndarray, max_turn_rad: float, step: float = 0.5, max_pitch_rad: float | None = None) -> tuple[float, dict]:
    hard, soft, _feasible, detail = split_violation(
        env,
        path,
        hard_turn_rad=max_turn_rad,
        step=step,
        hard_pitch_rad=max_pitch_rad,
    )
    total = float(hard + max(0.0, soft - float(detail['tau_soft'])))
    detail['weighted_total'] = total
    return total, detail
