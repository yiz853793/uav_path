from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import sampled_polyline_points_array


@dataclass
class EvalResult:
    obj: np.ndarray
    feasible: bool
    violation: float
    detail: dict
    hard_violation: float = 0.0
    soft_violation: float = 0.0


def evaluate_path(
    env: GridEnv,
    path: np.ndarray,
    max_turn_deg: float = 90.0,
    soft_turn_deg: float = 60.0,
    max_pitch_deg: float = 35.0,
    soft_pitch_deg: float = 25.0,
    desired_clearance_margin: float = 2.0,
    tau_soft: float = 25.0,
    sample_step: float = 0.5,
    k_energy_turn: float = 5.0,
    k_energy_climb: float = 2.0,
) -> EvalResult:
    assert path.ndim == 2 and path.shape[1] in (2, 3)
    if path.shape[1] == 2:
        path = env.lift_path_to_3d(path)

    max_turn_rad = float(max_turn_deg) * np.pi / 180.0
    soft_turn_rad = min(float(soft_turn_deg) * np.pi / 180.0, max_turn_rad)
    max_pitch_rad = float(max_pitch_deg) * np.pi / 180.0
    soft_pitch_rad = min(float(soft_pitch_deg) * np.pi / 180.0, max_pitch_rad)
    desired_clearance = float(env.min_clearance) + max(0.0, float(desired_clearance_margin))
    tau_soft = float(tau_soft)

    mpath = env.metric_path(path)
    seg = mpath[1:] - mpath[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    f1 = float(np.sum(seg_len, dtype=np.float64))

    proj = mpath[:, :2]
    if len(path) >= 3:
        v1 = proj[1:-1] - proj[:-2]
        v2 = proj[2:] - proj[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        ok = (n1 > 1e-9) & (n2 > 1e-9)
        if np.any(ok):
            c = np.einsum('ij,ij->i', v1[ok], v2[ok]) / (n1[ok] * n2[ok])
            c = np.clip(c, -1.0, 1.0)
            ang = np.arccos(c)
            smooth = float(np.sum(ang, dtype=np.float64))
            vt_soft = float(np.sum(np.maximum(0.0, ang - soft_turn_rad), dtype=np.float64))
            vt_hard = float(np.sum(np.maximum(0.0, ang - max_turn_rad), dtype=np.float64))
        else:
            smooth = 0.0
            vt_soft = 0.0
            vt_hard = 0.0
    else:
        smooth = 0.0
        vt_soft = 0.0
        vt_hard = 0.0

    dxy = np.linalg.norm(np.diff(mpath[:, :2], axis=0), axis=1)
    dz = np.abs(np.diff(path[:, 2]))
    pitch = np.arctan2(dz, np.maximum(1e-9, dxy)) if len(path) >= 2 else np.zeros((0,), dtype=np.float32)
    vp_soft = float(np.sum(np.maximum(0.0, pitch - soft_pitch_rad), dtype=np.float64))
    vp_hard = float(np.sum(np.maximum(0.0, pitch - max_pitch_rad), dtype=np.float64))

    x = path[:, 0]
    y = path[:, 1]
    z = path[:, 2]
    vb = 0.0
    vb += float(np.sum(np.maximum(0.0, -x), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, -y), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, x - (env.W - 1)), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, y - (env.H - 1)), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, env.z_min - z), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, z - env.z_max), dtype=np.float64))

    # Fast vectorized sampling over all segments at once.
    if len(path) >= 2:
        pts = sampled_polyline_points_array(path, step=sample_step, xy_resolution=env.resolution)
    else:
        pts = path[:1]

    ix = np.rint(pts[:, 0]).astype(np.int32)
    iy = np.rint(pts[:, 1]).astype(np.int32)
    oob = (ix < 0) | (ix >= env.W) | (iy < 0) | (iy >= env.H)
    collision = bool(np.any(oob))

    ix2 = np.clip(ix, 0, env.W - 1)
    iy2 = np.clip(iy, 0, env.H - 1)

    if path.shape[1] < 3:
        if (not collision) and bool(np.any(env.occupancy[iy2, ix2])):
            collision = True
        clearance_hard = 0.0
        clearance_soft = 0.0
        f2 = float(np.sum(env.threat[iy2, ix2], dtype=np.float64))
    else:
        ground = env.height[iy2, ix2]
        collision_depth = float(np.sum(np.maximum(0.0, ground - pts[:, 2]), dtype=np.float64))
        if (not collision) and bool(np.any(pts[:, 2] < ground)):
            collision = True
        req_min = ground + float(env.min_clearance)
        req_safe = ground + desired_clearance
        clearance_hard = float(np.sum(np.maximum(0.0, req_min - pts[:, 2]), dtype=np.float64))
        clearance_soft = float(np.sum(np.maximum(0.0, req_safe - pts[:, 2]), dtype=np.float64))
        clearance = np.maximum(0.0, pts[:, 2] - ground)
        weight = np.exp(-0.06 * clearance)
        f2 = float(np.sum(env.threat[iy2, ix2] * weight, dtype=np.float64))
        if collision_depth > 0.0:
            clearance_hard += collision_depth

    vc = 1.0 if collision else 0.0
    climb = float(np.sum(np.maximum(0.0, np.diff(path[:, 2])), dtype=np.float64))
    f3 = float(f1 + float(k_energy_turn) * smooth + float(k_energy_climb) * climb)
    hard_violation = float(vb + 1000.0 * vc + 10.0 * clearance_hard + 10.0 * vt_hard + 10.0 * vp_hard)
    soft_violation = float(5.0 * clearance_soft + 10.0 * vt_soft + 10.0 * vp_soft)
    soft_excess = max(0.0, soft_violation - tau_soft)
    viol = float(hard_violation + soft_excess)
    detail = {
        'bounds': float(vb),
        'collision': float(vc),
        'turn': float(vt_hard),
        'pitch': float(vp_hard),
        'clearance': float(clearance_hard),
        'turn_soft': float(vt_soft),
        'pitch_soft': float(vp_soft),
        'clearance_soft': float(clearance_soft),
        'hard_violation': float(hard_violation),
        'soft_violation': float(soft_violation),
        'tau_soft': float(tau_soft),
        'weighted_total': float(viol),
        'smoothness': float(smooth),
        'climb': float(climb),
    }
    feasible = (hard_violation < 1e-6) and (soft_violation <= tau_soft + 1e-9)
    return EvalResult(
        obj=np.array([f1, f2, f3], dtype=np.float64),
        feasible=bool(feasible),
        violation=float(viol),
        detail=detail,
        hard_violation=float(hard_violation),
        soft_violation=float(soft_violation),
    )
