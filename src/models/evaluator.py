from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import sampled_points_array


@dataclass
class EvalResult:
    obj: np.ndarray
    feasible: bool
    violation: float
    detail: dict


def evaluate_path(
    env: GridEnv,
    path: np.ndarray,
    max_turn_deg: float = 90.0,
    max_pitch_deg: float = 35.0,
    sample_step: float = 0.5,
    k_energy_turn: float = 5.0,
    k_energy_climb: float = 2.0,
) -> EvalResult:
    assert path.ndim == 2 and path.shape[1] in (2, 3)
    if path.shape[1] == 2:
        path = env.lift_path_to_3d(path)

    max_turn_rad = float(max_turn_deg) * np.pi / 180.0
    max_pitch_rad = float(max_pitch_deg) * np.pi / 180.0

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
            vt = float(np.sum(np.maximum(0.0, ang - max_turn_rad), dtype=np.float64))
        else:
            smooth = 0.0
            vt = 0.0
    else:
        smooth = 0.0
        vt = 0.0

    dxy = np.linalg.norm(np.diff(mpath[:, :2], axis=0), axis=1)
    dz = np.abs(np.diff(path[:, 2]))
    pitch = np.arctan2(dz, np.maximum(1e-9, dxy)) if len(path) >= 2 else np.zeros((0,), dtype=np.float32)
    vp = float(np.sum(np.maximum(0.0, pitch - max_pitch_rad), dtype=np.float64))

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
        seg_pts = [sampled_points_array(path[i - 1], path[i], step=sample_step, xy_resolution=env.resolution) for i in range(1, len(path))]
        pts = np.concatenate(seg_pts, axis=0) if len(seg_pts) > 1 else seg_pts[0]
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
        clearance_violation = 0.0
        f2 = float(np.sum(env.threat[iy2, ix2], dtype=np.float64))
    else:
        ground = env.height[iy2, ix2]
        req = ground + env.min_clearance
        clearance_violation = float(np.sum(np.maximum(0.0, req - pts[:, 2]), dtype=np.float64))
        if (not collision) and bool(np.any(pts[:, 2] < req)):
            collision = True
        clearance = np.maximum(0.0, pts[:, 2] - ground)
        weight = np.exp(-0.06 * clearance)
        f2 = float(np.sum(env.threat[iy2, ix2] * weight, dtype=np.float64))

    vc = 1.0 if collision else 0.0
    climb = float(np.sum(np.maximum(0.0, np.diff(path[:, 2])), dtype=np.float64))
    f3 = float(f1 + float(k_energy_turn) * smooth + float(k_energy_climb) * climb)
    viol = float(vb + 1000.0 * vc + 10.0 * vt + 10.0 * vp + 10.0 * clearance_violation)
    detail = {
        'bounds': float(vb),
        'collision': float(vc),
        'turn': float(vt),
        'pitch': float(vp),
        'clearance': float(clearance_violation),
        'weighted_total': float(viol),
        'smoothness': float(smooth),
        'climb': float(climb),
    }
    feasible = (vc < 0.5) and (vb < 1e-6) and (vt < 1e-6) and (vp < 1e-6) and (clearance_violation < 1e-6)
    return EvalResult(obj=np.array([f1, f2, f3], dtype=np.float64), feasible=bool(feasible), violation=float(viol), detail=detail)
