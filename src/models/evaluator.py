# src/models/evaluator.py

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..env.grid_env import GridEnv
from ..env.collision import sampled_cells_array


@dataclass
class EvalResult:
    obj: np.ndarray        # shape [M]
    feasible: bool
    violation: float
    detail: dict


def evaluate_path(
    env: GridEnv,
    path: np.ndarray,
    max_turn_deg: float = 90.0,
    sample_step: float = 0.5,
    k_energy_turn: float = 5.0,
) -> EvalResult:
    """
    统一评估入口（高性能版本）：
      目标：f1=长度, f2=威胁, f3=能耗近似(长度 + k*转角累计)
      约束：边界/碰撞/最大转角

    关键优化：
    1) 避免 “约束检测”和“目标计算”重复遍历路径（旧实现会重复做采样与转角计算）
    2) 将逐点采样改为向量化采样 + numpy 索引
    """
    assert path.ndim == 2 and path.shape[1] == 2
    max_turn_rad = float(max_turn_deg) * np.pi / 180.0

    # ---------- f1: length ----------
    seg = path[1:] - path[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    f1 = float(np.sum(seg_len, dtype=np.float64))

    # ---------- angles (smoothness + turn constraint) ----------
    if len(path) >= 3:
        v1 = path[1:-1] - path[:-2]
        v2 = path[2:] - path[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        ok = (n1 > 1e-9) & (n2 > 1e-9)
        if np.any(ok):
            v1o = v1[ok]
            v2o = v2[ok]
            n1o = n1[ok]
            n2o = n2[ok]
            c = np.einsum("ij,ij->i", v1o, v2o) / (n1o * n2o)
            c = np.clip(c, -1.0, 1.0)
            ang = np.arccos(c)
            smooth = float(np.sum(ang, dtype=np.float64))
            over = np.maximum(0.0, ang - float(max_turn_rad))
            vt = float(np.sum(over, dtype=np.float64))
        else:
            smooth = 0.0
            vt = 0.0
    else:
        smooth = 0.0
        vt = 0.0

    # ---------- bounds violation ----------
    x = path[:, 0]
    y = path[:, 1]
    vb = 0.0
    vb += float(np.sum(np.maximum(0.0, -x), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, -y), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, x - (env.W - 1)), dtype=np.float64))
    vb += float(np.sum(np.maximum(0.0, y - (env.H - 1)), dtype=np.float64))

    # ---------- collision + threat (shared sampling) ----------
    collision = False
    f2 = 0.0
    for i in range(1, len(path)):
        ix, iy = sampled_cells_array(path[i - 1], path[i], step=sample_step)

        # collision: oob treated as collision
        oob = (ix < 0) | (ix >= env.W) | (iy < 0) | (iy >= env.H)
        if (not collision) and bool(np.any(oob)):
            collision = True
        if (not collision) and bool(np.any(env.occupancy[iy, ix])):
            collision = True

        # threat: keep old behavior (clip to boundary then sum)
        ix2 = np.clip(ix, 0, env.W - 1)
        iy2 = np.clip(iy, 0, env.H - 1)
        f2 += float(env.threat[iy2, ix2].sum(dtype=np.float64))

    vc = 1.0 if collision else 0.0

    # ---------- f3: energy approx ----------
    f3 = float(f1 + float(k_energy_turn) * smooth)

    # ---------- total violation ----------
    viol = float(vb + 1000.0 * vc + 10.0 * vt)
    detail = {"bounds": float(vb), "collision": float(vc), "turn": float(vt), "weighted_total": float(viol), "smoothness": float(smooth)}
    feasible = (vc < 0.5) and (vb < 1e-6) and (vt < 1e-6)

    return EvalResult(
        obj=np.array([f1, f2, f3], dtype=np.float64),
        feasible=bool(feasible),
        violation=float(viol),
        detail=detail,
    )
