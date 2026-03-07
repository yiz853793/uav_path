# src/models/constraints.py

from __future__ import annotations
import numpy as np
from ..env.grid_env import GridEnv
from ..env.collision import path_collision


def violation_bounds(env: GridEnv, path: np.ndarray) -> float:
    """
    越界违约程度：超出边界的 L1 距离总和（向量化）
    """
    x = path[:, 0]
    y = path[:, 1]
    v = 0.0
    v += float(np.sum(np.maximum(0.0, -x), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, -y), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, x - (env.W - 1)), dtype=np.float64))
    v += float(np.sum(np.maximum(0.0, y - (env.H - 1)), dtype=np.float64))
    return float(v)


def violation_collision(env: GridEnv, path: np.ndarray, step: float = 0.5) -> float:
    """
    碰撞违约：碰撞则 1，否则 0
    """
    return 1.0 if path_collision(env, path, step=step) else 0.0


def violation_max_turn(path: np.ndarray, max_turn_rad: float) -> float:
    """
    最大转角约束：超过部分累计（弧度，向量化）
    """
    if len(path) < 3:
        return 0.0
    v1 = path[1:-1] - path[:-2]
    v2 = path[2:] - path[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    ok = (n1 > 1e-9) & (n2 > 1e-9)
    if not np.any(ok):
        return 0.0
    v1 = v1[ok]
    v2 = v2[ok]
    n1 = n1[ok]
    n2 = n2[ok]
    c = np.einsum("ij,ij->i", v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    over = ang - float(max_turn_rad)
    over = np.maximum(0.0, over)
    return float(np.sum(over, dtype=np.float64))


def total_violation(env: GridEnv, path: np.ndarray, max_turn_rad: float, step: float = 0.5) -> tuple[float, dict]:
    vb = violation_bounds(env, path)
    vc = violation_collision(env, path, step=step)
    vt = violation_max_turn(path, max_turn_rad=max_turn_rad)

    # 权重：碰撞惩罚最大，其次转角，再次边界
    total = vb + 1000.0 * vc + 10.0 * vt
    detail = {"bounds": vb, "collision": vc, "turn": vt, "weighted_total": total}
    return float(total), detail
