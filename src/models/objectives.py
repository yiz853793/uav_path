# src/models/objectives.py

from __future__ import annotations
import numpy as np
from ..env.grid_env import GridEnv
from ..env.collision import sampled_cells_array


def objective_length(path: np.ndarray) -> float:
    seg = path[1:] - path[:-1]
    # dtype=float64 以减少累计误差（并与旧实现一致/更稳定）
    return float(np.sum(np.linalg.norm(seg, axis=1, ord=2), dtype=np.float64))


def objective_threat(env: GridEnv, path: np.ndarray, sample_step: float = 0.5) -> float:
    """
    沿路径采样累计威胁（越大越危险）

    旧实现逐点循环 + threat 索引；这里改为向量化采样 + numpy 索引求和。
    注意：保持旧行为：每个 segment 都包含两端点，所以连接点会被重复计数。
    """
    total = 0.0
    for i in range(1, len(path)):
        ix, iy = sampled_cells_array(path[i - 1], path[i], step=sample_step)
        ix2 = np.clip(ix, 0, env.W - 1)
        iy2 = np.clip(iy, 0, env.H - 1)
        total += float(env.threat[iy2, ix2].sum(dtype=np.float64))
    return float(total)


def objective_smoothness(path: np.ndarray) -> float:
    """
    平滑度：累计转角（弧度），越小越平滑
    向量化实现（避免 Python for-loop）。
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
    return float(np.sum(ang, dtype=np.float64))


def objective_energy_approx(path: np.ndarray, k_turn: float = 5.0) -> float:
    """
    飞行能耗近似（2D 工程化版本）：
      E ≈ L + k_turn * (累计转角)
    """
    L = objective_length(path)
    S = objective_smoothness(path)
    return float(L + float(k_turn) * S)
