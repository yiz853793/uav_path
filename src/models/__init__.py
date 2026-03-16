# src/model/__init__.py

from __future__ import annotations
import numpy as np


def resample_polyline(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    折线等弧长重采样到固定点数（含首尾）
    """
    assert path.ndim == 2 and path.shape[1] == 2
    n_points = int(n_points)
    if n_points <= 2:
        return np.vstack([path[0], path[-1]]).astype(np.float32)

    seg = path[1:] - path[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])

    if total < 1e-9:
        return np.repeat(path[:1], n_points, axis=0).astype(np.float32)

    targets = np.linspace(0.0, total, n_points)
    out = []
    j = 0
    for t in targets:
        while j < len(seglen) - 1 and s[j + 1] < t:
            j += 1
        t0, t1 = s[j], s[j + 1]
        if t1 - t0 < 1e-9:
            out.append(path[j].copy())
        else:
            a = (t - t0) / (t1 - t0)
            out.append(path[j] * (1 - a) + path[j + 1] * a)

    return np.asarray(out, dtype=np.float32)


def shortcut_smooth(path: np.ndarray, n_try: int, rng: np.random.Generator, collision_fn) -> np.ndarray:
    """
    随机捷径平滑：
      若 i->j 直连不碰撞，则删除中间点
    collision_fn(p,q)->bool
    """
    p = path.copy()
    if len(p) <= 2:
        return p

    for _ in range(int(n_try)):
        if len(p) <= 2:
            break
        i = int(rng.integers(0, len(p) - 1))
        j = int(rng.integers(i + 1, len(p)))
        if j <= i + 1:
            continue
        if not collision_fn(p[i], p[j]):
            p = np.vstack([p[: i + 1], p[j:]])

    return p.astype(np.float32)
