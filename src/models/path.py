# src/models/path

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

def densify_polyline_to_K(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    在保留原始折线拐点（顶点）的前提下，把折线加密/补点到固定点数（含首尾）。

    设计动机：
      - resample_polyline 是等弧长重采样，会“跨拐点切角”，在窄通道场景可能把可行路径重采样成不可行。
      - 这里采用“按段长度分配插点”的方式，只在每段直线内部插值，保证所有原始顶点都被保留。

    规则：
      - len(path) == n_points: 原样返回
      - len(path) < n_points: 按段长度分配插点
      - len(path) > n_points: 回退到 resample_polyline（此时点太多，通常来自栅格路径）
    """
    assert path.ndim == 2 and path.shape[1] == 2
    n_points = int(n_points)
    path = path.astype(np.float32)

    if n_points <= 2:
        return np.vstack([path[0], path[-1]]).astype(np.float32)

    if len(path) == n_points:
        return path.astype(np.float32)

    if len(path) > n_points:
        # 点数过多：退回等弧长重采样（通常来自 A* 的栅格路径）
        return resample_polyline(path, n_points)

    # len(path) < n_points: 需要补点
    seg = path[1:] - path[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    total = float(seglen.sum())
    extra = n_points - len(path)

    if total < 1e-9 or extra <= 0:
        out = np.vstack([path, np.repeat(path[-1][None, :], max(0, extra), axis=0)])
        return out[:n_points].astype(np.float32)

    raw = seglen / total * extra
    cnt = np.floor(raw).astype(int)
    rem = extra - int(cnt.sum())
    if rem > 0:
        frac = raw - cnt
        idx = np.argsort(-frac)[:rem]
        cnt[idx] += 1

    out = [path[0]]
    for i in range(len(seglen)):
        m = int(cnt[i])
        p0, p1 = path[i], path[i + 1]
        # 本段插 m 个点，且添加段终点
        for j in range(1, m + 2):
            t = j / (m + 1)
            out.append(p0 * (1 - t) + p1 * t)

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
