# src/algorithms/moead.py

from __future__ import annotations
import os
import sys
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque, defaultdict
from scipy.stats import chi2
from ..env.grid_env import GridEnv
from ..env.collision import segment_collision, sampled_points_array, sampled_polyline_points_array
from ..models.evaluator import evaluate_path, EvalResult
from ..models.path import shortcut_smooth, resample_polyline, densify_polyline_to_K
from .a_star import astar



class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each record (better for 'tail -f' / real-time debugging)."""

    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after each record so terminal shows output immediately."""

    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def _make_file_logger(log_path: str, console: bool = False) -> logging.Logger:
    """Create a dedicated file logger for MOEA/D debug.

    MOEA/D is called many times in a benchmark loop; we must avoid adding
    duplicate handlers. If console=True, also attach a StreamHandler so
    debug lines are printed to stdout in real time.
    """
    log_path = os.path.abspath(log_path)
    name = f"moead_debug::{log_path}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    has_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
        for h in logger.handlers
    )
    has_console = any(isinstance(h, _FlushingStreamHandler) for h in logger.handlers)
    if has_file and (not console or has_console):
        logger.propagate = False
        return logger

    if not has_file:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = _FlushingFileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    if console and not has_console:
        ch = _FlushingStreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    logger.propagate = False
    return logger

def _close_logger_handlers(logger: logging.Logger | None) -> None:
    """Flush and close handlers so Windows can rename/delete log files safely."""
    if logger is None:
        return
    for h in list(logger.handlers):
        try:
            h.flush()
        except Exception:
            pass
        try:
            h.close()
        except Exception:
            pass
        try:
            logger.removeHandler(h)
        except Exception:
            pass

def _pop_stats(pop_inds: List["Individual"]):
    """Lightweight population statistics for debug logging."""
    n = len(pop_inds)
    if n <= 0:
        return {
            "n": 0,
            "n_feasible": 0,
            "feasible_ratio": 0.0,
            "min_violation": 0.0,
            "mean_violation": 0.0,
            "best_obj_feasible": None,
        }

    viols = np.array([it.er.violation for it in pop_inds], dtype=np.float64)
    n_feas = int(sum(1 for it in pop_inds if it.er.feasible))
    best_obj = None
    if n_feas > 0:
        feas_objs = np.stack([it.er.obj for it in pop_inds if it.er.feasible], axis=0)
        best_obj = np.min(feas_objs, axis=0).tolist()

    return {
        "n": int(n),
        "n_feasible": int(n_feas),
        "feasible_ratio": float(n_feas / max(1, n)),
        "min_violation": float(np.min(viols)),
        "mean_violation": float(np.mean(viols)),
        "best_obj_feasible": best_obj,
    }


def _estimate_sampled_cells_count(path: np.ndarray, step: float) -> int:
    """Estimate how many sampled grid cells are visited by sampled_cells() over a polyline.

    sampled_cells() uses:
      n = max(1, ceil(dist/step))
      yields (n + 1) samples
    """
    step = float(step)
    step = max(1e-6, step)
    seg = path[1:] - path[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    n = np.ceil(seglen / step).astype(np.int64)
    n = np.maximum(1, n)
    return int(np.sum(n + 1))


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def tchebycheff(f: np.ndarray, w: np.ndarray, z: np.ndarray) -> float:
    # g(x|w,z*) = max_i w_i * |f_i - z_i|
    return float(np.max(w * np.abs(f - z)))


def uniform_weights(m: int, n: int, seed: int = 0, extreme_bias: float = 0.20) -> np.ndarray:
    """Generate simplex weights with corner intensification.

    For m=3 we explicitly make the corners denser than the center, following the
    spirit of corner-weight intensification: a dense boundary/corner bank is
    mixed with a much sparser interior bank. This gives better extreme-point
    coverage while still keeping central trade-off subproblems.
    """
    rng = np.random.default_rng(seed)
    if m != 3:
        w = rng.random((n, m))
        w = w / np.sum(w, axis=1, keepdims=True)
        return w.astype(np.float64)

    n = int(max(1, n))
    extreme_bias = float(np.clip(extreme_bias, 0.0, 0.60))

    def _simplex_grid(level: int) -> list[list[float]]:
        level = int(max(1, level))
        pts: list[list[float]] = []
        for i in range(level + 1):
            for j in range(level + 1 - i):
                k = level - i - j
                pts.append([i / level, j / level, k / level])
        return pts

    dense_level = max(6, int(np.sqrt(max(9, n))) + 4)
    sparse_level = max(2, dense_level // 3)
    dense = np.array(_simplex_grid(dense_level), dtype=np.float64)
    sparse = np.array(_simplex_grid(sparse_level), dtype=np.float64)

    # corner bank: points where one weight dominates strongly
    corner_thr = 1.0 - max(0.10, 0.55 * (1.0 - extreme_bias))
    dense_corner_mask = np.max(dense, axis=1) >= corner_thr
    dense_corner = dense[dense_corner_mask]
    dense_other = dense[~dense_corner_mask]

    # interior bank: prefer central / non-corner weights from the sparse grid
    sparse_center_mask = np.max(sparse, axis=1) <= 0.80
    sparse_center = sparse[sparse_center_mask] if np.any(sparse_center_mask) else sparse

    n_corner = int(np.clip(round(n * (0.30 + 0.90 * extreme_bias)), 3, n))
    n_center = max(0, n - n_corner)

    ws: list[np.ndarray] = []
    if len(dense_corner) > 0:
        idx = rng.choice(len(dense_corner), size=min(n_corner, len(dense_corner)), replace=False)
        ws.append(dense_corner[idx])
    if len(sparse_center) > 0 and n_center > 0:
        replace = len(sparse_center) < n_center
        idx = rng.choice(len(sparse_center), size=n_center, replace=replace)
        ws.append(sparse_center[idx])

    if ws:
        w = np.vstack(ws)
    else:
        w = rng.random((n, 3))

    # fill any missing slots with boundary/interior leftovers, then random points.
    if len(w) < n:
        leftovers = np.vstack([dense_other, sparse]) if len(dense_other) > 0 else sparse
        if len(leftovers) > 0:
            take = min(n - len(w), len(leftovers))
            idx = rng.choice(len(leftovers), size=take, replace=False)
            w = np.vstack([w, leftovers[idx]])
    while len(w) < n:
        extra = rng.random((1, 3))
        extra = extra / np.sum(extra, axis=1, keepdims=True)
        w = np.vstack([w, extra])

    # exact de-dup before truncation
    _, uniq_idx = np.unique(np.round(w, 12), axis=0, return_index=True)
    w = w[np.sort(uniq_idx)]
    if len(w) > n:
        rng.shuffle(w)
        w = w[:n]
    elif len(w) < n:
        extra = rng.random((n - len(w), 3))
        extra = extra / np.sum(extra, axis=1, keepdims=True)
        w = np.vstack([w, extra])

    w = np.clip(w.astype(np.float64), 1e-6, None)
    w = w / np.sum(w, axis=1, keepdims=True)
    return w


def build_neighbors(weights: np.ndarray, T: int) -> np.ndarray:
    dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    idx = np.argsort(dist, axis=1)[:, :T]
    return idx.astype(np.int32)


@dataclass
class Individual:
    x: np.ndarray  # decision vars: [K,D]
    er: EvalResult


class Archive:
    """Pareto archive with diversity-aware truncation.

    Notes
    -----
    - Dominance maintenance is still exact.
    - When the archive exceeds ``max_size``, truncation is *not* a plain kNN
      sparsity filter anymore. We first protect objective extremes, then prefer
      one representative per objective-space grid cell, and only then fill the
      remaining slots by a density-aware score. This makes the returned archive
      much more evenly spread on the Pareto front.
    """

    def __init__(
        self,
        max_size: int = 200,
        *,
        grid_bins: int = 0,
        protect_extremes: bool = True,
        crowd_k: int = 5,
        overflow_margin: int = 64,
    ):
        self.max_size = None if int(max_size) <= 0 else int(max_size)
        self.grid_bins = int(grid_bins)
        self.protect_extremes = bool(protect_extremes)
        self.crowd_k = int(max(1, crowd_k))
        self.overflow_margin = int(max(1, overflow_margin)) if self.max_size is not None else 0
        self.items: List[Individual] = []
        self._objs = np.empty((0, 3), dtype=np.float64)
        self._keys: set[tuple[float, float, float]] = set()

    @staticmethod
    def _obj_key(obj: np.ndarray) -> tuple[float, float, float]:
        arr = np.asarray(obj, dtype=np.float64)
        return (round(float(arr[0]), 12), round(float(arr[1]), 12), round(float(arr[2]), 12))

    def add(self, ind: Individual):
        obj = np.asarray(ind.er.obj, dtype=np.float64)
        key = self._obj_key(obj)
        if key in self._keys:
            return
        n = len(self.items)
        if n == 0:
            self.items = [ind]
            self._objs = obj[None, :].copy()
            self._keys.add(key)
            return

        f1 = self._objs[:, 0]
        left = int(np.searchsorted(f1, obj[0], side="right"))
        right = int(np.searchsorted(f1, obj[0], side="left"))

        # only prefix [0:left) can dominate the candidate because domination
        # requires existing f1 <= obj[0].
        if left > 0:
            pref = self._objs[:left]
            dom_mask = np.all(pref <= obj[None, :], axis=1) & np.any(pref < obj[None, :], axis=1)
            if np.any(dom_mask):
                return

        # only suffix [right:n) can be dominated by the candidate because
        # candidate domination requires obj[0] <= existing f1.
        keep_mask = np.ones(n, dtype=bool)
        if right < n:
            suff = self._objs[right:]
            dominated_mask = np.all(obj[None, :] <= suff, axis=1) & np.any(obj[None, :] < suff, axis=1)
            if np.any(dominated_mask):
                keep_mask[right:] = ~dominated_mask

        if not np.all(keep_mask):
            self.items = [it for it, keep in zip(self.items, keep_mask) if keep]
            removed_keys = {self._obj_key(o) for o in self._objs[~keep_mask]}
            self._keys.difference_update(removed_keys)
            self._objs = self._objs[keep_mask]
            f1 = self._objs[:, 0]

        pos = int(np.searchsorted(f1, obj[0], side="right"))
        self.items.insert(pos, ind)
        self._objs = np.insert(self._objs, pos, obj, axis=0)
        self._keys.add(key)

        if self.max_size is not None and len(self.items) > (self.max_size + self.overflow_margin):
            self._truncate()

    def compact(self, force: bool = False):
        if self.max_size is None:
            return
        limit = self.max_size if force else (self.max_size + self.overflow_margin)
        if len(self.items) > limit:
            self._truncate()

    def density(self, obj: np.ndarray, k: int = 5) -> float:
        if len(self._objs) <= 1:
            return 0.0
        F = self._objs
        mn = np.min(F, axis=0)
        mx = np.max(F, axis=0)
        denom = np.maximum(1e-9, mx - mn)
        q = (np.asarray(obj, dtype=np.float64) - mn) / denom
        Fn = (F - mn) / denom
        d = np.linalg.norm(Fn - q[None, :], axis=1)
        d = np.sort(d)
        kk = min(max(1, int(k)), len(d) - 1)
        return float(1.0 / max(1e-9, d[kk]))

    def _truncate(self):
        if self.max_size is None or len(self.items) <= self.max_size:
            return
        F = self._objs.astype(np.float64)
        n, m = F.shape
        mn = F.min(axis=0)
        mx = F.max(axis=0)
        denom = np.maximum(1e-9, mx - mn)
        Fn = (F - mn) / denom

        dist = np.linalg.norm(Fn[:, None, :] - Fn[None, :, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        k = min(max(1, int(self.crowd_k)), max(1, n - 1))
        knn = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        mean_knn = np.mean(knn, axis=1)  # larger = sparser

        bins = self.grid_bins if self.grid_bins > 1 else max(4, int(np.ceil((self.max_size * 2.0) ** (1.0 / max(1, m)))))
        coords = np.floor(Fn * bins).astype(np.int32)
        coords = np.clip(coords, 0, bins - 1)
        cell_keys = [tuple(int(v) for v in row) for row in coords]

        cell_to_indices: dict[tuple[int, ...], list[int]] = {}
        for i, ck in enumerate(cell_keys):
            cell_to_indices.setdefault(ck, []).append(i)
        cell_occ = np.array([len(cell_to_indices[ck]) for ck in cell_keys], dtype=np.float64)
        density_score = mean_knn / np.maximum(1.0, cell_occ)

        protected: list[int] = []
        if self.protect_extremes:
            for j in range(m):
                protected.append(int(np.argmin(F[:, j])))
        protected = sorted(set(protected))

        keep_set: set[int] = set(protected)

        # Stage 1: keep at most one representative per occupied cell.
        reps: list[int] = []
        for ck, idxs in cell_to_indices.items():
            center = (np.asarray(ck, dtype=np.float64) + 0.5) / float(bins)
            best = max(
                idxs,
                key=lambda i: (density_score[i], -float(np.linalg.norm(Fn[i] - center))),
            )
            reps.append(int(best))
        reps = sorted(set(reps), key=lambda i: (-density_score[i], int(cell_occ[i]), i))
        for i in reps:
            if len(keep_set) >= self.max_size:
                break
            keep_set.add(int(i))

        # Stage 2: round-robin fill, prioritizing cells that are still under-represented.
        if len(keep_set) < self.max_size:
            per_cell_sorted: dict[tuple[int, ...], list[int]] = {
                ck: sorted(idxs, key=lambda i: (-density_score[i], i)) for ck, idxs in cell_to_indices.items()
            }
            selected_per_cell = {ck: 0 for ck in cell_to_indices}
            for i in keep_set:
                selected_per_cell[cell_keys[i]] += 1

            made_progress = True
            while len(keep_set) < self.max_size and made_progress:
                made_progress = False
                cell_order = sorted(
                    cell_to_indices.keys(),
                    key=lambda ck: (selected_per_cell[ck] / max(1, len(cell_to_indices[ck])), selected_per_cell[ck], -len(cell_to_indices[ck])),
                )
                for ck in cell_order:
                    for i in per_cell_sorted[ck]:
                        if i in keep_set:
                            continue
                        keep_set.add(int(i))
                        selected_per_cell[ck] += 1
                        made_progress = True
                        break
                    if len(keep_set) >= self.max_size:
                        break

        keep = np.array(sorted(keep_set))
        if len(keep) > self.max_size:
            unprotected = [i for i in keep if i not in protected]
            unprotected = sorted(unprotected, key=lambda i: (-density_score[i], i))
            final_keep = protected + [i for i in unprotected if i not in protected]
            keep = np.array(sorted(final_keep[: self.max_size]), dtype=np.int64)

        self.items = [self.items[i] for i in keep]
        self._objs = self._objs[keep]
        self._keys = {self._obj_key(o) for o in self._objs}


def _clip_bounds(env: GridEnv, x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    y[:, 0] = np.clip(y[:, 0], 0, env.W - 1)
    y[:, 1] = np.clip(y[:, 1], 0, env.H - 1)
    if y.shape[1] >= 3:
        for i in range(len(y)):
            y[i] = env.clamp_point(y[i], clearance=env.min_clearance + 2.0)
    return y.astype(np.float32)


def _enforce_altitude_profile(env: GridEnv, path: np.ndarray, clearance_margin: float = 2.0, max_pitch_deg: float = 35.0, n_pass: int = 4) -> np.ndarray:
    y = np.asarray(path, dtype=np.float32).copy()
    if y.shape[1] < 3:
        return y
    safe = np.array([
        env.min_safe_altitude_at(float(px), float(py), clearance=env.min_clearance + float(clearance_margin))
        for px, py in y[:, :2]
    ], dtype=np.float32)
    y[:, 2] = np.maximum(y[:, 2], safe)
    if len(y) <= 1:
        return y
    dxy = np.linalg.norm(np.diff(y[:, :2], axis=0), axis=1)
    dzmax = np.maximum(0.5, np.tan(np.deg2rad(float(max_pitch_deg))) * np.maximum(dxy, 1e-3)).astype(np.float32)
    for _ in range(int(max(1, n_pass))):
        for i in range(1, len(y)):
            y[i, 2] = max(y[i, 2], y[i - 1, 2] - dzmax[i - 1], safe[i])
        for i in range(len(y) - 2, -1, -1):
            y[i, 2] = max(y[i, 2], y[i + 1, 2] - dzmax[i], safe[i])
    y[:, 2] = np.clip(y[:, 2], env.z_min, env.z_max)
    return y.astype(np.float32)


def _repair_segment_clearance(env: GridEnv, path: np.ndarray, step: float = 0.5, clearance_margin: float = 2.0) -> np.ndarray:
    y = np.asarray(path, dtype=np.float32).copy()
    if y.shape[1] < 3:
        return y
    for i in range(1, len(y)):
        pts = sampled_points_array(y[i - 1], y[i], step=step)
        ix = np.clip(np.rint(pts[:, 0]).astype(np.int32), 0, env.W - 1)
        iy = np.clip(np.rint(pts[:, 1]).astype(np.int32), 0, env.H - 1)
        req = env.height[iy, ix] + env.min_clearance + float(clearance_margin)
        need = float(np.max(req - pts[:, 2]))
        if need > 0.0:
            bump = need + 0.5
            y[i - 1, 2] += bump
            y[i, 2] += bump
    return y.astype(np.float32)


def _simplify_polyline_collision_aware(path: np.ndarray, K: int, collision_fn, rng: np.random.Generator, n_try: int = 2000) -> np.ndarray:
    p = np.asarray(path, dtype=np.float32).copy()
    if len(p) <= K:
        return p
    tries = 0
    while len(p) > K and tries < int(n_try):
        tries += 1
        i = int(rng.integers(0, len(p) - 2))
        j = int(rng.integers(i + 2, len(p)))
        if len(p) - (j - i - 1) < K:
            continue
        if not collision_fn(p[i], p[j]):
            p = np.vstack([p[: i + 1], p[j:]]).astype(np.float32)
    if len(p) > K:
        idx = np.linspace(0, len(p) - 1, K).round().astype(np.int32)
        idx[0] = 0
        idx[-1] = len(p) - 1
        cand = p[idx]
        if all(not collision_fn(cand[t - 1], cand[t]) for t in range(1, len(cand))):
            p = cand.astype(np.float32)
    return p.astype(np.float32)


def _path_local_frame(start: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    s = np.asarray(start, dtype=np.float32)
    g = np.asarray(goal, dtype=np.float32)
    d = (g[:2] - s[:2]).astype(np.float32)
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        u = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u = d / L
    v = np.array([-u[1], u[0]], dtype=np.float32)
    return u, v, L


def _build_stratified_candidate(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    K: int,
    rng: np.random.Generator,
    *,
    lateral_frac: float = 0.30,
    n_bands: int = 5,
    progress_jitter: float = 0.08,
    global_mix_prob: float = 0.10,
) -> np.ndarray:
    """Structured-random initialization with coverage along the start-goal corridor.

    The path is sampled layer-by-layer along the main axis from start to goal.
    Inside each layer, a lateral band is chosen and then jittered locally.
    This keeps randomness while preventing all initial paths from collapsing to
    the same corridor.
    """
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    x = np.linspace(start, goal, int(K)).astype(np.float32)
    if K <= 2:
        return x

    u, v, L = _path_local_frame(start, goal)
    lateral_range = max(float(env.resolution) * 4.0, float(L) * float(lateral_frac))
    n_bands = int(max(3, n_bands))
    band_edges = np.linspace(-lateral_range, lateral_range, n_bands + 1)
    base_t = np.linspace(0.0, 1.0, K)
    center_band = (n_bands - 1) / 2.0

    for i in range(1, K - 1):
        t0 = float(base_t[i])
        t = float(np.clip(t0 + rng.uniform(-progress_jitter, progress_jitter), 0.02, 0.98))
        if rng.random() < float(global_mix_prob):
            # retain a few purely exploratory samples
            cand_xy = np.array([
                rng.uniform(0.0, env.W - 1.0),
                rng.uniform(0.0, env.H - 1.0),
            ], dtype=np.float32)
        else:
            # use a permuted band order so different individuals emphasize different corridors
            frac = i / max(1, K - 1)
            preferred = (frac - 0.5) * 0.8
            band_shift = int(np.round(preferred * center_band))
            band_center_idx = int(np.clip(np.round(center_band + band_shift), 0, n_bands - 1))
            band_candidates = list(range(n_bands))
            rng.shuffle(band_candidates)
            if band_center_idx in band_candidates:
                band_candidates.remove(band_center_idx)
                band_candidates.insert(0, band_center_idx)
            chosen = band_candidates[0]
            d = float(rng.uniform(band_edges[chosen], band_edges[chosen + 1]))
            center = start[:2] + t * (goal[:2] - start[:2])
            tangential_jitter = float(rng.normal(0.0, 0.04 * L))
            cand_xy = (center + tangential_jitter * u + d * v).astype(np.float32)

        if x.shape[1] >= 3:
            base_z = env.min_safe_altitude_at(float(cand_xy[0]), float(cand_xy[1]), clearance=env.min_clearance + 2.0)
            z = max(base_z, float(start[2] + t * (goal[2] - start[2]) + rng.normal(1.0, 1.5)))
            x[i] = np.array([cand_xy[0], cand_xy[1], z], dtype=np.float32)
        else:
            x[i, :2] = cand_xy

    x = _clip_bounds(env, x)
    x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
    x = _repair_segment_clearance(env, x, step=0.5, clearance_margin=2.0)
    x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
    x[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
    x[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
    return x.astype(np.float32)


def make_initial_population(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    pop: int,
    K: int,
    seed: int = 0,
    *,
    astar_ratio: float = 0.25,
    astar_threat_weight: float = 0.0,
    astar_jitter_sigma: float = 1.5,
    astar_max_paths: int = 1,
    astar_penalty_step: float = 2.5,
    astar_max_expansions: Optional[int] = None,
    stratified_ratio: float = 0.60,
    stratified_lateral_frac: float = 0.30,
    stratified_n_bands: int = 5,
    stratified_progress_jitter: float = 0.08,
    global_random_ratio: float = 0.15,
    eval_sample_step: float = 0.75,
    smooth_collision_step: float = 0.75,
) -> Tuple[List[Individual], dict]:
    """Initialize population and return a lightweight init profile."""

    rng = np.random.default_rng(seed)
    init: List[Individual] = []
    profile = {
        "astar_total_s": 0.0,
        "astar_search_s": 0.0,
        "astar_backbone_post_s": 0.0,
        "astar_candidates": 0,
        "astar_found": 0,
        "astar_jitter_s": 0.0,
        "astar_jitter_n": 0,
        "stratified_s": 0.0,
        "stratified_n": 0,
        "global_random_s": 0.0,
        "global_random_n": 0,
        "repair_s": 0.0,
        "eval_s": 0.0,
    }

    pop = int(pop)
    K = int(K)
    n_astar = int(np.clip(round(pop * float(astar_ratio)), 0, pop))
    remaining_after_astar = max(0, pop - n_astar)
    n_stratified = int(np.clip(round(pop * float(stratified_ratio)), 0, remaining_after_astar))
    n_global_random = max(0, remaining_after_astar - n_stratified)
    requested_random = int(np.clip(round(pop * float(global_random_ratio)), 0, remaining_after_astar))
    if requested_random > n_global_random:
        take = min(requested_random - n_global_random, n_stratified)
        n_stratified -= take
        n_global_random += take

    astar_paths: List[np.ndarray] = []
    if n_astar > 0:
        t_astar_total0 = time.perf_counter()

        def _downsample_env_maxpool(src_env: GridEnv, factor: int) -> GridEnv:
            """Build a coarse 2D planning grid for A* seeding.

            Important: using `.any()` max-pooling on occupancy is far too conservative
            on dense city maps. For example, when the fine-grid occupancy ratio is around
            0.48, a 4x4 max-pool makes almost every coarse cell occupied, so A* spends a
            long time proving failure and returns no seed at all. We instead use an
            occupancy *fraction* threshold, which preserves wide free corridors while still
            blocking truly dense obstacle regions.
            """
            f = int(max(1, factor))
            if f == 1:
                return src_env
            occ = src_env.occupancy.astype(np.float32, copy=False)
            thr = src_env.threat.astype(np.float32, copy=False)
            H, W = occ.shape
            Hp = ((H + f - 1) // f) * f
            Wp = ((W + f - 1) // f) * f
            if Hp != H or Wp != W:
                occ_pad = np.ones((Hp, Wp), dtype=np.float32)
                occ_pad[:H, :W] = occ
                thr_pad = np.zeros((Hp, Wp), dtype=np.float32)
                thr_pad[:H, :W] = thr
            else:
                occ_pad = occ
                thr_pad = thr
            occ_frac = occ_pad.reshape(Hp // f, f, Wp // f, f).mean(axis=(1, 3))
            thr_ds = thr_pad.reshape(Hp // f, f, Wp // f, f).mean(axis=(1, 3))
            occ_thr = 0.60 if f >= 4 else 0.75
            occ_ds = occ_frac >= occ_thr
            # Keep the coarse threat field aware of clutter so f2-oriented A* still avoids
            # dense urban blocks even when they are not hard-blocked at the coarse scale.
            thr_ds = thr_ds + 0.50 * occ_frac
            return GridEnv(occ_ds, thr_ds, resolution=float(src_env.resolution) * float(f))

        area = int(env.H) * int(env.W)
        factors = (4, 2, 1) if area >= 1_000_000 else ((2, 1) if area >= 300_000 else (1,))
        max_paths = max(1, int(astar_max_paths))

        for f in factors:
            env_astar = _downsample_env_maxpool(env, int(f)) if int(f) > 1 else env
            coarse_occ_ratio = float(np.mean(env_astar.occupancy))
            # If the coarse grid is almost fully blocked, A* will only burn expansions and
            # still fail. Skip that factor and move to a finer one.
            if int(f) > 1 and coarse_occ_ratio >= 0.90:
                continue
            start_astar = start / float(f) if int(f) > 1 else start
            goal_astar = goal / float(f) if int(f) > 1 else goal
            max_exp = astar_max_expansions
            if max_exp is None:
                scale = 0.35 if int(f) >= 4 else (0.60 if int(f) == 2 else 1.0)
                max_exp = min(2_000_000, max(60_000, int(env_astar.H * env_astar.W * scale)))
            penalty_map = np.zeros_like(env_astar.threat, dtype=np.float32)
            for _k in range(max_paths):
                profile["astar_candidates"] += 1
                t_search0 = time.perf_counter()
                res = astar(
                    env_astar,
                    start_astar,
                    goal_astar,
                    threat_weight=float(astar_threat_weight),
                    penalty_map=penalty_map,
                    allow_diagonal=True,
                    max_expansions=int(max_exp),
                    heuristic_weight=(1.15 if int(f) > 1 else 1.05),
                )
                profile["astar_search_s"] += time.perf_counter() - t_search0
                if res.path is None or len(res.path) < 2:
                    break
                t_post0 = time.perf_counter()
                raw = res.path.astype(np.float32)
                if int(f) > 1:
                    raw[:, 0] *= float(f)
                    raw[:, 1] *= float(f)
                raw3d = env.lift_path_to_3d(
                    raw,
                    start_z=float(start[2]) if len(start) >= 3 else None,
                    goal_z=float(goal[2]) if len(goal) >= 3 else None,
                )
                if len(raw3d) > K:
                    idx = np.linspace(0, len(raw3d) - 1, K).round().astype(np.int32)
                    idx[0] = 0
                    idx[-1] = len(raw3d) - 1
                    base = raw3d[idx].astype(np.float32)
                elif len(raw3d) < K:
                    base = densify_polyline_to_K(raw3d, K)
                else:
                    base = raw3d.astype(np.float32)
                base = _clip_bounds(env, base)
                if base.shape[1] >= 3:
                    base = _enforce_altitude_profile(env, base, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=2)
                base[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
                base[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
                profile["astar_backbone_post_s"] += time.perf_counter() - t_post0
                astar_paths.append(base)
                profile["astar_found"] += 1
                cells = res.path.astype(np.int32)
                if len(cells) > 2:
                    cells = cells[1:-1]
                for xx, yy in cells:
                    if 0 <= yy < penalty_map.shape[0] and 0 <= xx < penalty_map.shape[1]:
                        penalty_map[yy, xx] += float(astar_penalty_step)
            if len(astar_paths) > 0:
                break
        profile["astar_total_s"] = time.perf_counter() - t_astar_total0
        uniq: List[np.ndarray] = []
        for p in astar_paths:
            if not any(np.allclose(p, q, atol=1e-3, rtol=0.0) for q in uniq):
                uniq.append(p)
        astar_paths = uniq

    if astar_paths:
        for t in range(n_astar):
            t0 = time.perf_counter()
            base = astar_paths[t % len(astar_paths)]
            x = base.copy()
            noise = rng.normal(0.0, float(astar_jitter_sigma), size=x.shape).astype(np.float32)
            noise[:, 0:2] *= 0.35
            if x.shape[1] >= 3:
                noise[:, 2] *= 0.20
            noise[0] = 0
            noise[-1] = 0
            x = _clip_bounds(env, x + noise)
            x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
            x = _repair_segment_clearance(env, x, step=0.5, clearance_margin=2.0)
            x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
            x[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
            x[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
            t_r0 = time.perf_counter()
            x = repair_light(env, x, rng, tries=8)
            profile["repair_s"] += time.perf_counter() - t_r0
            t_e0 = time.perf_counter()
            er = evaluate_path(env, x, sample_step=eval_sample_step)
            profile["eval_s"] += time.perf_counter() - t_e0
            init.append(Individual(x=x, er=er))
            profile["astar_jitter_s"] += time.perf_counter() - t0
            profile["astar_jitter_n"] += 1

    for _ in range(n_stratified):
        t0 = time.perf_counter()
        x = _build_stratified_candidate(
            env, start, goal, K, rng,
            lateral_frac=float(stratified_lateral_frac),
            n_bands=int(stratified_n_bands),
            progress_jitter=float(stratified_progress_jitter),
            global_mix_prob=0.10,
        )
        t_r0 = time.perf_counter()
        x = repair_light(env, x, rng, tries=6)
        profile["repair_s"] += time.perf_counter() - t_r0
        t_e0 = time.perf_counter()
        er = evaluate_path(env, x, sample_step=eval_sample_step)
        profile["eval_s"] += time.perf_counter() - t_e0
        init.append(Individual(x=x, er=er))
        profile["stratified_s"] += time.perf_counter() - t0
        profile["stratified_n"] += 1

    while len(init) < pop:
        t0 = time.perf_counter()
        x = np.linspace(start, goal, K).astype(np.float32)
        noise = rng.normal(0.0, 3.0, size=x.shape).astype(np.float32)
        if x.shape[1] >= 3:
            noise[:, 0:2] *= 0.85
            noise[:, 2] *= 0.30
        noise[0] = 0
        noise[-1] = 0
        x = _clip_bounds(env, x + noise)
        x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
        x = _repair_segment_clearance(env, x, step=0.5, clearance_margin=2.0)
        x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
        x[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
        x[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
        t_r0 = time.perf_counter()
        x = repair_light(env, x, rng, tries=6)
        profile["repair_s"] += time.perf_counter() - t_r0
        t_e0 = time.perf_counter()
        er = evaluate_path(env, x, sample_step=eval_sample_step)
        profile["eval_s"] += time.perf_counter() - t_e0
        init.append(Individual(x=x, er=er))
        profile["global_random_s"] += time.perf_counter() - t0
        profile["global_random_n"] += 1

    return init[:pop], profile


def crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    K = len(p1)
    if K <= 4:
        return p1.copy()
    a = int(rng.integers(1, K - 2))
    b = int(rng.integers(a + 1, K - 1))
    child = p1.copy()
    child[a:b] = p2[a:b]
    # 保持首尾
    child[0] = p1[0]
    child[-1] = p1[-1]
    return child.astype(np.float32)


def mutate(rng: np.random.Generator, x: np.ndarray, sigma: float = 2.5, p_mut: float = 0.25) -> np.ndarray:
    y = x.copy().astype(np.float32)
    D = y.shape[1]
    for i in range(1, len(y) - 1):
        if rng.random() < p_mut:
            delta = rng.normal(0.0, sigma, size=(D,)).astype(np.float32)
            if D >= 3:
                delta[2] *= 0.5
            y[i] += delta
    y[0] = x[0]
    y[-1] = x[-1]
    return y


def repair(env: GridEnv, x: np.ndarray, tries: int = 15, rng: Optional[np.random.Generator] = None, n_pass: int = 8) -> np.ndarray:
    """Stronger repair: clamp -> fix bad points -> raise colliding segments -> enforce pitch-limited altitude."""
    y = _clip_bounds(env, x)
    rng = np.random.default_rng() if rng is None else rng
    D = y.shape[1]
    for i in range(1, len(y) - 1):
        bad = env.is_occupied(int(round(y[i, 0])), int(round(y[i, 1]))) if D < 3 else False
        if D >= 3:
            bad = bad or (not env.is_free_point(y[i], clearance=env.min_clearance + 2.0))
        if bad:
            for _ in range(int(tries)):
                if D >= 3:
                    base_z = env.min_safe_altitude_at(float(y[i, 0]), float(y[i, 1]), clearance=env.min_clearance + 2.0)
                    cand = y[i].copy()
                    cand[:2] += rng.normal(0.0, 2.0, size=(2,)).astype(np.float32)
                    cand[2] = max(base_z, cand[2] + float(rng.normal(2.0, 2.0)))
                else:
                    cand = y[i] + rng.normal(0.0, 4.0, size=(D,)).astype(np.float32)
                cand = _clip_bounds(env, cand[None, :])[0]
                good = env.is_free_point(cand, clearance=env.min_clearance + 2.0) if D >= 3 else (not env.is_occupied(int(round(cand[0])), int(round(cand[1]))))
                if good:
                    y[i] = cand
                    break
    if D >= 3:
        y = _repair_segment_clearance(env, y, step=0.75, clearance_margin=2.0)
        y = _enforce_altitude_profile(env, y, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=n_pass)
        if n_pass >= 6:
            y = _repair_segment_clearance(env, y, step=0.75, clearance_margin=2.0)
            y = _enforce_altitude_profile(env, y, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=n_pass)
    return y.astype(np.float32)



def repair_light(env: GridEnv, x: np.ndarray, rng: np.random.Generator, tries: int = 6) -> np.ndarray:
    """Cheaper repair for the evolutionary loop.

    Key idea:
      - keep initialization heavy and robust
      - make offspring repair much lighter, because it runs pop*n_gen times
      - first detect clearly bad interior waypoints in a vectorized way
    """
    y = _clip_bounds(env, x)
    D = y.shape[1]

    if len(y) > 2:
        ix = np.rint(y[:, 0]).astype(np.int32)
        iy = np.rint(y[:, 1]).astype(np.int32)
        ix = np.clip(ix, 0, env.W - 1)
        iy = np.clip(iy, 0, env.H - 1)
        if D >= 3:
            safe_z = env.height[iy, ix] + (env.min_clearance + 1.5)
            bad_mask = y[:, 2] < safe_z
        else:
            bad_mask = env.occupancy[iy, ix]
        bad_idx = (np.flatnonzero(bad_mask[1:-1]) + 1).tolist()
    else:
        bad_idx = []

    for i in bad_idx:
        for _ in range(int(tries)):
            cand = y[i].copy()
            cand[:2] += rng.normal(0.0, 1.5, size=(2,)).astype(np.float32)
            if D >= 3:
                base_z = env.min_safe_altitude_at(float(cand[0]), float(cand[1]), clearance=env.min_clearance + 1.5)
                cand[2] = max(base_z, float(cand[2]) + float(rng.normal(1.5, 1.0)))
            cand = _clip_bounds(env, cand[None, :])[0]
            if D >= 3:
                cix = int(np.clip(round(float(cand[0])), 0, env.W - 1))
                ciy = int(np.clip(round(float(cand[1])), 0, env.H - 1))
                good = float(cand[2]) >= float(env.height[ciy, cix] + env.min_clearance + 1.5)
            else:
                good = not env.is_occupied(int(round(cand[0])), int(round(cand[1])))
            if good:
                y[i] = cand
                break
    if D >= 3:
        safe_z = env.height[iy, ix] + (env.min_clearance + 1.5)
        needs_segment_repair = len(bad_idx) > 0
        if not needs_segment_repair:
            dxy = np.linalg.norm(np.diff(y[:, :2], axis=0), axis=1)
            dz = np.abs(np.diff(y[:, 2]))
            pitch = np.arctan2(dz, np.maximum(1e-6, dxy)) if len(y) >= 2 else np.zeros((0,), dtype=np.float32)
            if np.any(pitch > np.deg2rad(35.0)):
                needs_segment_repair = True
            else:
                coarse_pts = sampled_polyline_points_array(y, step=max(2.0, 4.0 * float(env.resolution)), xy_resolution=env.resolution)
                cix = np.clip(np.rint(coarse_pts[:, 0]).astype(np.int32), 0, env.W - 1)
                ciy = np.clip(np.rint(coarse_pts[:, 1]).astype(np.int32), 0, env.H - 1)
                req = env.height[ciy, cix] + env.min_clearance + 1.5
                if bool(np.any(coarse_pts[:, 2] < req)):
                    needs_segment_repair = True
        if needs_segment_repair:
            y = _repair_segment_clearance(env, y, step=0.75, clearance_margin=1.5)
            y = _enforce_altitude_profile(env, y, clearance_margin=1.5, max_pitch_deg=35.0, n_pass=3)
    return y.astype(np.float32)


def _best_feasible_obj(pop_inds: List[Individual]) -> Optional[np.ndarray]:
    feas = [it.er.obj for it in pop_inds if it.er.feasible]
    if not feas:
        return None
    return np.min(np.stack(feas, axis=0), axis=0).astype(np.float64)


def _elite_by_obj(pop_inds: List[Individual], obj_idx: int, k: int = 3) -> List[Individual]:
    feas = [it for it in pop_inds if it.er.feasible]
    if not feas:
        cand = sorted(pop_inds, key=lambda it: it.er.violation)
        return cand[: max(1, k)]
    feas.sort(key=lambda it: float(it.er.obj[obj_idx]))
    return feas[: max(1, k)]


def _threat_grad(env: GridEnv, x: float, y: float) -> np.ndarray:
    ix = int(np.clip(round(x), 1, env.W - 2))
    iy = int(np.clip(round(y), 1, env.H - 2))
    gx = 0.5 * (float(env.threat[iy, ix + 1]) - float(env.threat[iy, ix - 1]))
    gy = 0.5 * (float(env.threat[iy + 1, ix]) - float(env.threat[iy - 1, ix]))
    return np.array([gx, gy], dtype=np.float32)


def _objective_local_search(
    env: GridEnv,
    base: np.ndarray,
    obj_idx: int,
    rng: np.random.Generator,
    K: int,
    smooth_tries: int,
    eval_sample_step: float,
    max_turn_deg: float,
    *,
    mode: str = "exploit",
) -> np.ndarray:
    y = np.asarray(base, dtype=np.float32).copy()
    n = len(y)
    if n <= 2:
        return y

    if obj_idx == 0:
        # f1: shorten path. exploit=gentle shortcut, escape=larger restructuring.
        if smooth_tries > 0:
            tries = max(4, int(smooth_tries) * (3 if mode == "escape" else 2))
            y = shortcut_smooth(y, n_try=tries, rng=rng, collision_fn=lambda p, q: segment_collision(env, p, q, step=0.75))
            y = densify_polyline_to_K(y, K)
        avg_rounds = 1 if mode == "exploit" else 3
        for _ in range(avg_rounds):
            i = int(rng.integers(1, n - 1))
            alpha = 0.5 if mode == "exploit" else 0.7
            y[i] = (1.0 - alpha) * y[i] + 0.5 * alpha * (y[i - 1] + y[i + 1])
    elif obj_idx == 1:
        # f2: push hotspot points away from threat gradient / local corridor shift
        threat_vals = np.array([float(env.threat[int(np.clip(round(p[1]), 0, env.H - 1)), int(np.clip(round(p[0]), 0, env.W - 1))]) for p in y], dtype=np.float32)
        hotspot_idx = np.argsort(-threat_vals[1:-1])[: max(1, min(3, n - 2))] + 1
        for i in hotspot_idx:
            g = _threat_grad(env, float(y[i, 0]), float(y[i, 1]))
            gn = float(np.linalg.norm(g))
            if gn < 1e-6:
                prev = y[i] - y[i - 1]
                nxt = y[i + 1] - y[i]
                d = prev[:2] + nxt[:2]
                dn = float(np.linalg.norm(d))
                if dn > 1e-6:
                    side = np.array([-d[1], d[0]], dtype=np.float32) / dn
                else:
                    side = rng.normal(0.0, 1.0, size=(2,)).astype(np.float32)
                    side /= max(1e-6, float(np.linalg.norm(side)))
                lo, hi = (2.0, 6.0) if mode == "exploit" else (4.0, 10.0)
                shift = side * float(rng.uniform(lo, hi))
            else:
                lo, hi = (2.0, 6.0) if mode == "exploit" else (4.0, 10.0)
                shift = (-g / gn) * float(rng.uniform(lo, hi))
            y[i, :2] += shift
        if n > 6 and rng.random() < (0.35 if mode == "exploit" else 0.75):
            a = int(rng.integers(1, max(2, n // 3)))
            b = int(rng.integers(max(a + 1, n // 2), n - 1))
            alpha = rng.uniform(-0.10, 0.10) if mode == "exploit" else rng.uniform(-0.25, 0.25)
            seg = y[b, :2] - y[a, :2]
            seg_n = float(np.linalg.norm(seg))
            if seg_n > 1e-6:
                side = np.array([-seg[1], seg[0]], dtype=np.float32) / seg_n
                y[a:b, :2] += alpha * side * max(env.H, env.W) * 0.05
    else:
        # f3: reduce turning / altitude oscillation. escape-mode smooths a wider band.
        rounds = 2 if mode == "exploit" else 4
        for _ in range(rounds):
            i = int(rng.integers(1, n - 1))
            y[i, :2] = 0.25 * y[i - 1, :2] + 0.5 * y[i, :2] + 0.25 * y[i + 1, :2]
            if y.shape[1] >= 3:
                y[i, 2] = 0.25 * y[i - 1, 2] + 0.5 * y[i, 2] + 0.25 * y[i + 1, 2]

    y[0] = base[0]
    y[-1] = base[-1]
    y = repair_light(env, y, rng, tries=6)
    if y.shape[1] >= 3:
        y = _enforce_altitude_profile(env, y, clearance_margin=1.5, max_pitch_deg=35.0, n_pass=3)
    return y.astype(np.float32)


def _subproblem_density(pop_inds: List[Individual], center_idx: int, k: int = 5) -> float:
    if len(pop_inds) <= 1:
        return 0.0
    feats = np.stack([it.er.obj for it in pop_inds], axis=0).astype(np.float64)
    mn = np.min(feats, axis=0)
    mx = np.max(feats, axis=0)
    denom = np.maximum(1e-9, mx - mn)
    fn = (feats - mn) / denom
    q = fn[int(center_idx)]
    d = np.linalg.norm(fn - q[None, :], axis=1)
    d = np.sort(d)
    kk = min(max(1, int(k)), len(d) - 1)
    return float(1.0 / max(1e-9, d[kk]))


def _update_subproblem_utility(
    pop_inds: List[Individual],
    W: np.ndarray,
    z: np.ndarray,
    utility: np.ndarray,
    last_g: np.ndarray,
    stall: np.ndarray,
    replace_counts: np.ndarray,
    archive: Archive,
    *,
    k_density: int = 5,
    use_archive_density: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pop_inds)
    cur_g = np.empty(n, dtype=np.float64)
    feats = np.stack([it.er.obj for it in pop_inds], axis=0).astype(np.float64)
    feas_mask = np.array([it.er.feasible for it in pop_inds], dtype=bool)
    viol = np.array([float(it.er.violation) for it in pop_inds], dtype=np.float64)
    cur_g[feas_mask] = np.max(W[feas_mask] * np.abs(feats[feas_mask] - z[None, :]), axis=1)
    cur_g[~feas_mask] = 1e12 + viol[~feas_mask]

    if n <= 1:
        density = np.zeros(n, dtype=np.float64)
    else:
        mn = np.min(feats, axis=0)
        mx = np.max(feats, axis=0)
        denom = np.maximum(1e-9, mx - mn)
        fn = (feats - mn) / denom
        diff = fn[:, None, :] - fn[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        kk = min(max(1, int(k_density)), n - 1)
        kth = np.partition(dist, kk - 1, axis=1)[:, kk - 1]
        density = 1.0 / np.maximum(1e-9, kth)

    if use_archive_density and archive.items:
        for i in np.flatnonzero(feas_mask):
            density[i] = 0.5 * density[i] + 0.5 * archive.density(feats[i], k=k_density)

    improve = np.maximum(0.0, last_g - cur_g)
    improved = improve > 1e-12
    stall = np.where(improved, 0.0, stall + 1.0)

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    u = (
        0.42 * _norm(improve)
        + 0.26 * _norm(replace_counts)
        + 0.20 * (1.0 - _norm(density))
        + 0.20 * utility
        - 0.18 * _norm(stall)
    )
    u = np.clip(u + 1e-6, 1e-6, None)
    return u, cur_g, stall


def _current_subproblem_scalar_values(pop_inds: List[Individual], W: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Current per-subproblem scalar objective used by MOEA/D and MTOE.

    Feasible individuals use Tchebycheff scalarization; infeasible individuals
    receive a large penalty plus constraint violation, consistent with the rest
    of this implementation.
    """
    n = len(pop_inds)
    cur_g = np.empty(n, dtype=np.float64)
    feats = np.stack([it.er.obj for it in pop_inds], axis=0).astype(np.float64)
    feas_mask = np.array([it.er.feasible for it in pop_inds], dtype=bool)
    viol = np.array([float(it.er.violation) for it in pop_inds], dtype=np.float64)
    if np.any(feas_mask):
        cur_g[feas_mask] = np.max(W[feas_mask] * np.abs(feats[feas_mask] - z[None, :]), axis=1)
    if np.any(~feas_mask):
        cur_g[~feas_mask] = 1e12 + viol[~feas_mask]
    return cur_g



def _resample_polyline_xy(path: np.ndarray, n_samples: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or len(path) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xy = path[:, :2]
    if len(xy) == 1:
        return np.repeat(xy, max(1, int(n_samples)), axis=0)
    n_samples = int(max(2, n_samples))
    seg = xy[1:] - xy[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(xy[:1], n_samples, axis=0)
    targets = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            out[i] = xy[-1]
            continue
        denom = max(1e-12, cum[j + 1] - cum[j])
        alpha = float((t - cum[j]) / denom)
        out[i] = (1.0 - alpha) * xy[j] + alpha * xy[j + 1]
    return out


def _path_basin_signature(
    path: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    *,
    n_bands: int = 7,
    n_samples: int = 9,
) -> Optional[str]:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or len(path) < 2:
        return None
    pts = _resample_polyline_xy(path, n_samples=max(3, int(n_samples)))
    s = np.asarray(start[:2], dtype=np.float64)
    g = np.asarray(goal[:2], dtype=np.float64)
    d = g - s
    L = float(np.linalg.norm(d))
    if L <= 1e-9:
        return None
    t = d / L
    n = np.array([-t[1], t[0]], dtype=np.float64)
    half_width = max(6.0, 0.22 * L)
    signed = (pts - s[None, :]) @ n
    u = np.clip(signed / half_width, -0.999, 0.999)
    n_bands = int(max(3, n_bands))
    bands = np.floor((u + 1.0) * 0.5 * n_bands).astype(np.int32)
    bands = np.clip(bands, 0, n_bands - 1)
    return '-'.join(str(int(v)) for v in bands.tolist())


def _best_feasible_individual(pop_inds: List[Individual], archive: "Archive") -> Optional[Individual]:
    best = None
    best_f2 = None
    for it in pop_inds:
        if not it.er.feasible:
            continue
        v = float(it.er.obj[1])
        if best is None or v < best_f2:
            best = it
            best_f2 = v
    for it in getattr(archive, 'items', []):
        if not it.er.feasible:
            continue
        v = float(it.er.obj[1])
        if best is None or v < best_f2:
            best = it
            best_f2 = v
    return best


def _basin_probe_from_state(
    cur_best_ind: Optional[Individual],
    *,
    start: np.ndarray,
    goal: np.ndarray,
    basin_registry: dict,
    basin_best_f2: dict,
    basin_histories: dict,
    basin_visit_counts: dict,
    distinct_sequence: deque,
    n_bands: int,
    n_samples: int,
    stagnation_window: int,
    f2_tol_abs: float,
    f2_tol_rel: float,
    reference_f2: Optional[float],
    ref_gap_tol: float,
    min_distinct: int,
) -> Optional[dict]:
    if cur_best_ind is None or (not cur_best_ind.er.feasible):
        return None
    sig = _path_basin_signature(cur_best_ind.x, start, goal, n_bands=n_bands, n_samples=n_samples)
    if sig is None:
        return None
    entered_new = sig not in basin_registry
    if entered_new:
        basin_registry[sig] = int(len(basin_registry))
    basin_id = int(basin_registry[sig])
    cur_f2 = float(cur_best_ind.er.obj[1])
    prev_best = basin_best_f2.get(basin_id)
    basin_best = cur_f2 if prev_best is None else min(float(prev_best), cur_f2)
    basin_best_f2[basin_id] = basin_best
    hist = basin_histories[basin_id]
    hist.append(basin_best)
    basin_visit_counts[basin_id] += 1
    if not distinct_sequence or distinct_sequence[-1] != basin_id:
        distinct_sequence.append(basin_id)
    hist_vals = list(hist)
    span = float(max(hist_vals) - min(hist_vals)) if hist_vals else 0.0
    eps = max(float(f2_tol_abs), abs(basin_best) * float(f2_tol_rel))
    basin_plateau = bool(len(hist) >= int(max(3, stagnation_window)) and span <= eps)
    reference_gap = None
    if reference_f2 is not None and np.isfinite(reference_f2):
        denom = max(1.0, abs(float(reference_f2)))
        reference_gap = float((basin_best - float(reference_f2)) / denom)
    action = 'normal'
    if basin_plateau:
        enough_distinct = len(set(distinct_sequence)) >= int(max(1, min_distinct))
        close_to_ref = (reference_gap is None) or (reference_gap <= float(ref_gap_tol))
        action = 'stop_candidate' if (enough_distinct and close_to_ref) else 'escape'
    return {
        'signature': sig,
        'basin_id': basin_id,
        'entered_new_basin': bool(entered_new),
        'current_best_f2': cur_f2,
        'basin_best_f2': float(basin_best),
        'basin_hist_len': int(len(hist)),
        'basin_span': float(span),
        'basin_plateau': bool(basin_plateau),
        'reference_f2': (None if reference_f2 is None or not np.isfinite(reference_f2) else float(reference_f2)),
        'reference_gap': reference_gap,
        'distinct_basins_seen': int(len(set(distinct_sequence))),
        'visit_count': int(basin_visit_counts[basin_id]),
        'action': action,
    }

def _mtoe_stop_decision(
    mtoe_hist: deque[float],
    *,
    tol_fun: float,
    confidence: float,
) -> tuple[bool, Optional[dict]]:
    """Return whether MTOE early stopping should trigger.

    Paper logic: over the last γ generations, compute the sample variance of the
    MTOE sequence and use a χ² test to check whether the underlying standard
    deviation is below ``Tol_fun``. Here ``chi2.sf`` gives the support
    probability for the hypothesis ``std(MTOE) <= Tol_fun``.

    Practical safeguard: variance alone can be misleading when the sequence is
    almost constant but still clearly non-zero (for example, a steady stream of
    similar improvements). Therefore we additionally require the window mean to
    be no larger than ``Tol_fun`` before declaring stagnation.
    """
    gamma = int(len(mtoe_hist))
    if gamma < 2:
        return False, None

    vals = np.asarray(mtoe_hist, dtype=np.float64)
    tol_fun = float(max(1e-12, tol_fun))
    confidence = float(np.clip(confidence, 0.0, 0.999999999))
    df = gamma - 1
    var = float(np.var(vals, ddof=1))
    std = float(np.sqrt(max(0.0, var)))
    stat = float(var * df / (tol_fun ** 2))
    p_support = float(chi2.sf(stat, df))
    critical_stat = float(chi2.isf(confidence, df))
    mean_guard = bool(float(np.mean(vals)) <= tol_fun)
    legacy_stop = bool(p_support >= confidence)
    stop = bool(legacy_stop and mean_guard)
    return stop, {
        "gamma": gamma,
        "variance": var,
        "window_std": std,
        "stat": stat,
        "critical_stat": critical_stat,
        "p_support": p_support,
        "tol_fun": tol_fun,
        "confidence": confidence,
        "mean_guard": mean_guard,
        "legacy_stop_without_mean_guard": legacy_stop,
        "mtoe": float(vals[-1]),
        "window_min": float(np.min(vals)),
        "window_max": float(np.max(vals)),
        "window_mean": float(np.mean(vals)),
    }


def _sample_active_subproblems(rng: np.random.Generator, utility: np.ndarray, n_select: int, phase: float) -> np.ndarray:
    n = len(utility)
    n_select = int(max(1, n_select))
    base = np.asarray(utility, dtype=np.float64).copy()
    if n <= n_select:
        return np.arange(n, dtype=np.int32)
    # early phase: flatter distribution for exploration; later phase: sharper focus
    tau = float(np.clip(1.15 - 0.75 * phase, 0.35, 1.15))
    probs = np.power(np.maximum(base, 1e-6), 1.0 / tau)
    probs /= np.sum(probs)
    core = int(min(n, max(1, round(0.75 * n_select))))
    idx_core = rng.choice(n, size=core, replace=False, p=probs)
    # keep some deterministic high-utility subproblems every generation
    remain = n_select - core
    if remain > 0:
        elite = np.argsort(-utility)[: min(remain, n)]
        idx = np.unique(np.concatenate([idx_core, elite])).astype(np.int32)
        if len(idx) < n_select:
            avail = np.setdiff1d(np.arange(n, dtype=np.int32), idx, assume_unique=False)
            extra = rng.choice(avail, size=n_select - len(idx), replace=False)
            idx = np.concatenate([idx, extra]).astype(np.int32)
        return idx[:n_select]
    return idx_core.astype(np.int32)


def _allocate_extreme_budget(
    total_extra: int,
    best_hist: List[deque],
    ls_success_ema: np.ndarray,
    ls_improve_ema: np.ndarray,
    min_per_obj: int = 1,
    max_frac_per_obj: float = 0.70,
) -> np.ndarray:
    if total_extra <= 0:
        return np.zeros(3, dtype=np.int32)
    scores = np.zeros(3, dtype=np.float64)
    for m in range(3):
        recent_gain = 0.0
        hist = list(best_hist[m])
        if len(hist) >= 2:
            recent_gain = max(0.0, float(hist[0] - hist[-1]))
        scores[m] = 0.55 * recent_gain + 0.25 * float(ls_success_ema[m]) + 0.20 * float(ls_improve_ema[m])
    scores += 1e-6
    alloc = np.zeros(3, dtype=np.int32)
    remaining = int(total_extra)
    if remaining >= 3 * min_per_obj:
        alloc[:] = int(min_per_obj)
        remaining -= int(np.sum(alloc))
    max_per = max(1, int(np.ceil(float(total_extra) * float(max_frac_per_obj))))
    if remaining > 0:
        probs = scores / np.sum(scores)
        for _ in range(remaining):
            order = np.argsort(-probs)
            placed = False
            for j in order:
                if alloc[j] < max_per:
                    alloc[j] += 1
                    placed = True
                    break
            if not placed:
                alloc[int(order[0])] += 1
    return alloc



def _cheap_candidate_precheck(
    env: GridEnv,
    child: np.ndarray,
    parent: Optional[np.ndarray] = None,
    *,
    threat_mean: Optional[float] = None,
    threat_std: Optional[float] = None,
) -> bool:
    """Fast reject rules before expensive evaluation.

    Return True when the child is *worth* a full evaluation.
    """
    y = np.asarray(child, dtype=np.float32)
    if len(y) <= 1:
        return False
    seg = y[1:] - y[:-1]
    seglen_xy = np.linalg.norm(seg[:, :2], axis=1)
    if np.any(seglen_xy < 1e-3):
        return False

    # absurdly long zig-zag offspring are usually wasted evaluations.
    path_len = float(np.sum(np.linalg.norm(seg, axis=1), dtype=np.float64))
    chord = float(np.linalg.norm(y[-1] - y[0]))
    if path_len > max(50.0, 4.0 * chord):
        return False

    if parent is not None:
        p = np.asarray(parent, dtype=np.float32)
        if p.shape == y.shape and float(np.mean(np.linalg.norm(y - p, axis=1), dtype=np.float64)) < 0.15:
            return False

    if y.shape[1] >= 3:
        if np.any(y[:, 2] < env.z_min) or np.any(y[:, 2] > env.z_max):
            return False
        dxy = np.maximum(1e-6, seglen_xy)
        pitch = np.arctan2(np.abs(seg[:, 2]), dxy)
        if np.any(pitch > np.deg2rad(45.0)):
            return False

    # coarse polyline sampling: much cheaper than full eval, but catches obvious OOB / clearance failures.
    coarse_step = max(2.0, 4.0 * float(env.resolution))
    pts = sampled_polyline_points_array(y, step=coarse_step, xy_resolution=env.resolution)
    ix = np.rint(pts[:, 0]).astype(np.int32)
    iy = np.rint(pts[:, 1]).astype(np.int32)
    oob = (ix < 0) | (ix >= env.W) | (iy < 0) | (iy >= env.H)
    if bool(np.any(oob)):
        return False
    ix = np.clip(ix, 0, env.W - 1)
    iy = np.clip(iy, 0, env.H - 1)

    if y.shape[1] >= 3:
        safe = env.height[iy, ix] + env.min_clearance
        miss_ratio = float(np.mean(pts[:, 2] < safe)) if len(pts) > 0 else 0.0
        if miss_ratio > 0.10:
            return False
        threat_vals = env.threat[iy, ix]
    else:
        if bool(np.any(env.occupancy[iy, ix])):
            return False
        threat_vals = env.threat[iy, ix]

    if threat_mean is None:
        threat_mean = float(np.mean(env.threat))
    if threat_std is None:
        threat_std = float(np.std(env.threat))
    if float(np.mean(threat_vals, dtype=np.float64)) > float(threat_mean + 2.5 * threat_std):
        return False
    return True


def _coarse_path_code(path: np.ndarray, n_samples: int = 9) -> np.ndarray:
    p = np.asarray(path, dtype=np.float32)
    if len(p) <= 1:
        return np.zeros((n_samples, 2), dtype=np.int32)
    idx = np.linspace(0, len(p) - 1, n_samples).round().astype(np.int32)
    idx[0] = 0
    idx[-1] = len(p) - 1
    q = p[idx, :2]
    return np.rint(q / 8.0).astype(np.int32)


def _path_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = _coarse_path_code(a)
    cb = _coarse_path_code(b)
    return float(np.mean(np.linalg.norm(ca.astype(np.float32) - cb.astype(np.float32), axis=1)))


def _pick_diverse_partner(pool: List[Individual], base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if not pool:
        return np.asarray(base, dtype=np.float32)
    if len(pool) == 1:
        return pool[0].x.astype(np.float32)
    cand_idx = rng.choice(len(pool), size=min(8, len(pool)), replace=False)
    best_j = int(cand_idx[0])
    best_d = -1.0
    for jj in cand_idx:
        d = _path_distance(base, pool[int(jj)].x)
        if d > best_d:
            best_d = d
            best_j = int(jj)
    return pool[best_j].x.astype(np.float32)


def _splice_with_candidate(base: np.ndarray, donor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(base, dtype=np.float32).copy()
    z = np.asarray(donor, dtype=np.float32)
    if len(y) <= 4 or y.shape != z.shape:
        return y
    a = int(rng.integers(1, max(2, len(y) // 3)))
    b = int(rng.integers(max(a + 1, len(y) // 2), len(y) - 1))
    y[a:b] = z[a:b]
    y[0] = base[0]
    y[-1] = base[-1]
    return y.astype(np.float32)


def _make_escape_candidate(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    K: int,
    rng: np.random.Generator,
    pool: List[Individual],
    archive: Archive,
    *,
    sigma: float = 8.0,
) -> np.ndarray:
    sources: List[Individual] = []
    if archive.items:
        sources.extend(archive.items[-min(len(archive.items), 24):])
    sources.extend(pool)
    if not sources:
        x = _build_stratified_candidate(env, start, goal, K, rng, lateral_frac=0.45, n_bands=7, progress_jitter=0.14, global_mix_prob=0.25)
        return repair_light(env, x, rng, tries=8)

    if any(it.er.feasible for it in sources):
        feasible_sources = [it for it in sources if it.er.feasible]
    else:
        feasible_sources = sources
    base_ind = feasible_sources[int(rng.integers(0, len(feasible_sources)))]
    base = base_ind.x.astype(np.float32)
    donor = _pick_diverse_partner(sources, base, rng)
    child = _splice_with_candidate(base, donor, rng)
    child = mutate(rng, child, sigma=float(sigma), p_mut=0.55)
    if rng.random() < 0.65:
        alt = _build_stratified_candidate(env, start, goal, K, rng, lateral_frac=0.50, n_bands=7, progress_jitter=0.16, global_mix_prob=0.30)
        child = _splice_with_candidate(child, alt, rng)
    child[0] = start
    child[-1] = goal
    child = repair_light(env, child, rng, tries=8)
    if child.shape[1] >= 3:
        child = _enforce_altitude_profile(env, child, clearance_margin=1.5, max_pitch_deg=35.0, n_pass=3)
    return child.astype(np.float32)


def better_feasible(a: EvalResult, b: EvalResult) -> bool:
    """
    可行解优先：
    - 可行 > 不可行
    - 都不可行：violation 更小更好
    - 都可行：这里只用于配合 scalar 值比较（外部逻辑做）
    """
    if a.feasible and (not b.feasible):
        return True
    if (not a.feasible) and b.feasible:
        return False
    if (not a.feasible) and (not b.feasible):
        return a.violation < b.violation
    return False  # 都可行时这里不直接判优


def moead(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_gen: int = 120,
    pop: int = 80,
    K: int = 30,
    T: int = 10,
    seed: int = 0,
    max_turn_deg: float = 90.0,
    archive_size: int = 0,
    archive_soft_limit: int = 320,
    archive_grid_bins: int = 0,
    archive_keep_extremes: bool = True,
    smooth_tries: int = 8,
    # --- evaluation & smoothing sampling step ---
    eval_sample_step: float = 0.5,
    smooth_collision_step: float = 0.5,
    # --- debug logging ---
    debug_log_path: Optional[str] = None,
    debug_every: int = 1,
    debug_level: int = 2,
    debug_console: bool = False,
    # --- init seeding ---
    init_astar_ratio: float = 0.25,
    init_astar_threat_weight: float = 0.0,
    init_astar_jitter_sigma: float = 1.5,
    init_astar_max_paths: int = 5,
    init_astar_penalty_step: float = 2.5,
    init_astar_max_expansions: Optional[int] = None,
    init_stratified_ratio: float = 0.60,
    init_stratified_lateral_frac: float = 0.30,
    init_stratified_n_bands: int = 5,
    init_stratified_progress_jitter: float = 0.08,
    init_global_random_ratio: float = 0.15,
    weight_extreme_bias: float = 0.20,
    extreme_offspring_ratio: float = 0.20,
    extreme_potential_window: int = 20,
    extreme_min_extra_per_obj: int = 1,
    extreme_max_frac_per_obj: float = 0.70,
    local_search_interval: int = 10,
    local_search_elite_k: int = 3,
    local_search_attempts_per_obj: int = 2,
    active_subproblem_ratio: float = 1.0,
    utility_update_interval: int = 3,
    utility_use_archive_density: bool = False,
    log_flush_every: int = 10,
    global_mating_prob_base: float = 0.10,
    global_mating_prob_stall: float = 0.35,
    archive_parent_prob: float = 0.20,
    escape_stall_window: int = 12,
    escape_injections: int = 6,
    escape_large_mut_sigma: float = 8.0,
    moead_min_gen: int = 20,
    moead_max_gen: Optional[int] = None,
    mtoe_tol_fun: float = 1e-5,
    mtoe_confidence: float = 0.99,
    disable_mtoe_stop: bool = False,
    basin_shadow_enable: bool = True,
    basin_band_count: int = 7,
    basin_signature_samples: int = 9,
    basin_stagnation_window: int = 10,
    basin_f2_tol_abs: float = 1.0,
    basin_f2_tol_rel: float = 0.01,
    basin_ref_gap_tol: float = 0.08,
    basin_min_distinct: int = 2,
    basin_escape_injections: int = 0,
    reference_f2: Optional[float] = None,
):
    """
    MOEA/D（Tchebycheff 标量化 + 外部档案）
    个体：固定 K 点的 2D 路径（含首尾固定）

    返回：
      pop_inds: List[Individual]
      archive: Archive
      log: dict
    """
    rng = np.random.default_rng(seed)

    logger = (
        _make_file_logger(debug_log_path, console=bool(debug_console))
        if (debug_log_path and str(debug_log_path).strip())
        else None
    )
    debug_every = int(max(1, debug_every))
    debug_level = int(debug_level)
    eval_sample_step = float(eval_sample_step)
    smooth_collision_step = float(smooth_collision_step)
    utility_update_interval = int(max(1, utility_update_interval))
    log_flush_every = int(max(1, log_flush_every))
    moead_min_gen = int(max(1, moead_min_gen))
    moead_max_gen = int(n_gen if moead_max_gen is None else max(1, moead_max_gen))
    if moead_max_gen < moead_min_gen:
        moead_max_gen = moead_min_gen
    mtoe_tol_fun = float(max(1e-12, mtoe_tol_fun))
    mtoe_confidence = float(np.clip(mtoe_confidence, 0.0, 0.999999999))
    disable_mtoe_stop = bool(disable_mtoe_stop)
    basin_shadow_enable = bool(basin_shadow_enable)
    basin_band_count = int(max(3, basin_band_count))
    basin_signature_samples = int(max(3, basin_signature_samples))
    basin_stagnation_window = int(max(3, basin_stagnation_window))
    basin_f2_tol_abs = float(max(0.0, basin_f2_tol_abs))
    basin_f2_tol_rel = float(max(0.0, basin_f2_tol_rel))
    basin_ref_gap_tol = float(max(0.0, basin_ref_gap_tol))
    basin_min_distinct = int(max(1, basin_min_distinct))
    basin_escape_injections = int(max(0, basin_escape_injections))
    reference_f2 = None if reference_f2 is None or (not np.isfinite(reference_f2)) else float(reference_f2)
    threat_mean = float(np.mean(env.threat))
    threat_std = float(np.std(env.threat))
    global_mating_prob_base = float(np.clip(global_mating_prob_base, 0.0, 1.0))
    global_mating_prob_stall = float(np.clip(global_mating_prob_stall, 0.0, 1.0))
    archive_parent_prob = float(np.clip(archive_parent_prob, 0.0, 1.0))
    escape_stall_window = int(max(3, escape_stall_window))
    escape_injections = int(max(0, escape_injections))
    escape_large_mut_sigma = float(max(1.0, escape_large_mut_sigma))
    M = 3
    W = uniform_weights(M, pop, seed=seed, extreme_bias=float(weight_extreme_bias))
    B = build_neighbors(W, T=T)

    t_init0 = time.perf_counter()
    pop_inds, init_profile = make_initial_population(
        env,
        start,
        goal,
        pop=pop,
        K=K,
        seed=seed,
        astar_ratio=float(init_astar_ratio),
        astar_threat_weight=float(init_astar_threat_weight),
        astar_jitter_sigma=float(init_astar_jitter_sigma),
        astar_max_paths=int(init_astar_max_paths),
        astar_penalty_step=float(init_astar_penalty_step),
        astar_max_expansions=init_astar_max_expansions,
        stratified_ratio=float(init_stratified_ratio),
        stratified_lateral_frac=float(init_stratified_lateral_frac),
        stratified_n_bands=int(init_stratified_n_bands),
        stratified_progress_jitter=float(init_stratified_progress_jitter),
        global_random_ratio=float(init_global_random_ratio),
        eval_sample_step=eval_sample_step,
        smooth_collision_step=smooth_collision_step,
    )
    init_s = time.perf_counter() - t_init0
    effective_archive_size = int(archive_size) if int(archive_size) > 0 else int(archive_soft_limit)
    archive = Archive(
        max_size=effective_archive_size,
        grid_bins=int(archive_grid_bins),
        protect_extremes=bool(archive_keep_extremes),
        crowd_k=5,
    )

    # ideal point z*
    # ideal point z*: 建议只由可行解更新（避免不可行解把 z 拉得过小导致标量化失真）
    feas_objs = [ind.er.obj for ind in pop_inds if ind.er.feasible]
    if len(feas_objs) > 0:
        z = np.min(np.stack(feas_objs, axis=0), axis=0).astype(np.float64)
    else:
        # 兜底：若初始全不可行（理论上 A* seeding 后很少发生），仍用全体最小值
        z = np.min(np.stack([ind.er.obj for ind in pop_inds], axis=0), axis=0).astype(np.float64)


    for ind in pop_inds:
        if ind.er.feasible:
            archive.add(ind)

    # debug counters (only meaningful if logger != None)
    coll_calls = 0
    coll_time_s = 0.0

    def collision_fn(p, q):
        """Collision function used in shortcut smoothing."""
        nonlocal coll_calls, coll_time_s
        coll_calls += 1
        if logger is None or debug_level < 3:
            return segment_collision(env, p, q, step=smooth_collision_step)
        t0 = time.perf_counter()
        hit = segment_collision(env, p, q, step=smooth_collision_step)
        coll_time_s += (time.perf_counter() - t0)
        return hit

    n_eval = 0
    best_hist = [deque(maxlen=max(2, int(extreme_potential_window))) for _ in range(3)]
    init_best = _best_feasible_obj(pop_inds)
    if init_best is not None:
        for m in range(3):
            best_hist[m].append(float(init_best[m]))
    ls_success_ema = np.zeros(3, dtype=np.float64)
    ls_improve_ema = np.zeros(3, dtype=np.float64)
    utility = np.ones(pop, dtype=np.float64)
    utility_last_g = _current_subproblem_scalar_values(pop_inds, W, z)
    mtoe_best_g = utility_last_g.copy()
    prev_z_mtoe = z.copy()
    stall = np.zeros(pop, dtype=np.float64)
    replace_counts = np.zeros(pop, dtype=np.float64)

    mtoe_window = 10
    mtoe_debug_tail: deque[dict] = deque(maxlen=25)
    mtoe_tests_run = 0
    shadow_stop_events: list[dict] = []
    basin_debug_tail: deque[dict] = deque(maxlen=50)
    basin_registry: dict[str, int] = {}
    basin_best_f2: dict[int, float] = {}
    basin_histories = defaultdict(lambda: deque(maxlen=max(3, int(basin_stagnation_window))))
    basin_visit_counts = defaultdict(int)
    basin_distinct_sequence: deque[int] = deque(maxlen=max(10, int(basin_stagnation_window) * 4))

    if logger is not None:
        st = _pop_stats(pop_inds)
        logger.info(
            "[start] env(H=%d,W=%d) seed=%d planner_seed=%d n_gen=%d pop=%d K=%d T=%d max_turn_deg=%.1f "
            "eval_step=%.3f smooth_step=%.3f init_s=%.3f init_feasible=%d/%d (%.1f%%) "
            "init_min_viol=%.3f init_mean_viol=%.3f init_archive=%d "
            "A* seeding: ratio=%.3f max_paths=%d penalty_step=%.3f threat_w=%.3f jitter=%.3f | "
            "stratified: ratio=%.3f lateral_frac=%.3f bands=%d progress_jitter=%.3f global_random=%.3f | weight_extreme_bias=%.3f | extra_ratio=%.3f potential_W=%d ls_interval=%d ls_elite=%d ls_attempts=%d active_ratio=%.3f utility_update_interval=%d utility_archive_density=%s log_flush_every=%d archive_cap=%s | MTOE(mode=best_so_far_delta window=%d min_gen=%d max_gen=%d tol_fun=%.3e confidence=%.4f disable_stop=%s) | BasinShadow(enable=%s bands=%d samples=%d window=%d f2_tol_abs=%.3f f2_tol_rel=%.4f ref_gap_tol=%.4f min_distinct=%d basin_escape_inj=%d ref_f2=%s)",
            int(env.H),
            int(env.W),
            int(seed),
            int(seed),
            int(n_gen),
            int(pop),
            int(K),
            int(T),
            float(max_turn_deg),
            float(eval_sample_step),
            float(smooth_collision_step),
            float(init_s),
            int(st["n_feasible"]),
            int(st["n"]),
            float(100.0 * st["feasible_ratio"]),
            float(st["min_violation"]),
            float(st["mean_violation"]),
            int(len(archive.items)),
            float(init_astar_ratio),
            int(init_astar_max_paths),
            float(init_astar_penalty_step),
            float(init_astar_threat_weight),
            float(init_astar_jitter_sigma),
            float(init_stratified_ratio),
            float(init_stratified_lateral_frac),
            int(init_stratified_n_bands),
            float(init_stratified_progress_jitter),
            float(init_global_random_ratio),
            float(weight_extreme_bias),
            float(extreme_offspring_ratio),
            int(extreme_potential_window),
            int(local_search_interval),
            int(local_search_elite_k),
            int(local_search_attempts_per_obj),
            float(active_subproblem_ratio),
            int(utility_update_interval),
            str(bool(utility_use_archive_density)),
            int(log_flush_every),
            str("unbounded" if archive.max_size is None else archive.max_size),
            int(mtoe_window),
            int(moead_min_gen),
            int(moead_max_gen),
            float(mtoe_tol_fun),
            float(mtoe_confidence),
            str(bool(disable_mtoe_stop)),
            str(bool(basin_shadow_enable)),
            int(basin_band_count),
            int(basin_signature_samples),
            int(basin_stagnation_window),
            float(basin_f2_tol_abs),
            float(basin_f2_tol_rel),
            float(basin_ref_gap_tol),
            int(basin_min_distinct),
            int(basin_escape_injections),
            ("None" if reference_f2 is None else f"{float(reference_f2):.6f}"),
        )
        init_total = float(init_s)
        init_other = max(0.0, init_total - float(init_profile.get("astar_total_s", 0.0)) - float(init_profile.get("stratified_s", 0.0)) - float(init_profile.get("global_random_s", 0.0)))
        logger.info(
            "[init] total=%.3fs astar_total=%.3fs (search=%.3fs backbone_post=%.3fs cand=%d found=%d) astar_jitter=%.3fs (n=%d) stratified=%.3fs (n=%d) global_random=%.3fs (n=%d) repair=%.3fs eval=%.3fs other=%.3fs",
            init_total,
            float(init_profile.get("astar_total_s", 0.0)),
            float(init_profile.get("astar_search_s", 0.0)),
            float(init_profile.get("astar_backbone_post_s", 0.0)),
            int(init_profile.get("astar_candidates", 0)),
            int(init_profile.get("astar_found", 0)),
            float(init_profile.get("astar_jitter_s", 0.0)),
            int(init_profile.get("astar_jitter_n", 0)),
            float(init_profile.get("stratified_s", 0.0)),
            int(init_profile.get("stratified_n", 0)),
            float(init_profile.get("global_random_s", 0.0)),
            int(init_profile.get("global_random_n", 0)),
            float(init_profile.get("repair_s", 0.0)),
            float(init_profile.get("eval_s", 0.0)),
            float(init_other),
        )
        denom = max(1e-9, init_total)
        logger.info(
            "[init_ratio] astar_total=%.1f%% astar_jitter=%.1f%% stratified=%.1f%% global_random=%.1f%% repair=%.1f%% eval=%.1f%%",
            100.0 * float(init_profile.get("astar_total_s", 0.0)) / denom,
            100.0 * float(init_profile.get("astar_jitter_s", 0.0)) / denom,
            100.0 * float(init_profile.get("stratified_s", 0.0)) / denom,
            100.0 * float(init_profile.get("global_random_s", 0.0)) / denom,
            100.0 * float(init_profile.get("repair_s", 0.0)) / denom,
            100.0 * float(init_profile.get("eval_s", 0.0)) / denom,
        )

    mtoe_hist: deque[float] = deque(maxlen=mtoe_window)
    stop_reason = "max_gen"
    stop_info = None
    actual_gens = 0
    best_f2_so_far = None
    best_f2_no_improve = 0
    last_escape_gen = -10**9

    for gen in range(int(moead_max_gen)):
        gen_t0 = time.perf_counter()
        gen_eval_s = 0.0
        gen_smooth_s = 0.0
        gen_neighbor_s = 0.0
        gen_repair_s = 0.0
        gen_eval_count = 0
        gen_precheck_reject = 0
        gen_regular_eval = 0
        gen_extra_eval = 0
        gen_ls_eval = 0
        gen_escape_eval = 0
        gen_n_smooth_in = 0
        gen_n_smooth_out = 0
        gen_coll_calls0 = coll_calls
        gen_coll_time0 = coll_time_s
        gen_extreme_alloc = np.zeros(3, dtype=np.int32)
        gen_ls_success = np.zeros(3, dtype=np.int32)
        gen_ls_attempts = np.zeros(3, dtype=np.int32)
        gen_escape_attempts = 0
        gen_escape_success = 0

        def _update_neighbors_for_child(child: np.ndarray, er_child: EvalResult, center_idx: int):
            nonlocal z, gen_neighbor_s, replace_counts
            if er_child.feasible:
                z = np.minimum(z, er_child.obj)
            t_n0 = time.perf_counter()
            nb = B[int(center_idx)]
            local_repl = 0
            for j in nb:
                jj = int(j)
                cur = pop_inds[jj]
                replaced = False
                if er_child.feasible and (not cur.er.feasible):
                    pop_inds[jj] = Individual(x=child, er=er_child)
                    replaced = True
                elif (not er_child.feasible) and cur.er.feasible:
                    replaced = False
                elif (not er_child.feasible) and (not cur.er.feasible):
                    if er_child.violation < cur.er.violation:
                        pop_inds[jj] = Individual(x=child, er=er_child)
                        replaced = True
                else:
                    g_child = tchebycheff(er_child.obj, W[jj], z)
                    g_cur = tchebycheff(cur.er.obj, W[jj], z)
                    if g_child <= g_cur:
                        pop_inds[jj] = Individual(x=child, er=er_child)
                        replaced = True
                if replaced:
                    local_repl += 1
                    replace_counts[jj] += 1.0
            gen_neighbor_s += (time.perf_counter() - t_n0)
            if er_child.feasible:
                archive.add(Individual(x=child, er=er_child))
            return local_repl

        archive.compact(force=False)
        # subproblem-level utility update + active subproblem sampling
        phase = float(gen) / max(1.0, float(n_gen - 1))
        if gen == 0 or (gen % utility_update_interval) == 0:
            utility, utility_last_g, stall = _update_subproblem_utility(
                pop_inds,
                W,
                z,
                utility,
                utility_last_g,
                stall,
                replace_counts,
                archive,
                use_archive_density=bool(utility_use_archive_density),
            )
            replace_counts *= 0.65
        else:
            replace_counts *= 0.80
        active_n = int(np.clip(round(float(active_subproblem_ratio) * float(pop)), 1, pop))
        if phase < 0.20:
            active_n = pop
        elif phase > 0.75:
            active_n = max(1, min(pop, int(round(0.70 * active_n))))
        active_idx = _sample_active_subproblems(rng, utility, n_select=active_n, phase=phase)
        utility_med = float(np.median(utility))
        stall_med = float(np.median(stall))
        stall_p60 = float(np.percentile(stall, 60))
        stall_p75 = float(np.percentile(stall, 75))

        # regular MOEA/D offspring, now focused on active subproblems
        for i in active_idx:
            i = int(i)
            nb = B[i]
            use_global = False
            stall_boost = 0.0 if stall_p75 <= 0 else float(np.clip((stall[i] - stall_med) / max(1e-6, stall_p75 - stall_med + 1e-6), 0.0, 1.0))
            p_global = min(0.95, global_mating_prob_base + global_mating_prob_stall * stall_boost)
            if rng.random() < p_global:
                use_global = True
            if use_global:
                pidx = rng.choice(pop, size=2, replace=False)
                p1 = pop_inds[int(pidx[0])].x
                if archive.items and rng.random() < archive_parent_prob:
                    p2 = archive.items[int(rng.integers(0, len(archive.items)))].x
                else:
                    p2 = _pick_diverse_partner(pop_inds, p1, rng)
            else:
                pidx = rng.choice(nb, size=2, replace=False)
                p1 = pop_inds[int(pidx[0])].x
                p2 = pop_inds[int(pidx[1])].x

            child = crossover(rng, p1, p2)
            sigma = 2.0 if utility[i] >= utility_med else 3.2
            p_mut = 0.20 if stall[i] <= stall_med else 0.35
            child = mutate(rng, child, sigma=sigma, p_mut=p_mut)
            t_r0 = time.perf_counter()
            child = repair_light(env, child, rng, tries=5 if utility[i] >= utility_med else 7)
            gen_repair_s += (time.perf_counter() - t_r0)

            do_smooth = (smooth_tries > 0) and ((i + gen) % 2 == 0) and (stall[i] <= stall_p75)
            if do_smooth:
                gen_n_smooth_in += int(len(child))
                t_s0 = time.perf_counter()
                child = shortcut_smooth(child, n_try=smooth_tries, rng=rng, collision_fn=collision_fn)
                gen_smooth_s += (time.perf_counter() - t_s0)
                gen_n_smooth_out += int(len(child))
                child = densify_polyline_to_K(child, K)

            child[0] = start
            child[-1] = goal
            if not _cheap_candidate_precheck(env, child, parent=p1, threat_mean=threat_mean, threat_std=threat_std):
                gen_precheck_reject += 1
                continue

            t_e0 = time.perf_counter()
            er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
            gen_eval_s += (time.perf_counter() - t_e0)
            n_eval += 1
            gen_eval_count += 1
            gen_regular_eval += 1
            _update_neighbors_for_child(child, er_child, i)

        # extra offspring with dynamic resource allocation across objective extremes
        total_extra = int(round(float(extreme_offspring_ratio) * float(pop)))
        if total_extra > 0:
            gen_extreme_alloc = _allocate_extreme_budget(
                total_extra,
                best_hist,
                ls_success_ema,
                ls_improve_ema,
                min_per_obj=int(extreme_min_extra_per_obj),
                max_frac_per_obj=float(extreme_max_frac_per_obj),
            )
            extreme_centers = [int(np.argmax(W[:, m])) for m in range(3)]
            for m in range(3):
                for _ in range(int(gen_extreme_alloc[m])):
                    elites = _elite_by_obj(pop_inds, m, k=max(1, int(local_search_elite_k)))
                    parent = elites[int(rng.integers(0, len(elites)))]
                    mode = "exploit" if utility[extreme_centers[m]] >= utility_med else "escape"
                    child = _objective_local_search(
                        env,
                        parent.x,
                        obj_idx=m,
                        rng=rng,
                        K=K,
                        smooth_tries=smooth_tries,
                        eval_sample_step=eval_sample_step,
                        max_turn_deg=max_turn_deg,
                        mode=mode,
                    )
                    child[0] = start
                    child[-1] = goal
                    if not _cheap_candidate_precheck(env, child, parent=parent.x, threat_mean=threat_mean, threat_std=threat_std):
                        gen_precheck_reject += 1
                        continue
                    t_e0 = time.perf_counter()
                    er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
                    gen_eval_s += (time.perf_counter() - t_e0)
                    n_eval += 1
                    gen_eval_count += 1
                    gen_extra_eval += 1
                    _update_neighbors_for_child(child, er_child, extreme_centers[m])

        # periodic directed local search on elite solutions
        if int(local_search_interval) > 0 and (((gen + 1) % int(local_search_interval)) == 0):
            extreme_centers = [int(np.argmax(W[:, m])) for m in range(3)]
            for m in range(3):
                center_idx = int(extreme_centers[m])
                ls_worth_try = (stall[center_idx] >= stall_p60) or (gen < max(10, int(local_search_interval))) or (ls_success_ema[m] >= 0.05) or (ls_improve_ema[m] > 1e-6)
                if not ls_worth_try:
                    continue
                elites = _elite_by_obj(pop_inds, m, k=max(1, int(local_search_elite_k)))
                for attempt in range(min(int(local_search_attempts_per_obj), len(elites))):
                    parent = elites[attempt]
                    base_val = float(parent.er.obj[m]) if parent.er.feasible else None
                    mode = "exploit" if utility[extreme_centers[m]] >= utility_med and stall[extreme_centers[m]] < stall_p60 else "escape"
                    child = _objective_local_search(
                        env,
                        parent.x,
                        obj_idx=m,
                        rng=rng,
                        K=K,
                        smooth_tries=smooth_tries,
                        eval_sample_step=eval_sample_step,
                        max_turn_deg=max_turn_deg,
                        mode=mode,
                    )
                    child[0] = start
                    child[-1] = goal
                    if not _cheap_candidate_precheck(env, child, parent=parent.x, threat_mean=threat_mean, threat_std=threat_std):
                        gen_precheck_reject += 1
                        continue
                    t_e0 = time.perf_counter()
                    er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
                    gen_eval_s += (time.perf_counter() - t_e0)
                    n_eval += 1
                    gen_eval_count += 1
                    gen_ls_eval += 1
                    gen_ls_attempts[m] += 1
                    improved = False
                    if er_child.feasible and parent.er.feasible and float(er_child.obj[m]) + 1e-9 < float(parent.er.obj[m]):
                        improved = True
                        gain = max(0.0, float(parent.er.obj[m] - er_child.obj[m]))
                        ls_improve_ema[m] = 0.8 * ls_improve_ema[m] + 0.2 * gain
                    elif er_child.feasible and (not parent.er.feasible):
                        improved = True
                        ls_improve_ema[m] = 0.8 * ls_improve_ema[m] + 0.2 * 1.0
                    else:
                        ls_improve_ema[m] = 0.9 * ls_improve_ema[m]
                    ls_success_ema[m] = 0.8 * ls_success_ema[m] + 0.2 * (1.0 if improved else 0.0)
                    if improved:
                        gen_ls_success[m] += 1
                    _update_neighbors_for_child(child, er_child, extreme_centers[m])

        cur_best = _best_feasible_obj(pop_inds)
        if cur_best is not None:
            cur_best_f2 = float(cur_best[1])
            if best_f2_so_far is None or cur_best_f2 + 1e-9 < float(best_f2_so_far):
                best_f2_so_far = cur_best_f2
                best_f2_no_improve = 0
            else:
                best_f2_no_improve += 1
        else:
            best_f2_no_improve += 1

        cur_best_ind = _best_feasible_individual(pop_inds, archive)
        basin_probe = None
        if basin_shadow_enable:
            basin_probe = _basin_probe_from_state(
                cur_best_ind,
                start=start,
                goal=goal,
                basin_registry=basin_registry,
                basin_best_f2=basin_best_f2,
                basin_histories=basin_histories,
                basin_visit_counts=basin_visit_counts,
                distinct_sequence=basin_distinct_sequence,
                n_bands=basin_band_count,
                n_samples=basin_signature_samples,
                stagnation_window=basin_stagnation_window,
                f2_tol_abs=basin_f2_tol_abs,
                f2_tol_rel=basin_f2_tol_rel,
                reference_f2=reference_f2,
                ref_gap_tol=basin_ref_gap_tol,
                min_distinct=basin_min_distinct,
            )
            if basin_probe is not None:
                basin_probe = dict(basin_probe)
                basin_probe['gen'] = int(gen)
                basin_debug_tail.append(dict(basin_probe))

        trigger_escape = bool(escape_injections > 0 and best_f2_no_improve >= escape_stall_window and (gen - last_escape_gen) >= max(3, escape_stall_window // 2))
        if trigger_escape:
            target_order = np.argsort(-(stall + 0.25 * (1.0 / np.maximum(utility, 1e-6))))
            for target in target_order[: min(int(escape_injections), len(target_order))]:
                target = int(target)
                child = _make_escape_candidate(env, start, goal, K, rng, pop_inds, archive, sigma=escape_large_mut_sigma)
                child[0] = start
                child[-1] = goal
                if not _cheap_candidate_precheck(env, child, parent=pop_inds[target].x, threat_mean=threat_mean, threat_std=threat_std):
                    gen_precheck_reject += 1
                    continue
                t_e0 = time.perf_counter()
                er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
                gen_eval_s += (time.perf_counter() - t_e0)
                n_eval += 1
                gen_eval_count += 1
                gen_escape_eval += 1
                gen_escape_attempts += 1
                prev_best_target = pop_inds[target].er
                _update_neighbors_for_child(child, er_child, target)
                if er_child.feasible and ((not prev_best_target.feasible) or float(er_child.obj[1]) + 1e-9 < float(prev_best_target.obj[1])):
                    gen_escape_success += 1
            last_escape_gen = gen
            if gen_escape_success > 0:
                best_f2_no_improve = 0

        if cur_best is not None:
            for m in range(3):
                best_hist[m].append(float(cur_best[m]))
        archive.compact(force=True)
        # --- per-generation debug ---
        if logger is not None and ((gen % debug_every) == 0 or gen == int(n_gen) - 1):
            st = _pop_stats(pop_inds)
            gen_total_s = time.perf_counter() - gen_t0
            n_eval_gen = max(1, int(gen_eval_count))
            avg_eval_ms = 1000.0 * gen_eval_s / max(1, n_eval_gen)
            avg_repair_ms = 1000.0 * gen_repair_s / max(1, n_eval_gen)
            avg_smooth_ms = 1000.0 * gen_smooth_s / max(1, n_eval_gen)
            avg_neighbor_ms = 1000.0 * gen_neighbor_s / max(1, n_eval_gen)
            best_obj = st["best_obj_feasible"]

            if debug_level <= 1:
                logger.info(
                    "[gen=%d] total=%.3fs archive=%d feasible=%d/%d (%.1f%%) min_viol=%.3f mean_viol=%.3f best_feas=%s extra=%s ls=%s escape=%d/%d",
                    int(gen),
                    float(gen_total_s),
                    int(len(archive.items)),
                    int(st["n_feasible"]),
                    int(st["n"]),
                    float(100.0 * st["feasible_ratio"]),
                    float(st["min_violation"]),
                    float(st["mean_violation"]),
                    str(best_obj),
                    gen_extreme_alloc.tolist(),
                    gen_ls_success.tolist(),
                    int(gen_escape_success),
                    int(gen_escape_attempts),
                )
            else:
                # Level>=2: add timing breakdown
                msg = (
                    "[gen=%d] total=%.3fs archive=%d feasible=%d/%d (%.1f%%) "
                    "eval=%.2fms(n=%d) repair=%.2fms smooth=%.2fms neigh=%.2fms pre_reject=%d eval_split=%d/%d/%d/%d best_feas=%s extra=%s ls=%s escape=%d/%d"
                )

                if debug_level >= 3:
                    # Level>=3: collision profiling + sampled cell estimation (for one example)
                    gen_coll_calls = int(coll_calls - gen_coll_calls0)
                    gen_coll_time = float(coll_time_s - gen_coll_time0)

                    # estimate sampled-cells count of evaluation cost (use one representative path: the first individual)
                    try:
                        sample_path = pop_inds[0].x
                        est_cells = _estimate_sampled_cells_count(sample_path, eval_sample_step)
                    except Exception:
                        est_cells = -1

                    msg += (
                        " coll_calls=%d coll_time=%.3fs est_cells(eval_step)= %d "
                        "smooth_vertices(avg_in->avg_out)=%.1f->%.1f"
                    )
                    avg_in = float(gen_n_smooth_in) / max(1, n_eval_gen)
                    avg_out = float(gen_n_smooth_out) / max(1, n_eval_gen)
                    logger.info(
                        msg,
                        int(gen),
                        float(gen_total_s),
                        int(len(archive.items)),
                        int(st["n_feasible"]),
                        int(st["n"]),
                        float(100.0 * st["feasible_ratio"]),
                        float(avg_eval_ms),
                        int(gen_eval_count),
                        float(avg_repair_ms),
                        float(avg_smooth_ms),
                        float(avg_neighbor_ms),
                        int(gen_precheck_reject),
                        int(gen_regular_eval),
                        int(gen_extra_eval),
                        int(gen_ls_eval),
                        int(gen_escape_eval),
                        str(best_obj),
                        gen_extreme_alloc.tolist(),
                        gen_ls_success.tolist(),
                        int(gen_escape_success),
                        int(gen_escape_attempts),
                        int(gen_coll_calls),
                        float(gen_coll_time),
                        int(est_cells),
                        float(avg_in),
                        float(avg_out),
                    )
                else:
                    logger.info(
                        msg,
                        int(gen),
                        float(gen_total_s),
                        int(len(archive.items)),
                        int(st["n_feasible"]),
                        int(st["n"]),
                        float(100.0 * st["feasible_ratio"]),
                        float(avg_eval_ms),
                        int(gen_eval_count),
                        float(avg_repair_ms),
                        float(avg_smooth_ms),
                        float(avg_neighbor_ms),
                        int(gen_precheck_reject),
                        int(gen_regular_eval),
                        int(gen_extra_eval),
                        int(gen_ls_eval),
                        int(gen_escape_eval),
                        str(best_obj),
                        gen_extreme_alloc.tolist(),
                        gen_ls_success.tolist(),
                        int(gen_escape_success),
                        int(gen_escape_attempts),
                    )
                if basin_probe is not None:
                    logger.info(
                        "[gen=%d][basin] basin_id=%d entered=%s action=%s distinct=%d basin_best_f2=%.6f current_best_f2=%.6f span=%.6f plateau=%s ref_f2=%s ref_gap=%s sig=%s",
                        int(gen),
                        int(basin_probe["basin_id"]),
                        str(bool(basin_probe["entered_new_basin"])),
                        str(basin_probe["action"]),
                        int(basin_probe["distinct_basins_seen"]),
                        float(basin_probe["basin_best_f2"]),
                        float(basin_probe["current_best_f2"]),
                        float(basin_probe["basin_span"]),
                        str(bool(basin_probe["basin_plateau"])),
                        ("None" if basin_probe.get("reference_f2") is None else f"{float(basin_probe['reference_f2']):.6f}"),
                        ("None" if basin_probe.get("reference_gap") is None else f"{float(basin_probe['reference_gap']):.6f}"),
                        str(basin_probe["signature"]),
                    )

        cur_g_raw = _current_subproblem_scalar_values(pop_inds, W, z)
        cur_g_best = np.minimum(mtoe_best_g, cur_g_raw)
        toe_vec = np.maximum(0.0, mtoe_best_g - cur_g_best)
        mtoe_idx = int(np.argmax(toe_vec)) if toe_vec.size > 0 else -1
        mtoe = float(toe_vec[mtoe_idx]) if mtoe_idx >= 0 else 0.0
        z_delta = np.abs(z - prev_z_mtoe)
        nonzero_count = int(np.count_nonzero(toe_vec > 1e-12))
        mtoe_probe = {
            "gen": int(gen),
            "mode": "best_so_far_delta",
            "idx": int(mtoe_idx),
            "value": float(mtoe),
            "delta_count": int(nonzero_count),
            "delta_mean": float(np.mean(toe_vec)) if toe_vec.size > 0 else 0.0,
            "delta_p90": float(np.percentile(toe_vec, 90)) if toe_vec.size > 0 else 0.0,
            "prev_best": float(mtoe_best_g[mtoe_idx]) if mtoe_idx >= 0 else None,
            "cur_raw": float(cur_g_raw[mtoe_idx]) if mtoe_idx >= 0 else None,
            "cur_best": float(cur_g_best[mtoe_idx]) if mtoe_idx >= 0 else None,
            "z_delta_inf": float(np.max(z_delta)) if z_delta.size > 0 else 0.0,
            "z_delta_l2": float(np.linalg.norm(z_delta)) if z_delta.size > 0 else 0.0,
        }
        mtoe_debug_tail.append(mtoe_probe)
        mtoe_hist.append(mtoe)
        mtoe_best_g = cur_g_best
        prev_z_mtoe = z.copy()
        actual_gens = gen + 1

        if (gen + 1) >= moead_min_gen and len(mtoe_hist) >= mtoe_window:
            mtoe_tests_run += 1
            should_stop, mtoe_stats = _mtoe_stop_decision(
                mtoe_hist,
                tol_fun=mtoe_tol_fun,
                confidence=mtoe_confidence,
            )
            if mtoe_stats is not None:
                mtoe_stats["probe"] = dict(mtoe_probe)
            if logger is not None and ((gen % debug_every) == 0 or gen == int(moead_max_gen) - 1):
                logger.info(
                    "[gen=%d][mtoe] value=%.6e idx=%d nonzero=%d mean=%.6e p90=%.6e std=%.6e var=%.6e p_support=%.6f mean_guard=%s tol_fun=%.6e confidence=%.6f z_shift_inf=%.6e window=[%.6e, %.6e]",
                    int(gen),
                    float(mtoe_stats["mtoe"]),
                    int(mtoe_probe["idx"]),
                    int(mtoe_probe["delta_count"]),
                    float(mtoe_probe["delta_mean"]),
                    float(mtoe_probe["delta_p90"]),
                    float(mtoe_stats["window_std"]),
                    float(mtoe_stats["variance"]),
                    float(mtoe_stats["p_support"]),
                    str(bool(mtoe_stats["mean_guard"])),
                    float(mtoe_stats["tol_fun"]),
                    float(mtoe_stats["confidence"]),
                    float(mtoe_probe["z_delta_inf"]),
                    float(mtoe_stats["window_min"]),
                    float(mtoe_stats["window_max"]),
                )
            if should_stop:
                event = {
                    "gen": int(gen),
                    "reason": "mtoe",
                    "enabled": bool(not disable_mtoe_stop),
                    "mtoe": float(mtoe_stats["mtoe"]),
                    "idx": int(mtoe_probe["idx"]),
                    "delta_count": int(mtoe_probe["delta_count"]),
                    "delta_mean": float(mtoe_probe["delta_mean"]),
                    "window_std": float(mtoe_stats["window_std"]),
                    "p_support": float(mtoe_stats["p_support"]),
                    "mean_guard": bool(mtoe_stats["mean_guard"]),
                    "tol_fun": float(mtoe_stats["tol_fun"]),
                    "confidence": float(mtoe_stats["confidence"]),
                    "basin": (dict(basin_probe) if basin_probe is not None else None),
                }
                shadow_stop_events.append(event)
                basin_action = str(basin_probe.get("action")) if basin_probe is not None else "normal"
                do_basin_escape = bool(basin_action == "escape" and basin_escape_injections > 0 and (gen - last_escape_gen) >= max(3, escape_stall_window // 2))
                if do_basin_escape:
                    target_order = np.argsort(-(stall + 0.25 * (1.0 / np.maximum(utility, 1e-6))))
                    emergency_success = 0
                    emergency_attempts = 0
                    for target in target_order[: min(int(basin_escape_injections), len(target_order))]:
                        target = int(target)
                        child = _make_escape_candidate(env, start, goal, K, rng, pop_inds, archive, sigma=escape_large_mut_sigma)
                        if not _cheap_candidate_precheck(env, child, parent=pop_inds[target].x, threat_mean=threat_mean, threat_std=threat_std):
                            gen_precheck_reject += 1
                            continue
                        er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
                        n_eval += 1
                        gen_eval_count += 1
                        gen_escape_eval += 1
                        emergency_attempts += 1
                        prev_best_target = pop_inds[target].er
                        _update_neighbors_for_child(child, er_child, target)
                        if er_child.feasible and ((not prev_best_target.feasible) or float(er_child.obj[1]) + 1e-9 < float(prev_best_target.obj[1])):
                            emergency_success += 1
                    last_escape_gen = gen
                    if emergency_success > 0:
                        best_f2_no_improve = 0
                    if logger is not None:
                        logger.info("[basin_escape] gen=%d action=%s success=%d/%d basin_id=%s", int(gen), basin_action, int(emergency_success), int(emergency_attempts), str(None if basin_probe is None else basin_probe.get('basin_id')))
                    if emergency_success > 0:
                        continue
                emergency_escape = bool(escape_injections > 0 and (gen - last_escape_gen) >= max(3, escape_stall_window // 2))
                if emergency_escape:
                    target_order = np.argsort(-(stall + 0.25 * (1.0 / np.maximum(utility, 1e-6))))
                    emergency_success = 0
                    emergency_attempts = 0
                    for target in target_order[: min(int(escape_injections), len(target_order))]:
                        target = int(target)
                        child = _make_escape_candidate(env, start, goal, K, rng, pop_inds, archive, sigma=escape_large_mut_sigma)
                        if not _cheap_candidate_precheck(env, child, parent=pop_inds[target].x, threat_mean=threat_mean, threat_std=threat_std):
                            gen_precheck_reject += 1
                            continue
                        er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
                        n_eval += 1
                        gen_eval_count += 1
                        gen_escape_eval += 1
                        emergency_attempts += 1
                        prev_best_target = pop_inds[target].er
                        _update_neighbors_for_child(child, er_child, target)
                        if er_child.feasible and ((not prev_best_target.feasible) or float(er_child.obj[1]) + 1e-9 < float(prev_best_target.obj[1])):
                            emergency_success += 1
                    last_escape_gen = gen
                    best_f2_no_improve = 0
                    if logger is not None:
                        logger.info("[escape] reason=pre_stop gen=%d success=%d/%d", int(gen), int(emergency_success), int(emergency_attempts))
                    if emergency_success > 0:
                        continue
                if disable_mtoe_stop:
                    if logger is not None:
                        logger.info(
                            "[shadow_stop] reason=mtoe gen=%d value=%.6e idx=%d nonzero=%d mean=%.6e std=%.6e p_support=%.6f mean_guard=%s tol_fun=%.6e confidence=%.6f enabled=%s basin_action=%s basin_id=%s",
                            int(gen),
                            float(mtoe_stats["mtoe"]),
                            int(mtoe_probe["idx"]),
                            int(mtoe_probe["delta_count"]),
                            float(mtoe_probe["delta_mean"]),
                            float(mtoe_stats["window_std"]),
                            float(mtoe_stats["p_support"]),
                            str(bool(mtoe_stats["mean_guard"])),
                            float(mtoe_stats["tol_fun"]),
                            float(mtoe_stats["confidence"]),
                            str(False),
                            str(basin_action),
                            str(None if basin_probe is None else basin_probe.get('basin_id')),
                        )
                    continue
                stop_reason = "mtoe"
                stop_info = dict(mtoe_stats)
                stop_info["basin_probe"] = (dict(basin_probe) if basin_probe is not None else None)
                if logger is not None:
                    logger.info(
                        "[stop] reason=mtoe gen=%d value=%.6e idx=%d nonzero=%d mean=%.6e std=%.6e p_support=%.6f mean_guard=%s tol_fun=%.6e confidence=%.6f basin_action=%s basin_id=%s",
                        int(gen),
                        float(mtoe_stats["mtoe"]),
                        int(mtoe_probe["idx"]),
                        int(mtoe_probe["delta_count"]),
                        float(mtoe_probe["delta_mean"]),
                        float(mtoe_stats["window_std"]),
                        float(mtoe_stats["p_support"]),
                        str(bool(mtoe_stats["mean_guard"])),
                        float(mtoe_stats["tol_fun"]),
                        float(mtoe_stats["confidence"]),
                        str(basin_action),
                        str(None if basin_probe is None else basin_probe.get('basin_id')),
                    )
                break

        # periodic flush to disk so logs still update in near real time without flushing every generation
        if logger is not None and ((((gen + 1) % log_flush_every) == 0) or gen == int(n_gen) - 1):
            for _h in list(logger.handlers):
                try:
                    _h.flush()
                except Exception:
                    pass


    if stop_reason == "max_gen":
        actual_gens = int(moead_max_gen)

    _close_logger_handlers(logger)

    log = {
        "n_gen": int(actual_gens),
        "configured_n_gen": int(moead_max_gen),
        "requested_n_gen": int(n_gen),
        "pop": int(pop),
        "K": int(K),
        "T": int(T),
        "n_eval": int(n_eval),
        "archive_size": len(archive.items),
        "max_gen": int(moead_max_gen),
        "moead_min_gen": int(moead_min_gen),
        "mtoe_enabled": True,
        "mtoe_mode": "best_so_far_delta",
        "mtoe_tol_fun": float(mtoe_tol_fun),
        "mtoe_confidence": float(mtoe_confidence),
        "stop_reason": str(stop_reason),
        "mtoe_window": int(mtoe_window),
        "mtoe_last": float(mtoe_hist[-1]) if len(mtoe_hist) > 0 else None,
        "mtoe_history_tail": [float(v) for v in list(mtoe_hist)],
        "mtoe_debug_tail": [dict(v) for v in list(mtoe_debug_tail)],
        "mtoe_tests_run": int(mtoe_tests_run),
        "mtoe_stop": stop_info,
        "disable_mtoe_stop": bool(disable_mtoe_stop),
        "shadow_stop_events": [dict(v) for v in shadow_stop_events],
        "first_shadow_stop_gen": (int(shadow_stop_events[0]["gen"]) if shadow_stop_events else None),
        "shadow_stop_count": int(len(shadow_stop_events)),
        "basin_shadow_enable": bool(basin_shadow_enable),
        "basin_band_count": int(basin_band_count),
        "basin_signature_samples": int(basin_signature_samples),
        "basin_stagnation_window": int(basin_stagnation_window),
        "basin_f2_tol_abs": float(basin_f2_tol_abs),
        "basin_f2_tol_rel": float(basin_f2_tol_rel),
        "basin_ref_gap_tol": float(basin_ref_gap_tol),
        "basin_min_distinct": int(basin_min_distinct),
        "basin_escape_injections": int(basin_escape_injections),
        "reference_f2": reference_f2,
        "distinct_basin_count": int(len(basin_registry)),
        "basin_debug_tail": [dict(v) for v in list(basin_debug_tail)],
        "ideal_point": z.tolist(),
        "init_profile": dict(init_profile),
        "init_s": float(init_s),
    }
    return pop_inds, archive, log
