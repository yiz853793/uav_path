# src/algorithms/moead.py

from __future__ import annotations
import os
import sys
import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from ..env.grid_env import GridEnv
from ..env.collision import segment_collision, sampled_points_array
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


def uniform_weights(m: int, n: int, seed: int = 0) -> np.ndarray:
    """
    生成 simplex 上的权重向量
    - m=3 时用规则网格 + 随机补齐（够毕设）
    """
    rng = np.random.default_rng(seed)
    if m != 3:
        w = rng.random((n, m))
        w = w / np.sum(w, axis=1, keepdims=True)
        return w.astype(np.float64)

    ws = []
    k = int(np.sqrt(n)) + 1
    for i in range(k):
        for j in range(k):
            a = i / (k - 1)
            b = j / (k - 1)
            if a + b <= 1.0:
                ws.append([a, b, 1.0 - a - b])

    rng.shuffle(ws)
    ws = ws[:n]
    if len(ws) < n:
        extra = rng.random((n - len(ws), 3))
        extra = extra / np.sum(extra, axis=1, keepdims=True)
        ws.extend(extra.tolist())

    w = np.array(ws, dtype=np.float64)
    w = np.clip(w, 1e-6, None)
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
    """
    非支配解集（简单维护 + 简单截断）
    """

    def __init__(self, max_size: int = 200):
        self.max_size = int(max_size)
        self.items: List[Individual] = []

    def add(self, ind: Individual):
        # 去重：避免目标向量完全相同导致 archive “挤满一堆重复点”
        for it in self.items:
            if np.allclose(it.er.obj, ind.er.obj, rtol=0.0, atol=1e-9):
                return
        new_items = []
        dominated = False
        for it in self.items:
            if dominates(it.er.obj, ind.er.obj):
                dominated = True
                break
            if not dominates(ind.er.obj, it.er.obj):
                new_items.append(it)
        if dominated:
            return
        new_items.append(ind)
        self.items = new_items
        if len(self.items) > self.max_size:
            self._truncate()

    def _truncate(self):
        F = np.array([it.er.obj for it in self.items], dtype=np.float64)
        mn = F.min(axis=0)
        mx = F.max(axis=0)
        denom = np.maximum(1e-9, mx - mn)
        Fn = (F - mn) / denom
        center = Fn.mean(axis=0)
        d = np.linalg.norm(Fn - center, axis=1)
        keep = np.argsort(-d)[: self.max_size]
        self.items = [self.items[i] for i in keep]


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
    eval_sample_step: float = 0.75,
    smooth_collision_step: float = 0.75,
) -> List[Individual]:
    """Initialize population.

    Strategy:
      1) Use A* to obtain one (or a few) feasible backbone paths on the grid.
      2) Create a portion of the population by jittering around the A* backbone,
         so MOEA/D starts with more feasible individuals (helps narrow valleys / saddles).
      3) Fill the rest with noisy straight-line polylines (original behavior).
    """

    rng = np.random.default_rng(seed)
    init: List[Individual] = []

    pop = int(pop)
    K = int(K)
    n_astar = int(np.clip(round(pop * float(astar_ratio)), 0, pop))

    # --- 1) Try to get multiple diverse A* path(s) as feasible backbones ---
    astar_paths: List[np.ndarray] = []
    if n_astar > 0:
        # We generate alternatives via *path-penalty re-planning*:
        #  - first run A* normally
        #  - then add a penalty on cells visited by the found path (excluding endpoints)
        #  - re-run A* with the penalty map to encourage a different corridor
        # This is a lightweight k-shortest-ish approach and works well for
        # narrow valleys/saddles where random init struggles.

        max_paths = max(1, int(astar_max_paths))

        # --- coarse-to-fine A* for large maps ---
        # A* with dict-based gscore/came_from is very slow on 1600x2000.
        # We downsample the occupancy/threat map (max-pool for obstacles) for seeding,
        # then upsample back and repair with collision-checked smoothing.
        def _downsample_env_maxpool(src_env: GridEnv, factor: int) -> GridEnv:
            f = int(max(1, factor))
            if f == 1:
                return src_env
            occ = src_env.occupancy
            thr = src_env.threat.astype(np.float32, copy=False)
            H, W = occ.shape
            Hp = ((H + f - 1) // f) * f
            Wp = ((W + f - 1) // f) * f

            if Hp != H or Wp != W:
                # pad obstacles as True to be safe (avoid creating fake corridors)
                occ_pad = np.ones((Hp, Wp), dtype=bool)
                occ_pad[:H, :W] = occ
                thr_pad = np.zeros((Hp, Wp), dtype=np.float32)
                thr_pad[:H, :W] = thr
            else:
                occ_pad = occ
                thr_pad = thr

            occ_ds = occ_pad.reshape(Hp // f, f, Wp // f, f).any(axis=(1, 3))
            thr_ds = thr_pad.reshape(Hp // f, f, Wp // f, f).mean(axis=(1, 3))
            return GridEnv(occ_ds, thr_ds, resolution=float(src_env.resolution) * float(f))

        area = int(env.H) * int(env.W)
        if area >= 1_000_000:
            factors = (4, 2, 1)
        elif area >= 300_000:
            factors = (2, 1)
        else:
            factors = (1,)

        for f in factors:
            env_astar = _downsample_env_maxpool(env, int(f)) if int(f) > 1 else env
            start_astar = start / float(f) if int(f) > 1 else start
            goal_astar = goal / float(f) if int(f) > 1 else goal

            # cap expansions by the *A* grid size
            max_exp = astar_max_expansions
            if max_exp is None:
                max_exp = min(2_000_000, max(200_000, env_astar.H * env_astar.W))

            penalty_map = np.zeros_like(env_astar.threat, dtype=np.float32)

            for k in range(max_paths):
                res = astar(
                    env_astar,
                    start_astar,
                    goal_astar,
                    threat_weight=float(astar_threat_weight),
                    penalty_map=penalty_map,
                    allow_diagonal=True,
                    max_expansions=int(max_exp),
                )
                if res.path is None or len(res.path) < 2:
                    break

                raw = res.path.astype(np.float32)
                # map back to full-res coordinates
                if int(f) > 1:
                    raw[:, 0] *= float(f)
                    raw[:, 1] *= float(f)

                raw3d = env.lift_path_to_3d(raw, start_z=float(start[2]) if len(start) >= 3 else None, goal_z=float(goal[2]) if len(goal) >= 3 else None)

                # IMPORTANT: for large maps / narrow corridors,
                # resample_polyline (equal-arc) may cut corners and turn a feasible grid path into an infeasible polyline.
                # We first shortcut-smooth with collision checks (so every segment is truly collision-free),
                # then densify while preserving vertices to exactly K points.
                collision_fn = lambda p, q: segment_collision(env, p, q, step=smooth_collision_step)
                base = _simplify_polyline_collision_aware(raw3d.copy(), K, collision_fn=collision_fn, rng=rng)
                if len(base) > K:
                    base = shortcut_smooth(base.copy(), n_try=800, rng=rng, collision_fn=collision_fn)
                    base = _simplify_polyline_collision_aware(base, K, collision_fn=collision_fn, rng=rng)
                if len(base) < K:
                    base = densify_polyline_to_K(base, K)
                base = _enforce_altitude_profile(env, base, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
                base = _repair_segment_clearance(env, base, step=0.5, clearance_margin=2.0)
                base = _enforce_altitude_profile(env, base, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
                base[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
                base[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
                er = evaluate_path(env, base, sample_step=eval_sample_step)
                if not er.feasible:
                    base = densify_polyline_to_K(raw3d, K)
                    base = _enforce_altitude_profile(env, base, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=8)
                    base = _repair_segment_clearance(env, base, step=0.5, clearance_margin=2.0)
                    base = _enforce_altitude_profile(env, base, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=8)
                    base[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
                    base[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
                    er = evaluate_path(env, base, sample_step=eval_sample_step)

                # still infeasible -> skip this backbone
                if not er.feasible:
                    # update penalty and continue trying another corridor
                    pass
                else:
                    astar_paths.append(base)

                # add penalty along found path to diversify (in the A* grid coords)
                cells = res.path.astype(np.int32)
                if len(cells) > 2:
                    cells = cells[1:-1]  # exclude start/goal
                for x, y in cells:
                    if 0 <= y < penalty_map.shape[0] and 0 <= x < penalty_map.shape[1]:
                        penalty_map[y, x] += float(astar_penalty_step)

            if len(astar_paths) > 0:
                break
# de-dup very similar backbones
        uniq: List[np.ndarray] = []
        for p in astar_paths:
            if not any(np.allclose(p, q, atol=1e-3, rtol=0.0) for q in uniq):
                uniq.append(p)
        astar_paths = uniq

    # --- 2) Build A* jittered individuals ---
    if astar_paths:
        for t in range(n_astar):
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
            x = repair_light(env, x, rng, tries=8)
            er = evaluate_path(env, x, sample_step=eval_sample_step)
            init.append(Individual(x=x, er=er))

    # --- 3) Fill the rest (noisy straight line) ---
    while len(init) < pop:
        x = np.linspace(start, goal, K).astype(np.float32)
        noise = rng.normal(0.0, 3.0, size=x.shape).astype(np.float32)
        if x.shape[1] >= 3:
            noise[:, 0:2] *= 0.65
            noise[:, 2] *= 0.25
        noise[0] = 0
        noise[-1] = 0
        x = _clip_bounds(env, x + noise)
        x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
        x = _repair_segment_clearance(env, x, step=0.5, clearance_margin=2.0)
        x = _enforce_altitude_profile(env, x, clearance_margin=2.0, max_pitch_deg=35.0, n_pass=6)
        x[0] = np.maximum(start, env.clamp_point(start, clearance=env.min_clearance + 2.0))
        x[-1] = np.maximum(goal, env.clamp_point(goal, clearance=env.min_clearance + 2.0))
        x = repair_light(env, x, rng, tries=6)
        er = evaluate_path(env, x, sample_step=eval_sample_step)
        init.append(Individual(x=x, er=er))

    return init


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
    """
    y = _clip_bounds(env, x)
    D = y.shape[1]
    # only fix points that are clearly invalid
    for i in range(1, len(y) - 1):
        if D >= 3:
            bad = not env.is_free_point(y[i], clearance=env.min_clearance + 1.5)
        else:
            bad = env.is_occupied(int(round(y[i, 0])), int(round(y[i, 1])))
        if not bad:
            continue
        for _ in range(int(tries)):
            cand = y[i].copy()
            cand[:2] += rng.normal(0.0, 1.5, size=(2,)).astype(np.float32)
            if D >= 3:
                base_z = env.min_safe_altitude_at(float(cand[0]), float(cand[1]), clearance=env.min_clearance + 1.5)
                cand[2] = max(base_z, float(cand[2]) + float(rng.normal(1.5, 1.0)))
            cand = _clip_bounds(env, cand[None, :])[0]
            if D >= 3:
                good = env.is_free_point(cand, clearance=env.min_clearance + 1.5)
            else:
                good = not env.is_occupied(int(round(cand[0])), int(round(cand[1])))
            if good:
                y[i] = cand
                break
    if D >= 3:
        y = _repair_segment_clearance(env, y, step=0.75, clearance_margin=1.5)
        y = _enforce_altitude_profile(env, y, clearance_margin=1.5, max_pitch_deg=35.0, n_pass=3)
    return y.astype(np.float32)

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
    archive_size: int = 200,
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
    M = 3
    W = uniform_weights(M, pop, seed=seed)
    B = build_neighbors(W, T=T)

    t_init0 = time.perf_counter()
    pop_inds = make_initial_population(
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
        eval_sample_step=eval_sample_step,
        smooth_collision_step=smooth_collision_step,
    )
    init_s = time.perf_counter() - t_init0
    archive = Archive(max_size=archive_size)

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

    if logger is not None:
        st = _pop_stats(pop_inds)
        logger.info(
            "[start] env(H=%d,W=%d) seed=%d n_gen=%d pop=%d K=%d T=%d max_turn_deg=%.1f "
            "eval_step=%.3f smooth_step=%.3f init_s=%.3f init_feasible=%d/%d (%.1f%%) "
            "init_min_viol=%.3f init_mean_viol=%.3f init_archive=%d "
            "A* seeding: ratio=%.3f max_paths=%d penalty_step=%.3f threat_w=%.3f jitter=%.3f",
            int(env.H),
            int(env.W),
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
        )

    for gen in range(int(n_gen)):
        gen_t0 = time.perf_counter()
        gen_eval_s = 0.0
        gen_smooth_s = 0.0
        gen_neighbor_s = 0.0
        gen_repair_s = 0.0
        gen_n_smooth_in = 0
        gen_n_smooth_out = 0
        gen_coll_calls0 = coll_calls
        gen_coll_time0 = coll_time_s

        for i in range(pop):
            # 从邻域选择两个父代
            nb = B[i]
            pidx = rng.choice(nb, size=2, replace=False)
            p1 = pop_inds[int(pidx[0])].x
            p2 = pop_inds[int(pidx[1])].x

            child = crossover(rng, p1, p2)
            child = mutate(rng, child, sigma=2.5, p_mut=0.25)
            t_r0 = time.perf_counter()
            child = repair_light(env, child, rng, tries=6)
            gen_repair_s += (time.perf_counter() - t_r0)

            # 可选：轻量捷径平滑（提升质量，但会增加评估耗时）
            do_smooth = (smooth_tries > 0) and ((i + gen) % 2 == 0)
            if do_smooth:
                gen_n_smooth_in += int(len(child))
                t_s0 = time.perf_counter()
                child = shortcut_smooth(child, n_try=smooth_tries, rng=rng, collision_fn=collision_fn)
                gen_smooth_s += (time.perf_counter() - t_s0)
                gen_n_smooth_out += int(len(child))

                # 平滑后点数可能变少，补回固定 K：保留拐点的按段插点，避免切角导致碰撞
                child = densify_polyline_to_K(child, K)

            child[0] = start
            child[-1] = goal

            t_e0 = time.perf_counter()
            er_child = evaluate_path(env, child, max_turn_deg=max_turn_deg, sample_step=eval_sample_step)
            gen_eval_s += (time.perf_counter() - t_e0)
            n_eval += 1

            # 更新 ideal point（只对目标值取 min）
            # 仅用可行解更新 ideal point；不可行解只用于约束比较，不影响 z*
            if er_child.feasible:
                z = np.minimum(z, er_child.obj)

            # 更新邻域解：Deb 可行性规则 + 可行解上的 Tchebycheff 标量化
            t_n0 = time.perf_counter()
            for j in nb:
                jj = int(j)
                cur = pop_inds[jj]

                # 1) 可行性优先
                if er_child.feasible and (not cur.er.feasible):
                    pop_inds[jj] = Individual(x=child, er=er_child)
                    continue
                if (not er_child.feasible) and cur.er.feasible:
                    continue

                # 2) 都不可行：违反程度更小者更优
                if (not er_child.feasible) and (not cur.er.feasible):
                    if er_child.violation < cur.er.violation:
                        pop_inds[jj] = Individual(x=child, er=er_child)
                    continue

                # 3) 都可行：按 Tchebycheff scalar 比较
                g_child = tchebycheff(er_child.obj, W[jj], z)
                g_cur = tchebycheff(cur.er.obj, W[jj], z)
                if g_child <= g_cur:
                    pop_inds[jj] = Individual(x=child, er=er_child)

            gen_neighbor_s += (time.perf_counter() - t_n0)

            if er_child.feasible:
                archive.add(Individual(x=child, er=er_child))

        # --- per-generation debug ---
        if logger is not None and ((gen % debug_every) == 0 or gen == int(n_gen) - 1):
            st = _pop_stats(pop_inds)
            gen_total_s = time.perf_counter() - gen_t0
            n_eval_gen = int(pop)
            avg_eval_ms = 1000.0 * gen_eval_s / max(1, n_eval_gen)
            avg_repair_ms = 1000.0 * gen_repair_s / max(1, n_eval_gen)
            avg_smooth_ms = 1000.0 * gen_smooth_s / max(1, n_eval_gen)
            avg_neighbor_ms = 1000.0 * gen_neighbor_s / max(1, n_eval_gen)
            best_obj = st["best_obj_feasible"]

            if debug_level <= 1:
                logger.info(
                    "[gen=%d] total=%.3fs archive=%d feasible=%d/%d (%.1f%%) min_viol=%.3f mean_viol=%.3f best_feas=%s",
                    int(gen),
                    float(gen_total_s),
                    int(len(archive.items)),
                    int(st["n_feasible"]),
                    int(st["n"]),
                    float(100.0 * st["feasible_ratio"]),
                    float(st["min_violation"]),
                    float(st["mean_violation"]),
                    str(best_obj),
                )
            else:
                # Level>=2: add timing breakdown
                msg = (
                    "[gen=%d] total=%.3fs archive=%d feasible=%d/%d (%.1f%%) "
                    "eval=%.2fms repair=%.2fms smooth=%.2fms neigh=%.2fms best_feas=%s"
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
                        float(avg_repair_ms),
                        float(avg_smooth_ms),
                        float(avg_neighbor_ms),
                        str(best_obj),
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
                        float(avg_repair_ms),
                        float(avg_smooth_ms),
                        float(avg_neighbor_ms),
                        str(best_obj),
                    )

        # force flush to disk so logs update in real time
        if logger is not None:
            for _h in list(logger.handlers):
                try:
                    _h.flush()
                except Exception:
                    pass


    log = {
        "n_gen": int(n_gen),
        "pop": int(pop),
        "K": int(K),
        "T": int(T),
        "n_eval": int(n_eval),
        "archive_size": len(archive.items),
        "ideal_point": z.tolist(),
    }
    return pop_inds, archive, log
