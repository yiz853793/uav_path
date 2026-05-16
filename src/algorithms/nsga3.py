from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from ..env.grid_env import GridEnv
from ..models.evaluator import evaluate_path
from .moead import (
    Archive,
    Individual,
    crossover,
    make_initial_population,
    mutate,
    repair_light,
    uniform_weights,
)


_EPS = 1e-12


def _dominates_obj(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b + _EPS) and np.any(a < b - _EPS))


def constrained_dominates(a: Individual, b: Individual) -> bool:
    """Feasibility-first dominance used by NSGA-style constrained sorting."""
    a_feas = bool(a.er.feasible)
    b_feas = bool(b.er.feasible)
    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    if not a_feas and not b_feas:
        return float(a.er.violation) < float(b.er.violation) - _EPS
    return _dominates_obj(np.asarray(a.er.obj, dtype=np.float64), np.asarray(b.er.obj, dtype=np.float64))


def fast_non_dominated_sort(pop: Sequence[Individual]) -> List[List[int]]:
    n = len(pop)
    if n == 0:
        return []
    dominates_list: List[List[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int32)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(p + 1, n):
            if constrained_dominates(pop[p], pop[q]):
                dominates_list[p].append(q)
                dominated_count[q] += 1
            elif constrained_dominates(pop[q], pop[p]):
                dominates_list[q].append(p)
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts


def _finite_objective_matrix(pop: Sequence[Individual], indices: Sequence[int]) -> np.ndarray:
    if not indices:
        return np.zeros((0, 3), dtype=np.float64)
    mat = np.array([np.asarray(pop[i].er.obj, dtype=np.float64) for i in indices], dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] != 3:
        mat = np.reshape(mat, (len(indices), 3))
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = np.isfinite(col)
        if np.any(finite):
            hi = float(np.max(col[finite]))
            lo = float(np.min(col[finite]))
            fill = hi + max(1.0, hi - lo)
        else:
            fill = 1.0
        col[~finite] = fill
        mat[:, j] = col
    return mat


def _normalize_objectives(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return F.astype(np.float64)
    zmin = np.min(F, axis=0)
    zmax = np.max(F, axis=0)
    denom = np.where((zmax - zmin) > _EPS, zmax - zmin, 1.0)
    return (F - zmin) / denom


def _associate_to_reference_dirs(F_norm: np.ndarray, ref_dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirs = np.asarray(ref_dirs, dtype=np.float64)
    dirs = np.clip(dirs, _EPS, None)
    dirs = dirs / np.maximum(_EPS, np.linalg.norm(dirs, axis=1, keepdims=True))
    F = np.asarray(F_norm, dtype=np.float64)
    proj = F @ dirs.T
    diff = F[:, None, :] - proj[:, :, None] * dirs[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    assoc = np.argmin(dist, axis=1).astype(np.int32)
    best_dist = dist[np.arange(len(F)), assoc]
    return assoc, best_dist


def _select_last_front(
    pool: Sequence[Individual],
    selected: List[int],
    last_front: List[int],
    slots: int,
    ref_dirs: np.ndarray,
    rng: np.random.Generator,
) -> List[int]:
    if slots <= 0 or not last_front:
        return []
    if slots >= len(last_front):
        return list(last_front)

    combined = list(selected) + list(last_front)
    F = _finite_objective_matrix(pool, combined)
    F_norm = _normalize_objectives(F)
    assoc, dist = _associate_to_reference_dirs(F_norm, ref_dirs)

    n_ref = int(len(ref_dirs))
    niche_count = np.zeros(n_ref, dtype=np.int32)
    for niche in assoc[: len(selected)]:
        niche_count[int(niche)] += 1

    cand_assoc = assoc[len(selected) :]
    cand_dist = dist[len(selected) :]
    remaining = list(range(len(last_front)))
    chosen: List[int] = []

    while len(chosen) < slots and remaining:
        active_niches = sorted({int(cand_assoc[pos]) for pos in remaining})
        min_count = min(int(niche_count[n]) for n in active_niches)
        best_niches = [n for n in active_niches if int(niche_count[n]) == min_count]
        niche = int(best_niches[int(rng.integers(0, len(best_niches)))])
        niche_positions = [pos for pos in remaining if int(cand_assoc[pos]) == niche]
        if not niche_positions:
            niche_count[niche] += 1
            continue
        if niche_count[niche] == 0:
            pos = min(niche_positions, key=lambda p: float(cand_dist[p]))
        else:
            pos = niche_positions[int(rng.integers(0, len(niche_positions)))]
        chosen.append(last_front[pos])
        remaining.remove(pos)
        niche_count[niche] += 1

    return chosen


def _environmental_select(
    pool: Sequence[Individual],
    pop_size: int,
    ref_dirs: np.ndarray,
    rng: np.random.Generator,
) -> List[Individual]:
    fronts = fast_non_dominated_sort(pool)
    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
            continue
        slots = pop_size - len(selected)
        selected.extend(_select_last_front(pool, selected, list(front), slots, ref_dirs, rng))
        break
    if len(selected) < pop_size:
        leftovers = [i for i in range(len(pool)) if i not in set(selected)]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: pop_size - len(selected)])
    return [pool[i] for i in selected[:pop_size]]


def _rank_lookup(fronts: Sequence[Sequence[int]], n: int) -> np.ndarray:
    rank = np.full(n, fill_value=max(1, n), dtype=np.int32)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[int(idx)] = int(r)
    return rank


def _tournament(pop: Sequence[Individual], rank: np.ndarray, rng: np.random.Generator) -> Individual:
    i = int(rng.integers(0, len(pop)))
    j = int(rng.integers(0, len(pop)))
    a, b = pop[i], pop[j]
    if constrained_dominates(a, b):
        return a
    if constrained_dominates(b, a):
        return b
    if int(rank[i]) < int(rank[j]):
        return a
    if int(rank[j]) < int(rank[i]):
        return b
    if float(a.er.violation) < float(b.er.violation) - _EPS:
        return a
    if float(b.er.violation) < float(a.er.violation) - _EPS:
        return b
    return a if rng.random() < 0.5 else b


def nsga3(
    env: GridEnv,
    start: np.ndarray,
    goal: np.ndarray,
    n_gen: int = 120,
    pop: int = 80,
    K: int = 30,
    seed: int = 0,
    ref_dirs_count: Optional[int] = None,
    crossover_prob: float = 0.90,
    mutation_prob: float = 0.25,
    mutation_sigma: float = 2.5,
    max_turn_deg: float = 90.0,
    soft_turn_deg: float = 60.0,
    max_pitch_deg: float = 35.0,
    soft_pitch_deg: float = 25.0,
    desired_clearance_margin: float = 2.0,
    tau_soft: float = 25.0,
    archive_size: int = 0,
    archive_soft_limit: int = 320,
    archive_grid_bins: int = 0,
    archive_keep_extremes: bool = True,
    eval_sample_step: float = 0.5,
    smooth_collision_step: float = 0.5,
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
):
    """Reference-direction NSGA-III baseline for the UAV path encoding.

    Returns
    -------
    pop_inds, archive, log
        Same shape as ``moead`` so experiment code can treat it as another
        multi-objective planner.
    """
    rng = np.random.default_rng(seed)
    pop = int(max(4, pop))
    K = int(max(2, K))
    n_gen = int(max(0, n_gen))
    crossover_prob = float(np.clip(crossover_prob, 0.0, 1.0))
    mutation_prob = float(np.clip(mutation_prob, 0.0, 1.0))
    mutation_sigma = float(max(0.0, mutation_sigma))
    ref_count = int(ref_dirs_count) if ref_dirs_count is not None and int(ref_dirs_count) > 0 else pop
    ref_dirs = uniform_weights(3, ref_count, seed=seed, extreme_bias=float(weight_extreme_bias))

    eval_kwargs = dict(
        max_turn_deg=float(max_turn_deg),
        soft_turn_deg=float(min(soft_turn_deg, max_turn_deg)),
        max_pitch_deg=float(max_pitch_deg),
        soft_pitch_deg=float(min(soft_pitch_deg, max_pitch_deg)),
        desired_clearance_margin=float(max(0.0, desired_clearance_margin)),
        tau_soft=float(max(0.0, tau_soft)),
        sample_step=float(eval_sample_step),
    )

    t0 = time.perf_counter()
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
        eval_sample_step=float(eval_sample_step),
        smooth_collision_step=float(smooth_collision_step),
        max_turn_deg=float(max_turn_deg),
        soft_turn_deg=float(soft_turn_deg),
        max_pitch_deg=float(max_pitch_deg),
        soft_pitch_deg=float(soft_pitch_deg),
        desired_clearance_margin=float(desired_clearance_margin),
        tau_soft=float(tau_soft),
    )
    init_s = time.perf_counter() - t0

    effective_archive_size = int(archive_size) if int(archive_size) > 0 else int(archive_soft_limit)
    archive = Archive(
        max_size=effective_archive_size,
        grid_bins=int(archive_grid_bins),
        protect_extremes=bool(archive_keep_extremes),
        crowd_k=5,
    )
    for ind in pop_inds:
        if ind.er.feasible:
            archive.add(ind)

    n_eval = len(pop_inds)
    front0_sizes: List[int] = []
    feasible_counts: List[int] = []

    for _gen in range(1, n_gen + 1):
        fronts = fast_non_dominated_sort(pop_inds)
        rank = _rank_lookup(fronts, len(pop_inds))
        offspring: List[Individual] = []
        while len(offspring) < pop:
            p1 = _tournament(pop_inds, rank, rng)
            p2 = _tournament(pop_inds, rank, rng)
            if rng.random() < crossover_prob:
                child_x = crossover(rng, p1.x, p2.x)
            else:
                child_x = p1.x.copy()
            child_x = mutate(rng, child_x, sigma=mutation_sigma, p_mut=mutation_prob)
            child_x = repair_light(env, child_x, rng, tries=6)
            er = evaluate_path(env, child_x, **eval_kwargs)
            child = Individual(x=child_x, er=er)
            offspring.append(child)
            n_eval += 1
            if er.feasible:
                archive.add(child)

        pool = list(pop_inds) + offspring
        pop_inds = _environmental_select(pool, pop, ref_dirs, rng)
        gen_fronts = fast_non_dominated_sort(pop_inds)
        front0_sizes.append(len(gen_fronts[0]) if gen_fronts else 0)
        feasible_counts.append(sum(1 for ind in pop_inds if ind.er.feasible))

    final_fronts = fast_non_dominated_sort(pop_inds)
    log = {
        "n_gen": int(n_gen),
        "configured_n_gen": int(n_gen),
        "requested_n_gen": int(n_gen),
        "pop": int(pop),
        "K": int(K),
        "n_eval": int(n_eval),
        "ref_dirs": int(len(ref_dirs)),
        "archive_size": int(len(archive.items)),
        "stop_reason": "max_gen",
        "init_s": float(init_s),
        "init_profile": init_profile,
        "front0_size": int(len(final_fronts[0])) if final_fronts else 0,
        "front0_size_last": int(front0_sizes[-1]) if front0_sizes else (int(len(final_fronts[0])) if final_fronts else 0),
        "feasible_count": int(sum(1 for ind in pop_inds if ind.er.feasible)),
        "feasible_count_last": int(feasible_counts[-1]) if feasible_counts else int(sum(1 for ind in pop_inds if ind.er.feasible)),
    }
    return pop_inds, archive, log
