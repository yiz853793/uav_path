import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.env.grid_env import GridEnv
from src.algorithms.rrt_star import rrt_star
from src.algorithms.prm import prm
from src.algorithms.moead import moead
from src.models.evaluator import evaluate_path
from src.experiment.vis_data import build_vis_payload, save_vis_payload
from src.experiment.benchmark_shared import (
    ensure_dir,
    map_size_name,
    astar_tag_from_args,
    moead_core_tag_from_args,
    default_start_goal_for_env,
    build_moead_metrics_block,
    parse_moead_debug_log_to_rows,
    _append_rows_csv,
    mtoe_summary_row_from_metrics,
    parse_mtoe_debug_dump_to_rows,
    debug_payload_to_rows,
    write_mtoe_debug_log,
)


def meters_to_cells(env, value_m: float) -> float:
    return float(value_m) / float(env.resolution)


def resolve_terrain_files(args) -> List[str]:
    if str(getattr(args, 'glob', '')).strip():
        import glob
        return sorted(glob.glob(args.glob))
    if args.terrain_type in ('city', 'hill_city'):
        terrain_subdir = os.path.join(args.terrain_dir, f"{args.terrain_type}_{args.city_density:.2f}")
        prefix = 'city' if args.terrain_type == 'city' else 'hill_city'
        return [os.path.join(terrain_subdir, f"{prefix}_seed{seed:04d}.npz") for seed in range(args.seed_from, args.seed_to)]
    size_dir = {'small': 'S', 'medium': 'M', 'large': 'L'}[args.size]
    terrain_subdir = os.path.join(args.terrain_dir, size_dir)
    return [os.path.join(terrain_subdir, f"mountain_seed{seed:04d}.npz") for seed in range(args.seed_from, args.seed_to)]



def run_single_case(
    args,
    terrain_path: str,
    planner_seed: int,
    *,
    out_dir_override: Optional[str] = None,
    moead_debug_csv_path: str = '',
    mtoe_csv_path: str = '',
    mtoe_debug_csv_path: str = '',
) -> Dict[str, Any]:
    if not os.path.exists(terrain_path):
        raise FileNotFoundError(terrain_path)

    base = os.path.basename(terrain_path)
    terrain_seed = None
    if 'seed' in base:
        try:
            terrain_seed = int(base.split('seed')[-1].split('.')[0])
        except Exception:
            terrain_seed = None

    env, height, meta = GridEnv.load_npz(terrain_path)
    inflate_cells = int(np.ceil(args.inflate / env.resolution)) if args.inflate > 0 else 0
    if inflate_cells > 0:
        env = env.inflate_obstacles(inflate_cells)

    if isinstance(meta, dict):
        size_tag = str(meta.get('map_size', {}).get('tag', map_size_name(env.H, env.W)))
    else:
        size_tag = map_size_name(env.H, env.W)

    if out_dir_override:
        out_dir = out_dir_override
    elif str(getattr(args, 'single_case_out_dir', '')).strip():
        out_dir = str(args.single_case_out_dir).strip()
    elif terrain_seed is None:
        out_dir = os.path.join(args.out_root, size_tag, os.path.splitext(base)[0])
    else:
        out_dir = os.path.join(args.out_root, size_tag, f'seed{terrain_seed:04d}')
    ensure_dir(out_dir)

    if args.start is None or args.goal is None:
        start, goal = default_start_goal_for_env(env, height, z_offset_m=args.start_goal_z_offset)
    else:
        sx_m, sy_m = float(args.start[0]), float(args.start[1])
        gx_m, gy_m = float(args.goal[0]), float(args.goal[1])
        sx = meters_to_cells(env, sx_m)
        sy = meters_to_cells(env, sy_m)
        gx = meters_to_cells(env, gx_m)
        gy = meters_to_cells(env, gy_m)
        sx_i, sy_i = int(np.clip(round(sx), 0, env.W - 1)), int(np.clip(round(sy), 0, env.H - 1))
        gx_i, gy_i = int(np.clip(round(gx), 0, env.W - 1)), int(np.clip(round(gy), 0, env.H - 1))
        sz = float(height[sy_i, sx_i]) + args.start_goal_z_offset
        gz = float(height[gy_i, gx_i]) + args.start_goal_z_offset
        start = np.array([float(sx_i), float(sy_i), sz], dtype=np.float32)
        goal = np.array([float(gx_i), float(gy_i), gz], dtype=np.float32)

    invalid_start = False
    invalid_goal = False
    reason: List[str] = []
    sx, sy = int(round(float(start[0]))), int(round(float(start[1])))
    gx, gy = int(round(float(goal[0]))), int(round(float(goal[1])))
    if sx < 0 or sx >= env.W or sy < 0 or sy >= env.H:
        invalid_start = True
        reason.append('start_oob')
    elif float(start[2]) < float(height[sy, sx]):
        invalid_start = True
        reason.append('start_below_ground')
    if gx < 0 or gx >= env.W or gy < 0 or gy >= env.H:
        invalid_goal = True
        reason.append('goal_oob')
    elif float(goal[2]) < float(height[gy, gx]):
        invalid_goal = True
        reason.append('goal_below_ground')
    invalid_case = bool(invalid_start or invalid_goal)

    astar_tag = astar_tag_from_args(args)
    moead_core_tag = moead_core_tag_from_args(args)
    base_noext = base.replace('.npz', '')
    file_tag = f"{size_tag}{moead_core_tag}{astar_tag}"
    out_json = os.path.join(out_dir, f"metrics_{base_noext}_pseed{planner_seed:04d}_{file_tag}.json")

    if invalid_case:
        moead_metric_block, _ = build_moead_metrics_block(args=args, log=None, runtime_ms=0.0, archive_size=0)
        data = {
            'terrain_file': terrain_path,
            'terrain_seed': terrain_seed,
            'planner_seed': planner_seed,
            'inflate': args.inflate,
            'map_size': {'H': int(env.H), 'W': int(env.W), 'tag': size_tag},
            'start': start.tolist(),
            'goal': goal.tolist(),
            'invalid_case': True,
            'invalid': {'start_invalid': bool(invalid_start), 'goal_invalid': bool(invalid_goal), 'reason': reason},
            'rrt': {'iter': args.rrt_iter, 'runtime_ms': 0.0, 'found': False, 'path_len': None, 'obj': None, 'feasible': False, 'violation': None, 'detail': None},
            'prm': {'samples': args.prm_samples, 'k': args.prm_k, 'max_edge_len': args.prm_max_edge_len, 'threat_weight': args.prm_threat_weight, 'runtime_ms': 0.0, 'found': False, 'path_len': None, 'obj': None, 'feasible': False, 'violation': None, 'detail': None},
            'moead': moead_metric_block,
            'meta': meta,
        }
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {
            'terrain_base': base, 'terrain_seed': terrain_seed, 'planner_seed': planner_seed,
            'H': int(env.H), 'W': int(env.W), 'size_tag': size_tag,
            'profile': getattr(args, 'profile_resolved', getattr(args, 'profile', 'auto')),
            'invalid_case': True, 'invalid_reason': reason,
            'rrt_found': False, 'rrt_ms': 0.0, 'rrt_obj': [float('inf')]*3, 'rrt_feasible': False,
            'prm_found': False, 'prm_ms': 0.0, 'prm_obj': [float('inf')]*3, 'prm_feasible': False,
            'moead_ms': 0.0, 'moead_archive_size': 0, 'moead_stop_reason': 'not_run', 'moead_n_gen': 0, 'moead_mtoe_last': None, 'rep_objs': None,
        }

    eval_step_cells = args.moead_eval_step / env.resolution
    smooth_step_cells = args.moead_smooth_step / env.resolution
    moead_only = bool(getattr(args, 'moead_only', False))

    path_rrt = None
    rrt_ms = 0.0
    rrt_obj = [float('inf')]*3
    rrt_feasible = False
    rrt_violation = float('inf')
    rrt_detail = None
    if not moead_only:
        t0 = time.time()
        path_rrt, _ = rrt_star(env, start, goal, n_iter=args.rrt_iter, seed=planner_seed)
        rrt_ms = (time.time() - t0) * 1000.0
        if path_rrt is not None:
            rrt_er = evaluate_path(env, path_rrt, sample_step=eval_step_cells)
            rrt_obj = [float(rrt_er.obj[0]), float(rrt_er.obj[1]), float(rrt_er.obj[2])]
            rrt_feasible = bool(rrt_er.feasible)
            rrt_violation = float(rrt_er.violation)
            rrt_detail = dict(rrt_er.detail)

    path_prm = None
    prm_graph = None
    prm_ms = 0.0
    prm_obj = [float('inf')]*3
    prm_feasible = False
    prm_violation = float('inf')
    prm_detail = None
    prm_stats = {}
    if not moead_only:
        t0b = time.time()
        path_prm, prm_graph = prm(env, start, goal, n_samples=args.prm_samples, k=args.prm_k, max_edge_len=args.prm_max_edge_len, threat_weight=args.prm_threat_weight, seed=planner_seed)
        prm_ms = (time.time() - t0b) * 1000.0
        if path_prm is not None:
            prm_er = evaluate_path(env, path_prm, sample_step=eval_step_cells)
            prm_obj = [float(prm_er.obj[0]), float(prm_er.obj[1]), float(prm_er.obj[2])]
            prm_feasible = bool(prm_er.feasible)
            prm_violation = float(prm_er.violation)
            prm_detail = dict(prm_er.detail)
        prm_stats = getattr(prm_graph, 'stats', {}) if prm_graph is not None else {}

    t1 = time.time()
    moead_debug_log = os.path.join(out_dir, 'moead_debug.log') if args.moead_debug else None
    _, arch, log = moead(
        env, start, goal,
        n_gen=args.moead_max_gen,
        moead_min_gen=args.moead_min_gen,
        moead_max_gen=args.moead_max_gen,
        mtoe_tol_fun=args.mtoe_tol_fun,
        mtoe_confidence=args.mtoe_confidence,
        disable_mtoe_stop=bool(args.disable_mtoe_stop),
        basin_shadow_enable=bool(args.basin_shadow_enable),
        basin_band_count=args.basin_band_count,
        basin_signature_samples=args.basin_signature_samples,
        basin_stagnation_window=args.basin_stagnation_window,
        basin_f2_tol_abs=args.basin_f2_tol_abs,
        basin_f2_tol_rel=args.basin_f2_tol_rel,
        basin_ref_gap_tol=args.basin_ref_gap_tol,
        basin_min_distinct=args.basin_min_distinct,
        basin_escape_injections=args.basin_escape_injections,
        reference_f2=(float(rrt_obj[1]) if np.isfinite(rrt_obj[1]) else None),
        pop=args.moead_pop,
        K=args.K,
        T=args.moead_T,
        seed=planner_seed,
        eval_sample_step=eval_step_cells,
        smooth_collision_step=smooth_step_cells,
        debug_log_path=moead_debug_log,
        debug_every=args.moead_debug_every,
        debug_level=args.moead_debug_level,
        init_astar_ratio=args.init_astar_ratio,
        init_astar_threat_weight=args.init_astar_threat_weight,
        init_astar_jitter_sigma=args.init_astar_jitter_sigma,
        init_astar_max_paths=args.init_astar_max_paths,
        init_astar_penalty_step=args.init_astar_penalty_step,
        init_stratified_ratio=args.init_stratified_ratio,
        init_stratified_lateral_frac=args.init_stratified_lateral_frac,
        init_stratified_n_bands=args.init_stratified_n_bands,
        init_stratified_progress_jitter=args.init_stratified_progress_jitter,
        init_global_random_ratio=args.init_global_random_ratio,
        weight_extreme_bias=args.weight_extreme_bias,
        extreme_offspring_ratio=args.extreme_offspring_ratio,
        extreme_potential_window=args.extreme_potential_window,
        extreme_min_extra_per_obj=args.extreme_min_extra_per_obj,
        extreme_max_frac_per_obj=args.extreme_max_frac_per_obj,
        local_search_interval=args.local_search_interval,
        local_search_elite_k=args.local_search_elite_k,
        local_search_attempts_per_obj=args.local_search_attempts_per_obj,
        archive_size=args.archive_size,
        archive_soft_limit=args.archive_soft_limit,
        archive_grid_bins=args.archive_grid_bins,
        archive_keep_extremes=bool(args.archive_keep_extremes),
        active_subproblem_ratio=args.active_subproblem_ratio,
    )
    moead_ms = (time.time() - t1) * 1000.0

    reps = None
    if len(arch.items) > 0:
        objs = np.array([it.er.obj for it in arch.items], dtype=float)
        idx_f1 = int(np.argmin(objs[:, 0]))
        idx_f2 = int(np.argmin(objs[:, 1]))
        idx_f3 = int(np.argmin(objs[:, 2]))
        mn = objs.min(axis=0)
        mx = objs.max(axis=0)
        denom = np.where((mx - mn) > 1e-12, (mx - mn), 1.0)
        score = ((objs - mn) / denom) @ (np.array([1.0,1.0,1.0]) / 3.0)
        reps = {'min_f1': idx_f1, 'min_f2': idx_f2, 'min_f3': idx_f3, 'compromise': int(np.argmin(score)), 'objs': objs}

    vis_json = os.path.join(out_dir, f"visdata_{base_noext}_pseed{planner_seed:04d}_{file_tag}.json")
    title = f"Paths | {base_noext} size={size_tag} gen={args.moead_min_gen}-{args.moead_max_gen} pop={args.moead_pop} K={args.K} planner_seed={planner_seed}"
    vis_payload = build_vis_payload(
        terrain_file=terrain_path, terrain_seed=terrain_seed, planner_seed=planner_seed, size_tag=size_tag,
        inflate=args.inflate, map_hw=(env.H, env.W), start=start, goal=goal,
        path_rrt=path_rrt, path_prm=path_prm, arch=arch, reps=reps, title=title, meta=meta,
        extra={'planner_args': {'profile': getattr(args, 'profile_resolved', getattr(args, 'profile', 'auto')), 'moead_only': moead_only}, 'runtime_ms': {'rrt': float(rrt_ms), 'prm': float(prm_ms), 'moead': float(moead_ms)}}
    )
    save_vis_payload(vis_json, vis_payload)

    moead_metric_block, moead_debug_payload = build_moead_metrics_block(args=args, log=log, runtime_ms=moead_ms, archive_size=len(arch.items))
    data = {
        'terrain_file': terrain_path,
        'terrain_seed': terrain_seed,
        'planner_seed': planner_seed,
        'inflate': args.inflate,
        'map_size': {'H': int(env.H), 'W': int(env.W), 'tag': size_tag},
        'start': start.tolist(), 'goal': goal.tolist(), 'invalid_case': False,
        'invalid': {'start_invalid': False, 'goal_invalid': False, 'reason': []},
        'rrt': {'iter': args.rrt_iter, 'runtime_ms': rrt_ms, 'found': path_rrt is not None, 'path_len': env.path_length(path_rrt) if path_rrt is not None else None, 'obj': rrt_obj if path_rrt is not None else None, 'feasible': rrt_feasible if path_rrt is not None else False, 'violation': rrt_violation if path_rrt is not None else None, 'detail': rrt_detail if path_rrt is not None else None, 'skipped': moead_only},
        'prm': {'samples': args.prm_samples, 'k': args.prm_k, 'max_edge_len': args.prm_max_edge_len, 'threat_weight': args.prm_threat_weight, 'runtime_ms': prm_ms, 'found': path_prm is not None, 'path_len': env.path_length(path_prm) if path_prm is not None else None, 'obj': prm_obj if path_prm is not None else None, 'feasible': prm_feasible if path_prm is not None else False, 'violation': prm_violation if path_prm is not None else None, 'detail': prm_detail if path_prm is not None else None, 'stats': prm_stats, 'skipped': moead_only},
        'moead': moead_metric_block,
        'meta': meta,
    }

    rep_objs = None
    if reps is not None:
        def obj_of(name):
            idx = int(reps[name])
            return [float(x) for x in arch.items[idx].er.obj]
        rep_objs = {'min_f1': obj_of('min_f1'), 'min_f2': obj_of('min_f2'), 'min_f3': obj_of('min_f3'), 'compromise': obj_of('compromise')}
        data['moead']['representatives'] = rep_objs

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if args.moead_debug:
        moead_debug_csv_path = moead_debug_csv_path or os.path.join(out_dir, 'moead_debug.csv')
        mtoe_csv_path = mtoe_csv_path or os.path.join(out_dir, 'mtoe.csv')
        mtoe_debug_csv_path = mtoe_debug_csv_path or os.path.join(out_dir, 'mtoe_debug.csv')
        moead_debug_path = os.path.join(out_dir, 'moead_debug.log')
        gen_rows, mtoe_rows = parse_moead_debug_log_to_rows(moead_debug_path, terrain_base=base, terrain_seed=terrain_seed, planner_seed=planner_seed, size_tag=size_tag)
        _append_rows_csv(moead_debug_csv_path, gen_rows, preset_fieldnames=['row_type','timestamp','gen','total_s','archive','feasible','feasible_pct','eval_ms','repair_ms','smooth_ms','neigh_ms','best_feas_f1','best_feas_f2','best_feas_f3','extra_1','extra_2','extra_3','ls_1','ls_2','ls_3'])
        if not mtoe_rows:
            mtoe_rows = [mtoe_summary_row_from_metrics(moead_metric_block)]
        _append_rows_csv(mtoe_csv_path, mtoe_rows, preset_fieldnames=['row_type','timestamp','gen','value','idx','nonzero','mean','p90','std','var','p_support','mean_guard','tol_fun','confidence','z_shift_inf','window_min','window_max','reason','stop_reason'])
        mtoe_dump_path = os.path.join(out_dir, 'mtoe_debug.log')
        if not os.path.exists(mtoe_dump_path) and moead_debug_payload:
            write_mtoe_debug_log(mtoe_dump_path, terrain_file=terrain_path, terrain_seed=terrain_seed, planner_seed=planner_seed, debug_payload=moead_debug_payload)
        mtoe_debug_rows = parse_mtoe_debug_dump_to_rows(mtoe_dump_path, terrain_base=base, terrain_seed=terrain_seed, planner_seed=planner_seed, size_tag=size_tag)
        if not mtoe_debug_rows:
            mtoe_debug_rows = debug_payload_to_rows(moead_debug_payload, terrain_base=base, terrain_seed=terrain_seed, planner_seed=planner_seed, size_tag=size_tag)
        _append_rows_csv(mtoe_debug_csv_path, mtoe_debug_rows, preset_fieldnames=['row_type','timestamp','gen','last','hist_len','tests_run','value','idx','nonzero','mean','p90','prev_best','cur_raw','cur_best','z_shift_inf','z_shift_l2','reason','gamma','std','var','p_support','mean_guard','tol_fun','confidence','window_min','window_max'])

    row = {
        'terrain_base': base,
        'terrain_seed': terrain_seed,
        'planner_seed': planner_seed,
        'H': int(env.H), 'W': int(env.W), 'size_tag': size_tag,
        'profile': getattr(args, 'profile_resolved', getattr(args, 'profile', 'auto')),
        'invalid_case': False,
        'rrt_found': path_rrt is not None, 'rrt_ms': float(rrt_ms), 'rrt_obj': rrt_obj, 'rrt_feasible': bool(rrt_feasible),
        'prm_found': path_prm is not None, 'prm_ms': float(prm_ms), 'prm_obj': prm_obj, 'prm_feasible': bool(prm_feasible),
        'moead_only': moead_only,
        'moead_ms': float(moead_ms), 'moead_archive_size': int(len(arch.items)), 'moead_stop_reason': log.get('stop_reason'),
        'moead_n_gen': int(log.get('n_gen', args.moead_max_gen)), 'moead_mtoe_last': None, 'rep_objs': rep_objs,
    }
    return row
