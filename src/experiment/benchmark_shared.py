import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from src.env.grid_env import GridEnv
from src.experiment.common_io import append_rows_csv as _append_rows_csv
from src.experiment.common_io import safe_float as _safe_float
from src.experiment.common_io import safe_int as _safe_int


RUN_PROFILES = {
    'quick': {
        'inflate': 1.0,
        'rrt_iter': 4000,
        'prm_samples': 4000,
        'prm_k': 24,
        'prm_max_edge_len': 120.0,
        'prm_threat_weight': 0.0,
        'moead_min_gen': 40,
        'moead_max_gen': 300,
        'moead_pop': 80,
        'moead_T': 10,
        'active_subproblem_ratio': 0.75,
        'archive_size': 0,
        'archive_soft_limit': 240,
        'utility_update_interval': 4,
        'log_flush_every': 20,
    },
    'balanced': {
        'inflate': 1.0,
        'rrt_iter': 6000,
        'prm_samples': 12000,
        'prm_k': 48,
        'prm_max_edge_len': 250.0,
        'prm_threat_weight': 0.0,
        'moead_min_gen': 60,
        'moead_max_gen': 700,
        'moead_pop': 128,
        'moead_T': 14,
        'active_subproblem_ratio': 0.80,
        'archive_size': 0,
        'archive_soft_limit': 320,
        'utility_update_interval': 3,
        'log_flush_every': 10,
    },
    'quality': {
        'inflate': 1.0,
        'rrt_iter': 8000,
        'prm_samples': 16000,
        'prm_k': 64,
        'prm_max_edge_len': 280.0,
        'prm_threat_weight': 0.0,
        'moead_min_gen': 80,
        'moead_max_gen': 1600,
        'moead_pop': 160,
        'moead_T': 16,
        'active_subproblem_ratio': 0.90,
        'archive_size': 0,
        'archive_soft_limit': 400,
        'utility_update_interval': 3,
        'log_flush_every': 10,
    },
}

PROFILE_OPTION_NAMES = {
    'inflate': ('--inflate', '-i'),
    'rrt_iter': ('--rrt_iter', '--rrt'),
    'prm_samples': ('--prm_samples', '--prm_n'),
    'prm_k': ('--prm_k',),
    'prm_max_edge_len': ('--prm_max_edge_len', '--prm_edge'),
    'prm_threat_weight': ('--prm_threat_weight',),
    'moead_min_gen': ('--moead_min_gen', '--gmin'),
    'moead_max_gen': ('--moead_max_gen', '--gmax'),
    'moead_pop': ('--moead_pop', '--pop'),
    'moead_T': ('--moead_T', '-T'),
    'active_subproblem_ratio': ('--active_subproblem_ratio', '--active_ratio'),
    'archive_size': ('--archive_size',),
    'archive_soft_limit': ('--archive_soft_limit', '--arch_soft'),
    'utility_update_interval': ('--utility_update_interval',),
    'log_flush_every': ('--log_flush_every',),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def meters_to_cells(env, value_m: float) -> float:
    return float(value_m) / float(env.resolution)


def _cli_option_used(argv: list[str], *option_names: str) -> bool:
    for token in argv:
        for opt in option_names:
            if token == opt or token.startswith(opt + '='):
                return True
    return False


def _resolve_profile_name(args) -> str:
    requested = str(getattr(args, 'profile', 'auto') or 'auto').strip().lower()
    if requested != 'auto':
        return requested
    terrain_type = str(getattr(args, 'terrain_type', 'mountain') or 'mountain').strip().lower()
    size = str(getattr(args, 'size', 'small') or 'small').strip().lower()
    if terrain_type in ('city', 'hill_city') or size in ('medium', 'large'):
        return 'balanced'
    return 'quick'


def apply_run_profile(args, argv: list[str]) -> None:
    resolved = _resolve_profile_name(args)
    if resolved not in RUN_PROFILES:
        raise ValueError(f'unknown run profile: {resolved}')
    args.profile_resolved = resolved
    for key, value in RUN_PROFILES[resolved].items():
        opt_names = PROFILE_OPTION_NAMES.get(key, (f'--{key}',))
        if not _cli_option_used(argv, *opt_names):
            setattr(args, key, value)


def parse_moead_debug_log_to_rows(log_path: str, *, terrain_base=None, terrain_seed=None, planner_seed=None, size_tag=None):
    if not log_path or not os.path.exists(log_path):
        return [], []
    import re
    common = {}
    rows_gen, rows_mtoe = [], []
    pat_start = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[start\].*?env\(H=(?P<H>\d+),W=(?P<W>\d+)\).*?seed=(?P<seed>\d+).*?n_gen=(?P<n_gen>\d+).*?pop=(?P<pop>\d+).*?K=(?P<K>\d+).*?T=(?P<T>\d+).*$')
    pat_mtoe = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[gen=(?P<gen>\d+)\]\[mtoe\].*?value=(?P<value>[-+eE0-9.]+).*?idx=(?P<idx>\d+).*?nonzero=(?P<nonzero>\d+).*?mean=(?P<mean>[-+eE0-9.]+).*?p90=(?P<p90>[-+eE0-9.]+).*?std=(?P<std>[-+eE0-9.]+).*?var=(?P<var>[-+eE0-9.]+).*?p_support=(?P<p_support>[-+eE0-9.]+).*?mean_guard=(?P<mean_guard>\S+).*?tol_fun=(?P<tol_fun>[-+eE0-9.]+).*?confidence=(?P<confidence>[-+eE0-9.]+).*?z_shift_inf=(?P<z_shift_inf>[-+eE0-9.]+).*?window=\[(?P<window_min>[-+eE0-9.]+), (?P<window_max>[-+eE0-9.]+)\]$')
    pat_stop = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[stop\].*?(?:reason=)?(?P<reason>[^,]+).*$')
    pat_gen_head = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[gen=(?P<gen>\d+)\].*?total=(?P<total>[0-9.]+)s.*?archive=(?P<archive>\d+).*?feasible=(?P<feas>\d+)\/(?P<pop>\d+)')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            m = pat_start.match(s)
            if m:
                rows_gen.append(dict(common, row_type='start', timestamp=m.group('ts')))
                continue
            m = pat_mtoe.match(s)
            if m:
                rows_mtoe.append(dict(common, row_type='mtoe', timestamp=m.group('ts'), gen=_safe_int(m.group('gen')), value=_safe_float(m.group('value')), idx=_safe_int(m.group('idx')), nonzero=_safe_int(m.group('nonzero')), mean=_safe_float(m.group('mean')), p90=_safe_float(m.group('p90')), std=_safe_float(m.group('std')), var=_safe_float(m.group('var')), p_support=_safe_float(m.group('p_support')), mean_guard=m.group('mean_guard'), tol_fun=_safe_float(m.group('tol_fun')), confidence=_safe_float(m.group('confidence')), z_shift_inf=_safe_float(m.group('z_shift_inf')), window_min=_safe_float(m.group('window_min')), window_max=_safe_float(m.group('window_max'))))
                continue
            m = pat_gen_head.match(s)
            if m:
                row = dict(common, row_type='gen', timestamp=m.group('ts'), gen=_safe_int(m.group('gen')), total_s=_safe_float(m.group('total')), archive=_safe_int(m.group('archive')), feasible=_safe_int(m.group('feas')), pop=_safe_int(m.group('pop')))
                kv_patterns = {'eval_ms': r'eval=([0-9.]+)ms', 'eval_n': r'eval=[0-9.]+ms\(n=(\d+)\)', 'repair_ms': r'repair=([0-9.]+)ms', 'smooth_ms': r'smooth=([0-9.]+)ms', 'neigh_ms': r'neigh=([0-9.]+)ms', 'pre_reject': r'pre_reject=(\d+)'}
                for key, pat in kv_patterns.items():
                    mm = re.search(pat, s)
                    row[key] = _safe_float(mm.group(1)) if (mm and key.endswith('_ms')) else (_safe_int(mm.group(1)) if mm else None)
                mm = re.search(r'eval_split=(\d+)\/(\d+)\/(\d+)\/(\d+)', s)
                if mm:
                    row['eval_split_regular'] = _safe_int(mm.group(1))
                    row['eval_split_extra'] = _safe_int(mm.group(2))
                    row['eval_split_ls'] = _safe_int(mm.group(3))
                    row['eval_split_escape'] = _safe_int(mm.group(4))
                if row['feasible'] is not None and row['pop']:
                    row['feasible_pct'] = 100.0 * row['feasible'] / row['pop']
                mm = re.search(r'best_feas=\[([^\]]+)\]', s)
                if mm:
                    best = [x.strip() for x in mm.group(1).split(',') if x.strip()]
                    for i in range(min(3, len(best))):
                        row[f'best_feas_f{i+1}'] = _safe_float(best[i])
                mm = re.search(r'extra=\[([^\]]+)\]', s)
                if mm:
                    extra = [x.strip() for x in mm.group(1).split(',') if x.strip()]
                    for i in range(min(3, len(extra))):
                        row[f'extra_{i+1}'] = _safe_int(extra[i])
                mm = re.search(r'ls=\[([^\]]+)\]', s)
                if mm:
                    ls = [x.strip() for x in mm.group(1).split(',') if x.strip()]
                    for i in range(min(3, len(ls))):
                        row[f'ls_{i+1}'] = _safe_int(ls[i])
                rows_gen.append(row)
                continue
            m = pat_stop.match(s)
            if m:
                rows_mtoe.append(dict(common, row_type='stop', timestamp=m.group('ts'), reason=(m.group('reason') or '').strip()))
    return rows_gen, rows_mtoe


def mtoe_summary_row_from_metrics(moead_metric_block: dict | None, *, terrain_base=None, terrain_seed=None, planner_seed=None, size_tag=None):
    mo = moead_metric_block or {}
    return {'row_type': 'summary', 'stop_reason': mo.get('stop_reason')}


def parse_mtoe_debug_dump_to_rows(log_path: str, *, terrain_base=None, terrain_seed=None, planner_seed=None, size_tag=None):
    if not log_path or not os.path.exists(log_path):
        return []
    import re
    rows = []
    common = {}
    pat_summary = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[summary\]\[mtoe_debug\].*?last=(?P<last>\S+).*?hist_len=(?P<hist_len>\d+).*?tests_run=(?P<tests_run>\S+).*$')
    pat_gen = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[gen=(?P<gen>\d+)\]\[mtoe_debug\].*?value=(?P<value>\S+).*?idx=(?P<idx>\S+).*?nonzero=(?P<nonzero>\S+).*?mean=(?P<mean>\S+).*?p90=(?P<p90>\S+).*?prev_best=(?P<prev_best>\S+).*?cur_raw=(?P<cur_raw>\S+).*?cur_best=(?P<cur_best>\S+).*?z_shift_inf=(?P<z_shift_inf>\S+).*?z_shift_l2=(?P<z_shift_l2>\S+).*$')
    pat_stop = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[stop\]\[mtoe_debug\].*?reason=(?P<reason>\S+).*?gamma=(?P<gamma>\S+).*?value=(?P<value>\S+).*?idx=(?P<idx>\S+).*?nonzero=(?P<nonzero>\S+).*?mean=(?P<mean>\S+).*?p90=(?P<p90>\S+).*?std=(?P<std>\S+).*?var=(?P<var>\S+).*?p_support=(?P<p_support>\S+).*?mean_guard=(?P<mean_guard>\S+).*?tol_fun=(?P<tol_fun>\S+).*?confidence=(?P<confidence>\S+).*?window=\[(?P<window_min>\S+),(?P<window_max>\S+)\].*$')
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            m = pat_summary.match(s)
            if m:
                rows.append(dict(common, row_type='summary', timestamp=m.group('ts'), last=_safe_float(m.group('last')), hist_len=_safe_int(m.group('hist_len')), tests_run=_safe_int(m.group('tests_run'))))
                continue
            m = pat_gen.match(s)
            if m:
                rows.append(dict(common, row_type='gen', timestamp=m.group('ts'), gen=_safe_int(m.group('gen')), value=_safe_float(m.group('value')), idx=_safe_int(m.group('idx')), nonzero=_safe_int(m.group('nonzero')), mean=_safe_float(m.group('mean')), p90=_safe_float(m.group('p90')), prev_best=_safe_float(m.group('prev_best')), cur_raw=_safe_float(m.group('cur_raw')), cur_best=_safe_float(m.group('cur_best')), z_shift_inf=_safe_float(m.group('z_shift_inf')), z_shift_l2=_safe_float(m.group('z_shift_l2'))))
                continue
            m = pat_stop.match(s)
            if m:
                rows.append(dict(common, row_type='stop', timestamp=m.group('ts'), reason=(m.group('reason') or '').strip(), gamma=_safe_int(m.group('gamma')), value=_safe_float(m.group('value')), idx=_safe_int(m.group('idx')), nonzero=_safe_int(m.group('nonzero')), mean=_safe_float(m.group('mean')), p90=_safe_float(m.group('p90')), std=_safe_float(m.group('std')), var=_safe_float(m.group('var')), p_support=_safe_float(m.group('p_support')), mean_guard=m.group('mean_guard'), tol_fun=_safe_float(m.group('tol_fun')), confidence=_safe_float(m.group('confidence')), window_min=_safe_float(m.group('window_min')), window_max=_safe_float(m.group('window_max'))))
                continue
            rows.append(dict(common, row_type='raw', value=s))
    return rows


def debug_payload_to_rows(debug_payload, *, terrain_base=None, terrain_seed=None, planner_seed=None, size_tag=None):
    if not isinstance(debug_payload, dict) or not debug_payload:
        return []
    rows = []
    common = {}
    for key, value in debug_payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            rows.append(dict(common, row_type='payload_scalar', section=key, value=value))
        else:
            rows.append(dict(common, row_type='payload_json', section=key, value=json.dumps(value, ensure_ascii=False)))
    return rows


def map_size_name(H: int, W: int) -> str:
    if (H, W) == (160, 200):
        return 'S'
    if (H, W) == (160 * 3, 200 * 3):
        return 'M'
    if (H, W) == (160 * 10, 200 * 10):
        return 'L'
    return f'H{H}W{W}'


def astar_tag_from_args(args) -> str:
    return f'_astarR{args.init_astar_ratio}_M{args.init_astar_max_paths}_P{args.init_astar_penalty_step}_TW{args.init_astar_threat_weight}_J{args.init_astar_jitter_sigma}'


def moead_core_tag_from_args(args) -> str:
    return f'_g{args.moead_min_gen}-{args.moead_max_gen}_pop{args.moead_pop}_K{args.K}_T{args.moead_T}'


def default_start_goal_for_env(env: GridEnv, height: np.ndarray, z_offset_m: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    W, H = env.W, env.H
    res = float(env.resolution)
    sx_m, sy_m = 25.0, 25.0
    gx_m, gy_m = float(max(25.0 + 50.0, (W - 60) * res)), float(max(25.0 + 50.0, (H - 70) * res))
    sx = meters_to_cells(env, sx_m)
    sy = meters_to_cells(env, sy_m)
    gx = meters_to_cells(env, gx_m)
    gy = meters_to_cells(env, gy_m)
    sx_i, sy_i = int(np.clip(round(sx), 0, W - 1)), int(np.clip(round(sy), 0, H - 1))
    gx_i, gy_i = int(np.clip(round(gx), 0, W - 1)), int(np.clip(round(gy), 0, H - 1))
    sz = float(height[sy_i, sx_i]) + float(z_offset_m)
    gz = float(height[gy_i, gx_i]) + float(z_offset_m)
    return np.array([float(sx_i), float(sy_i), sz], dtype=np.float32), np.array([float(gx_i), float(gy_i), gz], dtype=np.float32)


def _compact_mtoe_stop_info(stop_info):
    if not isinstance(stop_info, dict):
        return None
    keep = ['gamma', 'variance', 'window_std', 'stat', 'critical_stat', 'p_support', 'tol_fun', 'confidence', 'mean_guard', 'legacy_stop_without_mean_guard', 'mtoe', 'window_min', 'window_max', 'window_mean', 'probe', 'basin_probe']
    return {k: stop_info.get(k) for k in keep if k in stop_info}


def split_moead_log(log):
    log = log if isinstance(log, dict) else {}
    if not log:
        return {}, {}
    metric_keys = {'n_gen', 'configured_n_gen', 'requested_n_gen', 'pop', 'K', 'T', 'n_eval', 'archive_size', 'max_gen', 'moead_min_gen', 'mtoe_enabled', 'mtoe_mode', 'mtoe_tol_fun', 'mtoe_confidence', 'stop_reason', 'mtoe_window', 'disable_mtoe_stop', 'mtoe_stop', 'mtoe_last', 'mtoe_tests_run', 'mtoe_history_tail', 'mtoe_debug_tail', 'first_shadow_stop_gen', 'shadow_stop_count', 'basin_shadow_enable', 'basin_band_count', 'basin_signature_samples', 'basin_stagnation_window', 'basin_f2_tol_abs', 'basin_f2_tol_rel', 'basin_ref_gap_tol', 'basin_min_distinct', 'basin_escape_injections', 'reference_f2', 'distinct_basin_count', 'ideal_point'}
    metric_log = {k: log[k] for k in metric_keys if k in log}
    debug_log = {k: v for k, v in log.items() if k not in metric_keys}
    return metric_log, debug_log


def write_mtoe_debug_log(log_path: str, *, terrain_file: str = None, terrain_seed: int = None, planner_seed: int = None, debug_payload=None):
    if not log_path or not debug_payload:
        return
    def _fmt_float(x):
        return 'None' if x is None else f'{float(x):.6e}'
    lines = []
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f'{ts} INFO [start][mtoe_debug] terrain_file={terrain_file} terrain_seed={terrain_seed} planner_seed={planner_seed}')
    if isinstance(debug_payload, dict):
        hist = debug_payload.get('mtoe_history_tail') or []
        tests_run = debug_payload.get('mtoe_tests_run')
        last_val = debug_payload.get('mtoe_last')
        lines.append(f'{ts} INFO [summary][mtoe_debug] last={_fmt_float(last_val)} hist_len={len(hist)} tests_run={tests_run}')
        for item in debug_payload.get('mtoe_debug_tail') or []:
            if not isinstance(item, dict):
                continue
            gen = item.get('gen')
            lines.append(f"{ts} INFO [gen={gen}][mtoe_debug] value={_fmt_float(item.get('value'))} idx={item.get('idx')} nonzero={item.get('delta_count')} mean={_fmt_float(item.get('delta_mean'))} p90={_fmt_float(item.get('delta_p90'))} prev_best={_fmt_float(item.get('prev_best'))} cur_raw={_fmt_float(item.get('cur_raw'))} cur_best={_fmt_float(item.get('cur_best'))} z_shift_inf={_fmt_float(item.get('z_delta_inf'))} z_shift_l2={_fmt_float(item.get('z_delta_l2'))}")
        stop = debug_payload.get('mtoe_stop')
        if isinstance(stop, dict):
            probe = stop.get('probe') if isinstance(stop.get('probe'), dict) else {}
            basin_probe = stop.get('basin_probe') if isinstance(stop.get('basin_probe'), dict) else {}
            lines.append(f"{ts} INFO [stop][mtoe_debug] reason=mtoe gamma={stop.get('gamma')} value={_fmt_float(stop.get('mtoe'))} idx={probe.get('idx')} nonzero={probe.get('delta_count')} mean={_fmt_float(probe.get('delta_mean'))} p90={_fmt_float(probe.get('delta_p90'))} std={_fmt_float(stop.get('window_std'))} var={_fmt_float(stop.get('variance'))} p_support={_fmt_float(stop.get('p_support'))} mean_guard={stop.get('mean_guard')} tol_fun={_fmt_float(stop.get('tol_fun'))} confidence={_fmt_float(stop.get('confidence'))} window=[{_fmt_float(stop.get('window_min'))},{_fmt_float(stop.get('window_max'))}] basin_action={basin_probe.get('action')} basin_id={basin_probe.get('basin_id')}")
        shadow_events = debug_payload.get('shadow_stop_events') or []
        if shadow_events:
            first_gen = shadow_events[0].get('gen')
            gens = ','.join(str(int(ev.get('gen'))) for ev in shadow_events if ev.get('gen') is not None)
            lines.append(f'{ts} INFO [shadow_summary][mtoe_debug] count={len(shadow_events)} first_gen={first_gen} gens={gens}')
            for ev in shadow_events[-25:]:
                basin = ev.get('basin') if isinstance(ev.get('basin'), dict) else {}
                lines.append(f"{ts} INFO [shadow_stop][mtoe_debug] reason={ev.get('reason')} gen={ev.get('gen')} value={_fmt_float(ev.get('mtoe'))} idx={ev.get('idx')} nonzero={ev.get('delta_count')} mean={_fmt_float(ev.get('delta_mean'))} std={_fmt_float(ev.get('window_std'))} p_support={_fmt_float(ev.get('p_support'))} mean_guard={ev.get('mean_guard')} tol_fun={_fmt_float(ev.get('tol_fun'))} confidence={_fmt_float(ev.get('confidence'))} enabled={ev.get('enabled')} basin_action={basin.get('action')} basin_id={basin.get('basin_id')} ref_gap={_fmt_float(basin.get('reference_gap'))}")
        for item in debug_payload.get('basin_debug_tail') or []:
            if not isinstance(item, dict):
                continue
            lines.append(f"{ts} INFO [gen={item.get('gen')}][basin_debug] basin_id={item.get('basin_id')} entered={item.get('entered_new_basin')} action={item.get('action')} distinct={item.get('distinct_basins_seen')} basin_best_f2={_fmt_float(item.get('basin_best_f2'))} current_best_f2={_fmt_float(item.get('current_best_f2'))} span={_fmt_float(item.get('basin_span'))} plateau={item.get('basin_plateau')} ref_f2={_fmt_float(item.get('reference_f2'))} ref_gap={_fmt_float(item.get('reference_gap'))} sig={item.get('signature')}")
        init_profile = debug_payload.get('init_profile')
        if isinstance(init_profile, dict) and init_profile:
            compact = ' '.join(f'{k}={v}' for k, v in sorted(init_profile.items()))
            lines.append(f'{ts} INFO [init][mtoe_debug] {compact}')
        if 'init_s' in debug_payload:
            lines.append(f"{ts} INFO [init_ratio][mtoe_debug] init_s={_fmt_float(debug_payload.get('init_s'))}")
    else:
        lines.append(f'{ts} INFO [payload][mtoe_debug] value={str(debug_payload)}')
    lines.append(f'{ts} INFO [end][mtoe_debug]')
    Path(log_path).write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')


def build_moead_metrics_block(args, log, runtime_ms: float, archive_size: int):
    metric_log, debug_log = split_moead_log(log)
    stop_reason = metric_log.get('stop_reason')
    if stop_reason is None:
        stop_reason = 'not_run' if not metric_log and float(runtime_ms) <= 0.0 else None
    block = {
        'n_gen': int(metric_log.get('n_gen', args.moead_max_gen)),
        'configured_n_gen': int(metric_log.get('configured_n_gen', args.moead_max_gen)),
        'requested_n_gen': int(metric_log.get('requested_n_gen', args.moead_max_gen)),
        'moead_min_gen': int(args.moead_min_gen),
        'moead_max_gen': int(args.moead_max_gen),
        'stop_reason': stop_reason,
        'mtoe_tol_fun': float(args.mtoe_tol_fun),
        'mtoe_confidence': float(args.mtoe_confidence),
        'mtoe': {
            'enabled': True,
            'mode': metric_log.get('mtoe_mode', 'best_so_far_delta'),
            'window': int(metric_log.get('mtoe_window', 10)),
            'tol_fun': float(args.mtoe_tol_fun),
            'confidence': float(args.mtoe_confidence),
            'disable_stop': bool(metric_log.get('disable_mtoe_stop', False)),
            'first_shadow_stop_gen': metric_log.get('first_shadow_stop_gen'),
            'shadow_stop_count': metric_log.get('shadow_stop_count'),
        },
        'basin_shadow': {
            'enabled': bool(metric_log.get('basin_shadow_enable', False)),
            'band_count': metric_log.get('basin_band_count'),
            'signature_samples': metric_log.get('basin_signature_samples'),
            'stagnation_window': metric_log.get('basin_stagnation_window'),
            'f2_tol_abs': metric_log.get('basin_f2_tol_abs'),
            'f2_tol_rel': metric_log.get('basin_f2_tol_rel'),
            'ref_gap_tol': metric_log.get('basin_ref_gap_tol'),
            'min_distinct': metric_log.get('basin_min_distinct'),
            'escape_injections': metric_log.get('basin_escape_injections'),
            'reference_f2': metric_log.get('reference_f2'),
            'distinct_basin_count': metric_log.get('distinct_basin_count'),
        },
        'pop': int(args.moead_pop),
        'K': int(args.K),
        'T': int(args.moead_T),
        'init_astar_ratio': float(args.init_astar_ratio),
        'init_astar_threat_weight': float(args.init_astar_threat_weight),
        'init_astar_jitter_sigma': float(args.init_astar_jitter_sigma),
        'init_astar_max_paths': int(args.init_astar_max_paths),
        'init_astar_penalty_step': float(args.init_astar_penalty_step),
        'init_stratified_ratio': float(args.init_stratified_ratio),
        'init_stratified_lateral_frac': float(args.init_stratified_lateral_frac),
        'init_stratified_n_bands': int(args.init_stratified_n_bands),
        'init_stratified_progress_jitter': float(args.init_stratified_progress_jitter),
        'init_global_random_ratio': float(args.init_global_random_ratio),
        'weight_extreme_bias': float(args.weight_extreme_bias),
        'utility_update_interval': int(args.utility_update_interval),
        'utility_use_archive_density': int(args.utility_use_archive_density),
        'log_flush_every': int(args.log_flush_every),
        'runtime_ms': float(runtime_ms),
        'archive_size': int(archive_size),
    }
    stop = _compact_mtoe_stop_info(metric_log.get('mtoe_stop'))
    if stop is not None:
        block['mtoe']['last'] = metric_log.get('mtoe_last')
        block['mtoe']['stop'] = stop
    return block, debug_log
