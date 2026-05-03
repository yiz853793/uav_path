

# ===== Common IO =====
import csv
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def mean_std(values: Iterable[Any]) -> Tuple[Optional[float], Optional[float], int]:
    vals = [safe_float(v) for v in values if safe_float(v) is not None]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, var ** 0.5, n


def read_rows_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not csv_path or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        try:
            return list(csv.DictReader(f))
        except Exception:
            return []


def append_rows_csv(csv_path: str, rows: List[Dict[str, Any]], preset_fieldnames: Optional[List[str]] = None) -> None:
    if not csv_path:
        return
    ensure_dir(os.path.dirname(csv_path) or '.')
    rows = rows or []
    fieldnames: List[str] = list(preset_fieldnames or [])
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            try:
                existing = csv.DictReader(f)
                existing_fieldnames = list(existing.fieldnames or [])
                for name in existing_fieldnames:
                    if name not in fieldnames:
                        fieldnames.append(name)
            except Exception:
                pass
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        return

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if not file_exists:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        return

    old_rows = read_rows_csv(csv_path)
    old_fieldnames = list(old_rows[0].keys()) if old_rows else fieldnames
    if old_fieldnames != fieldnames:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in old_rows + rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        return

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def write_rows_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not csv_path:
        return
    ensure_dir(os.path.dirname(csv_path) or '.')
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if not fieldnames:
            f.write('')
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


_append_rows_csv = append_rows_csv
_safe_float = safe_float
_safe_int = safe_int


# ===== Benchmark shared helpers =====
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np

from src.env.grid_env import GridEnv


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
        'mtoe_window': int(getattr(args, 'mtoe_window', 10)),
        'mtoe': {
            'enabled': True,
            'mode': metric_log.get('mtoe_mode', 'best_so_far_delta'),
            'window': int(metric_log.get('mtoe_window', getattr(args, 'mtoe_window', 10))),
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


# ===== Benchmark argument parser =====
import argparse
from typing import List


def build_benchmark_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run benchmark on terrain datasets. Most users only need terrain selection, "
            "-n/--num_terrains, -o/--out_root and optionally -P/--profile."
        ),
    )

    ap.add_argument("--terrain_dir", type=str, default="terrains", help="directory containing *.npz terrains")
    ap.add_argument("--out_root", "-o", type=str, default="outputs", help="benchmark output root")
    ap.add_argument("--terrain_type", "--terrain", "-t", dest="terrain_type", type=str, default="mountain", choices=["mountain", "city", "hill_city"], help="terrain type when --glob is not provided")
    ap.add_argument("--city_density", "--density", "-d", dest="city_density", type=float, default=0.24, help="city density tag when terrain_type=city/hill_city and --glob is not provided")
    ap.add_argument("--size", "-s", type=str, default="small", choices=["small", "medium", "large"], help="mountain size when --glob is not provided")
    ap.add_argument("--profile", "-P", type=str, default="auto", choices=["auto", "quick", "balanced", "quality"], help="preset for common benchmark budgets; explicit CLI values override preset values")

    ap.add_argument("--seed_from", "--sf", dest="seed_from", type=int, default=0, help="inclusive terrain seed start")
    ap.add_argument("--seed_to", "--st", dest="seed_to", type=int, default=10, help="exclusive terrain seed end; used if no --glob")
    ap.add_argument("--glob", type=str, default="", help="optional terrain glob, e.g. 'terrains/*/mountain_seed*.npz'")

    ap.add_argument("--planner_seed", "--pseed", dest="planner_seed", type=int, default=0, help="single planner seed")
    ap.add_argument("--planner_seeds", type=int, nargs="+", default=None, help="optional list of planner seeds; if provided, overrides --planner_seed")
    ap.add_argument("--num_terrains", "-n", dest="num_terrains", type=int, default=None, help="alias for seed_to-seed_from when not using --glob")
    ap.add_argument("--inflate", "-i", type=float, default=1.0, help="inflate obstacle radius in meters")

    ap.add_argument("--rrt_iter", "--rrt", dest="rrt_iter", type=int, default=4000, help="RRT* iterations")
    ap.add_argument("--prm_samples", "--prm_n", dest="prm_samples", type=int, default=1200, help="PRM sample count")
    ap.add_argument("--prm_k", type=int, default=12, help="PRM neighbors per node")
    ap.add_argument("--prm_max_edge_len", "--prm_edge", dest="prm_max_edge_len", type=float, default=30.0, help="PRM max edge length in meters")
    ap.add_argument("--prm_threat_weight", type=float, default=0.0, help="PRM threat weight")

    ap.add_argument("--moead_min_gen", "--gmin", dest="moead_min_gen", type=int, default=20, help="minimum MOEA/D generations before early stop")
    ap.add_argument("--moead_max_gen", "--gmax", dest="moead_max_gen", type=int, default=None, help="maximum MOEA/D generations")
    ap.add_argument("--mtoe_tol_fun", type=float, default=1e-5)
    ap.add_argument("--mtoe_confidence", type=float, default=0.995)
    ap.add_argument("--mtoe_window", type=int, default=10, help="MTOE early-stop rolling window size")
    ap.add_argument("--disable_mtoe_stop", action="store_true", help="run to moead_max_gen but keep recording shadow MTOE stop events")
    ap.add_argument("--basin_shadow_enable", type=int, default=1, help="enable basin-aware shadow monitoring (1/0)")
    ap.add_argument("--basin_band_count", type=int, default=7, help="number of lateral bins for basin signature")
    ap.add_argument("--basin_signature_samples", type=int, default=9, help="number of progress samples used in basin signature")
    ap.add_argument("--basin_stagnation_window", type=int, default=10, help="window for basin saturation checks")
    ap.add_argument("--basin_f2_tol_abs", type=float, default=1.0, help="absolute f2 span tolerance for basin plateau")
    ap.add_argument("--basin_f2_tol_rel", type=float, default=0.01, help="relative f2 span tolerance for basin plateau")
    ap.add_argument("--basin_ref_gap_tol", type=float, default=0.08, help="reference-gap threshold for stop_candidate vs escape")
    ap.add_argument("--basin_min_distinct", type=int, default=2, help="minimum distinct basins before stop_candidate is allowed")
    ap.add_argument("--basin_escape_injections", type=int, default=0, help="extra escape injections when basin monitor says action=escape")
    ap.add_argument("--moead_pop", "--pop", dest="moead_pop", type=int, default=60, help="MOEA/D population size")
    ap.add_argument("--K", type=int, default=30, help="number of control points per path")

    ap.add_argument("--moead_T", "-T", dest="moead_T", type=int, default=10, help="MOEA/D neighborhood size T")
    ap.add_argument("--init_astar_ratio", type=float, default=0.25, help="fraction of population initialized from A* seeding")
    ap.add_argument("--init_astar_threat_weight", type=float, default=0.0, help="A* cost threat weight for seeding (0=ignore threat)")
    ap.add_argument("--init_astar_jitter_sigma", type=float, default=1.5, help="std of Gaussian jitter for A* path points")
    ap.add_argument("--init_astar_max_paths", type=int, default=5, help="max number of diverse A* backbone paths to generate")
    ap.add_argument("--init_astar_penalty_step", type=float, default=2.5, help="penalty added on visited cells after each A* to encourage diversity")
    ap.add_argument("--init_stratified_ratio", type=float, default=0.60, help="fraction of population initialized by stratified corridor sampling")
    ap.add_argument("--init_stratified_lateral_frac", type=float, default=0.30, help="half-width of stratified corridor as a fraction of start-goal distance")
    ap.add_argument("--init_stratified_n_bands", type=int, default=5, help="number of lateral bands for stratified sampling")
    ap.add_argument("--init_stratified_progress_jitter", type=float, default=0.08, help="jitter on normalized progress of each control point")
    ap.add_argument("--init_global_random_ratio", type=float, default=0.15, help="small exploratory tail kept as fully random/noisy individuals")
    ap.add_argument("--weight_extreme_bias", type=float, default=0.20, help="share of MOEA/D weights reserved near single-objective extremes")

    ap.add_argument("--extreme_offspring_ratio", type=float, default=0.20, help="extra offspring ratio allocated adaptively to extreme directions")
    ap.add_argument("--extreme_potential_window", type=int, default=20, help="window size for estimating per-objective improvement potential")
    ap.add_argument("--extreme_min_extra_per_obj", type=int, default=1, help="minimum extra offspring reserved for each objective when extra budget is active")
    ap.add_argument("--extreme_max_frac_per_obj", type=float, default=0.60, help="maximum fraction of extra budget assigned to any single objective")
    ap.add_argument("--local_search_interval", type=int, default=10, help="run directional local search every N generations (0 to disable)")
    ap.add_argument("--local_search_elite_k", type=int, default=3, help="top-k archive elites per objective used for directional local search")
    ap.add_argument("--local_search_attempts_per_obj", type=int, default=2, help="directional local-search attempts per objective each trigger")
    ap.add_argument("--max_turn_deg", type=float, default=90.0, help="hard turn-angle limit in degrees")
    ap.add_argument("--soft_turn_deg", type=float, default=60.0, help="soft preferred turn-angle limit in degrees")
    ap.add_argument("--max_pitch_deg", type=float, default=35.0, help="hard pitch-angle limit in degrees")
    ap.add_argument("--soft_pitch_deg", type=float, default=25.0, help="soft preferred pitch-angle limit in degrees")
    ap.add_argument("--desired_clearance_margin", type=float, default=2.0, help="extra preferred clearance above min_clearance")
    ap.add_argument("--tau_soft", type=float, default=25.0, help="maximum allowed weighted soft-constraint violation")
    ap.add_argument("--archive_size", type=int, default=0, help="MOEA/D archive upper bound (>0 uses hard cap; <=0 falls back to archive_soft_limit)")
    ap.add_argument("--archive_soft_limit", "--arch_soft", dest="archive_soft_limit", type=int, default=320, help="soft archive cap used when archive_size<=0; improves Pareto spread")
    ap.add_argument("--archive_grid_bins", type=int, default=0, help="objective-space grid bins for diversity-aware archive truncation (0=auto)")
    ap.add_argument("--archive_keep_extremes", type=int, default=1, help="protect objective extremes during archive truncation (1/0)")
    ap.add_argument("--active_subproblem_ratio", "--active_ratio", dest="active_subproblem_ratio", type=float, default=1.0, help="fraction of subproblems activated each generation (0,1]")
    ap.add_argument("--utility_update_interval", type=int, default=3, help="update MOEA/D utility every N generations")
    ap.add_argument("--utility_use_archive_density", type=int, default=0, help="whether to mix archive density into utility update (1/0)")
    ap.add_argument("--log_flush_every", type=int, default=10, help="flush MOEA/D debug log every N generations")

    ap.add_argument("--moead_debug", action="store_true", help="write MOEA/D debug log to out_dir/moead_debug.log and MTOE stats to mtoe_debug_*.log")
    ap.add_argument("--moead_debug_every", type=int, default=1, help="log every N generations")
    ap.add_argument("--moead_debug_level", type=int, default=2, help="1=coarse, 2=timing breakdown, 3=collision profiling")
    ap.add_argument("--moead_eval_step", type=float, default=2.5, help="sample step for evaluation (collision + threat), in meters. Larger => much faster but less precise.")
    ap.add_argument("--moead_smooth_step", type=float, default=2.5, help="sample step for shortcut-smooth collision checks, in meters. Larger => faster.")

    ap.add_argument("--start", type=float, nargs=2, default=None, help="start point in meters: x_m y_m")
    ap.add_argument("--goal", type=float, nargs=2, default=None, help="goal point in meters: x_m y_m")

    ap.add_argument("--outdir", type=str, default=None, help="alias of --out_root")
    ap.add_argument("--single_case_out_dir", type=str, default="", help="when set, write outputs of a single terrain directly into this directory")
    ap.add_argument("--summary_log", type=str, default="", help="append overall summary to this log file (default: out_root/benchmark_summary.log)")
    ap.add_argument("--summary_csv", type=str, default="", help="write one-row benchmark summary CSV (default: out_root/benchmark_summary.csv)")
    ap.add_argument("--moead_debug_csv", type=str, default="", help="optional CSV path for parsed per-generation rows; default writes into each seed folder as moead_debug.csv")
    ap.add_argument("--mtoe_csv", type=str, default="", help="optional CSV path for parsed MTOE rows; default writes into each seed folder as mtoe.csv")
    ap.add_argument("--mtoe_debug_csv", type=str, default="", help="optional CSV path for parsed original mtoe_debug log; default writes into each seed folder as mtoe_debug.csv")
    ap.add_argument("--start_goal_z_offset", type=float, default=15.0, help="default start/goal altitude offset above local ground, in meters")
    return ap


def parse_benchmark_args(argv: List[str]):
    parser = build_benchmark_parser()
    args = parser.parse_args(argv)
    apply_run_profile(args, argv)
    if args.moead_max_gen is None:
        args.moead_max_gen = 80
    if args.num_terrains is not None and not args.glob.strip():
        args.seed_to = int(args.seed_from) + int(max(0, args.num_terrains))
    if args.outdir is not None and str(args.outdir).strip():
        args.out_root = str(args.outdir).strip()
    return args


# ===== Single-case path planning =====
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
from src.experiment.terrain_result_plotting import build_vis_payload, save_vis_payload

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
    eval_kwargs = dict(
        max_turn_deg=args.max_turn_deg,
        soft_turn_deg=args.soft_turn_deg,
        max_pitch_deg=args.max_pitch_deg,
        soft_pitch_deg=args.soft_pitch_deg,
        desired_clearance_margin=args.desired_clearance_margin,
        tau_soft=args.tau_soft,
        sample_step=eval_step_cells,
    )
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
            rrt_er = evaluate_path(env, path_rrt, **eval_kwargs)
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
            prm_er = evaluate_path(env, path_prm, **eval_kwargs)
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
        mtoe_window=args.mtoe_window,
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
        max_turn_deg=args.max_turn_deg,
        soft_turn_deg=args.soft_turn_deg,
        max_pitch_deg=args.max_pitch_deg,
        soft_pitch_deg=args.soft_pitch_deg,
        desired_clearance_margin=args.desired_clearance_margin,
        tau_soft=args.tau_soft,
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


# ===== Multi-seed / batch path planning =====
import argparse
import ast
import json
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_seed_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    if 'seed' not in base:
        return None
    try:
        return int(base.split('seed')[-1].split('.')[0])
    except Exception:
        return None


def parse_forward_args(argv: List[str]):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--terrain_type', '-t', type=str, default='mountain')
    ap.add_argument('--city_density', '-d', type=float, default=0.24)
    ap.add_argument('--size', '-s', type=str, default='small')
    ap.add_argument('--seed_from', type=int, default=0)
    ap.add_argument('--seed_to', type=int, default=10)
    ap.add_argument('--num_terrains', '-n', type=int, default=None)
    ap.add_argument('--glob', type=str, default='')
    args, _ = ap.parse_known_args(argv)
    if args.num_terrains is not None and not str(args.glob).strip():
        args.seed_to = int(args.seed_from) + int(max(0, args.num_terrains))
    return args


def benchmark_style_subdir(argv: List[str]) -> str:
    a = parse_forward_args(argv)
    if str(a.glob).strip():
        return ''
    if a.terrain_type in ('city', 'hill_city'):
        return f'{a.terrain_type}_{a.city_density:.2f}'
    size_dir = {'small': 'S', 'medium': 'M', 'large': 'L'}.get(str(a.size).lower(), str(a.size))
    return size_dir


def split_args() -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(description='Run multi-terrain multi-planner-seed reproducibility sweep directly.')
    ap.add_argument('--out_root', type=str, required=True)
    ap.add_argument('--planner_seed_from', type=int, default=None)
    ap.add_argument('--planner_seed_to', type=int, default=None)
    ap.add_argument('--planner_seeds', type=int, nargs='+', default=None)
    ap.add_argument('--repeat_count', type=int, default=None)
    ap.add_argument('--planner_seed_base', type=int, default=0)
    ap.add_argument('--progress_every', type=int, default=1)
    ap.add_argument('--write_seed_cases', action='store_true', help='also write per-seed reproducibility_cases.csv')
    ap.add_argument('--write_seed_log', action='store_true', help='also write per-seed reproducibility_seed.log')
    ap.add_argument('--write_by_planner', action='store_true', help='also write global reproducibility_by_planner_seed.csv')
    ap.add_argument('--write_seed_mtoe_csv', action='store_true', help='also keep seed-level mtoe_all.csv')
    ap.add_argument('--keep_raw_debug_logs', action='store_true', help='keep rotated raw moead_debug/mtoe_debug logs after CSV extraction')
    args, unknown = ap.parse_known_args()
    bench_args = list(unknown)
    if bench_args and bench_args[0] == '--':
        bench_args = bench_args[1:]
    return args, bench_args


def resolve_planner_seeds(args: argparse.Namespace) -> List[int]:
    if args.planner_seeds:
        return sorted(dict.fromkeys(int(x) for x in args.planner_seeds))
    if args.repeat_count is not None:
        base = int(args.planner_seed_base)
        return list(range(base, base + int(args.repeat_count)))
    if args.planner_seed_from is not None and args.planner_seed_to is not None:
        return list(range(int(args.planner_seed_from), int(args.planner_seed_to)))
    if args.planner_seed_from is not None:
        return [int(args.planner_seed_from)]
    return [0]


def _maybe_list(v: Any):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return val
        except Exception:
            pass
    return None


def _group_key(row: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, ...]:
    return tuple(row.get(k) for k in keys)


def aggregate_rows(rows: List[Dict[str, Any]], group_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, group_keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for gkey, grows in sorted(groups.items(), key=lambda kv: kv[0]):
        item: Dict[str, Any] = {k: v for k, v in zip(group_keys, gkey)}
        item['updated_at'] = now_str()
        item['n_runs'] = len(grows)
        invalid = [r for r in grows if safe_int(r.get('invalid_case')) == 1]
        valid = [r for r in grows if safe_int(r.get('invalid_case')) != 1]
        item['n_invalid'] = len(invalid)
        item['n_valid'] = len(valid)
        if valid:
            item['archive_positive_rate'] = sum(1 for r in valid if safe_int(r.get('moead_archive_size')) and safe_int(r.get('moead_archive_size')) > 0) / len(valid)
            item['rrt_found_rate'] = sum(1 for r in valid if str(r.get('rrt_found')).lower() in ('1', 'true', 'yes')) / len(valid)
            item['prm_found_rate'] = sum(1 for r in valid if str(r.get('prm_found')).lower() in ('1', 'true', 'yes')) / len(valid)
            for col in ('moead_ms', 'moead_n_gen', 'moead_archive_size', 'rrt_ms', 'prm_ms'):
                mean, std, n = mean_std(r.get(col) for r in valid)
                item[f'{col}_mean'] = mean
                item[f'{col}_std'] = std
                item[f'{col}_n'] = n
            stop_counts: Dict[str, int] = {}
            for r in valid:
                reason = str(r.get('moead_stop_reason') or '')
                stop_counts[reason] = stop_counts.get(reason, 0) + 1
            item['stop_reason_counts'] = '; '.join(f'{k}:{v}' for k, v in sorted(stop_counts.items()))
            for rep in ('min_f1', 'min_f2', 'min_f3', 'compromise'):
                for j in range(3):
                    vals = []
                    for r in valid:
                        robj = _maybe_list(r.get(f'{rep}_obj'))
                        if robj is not None and len(robj) > j:
                            vals.append(robj[j])
                    mean, std, n = mean_std(vals)
                    item[f'{rep}_f{j+1}_mean'] = mean
                    item[f'{rep}_f{j+1}_std'] = std
                    item[f'{rep}_f{j+1}_n'] = n
        out.append(item)
    return out


def format_float(x: Any, nd: int = 4) -> str:
    v = safe_float(x)
    if v is None:
        return '-'
    return f'{v:.{nd}f}'


def write_seed_log(seed_dir: str, terrain_seed: int, seed_summary: Dict[str, Any]) -> None:
    log_path = os.path.join(seed_dir, 'reproducibility_seed.log')
    lines = [
        f'terrain_seed: {terrain_seed}',
        f'updated_at: {seed_summary.get("updated_at", now_str())}',
        f'n_runs: {seed_summary.get("n_runs", 0)}',
        f'n_valid: {seed_summary.get("n_valid", 0)}',
        f'n_invalid: {seed_summary.get("n_invalid", 0)}',
        f'archive_positive_rate: {format_float(seed_summary.get("archive_positive_rate"))}',
        f'rrt_found_rate: {format_float(seed_summary.get("rrt_found_rate"))}',
        f'prm_found_rate: {format_float(seed_summary.get("prm_found_rate"))}',
        f'moead_ms_mean±std: {format_float(seed_summary.get("moead_ms_mean"), 2)} ± {format_float(seed_summary.get("moead_ms_std"), 2)}',
        f'moead_n_gen_mean±std: {format_float(seed_summary.get("moead_n_gen_mean"), 2)} ± {format_float(seed_summary.get("moead_n_gen_std"), 2)}',
        f'moead_archive_size_mean±std: {format_float(seed_summary.get("moead_archive_size_mean"), 2)} ± {format_float(seed_summary.get("moead_archive_size_std"), 2)}',
        f'stop_reason_counts: {seed_summary.get("stop_reason_counts", "")}',
    ]
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).rstrip() + '\n')


def print_progress(cur_idx: int, total_runs: int, row: Optional[Dict[str, Any]], overall: Dict[str, Any], seed_summary: Optional[Dict[str, Any]] = None) -> None:
    row = row or {}
    print(f'[progress] {cur_idx}/{total_runs} done | terrain_seed={row.get("terrain_seed")} planner_seed={row.get("planner_seed")} returncode={row.get("returncode", 0)} invalid={row.get("invalid_case", 0)} archive={row.get("moead_archive_size")} stop={row.get("moead_stop_reason")}')
    print(f'          overall: valid={overall.get("n_valid",0)}/{overall.get("n_runs",0)} archive+={format_float(overall.get("archive_positive_rate"))} mean_ms={format_float(overall.get("moead_ms_mean"),2)}')
    if seed_summary:
        print(f'          seed-summary: valid={seed_summary.get("n_valid",0)}/{seed_summary.get("n_runs",0)} archive+={format_float(seed_summary.get("archive_positive_rate"))} mean_ms={format_float(seed_summary.get("moead_ms_mean"),2)}')


def maybe_rotate_debug_logs(seed_dir: str, planner_seed: int, *, retries: int = 5, sleep_s: float = 0.25) -> List[str]:
    rotated = []
    candidates = {
        'moead_debug.log': f'moead_debug_pseed{planner_seed:04d}.log',
        'mtoe_debug.log': f'mtoe_debug_pseed{planner_seed:04d}.log',
    }
    for src_name, dst_name in candidates.items():
        src = os.path.join(seed_dir, src_name)
        dst = os.path.join(seed_dir, dst_name)
        if not os.path.exists(src):
            continue
        last_err = None
        for _ in range(max(1, int(retries))):
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                rotated.append(dst)
                last_err = None
                break
            except PermissionError as e:
                last_err = e
                time.sleep(float(sleep_s))
        if last_err is not None:
            try:
                shutil.copy2(src, dst)
                rotated.append(dst)
            except Exception:
                pass
    return rotated


def cleanup_debug_logs(paths: List[str]) -> None:
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _extract_case_summary_from_metrics(run_dir: str, terrain_seed: Optional[int], planner_seed: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {'terrain_seed': terrain_seed, 'planner_seed': planner_seed, 'invalid_case': 0}
    try:
        prefix = f'_pseed{int(planner_seed):04d}_'
        candidates = [os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.startswith('metrics_') and prefix in name and name.endswith('.json')]
        if not candidates:
            return row
        metrics_path = sorted(candidates)[-1]
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        row['terrain_file'] = data.get('terrain_file')
        row['terrain_seed'] = data.get('terrain_seed', terrain_seed)
        row['planner_seed'] = data.get('planner_seed', planner_seed)
        row['invalid_case'] = int(bool(data.get('invalid_case')))
        rrt = data.get('rrt') or {}
        prm = data.get('prm') or {}
        mo = data.get('moead') or {}
        mtoe = mo.get('mtoe') or {}
        reps = mo.get('representatives') or {}
        row['rrt_found'] = int(bool(rrt.get('found')))
        row['rrt_feasible'] = int(bool(rrt.get('feasible')))
        row['rrt_ms'] = rrt.get('runtime_ms')
        row['rrt_obj'] = repr(rrt.get('obj')) if isinstance(rrt.get('obj'), list) else rrt.get('obj')
        row['prm_found'] = int(bool(prm.get('found')))
        row['prm_feasible'] = int(bool(prm.get('feasible')))
        row['prm_ms'] = prm.get('runtime_ms')
        row['prm_obj'] = repr(prm.get('obj')) if isinstance(prm.get('obj'), list) else prm.get('obj')
        row['moead_ms'] = mo.get('runtime_ms')
        row['moead_archive_size'] = mo.get('archive_size')
        row['moead_stop_reason'] = mo.get('stop_reason')
        row['moead_n_gen'] = mo.get('n_gen')
        row['moead_mtoe_last'] = mtoe.get('last')
        row['mtoe_enabled'] = int(bool(mtoe.get('enabled'))) if mtoe.get('enabled') is not None else None
        row['mtoe_mode'] = mtoe.get('mode')
        row['mtoe_tol_fun'] = mtoe.get('tol_fun')
        row['mtoe_confidence'] = mtoe.get('confidence')
        row['disable_mtoe_stop'] = int(bool(mtoe.get('disable_stop'))) if mtoe.get('disable_stop') is not None else None
        row['first_shadow_stop_gen'] = mtoe.get('first_shadow_stop_gen')
        row['shadow_stop_count'] = mtoe.get('shadow_stop_count')
        if isinstance(reps, dict):
            row['rep_objs'] = reps
            for name, obj in reps.items():
                row[f'{name}_obj'] = repr(obj)
                vals = _maybe_list(obj) or []
                for i, v in enumerate(vals[:3], start=1):
                    row[f'{name}_f{i}'] = v
        return row
    except Exception:
        return row


def enrich_case_row(row: Dict[str, Any], run_dir: str, terrain_path: str) -> Dict[str, Any]:
    row = dict(row)
    row['timestamp'] = now_str()
    row['run_dir'] = run_dir
    row['terrain_file'] = terrain_path
    row['returncode'] = int(row.get('returncode', 0))
    rep_objs = row.pop('rep_objs', None)
    if isinstance(rep_objs, dict):
        for name, obj in rep_objs.items():
            row[f'{name}_obj'] = repr(obj)
            vals = _maybe_list(obj) or []
            for i, v in enumerate(vals[:3], start=1):
                row[f'{name}_f{i}'] = v
    for prefix in ('rrt', 'prm'):
        obj = row.get(f'{prefix}_obj')
        if isinstance(obj, list):
            row[f'{prefix}_obj'] = repr(obj)
            for i, v in enumerate(obj[:3], start=1):
                row[f'{prefix}_f{i}'] = v
    row['invalid_case'] = int(bool(row.get('invalid_case')))
    row['rrt_found'] = int(bool(row.get('rrt_found')))
    row['prm_found'] = int(bool(row.get('prm_found')))
    row['rrt_feasible'] = int(bool(row.get('rrt_feasible')))
    row['prm_feasible'] = int(bool(row.get('prm_feasible')))
    return row


def rows_for_seed(rows: List[Dict[str, Any]], terrain_seed: int) -> List[Dict[str, Any]]:
    return [r for r in rows if safe_int(r.get('terrain_seed')) == int(terrain_seed)]


def run_reproducibility_sweep(args: argparse.Namespace, bench_args: List[str]) -> None:
    planner_seeds = resolve_planner_seeds(args)
    base_args = parse_benchmark_args(bench_args)
    setattr(base_args, 'moead_only', bool(getattr(args, 'moead_only', False)))
    terrain_files = resolve_terrain_files(base_args)
    if not terrain_files:
        raise SystemExit('No terrain files resolved from forwarded benchmark-style arguments.')
    missing = [p for p in terrain_files if not os.path.exists(p)]
    if missing:
        raise SystemExit('Missing terrain files:\n' + '\n'.join(missing[:10]))

    benchmark_subdir = benchmark_style_subdir(bench_args)
    effective_root = os.path.join(args.out_root, benchmark_subdir) if benchmark_subdir else args.out_root
    ensure_dir(effective_root)

    global_cases_csv = os.path.join(effective_root, 'reproducibility_cases.csv')
    global_overall_csv = os.path.join(effective_root, 'reproducibility_overall.csv')
    global_by_terrain_csv = os.path.join(effective_root, 'reproducibility_by_terrain.csv')
    global_by_planner_csv = os.path.join(effective_root, 'reproducibility_by_planner_seed.csv')

    all_case_rows: List[Dict[str, Any]] = read_rows_csv(global_cases_csv)
    total_runs = len(terrain_files) * len(planner_seeds)
    cur_idx = 0

    print('[repro] terrain_count =', len(terrain_files))
    print('[repro] planner_seed_count =', len(planner_seeds))
    print('[repro] total_runs =', total_runs)
    print('[repro] out_root =', effective_root)
    print('[repro] mode =', 'direct-call moead only' if getattr(base_args, 'moead_only', False) else 'direct-call moead/rrt*/prm')

    for terrain_path in terrain_files:
        terrain_seed = parse_seed_from_filename(terrain_path)
        if terrain_seed is None:
            raise RuntimeError(f'Cannot parse terrain seed from filename: {terrain_path}')
        seed_dir = os.path.join(effective_root, f'seed{terrain_seed:04d}')
        ensure_dir(seed_dir)
        seed_cases_csv = os.path.join(seed_dir, 'reproducibility_cases.csv')
        seed_summary_csv = os.path.join(seed_dir, 'reproducibility_summary.csv')
        seed_moead_debug_csv = os.path.join(seed_dir, 'moead_debug_all.csv')
        seed_mtoe_csv = os.path.join(seed_dir, 'mtoe_all.csv')
        seed_mtoe_debug_csv = os.path.join(seed_dir, 'mtoe_debug_all.csv')

        for planner_seed in planner_seeds:
            cur_idx += 1
            print(f'[run {cur_idx}/{total_runs}] terrain_seed={terrain_seed:04d} planner_seed={planner_seed:04d}')
            rotated_logs: List[str] = []
            try:
                row = run_single_case(
                    base_args,
                    terrain_path,
                    planner_seed,
                    out_dir_override=seed_dir,
                    moead_debug_csv_path=seed_moead_debug_csv,
                    mtoe_csv_path=(seed_mtoe_csv if getattr(args, 'write_seed_mtoe_csv', False) else ''),
                    mtoe_debug_csv_path=seed_mtoe_debug_csv,
                )
                rotated_logs = maybe_rotate_debug_logs(seed_dir, planner_seed)
                row = enrich_case_row(row, seed_dir, terrain_path)
            except Exception as e:
                row = {
                    'timestamp': now_str(), 'terrain_seed': terrain_seed, 'planner_seed': planner_seed,
                    'run_dir': seed_dir, 'terrain_file': terrain_path, 'returncode': 1, 'invalid_case': 0,
                    'moead_stop_reason': f'exception:{type(e).__name__}', 'error': repr(e),
                }
                recovered = _extract_case_summary_from_metrics(seed_dir, terrain_seed, planner_seed)
                for k, v in recovered.items():
                    if row.get(k) in (None, '', 0) and v not in (None, ''):
                        row[k] = v
                if row.get('moead_archive_size') is not None:
                    row['returncode'] = 0
            if not getattr(args, 'keep_raw_debug_logs', False):
                cleanup_debug_logs(rotated_logs)

            append_rows_csv(global_cases_csv, [row])
            if getattr(args, 'write_seed_cases', False):
                append_rows_csv(seed_cases_csv, [row])
            all_case_rows.append(row)

            seed_rows = rows_for_seed(all_case_rows, terrain_seed)
            seed_summary_rows = aggregate_rows(seed_rows, ('terrain_seed',))
            write_rows_csv(seed_summary_csv, seed_summary_rows)
            seed_summary = seed_summary_rows[0] if seed_summary_rows else {'terrain_seed': terrain_seed, 'updated_at': now_str()}
            if getattr(args, 'write_seed_log', False):
                write_seed_log(seed_dir, terrain_seed, seed_summary)

            overall_rows = aggregate_rows(all_case_rows, tuple())
            by_terrain_rows = aggregate_rows(all_case_rows, ('terrain_seed',))
            write_rows_csv(global_overall_csv, overall_rows)
            write_rows_csv(global_by_terrain_csv, by_terrain_rows)
            if getattr(args, 'write_by_planner', False):
                write_rows_csv(global_by_planner_csv, aggregate_rows(all_case_rows, ('planner_seed',)))

            if cur_idx % max(1, int(args.progress_every)) == 0:
                print_progress(cur_idx, total_runs, row, overall_rows[0] if overall_rows else {}, seed_summary)

    print('[done] reproducibility sweep finished')
    print('global cases   :', global_cases_csv)
    print('global overall :', global_overall_csv)
    print('global terrain :', global_by_terrain_csv)
    if getattr(args, 'write_by_planner', False):
        print('global planner :', global_by_planner_csv)


def main() -> None:
    args, bench_args = split_args()
    run_reproducibility_sweep(args, bench_args)


if __name__ == '__main__':
    main()


# -----------------------------
# Batch benchmark entry point
# -----------------------------

def main_benchmark(argv: List[str] | None = None) -> None:
    args = parse_benchmark_args(list(argv or []))
    terrain_files = resolve_terrain_files(args)
    if not terrain_files:
        raise SystemExit('No terrain files resolved.')
    missing = [p for p in terrain_files if not os.path.exists(p)]
    if missing:
        raise SystemExit('Missing terrain files:\n' + '\n'.join(missing[:10]))

    planner_seeds = list(args.planner_seeds) if args.planner_seeds else [int(args.planner_seed)]
    ensure_dir(args.out_root)
    summary_csv = args.summary_csv or os.path.join(args.out_root, 'benchmark_summary.csv')
    summary_log = args.summary_log or os.path.join(args.out_root, 'benchmark_summary.log')

    rows: List[Dict[str, Any]] = []
    total = len(terrain_files) * len(planner_seeds)
    idx = 0
    for terrain_path in terrain_files:
        for planner_seed in planner_seeds:
            idx += 1
            print(f'[benchmark {idx}/{total}] terrain={terrain_path} planner_seed={planner_seed}')
            try:
                row = run_single_case(args, terrain_path, int(planner_seed))
                row = enrich_case_row(row, row.get('run_dir', ''), terrain_path) if 'run_dir' in row else row
                row['returncode'] = int(row.get('returncode', 0))
            except Exception as exc:
                row = {
                    'timestamp': now_str(),
                    'terrain_file': terrain_path,
                    'planner_seed': int(planner_seed),
                    'returncode': 1,
                    'error': repr(exc),
                }
            rows.append(row)
            append_rows_csv(summary_csv, [row])
            with open(summary_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    overall = aggregate_rows(rows, tuple()) if rows else []
    if overall:
        write_rows_csv(os.path.join(args.out_root, 'benchmark_overall.csv'), overall)
    print('[done] benchmark finished')
    print('summary csv:', summary_csv)
