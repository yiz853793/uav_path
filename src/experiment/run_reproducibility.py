import argparse
import ast
import json
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.experiment.bench_core import resolve_terrain_files, run_single_case
from src.experiment.benchmark_parser import parse_benchmark_args
from src.experiment.common_io import (
    append_rows_csv,
    ensure_dir,
    mean_std,
    now_str,
    read_rows_csv,
    safe_float,
    safe_int,
    write_rows_csv,
)


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
