import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_rows_csv(csv_path: str, rows: List[Dict[str, Any]]):
    if not csv_path or not rows:
        return
    ensure_dir(os.path.dirname(csv_path) or '.')
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    existing_fields: List[str] = []
    existing_rows: List[Dict[str, Any]] = []
    if file_exists:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                existing_rows = list(reader)
        except Exception:
            existing_fields = []
            existing_rows = []

    if existing_fields:
        merged = list(existing_fields)
        for k in fieldnames:
            if k not in merged:
                merged.append(k)
        fieldnames = merged

    rewrite = (not file_exists) or (not existing_fields) or (existing_fields != fieldnames)
    if rewrite:
        all_rows = existing_rows + rows
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k) for k in fieldnames})
    else:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for r in rows:
                writer.writerow({k: r.get(k) for k in fieldnames})


def parse_seed_from_filename(path: str):
    m = re.search(r'_seed(\d+)\.npz$', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_forward_args(argv: List[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--terrain_dir', type=str, default='terrains')
    p.add_argument('--terrain_type', type=str, default='mountain', choices=['mountain', 'city', 'hill_city'])
    p.add_argument('--city_density', type=float, default=0.24)
    p.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    p.add_argument('--seed_from', type=int, default=0)
    p.add_argument('--seed_to', type=int, default=50)
    p.add_argument('--glob', type=str, default='')
    p.add_argument('--num_terrains', type=int, default=None)
    ns, _ = p.parse_known_args(argv)
    if ns.num_terrains is not None and not ns.glob.strip():
        ns.seed_to = int(ns.seed_from) + int(max(0, ns.num_terrains))
    return ns


def terrain_files_from_forward_args(argv: List[str]) -> List[str]:
    ns = parse_forward_args(argv)
    if ns.glob.strip():
        files = sorted(glob.glob(ns.glob.strip()))
    elif ns.terrain_type in ('city', 'hill_city'):
        terrain_subdir = os.path.join(ns.terrain_dir, f'{ns.terrain_type}_{ns.city_density:.2f}')
        prefix = 'city' if ns.terrain_type == 'city' else 'hill_city'
        files = [
            os.path.join(terrain_subdir, f'{prefix}_seed{seed:04d}.npz')
            for seed in range(ns.seed_from, ns.seed_to)
        ]
    else:
        size_dir = {'small': 'S', 'medium': 'M', 'large': 'L'}[ns.size]
        terrain_subdir = os.path.join(ns.terrain_dir, size_dir)
        files = [
            os.path.join(terrain_subdir, f'mountain_seed{seed:04d}.npz')
            for seed in range(ns.seed_from, ns.seed_to)
        ]
    return files


def benchmark_style_subdir(argv: List[str]) -> str:
    ns = parse_forward_args(argv)
    if ns.glob.strip():
        return ''
    if ns.terrain_type in ('city', 'hill_city'):
        return f'{ns.terrain_type}_{ns.city_density:.2f}'
    return {'small': 'S', 'medium': 'M', 'large': 'L'}[ns.size]


def extract_case_rows(run_dir: str, repeat_idx: int, terrain_seed: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mp in sorted(glob.glob(os.path.join(run_dir, 'metrics_*.json'))):
        with open(mp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        meta = data.get('meta') or {}
        rrt = data.get('rrt') or {}
        prm = data.get('prm') or {}
        mo = data.get('moead') or {}
        reps = mo.get('representatives') or {}
        row: Dict[str, Any] = {
            'timestamp': now_str(),
            'repeat_idx': repeat_idx,
            'terrain_seed': terrain_seed,
            'run_dir': run_dir,
            'metrics_path': mp,
            'terrain_file': meta.get('terrain_file'),
            'planner_seed': meta.get('planner_seed'),
            'H': meta.get('H'),
            'W': meta.get('W'),
            'size_tag': meta.get('size_tag'),
            'invalid_case': int(bool(data.get('invalid_case', False))),
            'rrt_found': int(bool(rrt.get('found', False))),
            'rrt_runtime_ms': rrt.get('runtime_ms'),
            'prm_found': int(bool(prm.get('found', False))),
            'prm_runtime_ms': prm.get('runtime_ms'),
            'moead_runtime_ms': mo.get('runtime_ms'),
            'moead_archive_size': mo.get('archive_size'),
            'moead_stop_reason': mo.get('stop_reason'),
            'moead_n_gen': mo.get('n_gen'),
            'mtoe_tol_fun': (mo.get('mtoe') or {}).get('tol_fun'),
            'mtoe_confidence': (mo.get('mtoe') or {}).get('confidence'),
        }
        for rep_name, vals in reps.items():
            if isinstance(vals, list) and len(vals) >= 3:
                row[f'{rep_name}_f1'] = vals[0]
                row[f'{rep_name}_f2'] = vals[1]
                row[f'{rep_name}_f3'] = vals[2]
        rows.append(row)
    return rows


def split_args() -> Tuple[argparse.Namespace, List[str]]:
    ap = argparse.ArgumentParser(
        description='Repeat single-seed benchmark runs and store outputs as <out_root>/<benchmark-subdir>/seedXXXX/runYY/'
    )
    ap.add_argument('--repeat_count', type=int, required=True, help='number of repeats for each terrain seed')
    ap.add_argument('--out_root', type=str, required=True, help='root output directory, e.g. outputs')
    args, unknown = ap.parse_known_args()
    bench_args = list(unknown)
    if bench_args and bench_args[0] == '--':
        bench_args = bench_args[1:]
    if not bench_args:
        raise SystemExit('Please provide run_benchmark arguments after the script args. You may optionally use -- before them.')
    return args, bench_args


def main():
    args, bench_args = split_args()

    terrain_files = terrain_files_from_forward_args(bench_args)
    if not terrain_files:
        raise SystemExit('No terrain files resolved from forwarded arguments.')

    missing = [p for p in terrain_files if not os.path.exists(p)]
    if missing:
        sample = '\n'.join(missing[:10])
        raise SystemExit(f'Missing terrain files:\n{sample}')

    benchmark_subdir = benchmark_style_subdir(bench_args)
    effective_root = os.path.join(args.out_root, benchmark_subdir) if benchmark_subdir else args.out_root
    ensure_dir(effective_root)
    summary_csv = os.path.join(effective_root, 'reproducibility_summary.csv')
    cases_csv = os.path.join(effective_root, 'reproducibility_cases.csv')

    forbidden = {
        '--out_root', '--outdir', '--summary_log', '--summary_csv', '--glob', '--seed_from', '--seed_to', '--num_terrains', '--single_case_out_dir'
    }
    filtered: List[str] = []
    i = 0
    while i < len(bench_args):
        tok = bench_args[i]
        if tok in forbidden:
            i += 2
            continue
        filtered.append(tok)
        i += 1

    for terrain_path in terrain_files:
        terrain_seed = parse_seed_from_filename(terrain_path)
        if terrain_seed is None:
            raise RuntimeError(f'Cannot parse terrain seed from filename: {terrain_path}')
        seed_dir = os.path.join(effective_root, f'seed{terrain_seed:04d}')
        ensure_dir(seed_dir)

        for repeat_idx in range(args.repeat_count):
            run_dir = os.path.join(seed_dir, f'run{repeat_idx:02d}')
            ensure_dir(run_dir)
            cmd = [
                sys.executable, '-m', 'experiments.run_benchmark',
                '--glob', terrain_path,
                '--seed_from', str(terrain_seed),
                '--seed_to', str(terrain_seed + 1),
                '--out_root', run_dir,
                '--single_case_out_dir', run_dir,
                '--summary_log', os.path.join(run_dir, 'benchmark_summary.log'),
                '--summary_csv', os.path.join(run_dir, 'benchmark_summary.csv'),
            ] + filtered
            print(f'[run] seed={terrain_seed:04d} repeat={repeat_idx:02d}')
            print('      ' + ' '.join(cmd))
            proc = subprocess.run(cmd)
            status_row = {
                'timestamp': now_str(),
                'terrain_seed': terrain_seed,
                'repeat_idx': repeat_idx,
                'run_dir': run_dir,
                'terrain_file': terrain_path,
                'returncode': proc.returncode,
                'command': ' '.join(cmd),
            }
            summary_path = os.path.join(run_dir, 'benchmark_summary.csv')
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        first = next(reader, None)
                    if first:
                        for k, v in first.items():
                            if k not in status_row:
                                status_row[k] = v
                except Exception as exc:
                    status_row['summary_read_error'] = str(exc)
            append_rows_csv(summary_csv, [status_row])

            if proc.returncode == 0:
                case_rows = extract_case_rows(run_dir, repeat_idx=repeat_idx, terrain_seed=terrain_seed)
                if case_rows:
                    append_rows_csv(cases_csv, case_rows)

    print('[done] reproducibility runs finished')
    print('root:', effective_root)
    print('summary:', summary_csv)
    print('cases:', cases_csv)


if __name__ == '__main__':
    main()
