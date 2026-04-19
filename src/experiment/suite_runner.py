import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import Callable, Dict, List

from src.experiment.common_io import ensure_dir, now_str
from src.experiment.run_reproducibility import run_reproducibility_sweep
from src.experiment.suite_variants import overlay_args


VariantBuilder = Callable[[str, int], List[Dict[str, object]]]


def split_args(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--out_root', type=str, required=True)
    ap.add_argument('--planner_seed_from', type=int, default=None)
    ap.add_argument('--planner_seed_to', type=int, default=None)
    ap.add_argument('--planner_seeds', type=int, nargs='+', default=None)
    ap.add_argument('--repeat_count', type=int, default=None)
    ap.add_argument('--planner_seed_base', type=int, default=0)
    ap.add_argument('--progress_every', type=int, default=1)
    ap.add_argument('--suite', type=str, default='paper', choices=['paper', 'all', 'smoke'])
    ap.add_argument('--full_escape_injections', type=int, default=0)
    ap.add_argument('--timestamp_format', type=str, default='%Y%m%d_%H%M%S')
    ap.add_argument('--time_tag', type=str, default='')
    ap.add_argument('--no_timestamp', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    args, unknown = ap.parse_known_args()
    bench_args = list(unknown)
    if bench_args and bench_args[0] == '--':
        bench_args = bench_args[1:]
    return args, bench_args


def run_variant_sweep(args, bench_args: List[str], variant_builder: VariantBuilder) -> str:
    variants = variant_builder(args.suite, args.full_escape_injections)
    time_tag = args.time_tag.strip() or datetime.now().strftime(args.timestamp_format)
    suite_root = args.out_root if args.no_timestamp else os.path.join(args.out_root, 'times', time_tag)
    ensure_dir(suite_root)

    manifest = {
        'created_at': now_str(),
        'suite': args.suite,
        'time_tag': time_tag,
        'suite_root': suite_root,
        'variants': [v['name'] for v in variants],
        'planner_seed_from': args.planner_seed_from,
        'planner_seed_to': args.planner_seed_to,
        'planner_seeds': args.planner_seeds,
        'repeat_count': args.repeat_count,
        'planner_seed_base': args.planner_seed_base,
        'bench_args': bench_args,
    }
    with open(os.path.join(suite_root, 'suite_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for idx, variant in enumerate(variants, start=1):
        variant_name = str(variant['name'])
        variant_root = os.path.join(suite_root, variant_name)
        repro_ns = SimpleNamespace(
            out_root=variant_root,
            planner_seed_from=args.planner_seed_from,
            planner_seed_to=args.planner_seed_to,
            planner_seeds=deepcopy(args.planner_seeds),
            repeat_count=args.repeat_count,
            planner_seed_base=args.planner_seed_base,
            progress_every=args.progress_every,
            write_seed_cases=False,
            write_seed_log=False,
            write_by_planner=False,
            write_seed_mtoe_csv=False,
            keep_raw_debug_logs=False,
            moead_only=False,
        )
        if variant_builder.__name__ == 'build_ablation_variants':
            repro_ns.moead_only = True
        print(f'[suite {idx}/{len(variants)}] variant={variant_name}')
        print('  out_root =', variant_root)
        mode_desc = 'direct-call reproducibility -> moead only' if repro_ns.moead_only else 'direct-call reproducibility -> moead/rrt*/prm'
        print('  mode =', mode_desc)
        variant_args = overlay_args(bench_args, list(variant['overlay']))
        if args.dry_run:
            print('  forwarded =', ' '.join(variant_args))
            continue
        run_reproducibility_sweep(repro_ns, variant_args)

    print('[done] suite finished ->', suite_root)
    return suite_root
