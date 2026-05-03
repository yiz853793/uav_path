from typing import Dict, List


VALUE_FLAGS = {
    '--out_root', '--terrain_dir', '--terrain_type', '--city_density', '--size',
    '--seed_from', '--seed_to', '--num_terrains', '--glob',
    '--planner_seed', '--planner_seed_from', '--planner_seed_to', '--planner_seed_base', '--planner_seeds',
    '--repeat_count', '--progress_every', '--summary_log', '--summary_csv', '--single_case_out_dir',
    '--rrt_iter', '--prm_samples', '--prm_k', '--prm_max_edge_len', '--prm_threat_weight',
    '--moead_min_gen', '--moead_max_gen', '--mtoe_tol_fun', '--mtoe_confidence', '--mtoe_window',
    '--moead_pop', '--K', '--moead_T',
    '--basin_shadow_enable', '--basin_band_count', '--basin_signature_samples', '--basin_stagnation_window',
    '--basin_f2_tol_abs', '--basin_f2_tol_rel', '--basin_ref_gap_tol', '--basin_min_distinct', '--basin_escape_injections',
    '--init_astar_ratio', '--init_astar_threat_weight', '--init_astar_jitter_sigma', '--init_astar_max_paths',
    '--init_astar_penalty_step', '--init_stratified_ratio', '--init_stratified_lateral_frac', '--init_stratified_n_bands',
    '--init_stratified_progress_jitter', '--init_global_random_ratio', '--weight_extreme_bias', '--extreme_offspring_ratio',
    '--extreme_potential_window', '--extreme_min_extra_per_obj', '--extreme_max_frac_per_obj', '--local_search_interval',
    '--local_search_elite_k', '--local_search_attempts_per_obj', '--archive_size', '--archive_soft_limit',
    '--archive_grid_bins', '--archive_keep_extremes', '--active_subproblem_ratio', '--utility_update_interval',
    '--utility_use_archive_density', '--log_flush_every', '--moead_debug_every', '--moead_debug_level',
    '--moead_eval_step', '--moead_smooth_step', '--start_goal_z_offset', '--moead_debug_csv', '--mtoe_csv',
    '--mtoe_debug_csv', '-o', '-t', '-d', '-s', '-n', '-P', '--gmax', '--pop', '--pseed', '--arch_soft'
}
BOOL_FLAGS = {'--disable_mtoe_stop', '--moead_debug'}
CONFLICT_FLAGS = {
    '--disable_mtoe_stop', '--basin_shadow_enable', '--basin_escape_injections', '--init_astar_ratio',
    '--init_stratified_ratio', '--init_global_random_ratio', '--weight_extreme_bias', '--extreme_offspring_ratio',
    '--local_search_interval', '--local_search_elite_k', '--local_search_attempts_per_obj', '--active_subproblem_ratio'
}


def remove_conflicting_flags(argv: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in CONFLICT_FLAGS or tok in BOOL_FLAGS:
            if tok in VALUE_FLAGS:
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def overlay_args(base_args: List[str], overlay: List[str]) -> List[str]:
    return remove_conflicting_flags(base_args) + overlay


def _overlay_targets(overlay: List[str]) -> set[str]:
    targets = set()
    i = 0
    while i < len(overlay):
        tok = overlay[i]
        if tok.startswith('-'):
            targets.add(tok)
            if tok in VALUE_FLAGS:
                i += 2
                continue
        i += 1
    return targets


def _remove_selected_flags(argv: List[str], targets: set[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in targets:
            if tok in VALUE_FLAGS:
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def merge_overlay(base_overlay: List[str], overlay: List[str]) -> List[str]:
    return _remove_selected_flags(base_overlay, _overlay_targets(overlay)) + overlay


def full_overlay(full_escape_injections: int) -> List[str]:
    return [
        '--basin_shadow_enable', '1',
        '--init_astar_ratio', '0.25',
        '--init_stratified_ratio', '0.60',
        '--init_global_random_ratio', '0.15',
        '--weight_extreme_bias', '0.20',
        '--extreme_offspring_ratio', '0.20',
        '--local_search_interval', '10',
        '--local_search_elite_k', '3',
        '--local_search_attempts_per_obj', '2',
        '--active_subproblem_ratio', '0.80',
        '--basin_escape_injections', str(int(full_escape_injections)),
    ]


def baseline_overlay() -> List[str]:
    return [
        '--disable_mtoe_stop',
        '--basin_shadow_enable', '0',
        '--basin_escape_injections', '0',
        '--init_astar_ratio', '0.0',
        '--init_stratified_ratio', '0.0',
        '--init_global_random_ratio', '1.0',
        '--weight_extreme_bias', '0.0',
        '--extreme_offspring_ratio', '0.0',
        '--local_search_interval', '0',
        '--local_search_elite_k', '0',
        '--local_search_attempts_per_obj', '0',
        '--active_subproblem_ratio', '1.0',
    ]


def build_variants(suite: str, full_escape_injections: int) -> List[Dict[str, object]]:
    full = full_overlay(full_escape_injections)
    base = baseline_overlay()
    variants: List[Dict[str, object]] = [
        {'name': 'baseline', 'group': 'main', 'desc': 'Plain MOEA/D baseline.', 'overlay': base},
        {'name': 'full', 'group': 'main', 'desc': 'Current full configuration.', 'overlay': full},
        {'name': 'wo_init', 'group': 'ablation_init', 'desc': 'Full without initialization enhancements.', 'overlay': merge_overlay(full, ['--init_astar_ratio', '0.0', '--init_stratified_ratio', '0.0', '--init_global_random_ratio', '1.0'])},
        {'name': 'wo_search', 'group': 'ablation_search', 'desc': 'Full without active sampling / local search / escape.', 'overlay': merge_overlay(full, ['--basin_shadow_enable', '0', '--basin_escape_injections', '0', '--weight_extreme_bias', '0.0', '--extreme_offspring_ratio', '0.0', '--local_search_interval', '0', '--local_search_elite_k', '0', '--local_search_attempts_per_obj', '0', '--active_subproblem_ratio', '1.0'])},
        {'name': 'wo_active_extreme', 'group': 'ablation_search_split', 'desc': 'Full without active-subproblem and extreme-offspring bias.', 'overlay': merge_overlay(full, ['--weight_extreme_bias', '0.0', '--extreme_offspring_ratio', '0.0', '--active_subproblem_ratio', '1.0'])},
        {'name': 'wo_local_escape', 'group': 'ablation_search_split', 'desc': 'Full without local search or escape injection.', 'overlay': merge_overlay(full, ['--basin_shadow_enable', '0', '--basin_escape_injections', '0', '--local_search_interval', '0', '--local_search_elite_k', '0', '--local_search_attempts_per_obj', '0'])},
        {'name': 'wo_mtoe', 'group': 'ablation_mtoe', 'desc': 'Full without MTOE stop.', 'overlay': merge_overlay(full, ['--disable_mtoe_stop'])},
        {'name': 'base_plus_init', 'group': 'cumulative', 'desc': 'Baseline plus initialization enhancements only.', 'overlay': merge_overlay(base, ['--basin_shadow_enable', '0', '--init_astar_ratio', '0.25', '--init_stratified_ratio', '0.60', '--init_global_random_ratio', '0.15'])},
        {'name': 'base_plus_init_search', 'group': 'cumulative', 'desc': 'Baseline + init + search, still without MTOE stop.', 'overlay': merge_overlay(base, ['--basin_shadow_enable', '1', '--init_astar_ratio', '0.25', '--init_stratified_ratio', '0.60', '--init_global_random_ratio', '0.15', '--weight_extreme_bias', '0.20', '--extreme_offspring_ratio', '0.20', '--local_search_interval', '10', '--local_search_elite_k', '3', '--local_search_attempts_per_obj', '2', '--active_subproblem_ratio', '0.80', '--basin_escape_injections', str(int(full_escape_injections))])},
    ]
    if suite == 'smoke':
        keep = {'baseline', 'full'}
        return [v for v in variants if v['name'] in keep]
    return variants


def build_ablation_variants(suite: str, full_escape_injections: int) -> List[Dict[str, object]]:
    base_suite = 'paper' if suite == 'smoke' else suite
    variants = build_variants(base_suite, full_escape_injections)
    keep = {'baseline', 'full', 'wo_init', 'wo_search', 'wo_active_extreme', 'wo_local_escape', 'wo_mtoe', 'base_plus_init', 'base_plus_init_search'}
    if suite == 'smoke':
        keep = {'baseline', 'full', 'wo_init', 'wo_search', 'wo_mtoe'}
    return [v for v in variants if v['name'] in keep]


# ===== Baseline/full suite runner =====
import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import Callable, Dict, List

from src.experiment.path_planning import ensure_dir, now_str
from src.experiment.path_planning import run_reproducibility_sweep


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



def main() -> None:
    args, bench_args = split_args('Run baseline/full suite directly.')
    run_variant_sweep(args, bench_args, build_variants)


if __name__ == '__main__':
    main()
