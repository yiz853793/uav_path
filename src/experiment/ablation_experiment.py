from typing import Dict, List

from src.experiment.baseline_experiment import build_variants, run_variant_sweep, split_args


def build_ablation_variants(suite: str, full_escape_injections: int) -> List[Dict[str, object]]:
    base_suite = 'paper' if suite == 'smoke' else suite
    variants = build_variants(base_suite, full_escape_injections)
    keep = {
        'baseline', 'full', 'wo_init', 'wo_search', 'wo_active_extreme',
        'wo_local_escape', 'wo_mtoe', 'base_plus_init', 'base_plus_init_search',
    }
    if suite == 'smoke':
        keep = {'baseline', 'full', 'wo_init', 'wo_search', 'wo_mtoe'}
    return [v for v in variants if v['name'] in keep]


def main() -> None:
    args, bench_args = split_args('Run ablation-focused suite directly.')
    run_variant_sweep(args, bench_args, build_ablation_variants)


if __name__ == '__main__':
    main()
