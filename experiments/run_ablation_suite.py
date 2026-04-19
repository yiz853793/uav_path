from src.experiment.suite_runner import run_variant_sweep, split_args
from src.experiment.suite_variants import build_ablation_variants


def main() -> None:
    args, bench_args = split_args('Run ablation-focused suite directly (shared direct-call runner).')
    run_variant_sweep(args, bench_args, build_ablation_variants)


if __name__ == '__main__':
    main()
