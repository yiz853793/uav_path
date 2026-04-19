import argparse
from typing import List

from src.experiment.benchmark_shared import apply_run_profile


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
