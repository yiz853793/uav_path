from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

from src.experiment.path_planning import ensure_dir, parse_benchmark_args, run_single_case


def split_args(argv: List[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Run one planning case from an existing terrain .npz file.",
    )
    parser.add_argument("--terrain", required=True, help="path to one .npz terrain file")
    parser.add_argument("--out_root", "-o", default="outputs", help="output root")
    parser.add_argument("--single_case_out_dir", default="", help="write this case directly into this directory")
    parser.add_argument("--planner_seed", "--seed", dest="planner_seed", type=int, default=0)
    parser.add_argument("--out_json", default="", help="optional compatibility alias for the generated visdata json path")
    parser.add_argument("--help", "-h", action="store_true")
    front, bench_args = parser.parse_known_args(argv)
    if front.help:
        parser.print_help()
        print("\nForwarded benchmark-style options include --profile, --inflate, --rrt_iter, --prm_samples, --moead_pop, --moead_max_gen, --K, --moead_T, etc.")
        raise SystemExit(0)
    if bench_args and bench_args[0] == "--":
        bench_args = bench_args[1:]
    return front, bench_args


def build_benchmark_args(front: argparse.Namespace, bench_args: List[str]):
    forwarded = list(bench_args)
    forwarded += ["--out_root", str(front.out_root)]
    forwarded += ["--planner_seed", str(front.planner_seed)]
    if str(front.single_case_out_dir).strip():
        forwarded += ["--single_case_out_dir", str(front.single_case_out_dir).strip()]
    args = parse_benchmark_args(forwarded)
    args.terrain_dir = str(Path(front.terrain).resolve().parent)
    args.glob = str(Path(front.terrain).resolve())
    return args


def maybe_copy_vis_json(case_dir: str, out_json: str) -> None:
    if not out_json:
        return
    candidates = sorted(Path(case_dir).glob("visdata_*.json"))
    if not candidates:
        return
    target = Path(out_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], target)


def main(argv: List[str] | None = None) -> None:
    front, bench_args = split_args(list(sys.argv[1:] if argv is None else argv))
    terrain_path = Path(front.terrain)
    if not terrain_path.exists():
        raise FileNotFoundError(str(terrain_path))

    bench = build_benchmark_args(front, bench_args)
    out_dir_override = str(front.single_case_out_dir).strip() or None
    row = run_single_case(bench, str(terrain_path), int(front.planner_seed), out_dir_override=out_dir_override)

    case_dir = out_dir_override
    if case_dir is None:
        size_tag = row.get("size_tag") or "case"
        terrain_seed = row.get("terrain_seed")
        if terrain_seed is None:
            case_dir = os.path.join(str(front.out_root), str(size_tag), terrain_path.stem)
        else:
            case_dir = os.path.join(str(front.out_root), str(size_tag), f"seed{int(terrain_seed):04d}")
    ensure_dir(case_dir)
    maybe_copy_vis_json(case_dir, str(front.out_json).strip())

    print("[plan_from_terrain] done")
    print("case_dir:", case_dir)
    if front.out_json:
        print("vis_json_copy:", front.out_json)


if __name__ == "__main__":
    main()
