import argparse
import os
from typing import Iterable, Tuple

from src.env.grid_env import GridEnv
from src.experiment.common_io import ensure_dir


CITY_MAP_SHAPE = (1600, 2000)


def map_size_to_hw(size: str) -> Tuple[int, int]:
    size = (size or '').lower()
    if size in ('s', 'small'):
        return 160, 200
    if size in ('m', 'medium'):
        return 160 * 3, 200 * 3
    if size in ('l', 'large'):
        return 160 * 10, 200 * 10
    raise ValueError(f'Unknown size: {size} (use small/medium/large)')


def map_size_name(H: int, W: int) -> str:
    if (H, W) == (160, 200):
        return 'S'
    if (H, W) == (160 * 3, 200 * 3):
        return 'M'
    if (H, W) == (160 * 10, 200 * 10):
        return 'L'
    return f'H{H}W{W}'


def build_generation_parser(single_seed: bool) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='terrains', help='root dir for terrains')
    if single_seed:
        ap.add_argument('--seed', type=int, default=0)
    else:
        ap.add_argument('--seed_from', type=int, default=0)
        ap.add_argument('--seed_to', type=int, default=50, help='exclusive')
    ap.add_argument('--terrain_type', type=str, default='mountain', choices=['mountain', 'city', 'hill_city'])
    ap.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'], help='mountain only')
    ap.add_argument('--terrace_levels', type=int, default=28)
    ap.add_argument('--obstacle_height', type=float, default=0.78)
    ap.add_argument('--city_density', type=float, default=0.24)
    ap.add_argument('--hill_scale', type=float, default=26.0)
    return ap


def resolve_output_layout(args) -> Tuple[int, int, str, str]:
    if args.terrain_type in ('city', 'hill_city'):
        H, W = CITY_MAP_SHAPE
        size_tag = f'{args.terrain_type}_{args.city_density:.2f}'
    else:
        H, W = map_size_to_hw(args.size)
        size_tag = map_size_name(H, W)
    out_subdir = os.path.join(args.out_dir, size_tag)
    ensure_dir(out_subdir)
    return H, W, size_tag, out_subdir


def generate_one(args, seed: int) -> str:
    H, W, size_tag, out_subdir = resolve_output_layout(args)
    if args.terrain_type == 'city':
        env, height, meta = GridEnv.city_map(H=H, W=W, seed=seed, city_density=args.city_density)
        out_path = os.path.join(out_subdir, f'city_seed{seed:04d}.npz')
        extra_meta = {
            'terrain_type': 'city',
            'city_density': args.city_density,
            'map_size': {'H': H, 'W': W, 'tag': size_tag},
        }
    elif args.terrain_type == 'hill_city':
        env, height, meta = GridEnv.hill_city_map(seed=seed, city_density=args.city_density, hill_scale=args.hill_scale)
        out_path = os.path.join(out_subdir, f'hill_city_seed{seed:04d}.npz')
        extra_meta = {
            'terrain_type': 'hill_city',
            'city_density': args.city_density,
            'hill_scale': args.hill_scale,
            'map_size': {'H': H, 'W': W, 'tag': size_tag},
        }
    else:
        env, height, meta = GridEnv.mountain_map(H=H, W=W, seed=seed, terrace_levels=args.terrace_levels, obstacle_height=args.obstacle_height)
        out_path = os.path.join(out_subdir, f'mountain_seed{seed:04d}.npz')
        extra_meta = {
            'terrain_type': 'mountain',
            'map_size': {'H': H, 'W': W, 'tag': size_tag},
        }
    env.save_npz(out_path, height=height, meta={**meta, **extra_meta})
    return out_path


def generate_many(args, seeds: Iterable[int]) -> Tuple[int, int, str, str, list[str]]:
    H, W, size_tag, out_subdir = resolve_output_layout(args)
    out_paths = []
    for seed in seeds:
        out_paths.append(generate_one(args, int(seed)))
    return H, W, size_tag, out_subdir, out_paths
