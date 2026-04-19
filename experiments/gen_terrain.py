<<<<<<< HEAD
from src.experiment.terrain_generation import build_generation_parser, generate_one, resolve_output_layout


def main():
    ap = build_generation_parser(single_seed=True)
    args = ap.parse_args()
    H, W, size_tag, _ = resolve_output_layout(args)
    out_path = generate_one(args, args.seed)
    print('saved:', out_path, flush=True)
    print('map_size:', {'H': H, 'W': W, 'tag': size_tag}, flush=True)
    print('terrain_type:', args.terrain_type, flush=True)


if __name__ == '__main__':
=======
import os
import argparse
from src.env.grid_env import GridEnv


def map_size_to_hw(size: str):
    size = (size or "").lower()
    if size in ("s", "small"):
        return 160, 200
    if size in ("m", "medium"):
        return 160 * 3, 200 * 3
    if size in ("l", "large"):
        return 160 * 10, 200 * 10
    raise ValueError(f"Unknown size: {size} (use small/medium/large)")


def map_size_name(H: int, W: int) -> str:
    if (H, W) == (160, 200):
        return "S"
    if (H, W) == (160 * 3, 200 * 3):
        return "M"
    if (H, W) == (160 * 10, 200 * 10):
        return "L"
    return f"H{H}W{W}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="terrains", help="root dir for terrains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain_type", type=str, default="mountain", choices=["mountain", "city", "hill_city"])

    # mountain only
    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"], help="mountain only")
    ap.add_argument("--terrace_levels", type=int, default=28)
    ap.add_argument("--obstacle_height", type=float, default=0.78)

    # city / hill_city only
    ap.add_argument("--city_density", type=float, default=0.24)
    ap.add_argument("--hill_scale", type=float, default=26.0)

    args = ap.parse_args()

    if args.terrain_type in ("city", "hill_city"):
        H, W = 1600, 2000
        size_tag = f"{args.terrain_type}_{args.city_density:.2f}"
        out_subdir = os.path.join(args.out_dir, size_tag)
        os.makedirs(out_subdir, exist_ok=True)
        if args.terrain_type == "city":
            env, height, meta = GridEnv.city_map(H=H, W=W, seed=args.seed, city_density=args.city_density)
            out_path = os.path.join(out_subdir, f"city_seed{args.seed:04d}.npz")
        else:
            env, height, meta = GridEnv.hill_city_map(seed=args.seed, city_density=args.city_density, hill_scale=args.hill_scale)
            out_path = os.path.join(out_subdir, f"hill_city_seed{args.seed:04d}.npz")
        extra_meta = {"terrain_type": args.terrain_type, "city_density": args.city_density, "hill_scale": (args.hill_scale if args.terrain_type == "hill_city" else None), "map_size": {"H": H, "W": W, "tag": size_tag}}
    else:
        H, W = map_size_to_hw(args.size)
        size_tag = map_size_name(H, W)
        out_subdir = os.path.join(args.out_dir, size_tag)
        os.makedirs(out_subdir, exist_ok=True)
        env, height, meta = GridEnv.mountain_map(H=H, W=W, seed=args.seed, terrace_levels=args.terrace_levels, obstacle_height=args.obstacle_height)
        out_path = os.path.join(out_subdir, f"mountain_seed{args.seed:04d}.npz")
        extra_meta = {"terrain_type": "mountain", "map_size": {"H": H, "W": W, "tag": size_tag}}

    env.save_npz(out_path, height=height, meta={**meta, **extra_meta})
    print("saved:", out_path)
    print("map_size:", {"H": H, "W": W, "tag": size_tag})
    print("terrain_type:", args.terrain_type)


if __name__ == "__main__":
>>>>>>> origin/feature/3d
    main()
