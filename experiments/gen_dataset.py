# experiments/gen_dataset.py

import os
import argparse
from src.env.grid_env import GridEnv


def map_size_to_hw(size: str):
    size = (size or "").lower()
    if size in ("s", "small"):
        return 160, 200
    if size in ("m", "medium"):
        # 中地图：小地图的 3 倍
        return 160 * 3, 200 * 3
    if size in ("l", "large"):
        # 大地图：小地图的 10 倍
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
    ap.add_argument("--seed_from", type=int, default=0)
    ap.add_argument("--seed_to", type=int, default=50, help="exclusive")

    ap.add_argument(
        "--size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="preset map size (small=160x200, medium=480x600, large=1600x2000)",
    )
    ap.add_argument("--H", type=int, default=0, help="override height (0 means use preset)")
    ap.add_argument("--W", type=int, default=0, help="override width (0 means use preset)")

    ap.add_argument("--terrace_levels", type=int, default=28)
    ap.add_argument("--obstacle_height", type=float, default=0.78)

    args = ap.parse_args()

    # 选 H/W：优先 override
    if args.H > 0 and args.W > 0:
        H, W = args.H, args.W
    else:
        H, W = map_size_to_hw(args.size)

    size_tag = map_size_name(H, W)

    # 自动分桶到 terrains/S|M|L（或自定义 H/W 则 HxxxWyyy）
    out_subdir = os.path.join(args.out_dir, size_tag)
    os.makedirs(out_subdir, exist_ok=True)

    total = 0
    for seed in range(args.seed_from, args.seed_to):
        env, height, meta = GridEnv.mountain_map(
            H=H, W=W,
            seed=seed,
            terrace_levels=args.terrace_levels,
            obstacle_height=args.obstacle_height,
        )

        out_path = os.path.join(out_subdir, f"mountain_seed{seed:04d}.npz")
        env.save_npz(
            out_path,
            height=height,
            meta={**meta, "map_size": {"H": H, "W": W, "tag": size_tag}},
        )
        print("saved:", out_path)
        total += 1

    print(f"Done. Generated {total} terrains into {out_subdir}")
    print("map_size:", {"H": H, "W": W, "tag": size_tag})


if __name__ == "__main__":
    main()
