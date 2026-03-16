from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.env.grid_env import GridEnv


RENDER_TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "render_terrain",
    [
        "#0B132B",
        "#1D4E89",
        "#2A9D8F",
        "#7CB342",
        "#D9C27A",
        "#F3E9D2",
    ],
    N=256,
)


def map_size_to_hw(size: str) -> tuple[int, int]:
    size = (size or "").lower()
    if size in ("s", "small"):
        return 160, 200
    if size in ("m", "medium"):
        return 160 * 3, 200 * 3
    if size in ("l", "large"):
        return 160 * 10, 200 * 10
    raise ValueError(f"Unknown size: {size} (use small/medium/large)")


def parse_meta(data) -> dict:
    if isinstance(data, dict):
        return data
    try:
        if hasattr(data, "item"):
            return data.item()
    except Exception:
        pass
    return {}


def build_env_from_args(args) -> tuple[GridEnv, np.ndarray, dict, str]:
    if args.terrain:
        env, height, meta = GridEnv.load_npz(args.terrain)
        if height is None:
            height = env.height.copy()
        label = Path(args.terrain).stem
        return env, height, meta, label

    terrain_type = args.terrain_type.lower()
    if terrain_type == "mountain":
        H, W = map_size_to_hw(args.size)
        env, height, meta = GridEnv.mountain_map(
            H=H,
            W=W,
            seed=args.seed,
            terrace_levels=args.terrace_levels,
            obstacle_height=args.obstacle_height,
        )
    elif terrain_type == "city":
        env, height, meta = GridEnv.city_map(
            H=1600,
            W=2000,
            seed=args.seed,
            city_density=args.city_density,
        )
    elif terrain_type == "hill_city":
        env, height, meta = GridEnv.hill_city_map(
            seed=args.seed,
            city_density=args.city_density,
            hill_scale=args.hill_scale,
        )
    else:
        raise ValueError(f"Unsupported terrain_type: {terrain_type}")

    label = f"{terrain_type}_seed{args.seed:04d}"
    return env, height, meta, label


def stride_for_limit(shape: tuple[int, int], max_side: int) -> int:
    h, w = shape
    return max(1, int(np.ceil(max(h, w) / float(max_side))))


def apply_crop(arr: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return arr
    x0, x1, y0, y1 = crop
    return arr[y0:y1, x0:x1]


def sanitize_crop(crop_vals, W: int, H: int):
    if crop_vals is None:
        return None
    x0, x1, y0, y1 = crop_vals
    x0 = int(np.clip(x0, 0, W - 1))
    x1 = int(np.clip(x1, x0 + 1, W))
    y0 = int(np.clip(y0, 0, H - 1))
    y1 = int(np.clip(y1, y0 + 1, H))
    return (x0, x1, y0, y1)


def infer_param_density(meta: dict, args) -> float | None:
    if isinstance(meta, dict):
        for key in ("city_density", "param_density", "density_param", "input_density"):
            if key in meta:
                try:
                    return float(meta[key])
                except Exception:
                    pass

    if hasattr(args, "terrain_type") and args.terrain_type in ("city", "hill_city"):
        return float(args.city_density)

    return None


def plot_2d(env: GridEnv, height: np.ndarray, meta: dict, title: str, args, crop=None):
    occ = apply_crop(env.occupancy, crop)
    threat = apply_crop(env.threat, crop)
    hmap = apply_crop(height, crop)
    h_show = np.ma.masked_where(~occ, hmap)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    im0 = axes[0].imshow(
        hmap,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        alpha=0.88,
        cmap=RENDER_TERRAIN_CMAP,
    )
    axes[0].set_title("Height")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Terrain height")

    axes[1].imshow(
        hmap,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        alpha=0.88,
        cmap=RENDER_TERRAIN_CMAP,
    )
    axes[1].imshow(h_show, origin="lower", cmap="magma", alpha=0.80)
    axes[1].set_title("Height + Occupancy")

    im2 = axes[2].imshow(threat, origin="lower", cmap="viridis", interpolation="nearest", aspect="auto")
    axes[2].imshow(np.ma.masked_where(~occ, occ.astype(np.float32)), origin="lower", cmap="gray_r", alpha=0.50)
    axes[2].set_title("Threat + Occupancy")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    actual_occupancy = float(np.mean(env.occupancy))
    zmax = float(np.max(height))
    param_density = infer_param_density(meta, args)

    if param_density is None:
        subtitle = f"shape={env.H}x{env.W}, actual_occupancy={actual_occupancy:.4f}, z_max={zmax:.2f}"
    else:
        subtitle = (
            f"shape={env.H}x{env.W}, "
            f"param_density={param_density:.4f}, "
            f"actual_occupancy={actual_occupancy:.4f}, "
            f"z_max={zmax:.2f}"
        )

    fig.suptitle(
        f"{title}\n{subtitle}",
        fontsize=15,
    )
    return fig


def plot_3d(env: GridEnv, height: np.ndarray, title: str, crop=None, max_side: int = 220):
    hmap = apply_crop(height, crop)
    occ = apply_crop(env.occupancy, crop)

    stride = stride_for_limit(hmap.shape, max_side=max_side)
    hds = hmap[::stride, ::stride]
    occ_ds = occ[::stride, ::stride]
    yy, xx = np.mgrid[0:hds.shape[0], 0:hds.shape[1]]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, hds, cmap="terrain", linewidth=0, antialiased=False, alpha=0.98)

    if np.any(occ_ds):
        z_occ = np.ma.masked_where(~occ_ds, hds + 2.0)
        ax.plot_surface(xx, yy, z_occ, cmap="magma", linewidth=0, antialiased=False, alpha=0.65)

    fig.colorbar(surf, ax=ax, fraction=0.035, pad=0.08, shrink=0.82, label="Terrain height")
    ax.set_title(f"{title}\n3D surface (stride={stride})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_box_aspect((hds.shape[1], hds.shape[0], max(hds.shape) * 0.20))
    ax.view_init(elev=42, azim=-58)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Preview terrain from npz or generate by terrain type + seed.")
    ap.add_argument("--terrain", type=str, default=None, help="existing .npz terrain path")
    ap.add_argument("--terrain_type", type=str, default="hill_city", choices=["mountain", "city", "hill_city"], help="generate terrain on the fly when --terrain is not given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="test_outputs")
    ap.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"], help="mountain only")
    ap.add_argument("--terrace_levels", type=int, default=28)
    ap.add_argument("--obstacle_height", type=float, default=0.78)
    ap.add_argument("--city_density", type=float, default=0.24)
    ap.add_argument("--hill_scale", type=float, default=26.0)
    ap.add_argument("--crop", nargs=4, type=int, metavar=("X0", "X1", "Y0", "Y1"), default=None, help="optional crop box for visualization")
    ap.add_argument("--save_only", action="store_true", help="save figures without opening an interactive window")
    args = ap.parse_args()

    env, height, meta, label = build_env_from_args(args)
    crop = sanitize_crop(args.crop, env.W, env.H)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if crop is None:
        crop_suffix = ""
    else:
        x0, x1, y0, y1 = crop
        crop_suffix = f"_crop{x0}-{x1}_{y0}-{y1}"

    fig2d = plot_2d(env, height, meta, title=label, args=args, crop=crop)
    out2d = out_dir / f"{label}{crop_suffix}_2d.png"
    fig2d.savefig(out2d, dpi=180, bbox_inches="tight")

    fig3d = plot_3d(env, height, title=label, crop=crop)
    out3d = out_dir / f"{label}{crop_suffix}_3d.png"
    fig3d.savefig(out3d, dpi=180, bbox_inches="tight")

    print(f"saved 2D preview: {out2d}")
    print(f"saved 3D preview: {out3d}")
    print("meta:", meta)

    if not args.save_only:
        plt.show()
    else:
        plt.close(fig2d)
        plt.close(fig3d)


if __name__ == "__main__":
    main()