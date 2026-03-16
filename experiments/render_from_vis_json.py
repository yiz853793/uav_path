# experiments/render_from_vis_json.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_path_array(path_like: Any) -> Optional[np.ndarray]:
    if path_like is None:
        return None
    arr = np.asarray(path_like, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr


def stem_without_prefix(path: Path) -> str:
    return path.stem


def safe_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(**kwargs)


def normalize_path_str(p: str) -> str:
    return str(p).replace("\\", "/").strip()


def try_load_terrain_surface(
    vis: Dict[str, Any], json_path: Path, project_root: Path
) -> Optional[np.ndarray]:
    terrain_file = vis.get("terrain_file", None)
    if not terrain_file:
        return None

    terrain_file = normalize_path_str(terrain_file)

    candidates = [
        project_root / terrain_file,
        json_path.parent / terrain_file,
        Path(terrain_file),
    ]

    terrain_path = None
    for c in candidates:
        if c.exists():
            terrain_path = c
            break

    if terrain_path is None:
        return None

    try:
        data = np.load(terrain_path, allow_pickle=True)
    except Exception:
        return None

    candidate_keys = [
        "Z", "z", "height", "heights", "height_map", "heightmap",
        "terrain", "terrain_z", "surface", "surface_z", "map"
    ]

    for k in candidate_keys:
        if k in data:
            arr = np.asarray(data[k], dtype=float)
            if arr.ndim == 2:
                return arr

    for k in data.files:
        arr = np.asarray(data[k])
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)

    return None


def get_map_size(vis: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    map_size = vis.get("map_size", None)
    if isinstance(map_size, dict):
        return map_size.get("H", None), map_size.get("W", None)

    meta = vis.get("meta", None)
    if isinstance(meta, dict):
        meta_map_size = meta.get("map_size", None)
        if isinstance(meta_map_size, dict):
            return meta_map_size.get("H", None), meta_map_size.get("W", None)

    map_hw = vis.get("map_hw", None)
    if isinstance(map_hw, (list, tuple)) and len(map_hw) >= 2:
        return int(map_hw[0]), int(map_hw[1])

    return None, None


def extract_paths(vis: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # baseline planners
    planners = vis.get("planners", None)
    if isinstance(planners, dict):
        for key in ["rrt", "prm"]:
            item = planners.get(key, None)
            if isinstance(item, dict):
                path = item.get("path", None)
                if path is not None:
                    arr = np.asarray(path, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        out[key] = arr

    # MOEA/D representatives
    representatives = vis.get("representatives", None)
    if isinstance(representatives, dict):
        for key in ["min_f1", "min_f2", "min_f3", "compromise"]:
            item = representatives.get(key, None)
            if isinstance(item, dict):
                path = item.get("path", None)
                if path is not None:
                    arr = np.asarray(path, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        out[key] = arr

    return out


def extract_pareto(vis: Dict[str, Any]) -> Optional[np.ndarray]:
    archive_points = vis.get("archive_points", None)
    if archive_points is not None:
        arr = np.asarray(archive_points, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
            return arr
    return None


PATH_STYLE = {
    "rrt": dict(label="RRT", linewidth=2.6, alpha=0.95),
    "prm": dict(label="PRM", linewidth=2.6, alpha=0.95),
    "min_f1": dict(label="MOEA/D min_f1", linewidth=2.0, alpha=0.92),
    "min_f2": dict(label="MOEA/D min_f2", linewidth=2.0, alpha=0.92),
    "min_f3": dict(label="MOEA/D min_f3", linewidth=2.0, alpha=0.92),
    "compromise": dict(label="MOEA/D compromise", linewidth=3.2, alpha=1.0),
}


def build_title(vis: Dict[str, Any], suffix: str = "") -> str:
    title = vis.get("title", "Path Planning Visualization")
    if suffix:
        title += f"\n{suffix}"
    return title


def save_paths_2d(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
    terrain_z: Optional[np.ndarray],
) -> Path:
    out_path = out_dir / f"{stem_without_prefix(json_path)}_paths2d.png"

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)

    if terrain_z is not None:
        im = ax.imshow(
            terrain_z,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            alpha=0.88,
            cmap="terrain",   # 仅修改地形颜色
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Terrain height")

    paths = extract_paths(vis)
    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        ax.plot(path[:, 0], path[:, 1], **st)

    start = np.asarray(vis.get("start", [0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0]), dtype=float)

    if start.size >= 2:
        ax.scatter(start[0], start[1], marker="o", s=90, label="Start", zorder=6)
    if goal.size >= 2:
        ax.scatter(goal[0], goal[1], marker="*", s=180, label="Goal", zorder=6)

    H, W = get_map_size(vis)
    if W is not None and H is not None:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(build_title(vis, "2D Path View"))
    ax.grid(True, alpha=0.22)
    safe_legend(ax, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_paths_3d_preview(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
    terrain_z: Optional[np.ndarray],
) -> Path:
    out_path = out_dir / f"{stem_without_prefix(json_path)}_paths3d_preview.png"

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if terrain_z is not None:
        H, W = terrain_z.shape
        step_h = max(1, H // 80)
        step_w = max(1, W // 80)
        ys = np.arange(0, H, step_h)
        xs = np.arange(0, W, step_w)
        XX, YY = np.meshgrid(xs, ys)
        ZZ = terrain_z[np.ix_(ys, xs)]
        ax.plot_surface(
            XX, YY, ZZ,
            linewidth=0,
            antialiased=True,
            alpha=0.30,
            cmap="terrain",   # 仅修改地形颜色
        )

    paths = extract_paths(vis)
    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        z = path[:, 2] if path.shape[1] >= 3 else np.zeros(len(path))
        ax.plot(
            path[:, 0],
            path[:, 1],
            z,
            label=st.get("label", name),
            linewidth=st.get("linewidth", 2.0),
            alpha=st.get("alpha", 0.95),
        )

    start = np.asarray(vis.get("start", [0, 0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0, 0]), dtype=float)

    if start.size >= 3:
        ax.scatter(start[0], start[1], start[2], marker="o", s=90, label="Start")
    if goal.size >= 3:
        ax.scatter(goal[0], goal[1], goal[2], marker="*", s=180, label="Goal")

    H, W = get_map_size(vis)
    if W is not None and H is not None:
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(build_title(vis, "3D Path Preview"))
    safe_legend(ax, loc="best", fontsize=8)

    ax.view_init(elev=35, azim=-60)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pareto_3d(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
) -> Path:
    out_path = out_dir / f"{stem_without_prefix(json_path)}_pareto3d.png"
    pts = extract_pareto(vis)

    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    if pts is not None and pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=28, alpha=0.85, label="Pareto points")
    else:
        ax.text2D(0.5, 0.5, "No Pareto data available", transform=ax.transAxes, ha="center")

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.set_title(build_title(vis, "Pareto Front (3D)"))
    safe_legend(ax, loc="best", fontsize=9)
    ax.view_init(elev=24, azim=-58)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_interactive_paths(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
    project_root: Path,
) -> Path:
    import plotly.graph_objects as go

    out_path = out_dir / f"{stem_without_prefix(json_path)}_interactive_paths.html"

    terrain_z = try_load_terrain_surface(vis, json_path, project_root)
    paths = extract_paths(vis)

    fig = go.Figure()

    if terrain_z is not None:
        H, W = terrain_z.shape
        ys = np.arange(H)
        xs = np.arange(W)
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=terrain_z,
                opacity=0.50,
                showscale=True,
                name="terrain",
                colorscale=[   # 仅修改地形颜色
                    [0.00, "#0B132B"],
                    [0.18, "#1D4E89"],
                    [0.38, "#2A9D8F"],
                    [0.58, "#7CB342"],
                    [0.78, "#D9C27A"],
                    [1.00, "#F3E9D2"],
                ],
            )
        )

    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        z = path[:, 2] if path.shape[1] >= 3 else np.zeros(len(path))
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=z,
                mode="lines+markers",
                name=st.get("label", name),
                marker=dict(size=3),
                line=dict(width=8 if name == "compromise" else 5),
            )
        )

    start = np.asarray(vis.get("start", [0, 0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0, 0]), dtype=float)

    if start.size >= 3:
        fig.add_trace(go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode="markers", name="Start",
            marker=dict(size=6, symbol="circle")
        ))

    if goal.size >= 3:
        fig.add_trace(go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]],
            mode="markers", name="Goal",
            marker=dict(size=8, symbol="diamond")
        ))

    fig.update_layout(
        title=build_title(vis, "Interactive 3D Paths"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            domain=dict(x=[0.0, 0.82], y=[0.0, 1.0]),

            # 关键：手动控制三轴显示比例
            aspectmode="manual",
            aspectratio=dict(
                x=1.0,
                y=1.25,
                z=0.20,   # 越小越扁平，建议先试 0.08~0.12
            ),

            # 关键：相机视角稍微压低
            camera=dict(
                eye=dict(x=1.55, y=1.55, z=0.55)
            ),
        ),
        legend=dict(
            x=0.84,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def save_interactive_pareto(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
) -> Path:
    import plotly.graph_objects as go

    out_path = out_dir / f"{stem_without_prefix(json_path)}_pareto3d.html"
    pts = extract_pareto(vis)

    fig = go.Figure()

    if pts is not None and pts.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name="Pareto points",
                marker=dict(size=4),
            )
        )

        pad_x = max(0.3, 0.08 * float(np.ptp(pts[:, 0])))
        pad_y = max(0.3, 0.08 * float(np.ptp(pts[:, 1])))
        pad_z = max(0.3, 0.08 * float(np.ptp(pts[:, 2])))

        x_range = [float(np.min(pts[:, 0]) - pad_x), float(np.max(pts[:, 0]) + pad_x)]
        y_range = [float(np.min(pts[:, 1]) - pad_y), float(np.max(pts[:, 1]) + pad_y)]
        z_range = [float(np.min(pts[:, 2]) - pad_z), float(np.max(pts[:, 2]) + pad_z)]
    else:
        x_range = None
        y_range = None
        z_range = None

    fig.update_layout(
        title=build_title(vis, "Interactive Pareto Front (3D)"),
        scene=dict(
            xaxis=dict(
                title="f1",
                range=x_range,
                showbackground=True,
                backgroundcolor="rgb(245,245,245)",
                gridcolor="rgb(200,200,200)",
                zerolinecolor="rgb(160,160,160)",
                showspikes=False,
            ),
            yaxis=dict(
                title="f2",
                range=y_range,
                showbackground=True,
                backgroundcolor="rgb(245,245,245)",
                gridcolor="rgb(200,200,200)",
                zerolinecolor="rgb(160,160,160)",
                showspikes=False,
            ),
            zaxis=dict(
                title="f3",
                range=z_range,
                showbackground=True,
                backgroundcolor="rgb(245,245,245)",
                gridcolor="rgb(200,200,200)",
                zerolinecolor="rgb(160,160,160)",
                showspikes=False,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.9),
            camera=dict(
                eye=dict(x=1.45, y=1.35, z=0.95)
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path

def save_static(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
    project_root: Path,
):
    terrain_z = try_load_terrain_surface(vis, json_path, project_root)

    p1 = save_paths_2d(vis, json_path, out_dir, terrain_z)
    p2 = save_paths_3d_preview(vis, json_path, out_dir, terrain_z)
    p3 = save_pareto_3d(vis, json_path, out_dir)
    return p1, p2, p3


def collect_json_files(input_path: Path):
    if input_path.is_file():
        if input_path.name.startswith("visdata_") and input_path.suffix.lower() == ".json":
            return [input_path]
        return []
    return sorted(input_path.rglob("visdata_*.json"))


def parse_args():
    parser = argparse.ArgumentParser(description="Render benchmark visdata json files.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["static", "interactive", "all"],
    )
    parser.add_argument("--project_root", type=str, default=".")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    project_root = Path(args.project_root).resolve()

    json_files = collect_json_files(input_path)
    print(f"Found {len(json_files)} json file(s).")

    if not json_files:
        print("No visdata_*.json found.")
        return

    for jp in json_files:
        print(f"[render] {jp}")

        try:
            vis = load_json(jp)
        except Exception as e:
            print(f"  [skip] failed to load json: {e}")
            continue

        out_dir = jp.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode in ("static", "all"):
            try:
                p1, p2, p3 = save_static(vis, jp, out_dir, project_root)
                print(f"  static -> {p1}")
                print(f"  static -> {p2}")
                print(f"  static -> {p3}")
            except Exception as e:
                print(f"  [static failed] {e}")

        if args.mode in ("interactive", "all"):
            try:
                p4 = save_interactive_paths(vis, jp, out_dir, project_root)
                p5 = save_interactive_pareto(vis, jp, out_dir)
                print(f"  interactive -> {p4}")
                print(f"  interactive -> {p5}")
            except Exception as e:
                print(f"  [interactive failed] {e}")


if __name__ == "__main__":
    main()