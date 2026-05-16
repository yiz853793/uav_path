# experiments/vis_data.py
# -*- coding: utf-8 -*-


import json
from typing import Any, Dict, Optional

import numpy as np


def _to_serializable(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _path_to_list(path):
    if path is None:
        return None
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2:
        return None
    return arr.tolist()


def _point_to_list(pt):
    if pt is None:
        return None
    arr = np.asarray(pt, dtype=float).reshape(-1)
    if arr.size < 2:
        return None
    return arr.tolist()


def _repr_item_from_arch_item(arch_item) -> Dict[str, Any]:
    # 鍏抽敭淇锛歁OEA/D 鐨?Individual 璺緞瀛楁鏄?x锛屼笉鏄?path
    path = getattr(arch_item, "x", None)

    return {
        "path": _path_to_list(path),
        "f": [float(v) for v in arch_item.er.obj],
        "feasible": bool(arch_item.er.feasible),
        "violation": float(arch_item.er.violation),
        "detail": {k: float(v) for k, v in arch_item.er.detail.items()},
    }


def _repr_item_from_baseline_item(item: Dict[str, Any]) -> Dict[str, Any]:
    obj = item.get("obj") if isinstance(item, dict) else None
    detail = item.get("detail") if isinstance(item, dict) else None
    out: Dict[str, Any] = {
        "path": _path_to_list(item.get("path") if isinstance(item, dict) else None),
        "f": [float(v) for v in (obj or [])[:3]],
        "feasible": bool(item.get("feasible", False)) if isinstance(item, dict) else False,
        "violation": item.get("violation") if isinstance(item, dict) else None,
        "runtime_ms": item.get("runtime_ms") if isinstance(item, dict) else None,
    }
    if isinstance(detail, dict):
        out["detail"] = {k: float(v) for k, v in detail.items()}
    return out


def build_vis_payload(
    terrain_file: str,
    terrain_seed: Optional[int],
    planner_seed: int,
    size_tag: str,
    inflate: int,
    map_hw,
    start,
    goal,
    path_rrt=None,
    path_improved_rrt=None,
    path_prm=None,
    rrt_reps: Optional[Dict[str, Any]] = None,
    prm_reps: Optional[Dict[str, Any]] = None,
    rrt_archive_stats: Optional[Dict[str, Any]] = None,
    prm_archive_stats: Optional[Dict[str, Any]] = None,
    arch=None,
    reps=None,
    title: str = "",
    meta: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {
        "terrain_file": terrain_file,
        "terrain_seed": terrain_seed,
        "planner_seed": planner_seed,
        "size_tag": size_tag,
        "inflate": inflate,
        "map_hw": [int(map_hw[0]), int(map_hw[1])],
        "start": _point_to_list(start),
        "goal": _point_to_list(goal),
        "title": title,
        "planners": {
            "rrt": {
                "path": _path_to_list(path_rrt),
                "archive_stats": dict(rrt_archive_stats or {}),
                "representatives": {
                    key: _repr_item_from_baseline_item(item)
                    for key, item in (rrt_reps or {}).items()
                    if key in {"min_f1", "min_f2", "min_f3", "compromise"}
                },
            },
            "improved_rrt": {
                "path": _path_to_list(path_improved_rrt),
            },
            "prm": {
                "path": _path_to_list(path_prm),
                "archive_stats": dict(prm_archive_stats or {}),
                "representatives": {
                    key: _repr_item_from_baseline_item(item)
                    for key, item in (prm_reps or {}).items()
                    if key in {"min_f1", "min_f2", "min_f3", "compromise"}
                },
            },
        },
    }
    if payload["planners"]["rrt"]["archive_stats"]:
        payload["planners"]["rrt"]["archive_stats"]["representative_unique_count"] = len({
            tuple(round(float(v), 6) for v in item.get("obj", [])[:3])
            for item in (rrt_reps or {}).values()
            if isinstance(item, dict) and isinstance(item.get("obj"), (list, tuple)) and len(item.get("obj")) >= 3
        })
    if payload["planners"]["prm"]["archive_stats"]:
        payload["planners"]["prm"]["archive_stats"]["representative_unique_count"] = len({
            tuple(round(float(v), 6) for v in item.get("obj", [])[:3])
            for item in (prm_reps or {}).values()
            if isinstance(item, dict) and isinstance(item.get("obj"), (list, tuple)) and len(item.get("obj")) >= 3
        })

    # 涓嶅啀淇濆瓨 archive 鏄庣粏锛屽彧淇濈暀 3 鐩爣鐐?
    archive_points = []
    if arch is not None and hasattr(arch, "items") and len(arch.items) > 0:
        for item in arch.items:
            archive_points.append([float(v) for v in item.er.obj])
    payload["archive_points"] = archive_points

    # representatives锛氫繚瀛樺洓涓唬琛ㄨВ锛屽苟涓?path 蹇呴』浠?x 璇诲彇
    representatives: Dict[str, Any] = {}
    if reps is not None and arch is not None and hasattr(arch, "items") and len(arch.items) > 0:
        for key in ["min_f1", "min_f2", "min_f3", "compromise"]:
            if key in reps:
                idx = int(reps[key])
                arch_item = arch.items[idx]
                representatives[key] = _repr_item_from_arch_item(arch_item)

    payload["representatives"] = representatives

    if meta is not None:
        payload["meta"] = meta

    if extra is not None:
        for k, v in extra.items():
            payload[k] = _to_serializable(v)

    return payload


def save_vis_payload(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ===== Render planning-result terrain/path figures =====
# src/experiment/terrain_result_plotting.py
# -*- coding: utf-8 -*-


import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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


def get_xy_resolution(vis: Dict[str, Any]) -> float:
    map_size = vis.get("map_size", None)
    if isinstance(map_size, dict):
        if "xy_resolution_m" in map_size:
            return float(map_size["xy_resolution_m"])
        if "resolution" in map_size:
            return float(map_size["resolution"])

    meta = vis.get("meta", None)
    if isinstance(meta, dict):
        if "xy_resolution_m" in meta:
            return float(meta["xy_resolution_m"])
        if "resolution" in meta:
            return float(meta["resolution"])

        meta_map_size = meta.get("map_size", None)
        if isinstance(meta_map_size, dict):
            if "xy_resolution_m" in meta_map_size:
                return float(meta_map_size["xy_resolution_m"])
            if "resolution" in meta_map_size:
                return float(meta_map_size["resolution"])

    return 5.0


def to_metric_xy(path: np.ndarray, xy_res: float) -> np.ndarray:
    arr = np.asarray(path, dtype=float).copy()
    if arr.ndim == 2 and arr.shape[1] >= 2:
        arr[:, 0] *= xy_res
        arr[:, 1] *= xy_res
    return arr


def extract_paths(vis: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    planners = vis.get("planners", None)
    if isinstance(planners, dict):
        for key in ["rrt", "improved_rrt", "prm"]:
            item = planners.get(key, None)
            if isinstance(item, dict):
                planner_reps = item.get("representatives", None)
                has_reps = isinstance(planner_reps, dict) and len(planner_reps) > 0
                path = item.get("path", None)
                if path is not None and not has_reps:
                    arr = np.asarray(path, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        out[key] = arr
                if isinstance(planner_reps, dict):
                    for rep_key in ["min_f1", "min_f2", "min_f3", "compromise"]:
                        rep_item = planner_reps.get(rep_key, None)
                        if not isinstance(rep_item, dict):
                            continue
                        rep_path = rep_item.get("path", None)
                        if rep_path is None:
                            continue
                        arr = np.asarray(rep_path, dtype=float)
                        if arr.ndim == 2 and arr.shape[1] >= 2:
                            out[f"{key}_{rep_key}"] = arr

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


def extract_representative_points(vis: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    representatives = vis.get("representatives", None)
    if not isinstance(representatives, dict):
        return out

    for key in ["min_f1", "min_f2", "min_f3", "compromise"]:
        item = representatives.get(key, None)
        if not isinstance(item, dict):
            continue

        f = item.get("f", None)
        if f is None:
            continue

        arr = np.asarray(f, dtype=float).reshape(-1)
        if arr.size >= 3:
            out[key] = arr[:3]

    return out


def representative_display_name(key: str) -> str:
    mapping = {
        "min_f1": "min_f1",
        "min_f2": "min_f2",
        "min_f3": "min_f3",
        "compromise": "compromise",
    }
    return mapping.get(key, key)


def representative_color(key: str) -> str:
    mapping = {
        "min_f1": "#7B61FF",
        "min_f2": "#F2994A",
        "min_f3": "#2D9CDB",
        "compromise": "#FF5C8A",
    }
    return mapping.get(key, "#111111")


PATH_STYLE = {
    "rrt": dict(label="RRT*", linewidth=2.6, alpha=0.95),
    "improved_rrt": dict(label="Improved RRT*", linewidth=2.8, alpha=0.95, linestyle="-."),
    "prm": dict(label="PRM", linewidth=2.6, alpha=0.95),
    "rrt_min_f1": dict(label="RRT* min_f1", linewidth=2.4, alpha=0.95),
    "rrt_min_f2": dict(label="RRT* min_f2", linewidth=2.2, alpha=0.88, linestyle="--"),
    "rrt_min_f3": dict(label="RRT* min_f3", linewidth=2.2, alpha=0.88, linestyle=":"),
    "rrt_compromise": dict(label="RRT* compromise", linewidth=2.7, alpha=0.95, linestyle="-."),
    "prm_min_f1": dict(label="PRM min_f1", linewidth=2.4, alpha=0.95),
    "prm_min_f2": dict(label="PRM min_f2", linewidth=2.2, alpha=0.88, linestyle="--"),
    "prm_min_f3": dict(label="PRM min_f3", linewidth=2.2, alpha=0.88, linestyle=":"),
    "prm_compromise": dict(label="PRM compromise", linewidth=2.7, alpha=0.95, linestyle="-."),
    "min_f1": dict(label="MOEA/D min_f1", linewidth=2.1, alpha=0.92),
    "min_f2": dict(label="MOEA/D min_f2", linewidth=2.1, alpha=0.92),
    "min_f3": dict(label="MOEA/D min_f3", linewidth=2.1, alpha=0.92),
    "compromise": dict(label="MOEA/D compromise", linewidth=3.4, alpha=1.0),
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
    xy_res = get_xy_resolution(vis)

    if terrain_z is not None:
        Ht, Wt = terrain_z.shape
        im = ax.imshow(
            terrain_z,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
            alpha=0.86,
            cmap="terrain",
            extent=[0, Wt * xy_res, 0, Ht * xy_res],
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Terrain height (m)")

    paths = extract_paths(vis)
    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        path_m = to_metric_xy(path, xy_res)
        ax.plot(path_m[:, 0], path_m[:, 1], **st)

    start = np.asarray(vis.get("start", [0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0]), dtype=float)

    if start.size >= 2:
        ax.scatter(start[0] * xy_res, start[1] * xy_res, marker="o", s=70, label="Start", zorder=6)
    if goal.size >= 2:
        ax.scatter(goal[0] * xy_res, goal[1] * xy_res, marker="*", s=140, label="Goal", zorder=6)

    H, W = get_map_size(vis)
    if W is not None and H is not None:
        ax.set_xlim(0, W * xy_res)
        ax.set_ylim(0, H * xy_res)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(build_title(vis, "2D Path View"))
    ax.grid(True, alpha=0.18)
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
    xy_res = get_xy_resolution(vis)

    if terrain_z is not None:
        Ht, Wt = terrain_z.shape
        step_h = max(1, Ht // 80)
        step_w = max(1, Wt // 80)
        ys = np.arange(0, Ht, step_h)
        xs = np.arange(0, Wt, step_w)
        XX, YY = np.meshgrid(xs * xy_res, ys * xy_res)
        ZZ = terrain_z[np.ix_(ys, xs)]
        ax.plot_surface(
            XX, YY, ZZ,
            linewidth=0,
            antialiased=True,
            alpha=0.24,
            cmap="terrain",
        )

    paths = extract_paths(vis)
    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        path_m = to_metric_xy(path, xy_res)
        z = path[:, 2] if path.shape[1] >= 3 else np.zeros(len(path))
        ax.plot(
            path_m[:, 0],
            path_m[:, 1],
            z,
            label=st.get("label", name),
            linewidth=st.get("linewidth", 2.0),
            alpha=st.get("alpha", 0.95),
        )

    start = np.asarray(vis.get("start", [0, 0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0, 0]), dtype=float)

    if start.size >= 3:
        ax.scatter(start[0] * xy_res, start[1] * xy_res, start[2], marker="o", s=70, label="Start")
    if goal.size >= 3:
        ax.scatter(goal[0] * xy_res, goal[1] * xy_res, goal[2], marker="*", s=140, label="Goal")

    H, W = get_map_size(vis)
    if W is not None and H is not None:
        ax.set_xlim(0, W * xy_res)
        ax.set_ylim(0, H * xy_res)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(build_title(vis, "3D Path Preview"))
    safe_legend(ax, loc="best", fontsize=8)

    ax.set_box_aspect((1.0, 1.25, 0.20))
    ax.view_init(elev=30, azim=-58)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_pareto_2d_f1_f3_color_f2(
    vis: Dict[str, Any],
    json_path: Path,
    out_dir: Path,
) -> Path:
    out_path = out_dir / f"{stem_without_prefix(json_path)}_pareto2d_f1_f3_color_f2.png"
    pts = extract_pareto(vis)
    reps = extract_representative_points(vis)

    fig, ax = plt.subplots(figsize=(8.8, 6.8), dpi=180)

    if pts is None or pts.shape[0] == 0:
        ax.text(0.5, 0.5, "No Pareto data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(build_title(vis, "Pareto Front (f1-f3, color=f2)"))
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    vmin = float(np.min(pts[:, 1]))
    vmax = float(np.max(pts[:, 1]))

    # 璁╅鑹插彉鍖栨洿鍓х儓锛氱敤 PowerNorm 鎷夊紑宸紓
    norm = mcolors.PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")   # 姣?viridis 鏇粹€滄縺鐑堚€?

    sc = ax.scatter(
        pts[:, 0],
        pts[:, 2],
        c=pts[:, 1],
        s=24,
        alpha=0.88,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        label="Pareto points",
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("f2")

    dx = max(1e-9, float(np.ptp(pts[:, 0])))
    dy = max(1e-9, float(np.ptp(pts[:, 2])))

    for key, pt in reps.items():
        label = representative_display_name(key)

        # pt = [f1, f2, f3]
        point_color = cmap(norm(pt[1]))

        ax.scatter(
            pt[0], pt[2],
            s=24,
            c=[point_color],
            edgecolors="black",
            linewidths=0.8,
            alpha=1.0,
            zorder=6,
        )

        if key == "compromise":
            ax.scatter(
                pt[0], pt[2],
                s=36,
                facecolors="none",
                edgecolors="black",
                linewidths=1.1,
                zorder=7,
            )

        ax.text(
            pt[0] + 0.012 * dx,
            pt[2] + 0.012 * dy,
            label,
            fontsize=9,
            color="#222222",
            weight="bold",
            zorder=7,
        )

    ax.set_xlabel("f1")
    ax.set_ylabel("f3")
    ax.set_title(build_title(vis, "Pareto Front (2D): x=f1, y=f3, color=f2"))
    ax.grid(True, alpha=0.22)

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
    xy_res = get_xy_resolution(vis)

    fig = go.Figure()

    if terrain_z is not None:
        Ht, Wt = terrain_z.shape
        ys = np.arange(Ht) * xy_res
        xs = np.arange(Wt) * xy_res
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=terrain_z,
                opacity=0.34,
                showscale=True,
                name="Terrain",
                colorscale=[
                    [0.00, "#173B6C"],
                    [0.18, "#225E8A"],
                    [0.38, "#2F8C7A"],
                    [0.58, "#76B041"],
                    [0.78, "#C7B86A"],
                    [1.00, "#E9DFC8"],
                ],
                colorbar=dict(
                    title="Z (m)",
                    thickness=18,
                    len=0.78,
                    x=0.965,
                    y=0.50,
                ),
                hoverinfo="skip",
            )
        )

    for name, path in paths.items():
        st = PATH_STYLE.get(name, {})
        path_m = to_metric_xy(path, xy_res)
        z = path[:, 2] if path.shape[1] >= 3 else np.zeros(len(path))

        width = 9 if name == "compromise" else 5
        if name in ("rrt", "prm"):
            width = 6

        fig.add_trace(
            go.Scatter3d(
                x=path_m[:, 0],
                y=path_m[:, 1],
                z=z,
                mode="lines",
                name=st.get("label", name),
                line=dict(width=width),
                hovertemplate="x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m<extra>%{fullData.name}</extra>",
            )
        )

    start = np.asarray(vis.get("start", [0, 0, 0]), dtype=float)
    goal = np.asarray(vis.get("goal", [0, 0, 0]), dtype=float)

    if start.size >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=[start[0] * xy_res],
                y=[start[1] * xy_res],
                z=[start[2]],
                mode="markers",
                name="Start",
                marker=dict(size=4, symbol="circle"),
                hovertemplate="Start<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m<extra></extra>",
            )
        )

    if goal.size >= 3:
        fig.add_trace(
            go.Scatter3d(
                x=[goal[0] * xy_res],
                y=[goal[1] * xy_res],
                z=[goal[2]],
                mode="markers",
                name="Goal",
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="Goal<br>x=%{x:.1f} m<br>y=%{y:.1f} m<br>z=%{z:.1f} m<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text=build_title(vis, "Interactive 3D Paths"),
            x=0.05,
            y=0.97,
            xanchor="left",
            yanchor="top",
        ),
        scene=dict(
            xaxis=dict(
                title="X (m)",
                showbackground=True,
                backgroundcolor="rgb(240,244,250)",
                gridcolor="rgba(180,190,205,0.45)",
                zerolinecolor="rgba(160,170,185,0.55)",
            ),
            yaxis=dict(
                title="Y (m)",
                showbackground=True,
                backgroundcolor="rgb(240,244,250)",
                gridcolor="rgba(180,190,205,0.45)",
                zerolinecolor="rgba(160,170,185,0.55)",
            ),
            zaxis=dict(
                title="Z (m)",
                showbackground=True,
                backgroundcolor="rgb(240,244,250)",
                gridcolor="rgba(180,190,205,0.45)",
                zerolinecolor="rgba(160,170,185,0.55)",
            ),
            domain=dict(x=[0.02, 0.86], y=[0.06, 0.96]),
            aspectmode="manual",
            aspectratio=dict(x=1.00, y=1.28, z=0.24),
            camera=dict(
                eye=dict(x=1.18, y=1.55, z=0.48),
                center=dict(x=0.0, y=0.0, z=-0.04),
            ),
        ),
        legend=dict(
            x=0.80,
            y=0.94,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.65)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=10, r=10, b=10, t=55),
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
    reps = extract_representative_points(vis)

    fig = go.Figure()

    if pts is not None and pts.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 2],
                z=pts[:, 1],
                mode="markers",
                name="Pareto points",
                marker=dict(size=2),
            )
        )

        pad_x = max(0.3, 0.08 * float(np.ptp(pts[:, 0])))
        pad_y = max(0.3, 0.08 * float(np.ptp(pts[:, 2])))
        pad_z = max(0.3, 0.08 * float(np.ptp(pts[:, 1])))

        x_range = [float(np.min(pts[:, 0]) - pad_x), float(np.max(pts[:, 0]) + pad_x)]
        y_range = [float(np.min(pts[:, 2]) - pad_y), float(np.max(pts[:, 2]) + pad_y)]
        z_range = [float(np.min(pts[:, 1]) - pad_z), float(np.max(pts[:, 1]) + pad_z)]
    else:
        x_range = None
        y_range = None
        z_range = None

    for key, pt in reps.items():
        color = representative_color(key)
        label = representative_display_name(key)

        fig.add_trace(
            go.Scatter3d(
                x=[pt[0]],
                y=[pt[2]],
                z=[pt[1]],
                mode="markers+text",
                name=label,
                text=[label],
                textposition="top center",
                marker=dict(
                    size=4 if key != "compromise" else 5,
                    color=color,
                    line=dict(color="black", width=1),
                    opacity=1.0,
                ),
                hovertemplate=(
                    f"{label}<br>"
                    "f1=%{x:.3f}<br>"
                    "f3=%{y:.3f}<br>"
                    "f2=%{z:.3f}<extra></extra>"
                ),
            )
        )
    
    comp_pt = reps.get("compromise", None)
    if comp_pt is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[comp_pt[0]],
                y=[comp_pt[2]],
                z=[comp_pt[1]],
                mode="markers",
                name="compromise_ring",
                showlegend=False,
                marker=dict(
                    size=6,
                    color="rgba(0,0,0,0)",
                    line=dict(color="black", width=2),
                ),
                hoverinfo="skip",
            )
        )

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
                title="f3",
                range=y_range,
                showbackground=True,
                backgroundcolor="rgb(245,245,245)",
                gridcolor="rgb(200,200,200)",
                zerolinecolor="rgb(160,160,160)",
                showspikes=False,
            ),
            zaxis=dict(
                title="f2",
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
                eye=dict(x=1.55, y=1.35, z=0.95)
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
    p3 = save_pareto_2d_f1_f3_color_f2(vis, json_path, out_dir)
    return p1, p2, p3


def collect_json_files(input_path: Path):
    if input_path.is_file():
        if input_path.name.startswith("visdata_") and input_path.suffix.lower() == ".json":
            return [input_path]
        return []
    return sorted(input_path.rglob("visdata_*.json"))


def resolve_output_dir(json_path: Path, input_path: Path, out_root: Optional[Path]) -> Path:
    """Return where rendered files for one visdata json should be written.

    When out_root is None, keep the historical behavior and write beside the
    source json. When out_root is provided and input_path is a directory, preserve
    the json's relative parent directory under out_root, so rendered files from
    different seeds/planner seeds do not overwrite each other.
    """
    if out_root is None:
        return json_path.parent

    if input_path.is_file():
        return out_root

    try:
        rel_parent = json_path.parent.resolve().relative_to(input_path.resolve())
    except ValueError:
        rel_parent = Path(json_path.parent.name)

    return out_root / rel_parent


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
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Optional output root for rendered files. If omitted, files are written "
            "beside each visdata_*.json as before. If provided, the relative "
            "directory structure under --input is preserved."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else None

    json_files = collect_json_files(input_path)
    print(f"Found {len(json_files)} json file(s).", flush=True)

    if not json_files:
        print("No visdata_*.json found.", flush=True)
        return

    for jp in json_files:
        print(f"[render] {jp}", flush=True)

        try:
            vis = load_json(jp)
        except Exception as e:
            print(f"  [skip] failed to load json: {e}", flush=True)
            continue

        out_dir = resolve_output_dir(jp, input_path, out_root)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.mode in ("static", "all"):
            try:
                p1, p2, p3 = save_static(vis, jp, out_dir, project_root)
                print(f"  static -> {p1}", flush=True)
                print(f"  static -> {p2}", flush=True)
                print(f"  static -> {p3}", flush=True)
            except Exception as e:
                print(f"  [static failed] {e}", flush=True)

        if args.mode in ("interactive", "all"):
            try:
                p5 = save_interactive_paths(vis, jp, out_dir, project_root)
                p6 = save_interactive_pareto(vis, jp, out_dir)
                print(f"  interactive -> {p5}", flush=True)
                print(f"  interactive -> {p6}", flush=True)
            except Exception as e:
                print(f"  [interactive failed] {e}", flush=True)


if __name__ == "__main__":
    main()
