# experiments/vis_data.py
# -*- coding: utf-8 -*-

from __future__ import annotations

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
    # 关键修复：MOEA/D 的 Individual 路径字段是 x，不是 path
    path = getattr(arch_item, "x", None)

    return {
        "path": _path_to_list(path),
        "f": [float(v) for v in arch_item.er.obj],
        "feasible": bool(arch_item.er.feasible),
        "violation": float(arch_item.er.violation),
        "detail": {k: float(v) for k, v in arch_item.er.detail.items()},
    }


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
    path_prm=None,
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
            },
            "prm": {
                "path": _path_to_list(path_prm),
            },
        },
    }

    # 不再保存 archive 明细，只保留 3 目标点
    archive_points = []
    if arch is not None and hasattr(arch, "items") and len(arch.items) > 0:
        for item in arch.items:
            archive_points.append([float(v) for v in item.er.obj])
    payload["archive_points"] = archive_points

    # representatives：保存四个代表解，并且 path 必须从 x 读取
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