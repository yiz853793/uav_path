# src/viz/ploy.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..env.grid_env import GridEnv


def plot_env(env: GridEnv, ax=None, show_threat: bool = True):
    if ax is None:
        fig, ax = plt.subplots()
    if show_threat:
        ax.imshow(env.threat, origin="lower")
    else:
        ax.imshow(np.zeros_like(env.occupancy, dtype=np.float32), origin="lower")

    # 障碍叠加
    occ = env.occupancy.astype(np.float32)
    ax.imshow(np.ma.masked_where(occ < 0.5, occ), origin="lower", alpha=0.9)
    ax.set_xlim(0, env.W - 1)
    ax.set_ylim(0, env.H - 1)
    ax.set_aspect("equal")
    ax.set_title("Env (threat + obstacles)")
    return ax


def plot_path(path: np.ndarray, ax=None, label: str | None = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(path[:, 0], path[:, 1], marker="o", markersize=2.5, linewidth=1.2, label=label)
    if label is not None:
        ax.legend()
    return ax


def plot_pareto(objs: np.ndarray, ax=None):
    """
    objs: [N,3]
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(objs[:, 0], objs[:, 1], s=12)
    ax.set_xlabel("f1: length")
    ax.set_ylabel("f2: threat")
    ax.set_title("Pareto (f1-f2 projection)")
    return ax

def plot_pareto_3d(objs: np.ndarray, ax=None, title="Pareto (f1-f2-f3)"):
    """
    objs: [N,3]
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], s=12)
    ax.set_xlabel("f1: length")
    ax.set_ylabel("f2: threat")
    ax.set_zlabel("f3: energy")
    ax.set_title(title)
    return ax
