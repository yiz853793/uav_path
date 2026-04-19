from __future__ import annotations
import json
import numpy as np


class GridEnv:
    """
    地形驱动的 2.5D / 3D 栅格环境。

    - occupancy[y, x] = True 表示该柱体在平面上不可穿越（禁飞 / 实体障碍）
    - threat[y, x] = 平面威胁强度
    - height[y, x] = 地表高度（z_ground）

    坐标约定：
      p = (x, y) 或 (x, y, z)
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        threat: np.ndarray | None = None,
        resolution: float = 1.0,
        height: np.ndarray | None = None,
        z_min: float = 0.0,
        z_max: float | None = None,
        min_clearance: float = 6.0,
        default_cruise_altitude: float = 12.0,
        z_resolution: float = 1.0,
    ):
        assert occupancy.ndim == 2
        self.occupancy = occupancy.astype(bool)
        self.H, self.W = self.occupancy.shape
        self.resolution = float(resolution)  # horizontal resolution in meters per cell
        self.z_resolution = float(z_resolution)

        if threat is None:
            self.threat = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            assert threat.shape == occupancy.shape
            self.threat = threat.astype(np.float32)

        if height is None:
            self.height = np.zeros((self.H, self.W), dtype=np.float32)
        else:
            assert height.shape == occupancy.shape
            self.height = height.astype(np.float32)

        self.z_min = float(z_min)
        self.z_max = float(max(float(np.max(self.height)) + 64.0, z_max if z_max is not None else float(np.max(self.height)) + 64.0))
        self.min_clearance = float(min_clearance)
        self.default_cruise_altitude = float(default_cruise_altitude)

    def metric_point(self, p: np.ndarray) -> np.ndarray:
        q = np.asarray(p, dtype=np.float32).copy()
        if q.size >= 1:
            q[0] *= self.resolution
        if q.size >= 2:
            q[1] *= self.resolution
        return q

    def metric_delta(self, d: np.ndarray) -> np.ndarray:
        q = np.asarray(d, dtype=np.float32).copy()
        if q.size >= 1:
            q[0] *= self.resolution
        if q.size >= 2:
            q[1] *= self.resolution
        return q

    def metric_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(self.metric_delta(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32))))

    def metric_path(self, path: np.ndarray) -> np.ndarray:
        arr = np.asarray(path, dtype=np.float32).copy()
        if arr.ndim != 2:
            raise ValueError('path must be [N,D]')
        if arr.shape[1] >= 1:
            arr[:, 0] *= self.resolution
        if arr.shape[1] >= 2:
            arr[:, 1] *= self.resolution
        return arr

    def path_length(self, path: np.ndarray) -> float:
        arr = self.metric_path(path)
        if len(arr) < 2:
            return 0.0
        seg = arr[1:] - arr[:-1]
        return float(np.sum(np.linalg.norm(seg, axis=1), dtype=np.float64))

    def metric_step_from_cells(self, step_cells: float) -> float:
        return float(step_cells) * self.resolution

    @staticmethod
    def random_map(H: int = 120, W: int = 160, obstacle_prob: float = 0.18, seed: int = 0) -> "GridEnv":
        rng = np.random.default_rng(seed)
        occ = (rng.random((H, W)) < obstacle_prob)
        occ[:2, :] = False
        occ[-2:, :] = False
        occ[:, :2] = False
        occ[:, -2:] = False

        yy, xx = np.mgrid[0:H, 0:W]
        threat = np.zeros((H, W), dtype=np.float32)
        for _ in range(6):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            sx, sy = rng.uniform(12, 35), rng.uniform(12, 35)
            amp = rng.uniform(0.8, 2.5)
            threat += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy))).astype(np.float32)

        height = (0.35 * threat).astype(np.float32)
        threat = threat + occ.astype(np.float32) * 5.0
        return GridEnv(occ, threat, height=height)

    def in_bounds(self, p: np.ndarray) -> bool:
        x, y = float(p[0]), float(p[1])
        xy_ok = (0.0 <= x < self.W) and (0.0 <= y < self.H)
        if len(p) < 3:
            return xy_ok
        z = float(p[2])
        return xy_ok and (self.z_min <= z <= self.z_max)

    def is_occupied(self, ix: int, iy: int) -> bool:
        if ix < 0 or ix >= self.W or iy < 0 or iy >= self.H:
            return True
        return bool(self.occupancy[iy, ix])

    def threat_at(self, x: float, y: float) -> float:
        ix = int(np.clip(round(x), 0, self.W - 1))
        iy = int(np.clip(round(y), 0, self.H - 1))
        return float(self.threat[iy, ix])

    def ground_height_at(self, x: float, y: float) -> float:
        ix = int(np.clip(round(x), 0, self.W - 1))
        iy = int(np.clip(round(y), 0, self.H - 1))
        return float(self.height[iy, ix])

    def min_safe_altitude_at(self, x: float, y: float, clearance: float | None = None) -> float:
        c = self.min_clearance if clearance is None else float(clearance)
        return self.ground_height_at(x, y) + c

    def clamp_point(self, p: np.ndarray, clearance: float | None = None) -> np.ndarray:
        q = np.asarray(p, dtype=np.float32).copy()
        q[0] = np.clip(q[0], 0.0, self.W - 1.0)
        q[1] = np.clip(q[1], 0.0, self.H - 1.0)
        if len(q) >= 3:
            z_floor = self.min_safe_altitude_at(q[0], q[1], clearance=clearance)
            q[2] = np.clip(q[2], max(self.z_min, z_floor), self.z_max)
        return q.astype(np.float32)

    def is_free_point(self, p: np.ndarray, clearance: float | None = None) -> bool:
        if not self.in_bounds(p):
            return False
        ix = int(round(float(p[0])))
        iy = int(round(float(p[1])))
        # 2D: occupancy is a hard obstacle on the ground plane.
        if len(p) < 3:
            return not self.is_occupied(ix, iy)
        # 3D: terrain columns can be overflown; feasibility is governed by altitude.
        z_floor = self.min_safe_altitude_at(float(p[0]), float(p[1]), clearance=clearance)
        return float(p[2]) >= z_floor

    def altitude_from_ground(self, x: float, y: float, cruise_offset: float | None = None) -> float:
        offset = self.default_cruise_altitude if cruise_offset is None else float(cruise_offset)
        z = self.ground_height_at(x, y) + offset
        return float(np.clip(z, self.z_min, self.z_max))

    def route_cruise_altitude(self, path_xy: np.ndarray, clearance_margin: float = 2.0) -> float:
        path_xy = np.asarray(path_xy, dtype=np.float32)
        if path_xy.ndim != 2 or path_xy.shape[1] < 2:
            raise ValueError('path_xy must be [N,2] or [N,3]')
        xy = path_xy[:, :2]
        ground = np.array([self.ground_height_at(float(x), float(y)) for x, y in xy], dtype=np.float32)
        safe = ground + self.min_clearance + float(clearance_margin)
        cruise = float(np.max(safe))
        return float(np.clip(cruise, self.z_min, self.z_max))

    def lift_path_to_3d(self, path_xy: np.ndarray, start_z: float | None = None, goal_z: float | None = None, cruise_offset: float | None = None, clearance_margin: float = 2.0) -> np.ndarray:
        path_xy = np.asarray(path_xy, dtype=np.float32)
        if path_xy.ndim != 2:
            raise ValueError('path_xy must be 2D array')
        if path_xy.shape[1] == 3:
            out = path_xy.copy()
        elif path_xy.shape[1] == 2:
            z = np.array([self.altitude_from_ground(x, y, cruise_offset=cruise_offset) for x, y in path_xy], dtype=np.float32)
            out = np.concatenate([path_xy, z[:, None]], axis=1)
        else:
            raise ValueError(f'unsupported path dimension: {path_xy.shape[1]}')

        xy = out[:, :2]
        ground = np.array([self.ground_height_at(float(x), float(y)) for x, y in xy], dtype=np.float32)
        safe = ground + self.min_clearance + float(clearance_margin)
        cruise = self.route_cruise_altitude(xy, clearance_margin=clearance_margin)
        out[:, 2] = np.maximum(safe, cruise)

        if start_z is not None:
            out[0, 2] = max(float(start_z), float(safe[0]))
        if goal_z is not None:
            out[-1, 2] = max(float(goal_z), float(safe[-1]))
        for i in range(len(out)):
            out[i] = self.clamp_point(out[i], clearance=self.min_clearance + float(clearance_margin))
        return out.astype(np.float32)

    def inflate_obstacles(self, radius_cells: int) -> "GridEnv":
        r = int(max(0, radius_cells))
        if r == 0:
            return GridEnv(self.occupancy.copy(), self.threat.copy(), self.resolution, height=self.height.copy(), z_min=self.z_min, z_max=self.z_max, min_clearance=self.min_clearance, default_cruise_altitude=self.default_cruise_altitude, z_resolution=self.z_resolution)

        occ = self.occupancy.astype(np.uint8)
        H, W = occ.shape
        I = np.pad(occ, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(axis=0).cumsum(axis=1)
        y0 = np.clip(np.arange(H) - r, 0, H)
        y1 = np.clip(np.arange(H) + r + 1, 0, H)
        x0 = np.clip(np.arange(W) - r, 0, W)
        x1 = np.clip(np.arange(W) + r + 1, 0, W)
        win = I[y1[:, None], x1[None, :]] - I[y0[:, None], x1[None, :]] - I[y1[:, None], x0[None, :]] + I[y0[:, None], x0[None, :]]
        out = win > 0
        return GridEnv(out.astype(bool), self.threat.copy(), self.resolution, height=self.height.copy(), z_min=self.z_min, z_max=self.z_max, min_clearance=self.min_clearance, default_cruise_altitude=self.default_cruise_altitude, z_resolution=self.z_resolution)

    @staticmethod
    def mountain_map(H=160, W=200, seed=0, terrace_levels=24, obstacle_height=0.78):
        rng = np.random.default_rng(int(seed))
        base = fbm_2d(H, W, rng, base_grid=40, octaves=5)
        ridge = ridged_fbm_2d(H, W, rng, base_grid=55, octaves=5)
        height = 0.65 * base + 0.75 * ridge
        height = (height - height.min()) / (height.max() - height.min() + 1e-9)
        height = domain_warp(height, rng, strength=22.0)
        height = carve_valleys(height, iterations=7, strength=0.30)
        height = minecraft_terrace(height, levels=int(terrace_levels), jitter=0.01, rng=rng)
        threat = height.astype(np.float32)
        occ = (height > float(obstacle_height))
        occ[:2, :] = False; occ[-2:, :] = False
        occ[:, :2] = False; occ[:, -2:] = False
        threat = threat + occ.astype(np.float32) * 1.5
        height_metric = (height * 50.0).astype(np.float32)
        env = GridEnv(occ, threat, resolution=1.0, height=height_metric, z_min=0.0, z_max=float(np.max(height_metric)) + 80.0, min_clearance=4.0, default_cruise_altitude=24.0, z_resolution=1.0)
        meta = {
            "type": "mountain_map",
            "xy_resolution_m": 1.0,
            "z_resolution_m": 1.0,
            "seed": int(seed),
            "H": int(H),
            "W": int(W),
            "terrace_levels": int(terrace_levels),
            "obstacle_height": float(obstacle_height),
            "z_min": float(env.z_min),
            "z_max": float(env.z_max),
            "min_clearance": float(env.min_clearance),
            "default_cruise_altitude": float(env.default_cruise_altitude),
        }
        return env, height_metric, meta

    @staticmethod
    def city_map(H=1600, W=2000, seed=0, city_density=0.24):
        return GridEnv._structured_city_map(
            H=H,
            W=W,
            seed=seed,
            city_density=city_density,
            terrain_type='city',
            size_tag=f'city_{city_density:.2f}',
            use_hills=False,
        )

    @staticmethod
    def hill_city_map(H=1600, W=2000, seed=0, city_density=0.24, hill_scale=26.0):
        """
        hill_city 固定使用论文实验地图大小：H=1600, W=2000。
        这里保留 H / W 形参只是为了兼容旧调用方式，但内部会直接忽略。
        """
        H = 1600
        W = 2000
        return GridEnv._structured_city_map(
            H=H,
            W=W,
            seed=seed,
            city_density=city_density,
            terrain_type='hill_city',
            size_tag=f'hill_city_{city_density:.2f}',
            use_hills=True,
            hill_scale=hill_scale,
        )

    @staticmethod
    def _structured_city_map(H=1600, W=2000, seed=0, city_density=0.24, terrain_type='city', size_tag='city', use_hills=False, hill_scale=26.0):
        density_key = _density_key(city_density)
        morphology_seed = _mix_seed_with_density(seed, density_key, salt=11 if use_hills else 7)
        rng = np.random.default_rng(morphology_seed)
        density_morph_bias = float(np.clip((float(city_density) - 0.24) / 0.16, -1.0, 1.0))
        relief_scale = 1.0 - (0.20 * density_morph_bias if use_hills else 0.0)
        settlement_delta = int(np.round(2.0 * density_morph_bias))
        settlement_min_dist_scale = 1.0 - 0.14 * density_morph_bias
        urban_backbone_mix = float(np.clip(0.26 + 0.08 * density_morph_bias, 0.18, 0.36)) if use_hills else 0.0
        yy, xx = np.mgrid[0:H, 0:W]

        def box_mean(arr: np.ndarray, radius_y: int, radius_x: int | None = None) -> np.ndarray:
            radius_y = int(max(0, radius_y))
            radius_x = radius_y if radius_x is None else int(max(0, radius_x))
            if radius_y == 0 and radius_x == 0:
                return arr.astype(np.float32)
            src = arr.astype(np.float32)
            I = np.pad(src, ((1, 0), (1, 0)), mode='constant', constant_values=0).cumsum(axis=0).cumsum(axis=1)
            y0 = np.clip(np.arange(H) - radius_y, 0, H)
            y1 = np.clip(np.arange(H) + radius_y + 1, 0, H)
            x0 = np.clip(np.arange(W) - radius_x, 0, W)
            x1 = np.clip(np.arange(W) + radius_x + 1, 0, W)
            win = I[y1[:, None], x1[None, :]] - I[y0[:, None], x1[None, :]] - I[y1[:, None], x0[None, :]] + I[y0[:, None], x0[None, :]]
            area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
            return (win / np.maximum(area, 1.0)).astype(np.float32)

        def gaussian_field(centers: list[tuple[float, float, float, float, float]]) -> np.ndarray:
            field = np.zeros((H, W), dtype=np.float32)
            for cx, cy, sx, sy, amp in centers:
                field += (amp * np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))).astype(np.float32)
            return field

        def select_centers(score: np.ndarray, target_k: int, min_dist: float, margin_ratio: float = 0.08) -> list[tuple[float, float]]:
            score = score.astype(np.float32)
            flat_idx = np.argsort(score.reshape(-1))[::-1]
            y_margin = int(H * margin_ratio)
            x_margin = int(W * margin_ratio)
            centers_xy: list[tuple[float, float]] = []
            min_dist2 = float(min_dist * min_dist)
            for idx in flat_idx.tolist():
                y = idx // W
                x = idx % W
                if y < y_margin or y >= H - y_margin or x < x_margin or x >= W - x_margin:
                    continue
                good = True
                for cx, cy in centers_xy:
                    if (x - cx) * (x - cx) + (y - cy) * (y - cy) < min_dist2:
                        good = False
                        break
                if good:
                    centers_xy.append((float(x), float(y)))
                    if len(centers_xy) >= target_k:
                        break
            return centers_xy

        def stamp_disk(mask: np.ndarray, cx: float, cy: float, radius: int):
            radius = int(max(1, radius))
            x0 = max(0, int(round(cx)) - radius)
            x1 = min(W, int(round(cx)) + radius + 1)
            y0 = max(0, int(round(cy)) - radius)
            y1 = min(H, int(round(cy)) + radius + 1)
            if x0 >= x1 or y0 >= y1:
                return
            py, px = np.mgrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= ((px - cx) ** 2 + (py - cy) ** 2 <= float(radius * radius))

        def draw_road(mask: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: int):
            x0, y0 = p0
            x1, y1 = p1
            length = max(abs(x1 - x0), abs(y1 - y0))
            steps = max(2, int(length * 1.5))
            xs = np.linspace(x0, x1, steps, dtype=np.float32)
            ys = np.linspace(y0, y1, steps, dtype=np.float32)
            for x, y in zip(xs.tolist(), ys.tolist()):
                stamp_disk(mask, x, y, width)

        def draw_corridor_field(field: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: float, amp: float = 1.0):
            x0, y0 = p0
            x1, y1 = p1
            length = max(abs(x1 - x0), abs(y1 - y0))
            steps = max(8, int(length / max(4.0, width * 0.9)))
            xs = np.linspace(x0, x1, steps, dtype=np.float32)
            ys = np.linspace(y0, y1, steps, dtype=np.float32)
            sigma = max(2.0, float(width))
            rx = max(8, int(np.ceil(2.5 * sigma)))
            ry = rx
            for x, y in zip(xs.tolist(), ys.tolist()):
                x0i = max(0, int(round(x)) - rx)
                x1i = min(W, int(round(x)) + rx + 1)
                y0i = max(0, int(round(y)) - ry)
                y1i = min(H, int(round(y)) + ry + 1)
                if x0i >= x1i or y0i >= y1i:
                    continue
                py, px = np.mgrid[y0i:y1i, x0i:x1i]
                d2 = (px - x) ** 2 + (py - y) ** 2
                field[y0i:y1i, x0i:x1i] += (amp * np.exp(-d2 / (2.0 * sigma * sigma))).astype(np.float32)

        def q3(v: float) -> float:
            return float(max(3.0, 3.0 * round(v / 3.0)))

        if use_hills:
            ground = np.zeros((H, W), dtype=np.float32)

            # Broad southeastern-China hill city terrain: multiple developable lowlands,
            # wide hill masses, few landmark hills, and no single-corner dominance.
            theta = float(rng.uniform(-0.45, 0.45))

            # Very low-frequency undulation: broad, weak background relief instead of one global tilt.
            broad = fbm_2d(H, W, rng, base_grid=max(360, min(H, W) // 4), octaves=3)
            broad = domain_warp(broad, rng, strength=14.0)
            broad = (broad - broad.min()) / (broad.max() - broad.min() + 1e-9)
            ground += ((broad - 0.50) * rng.uniform(0.55, 0.95) * hill_scale * relief_scale).astype(np.float32)

            # Multi-basin lowland system: several moderate lowlands so different directions can develop.
            basin_field = np.zeros((H, W), dtype=np.float32)
            basin_count = int(rng.choice([3, 3, 4, 4]))
            basin_centers = []
            basin_attempts = 0
            min_basin_dist2 = (0.20 * min(H, W)) ** 2
            while len(basin_centers) < basin_count and basin_attempts < 80:
                basin_attempts += 1
                cx = float(rng.uniform(0.18 * W, 0.84 * W))
                cy = float(rng.uniform(0.22 * H, 0.82 * H))
                if any((cx - px) ** 2 + (cy - py) ** 2 < min_basin_dist2 for px, py in basin_centers):
                    continue
                basin_centers.append((cx, cy))
                sx = float(rng.uniform(0.13 * W, 0.22 * W))
                sy = float(rng.uniform(0.11 * H, 0.20 * H))
                amp = float(rng.uniform(0.12, 0.26) * hill_scale * (1.0 - 0.12 * density_morph_bias))
                basin_field += (amp * np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))).astype(np.float32)
            ground -= basin_field.astype(np.float32)

            # Broad hill masses: fewer, larger, and smoother than before.
            hill_mass_count = int(rng.choice([1, 2, 2, 3]))
            inner_hill_count = hill_mass_count
            hill_masses = []
            for _ in range(hill_mass_count):
                zone = int(rng.choice([0, 1, 2, 3, 4]))
                if zone == 0:      # north
                    cx = float(rng.uniform(0.10 * W, 0.88 * W)); cy = float(rng.uniform(0.10 * H, 0.30 * H))
                elif zone == 1:    # northeast / east
                    cx = float(rng.uniform(0.62 * W, 0.90 * W)); cy = float(rng.uniform(0.18 * H, 0.72 * H))
                elif zone == 2:    # west / northwest
                    cx = float(rng.uniform(0.10 * W, 0.30 * W)); cy = float(rng.uniform(0.18 * H, 0.72 * H))
                elif zone == 3:    # south interior
                    cx = float(rng.uniform(0.22 * W, 0.82 * W)); cy = float(rng.uniform(0.60 * H, 0.90 * H))
                else:              # central rolling hill belt
                    cx = float(rng.uniform(0.24 * W, 0.76 * W)); cy = float(rng.uniform(0.28 * H, 0.74 * H))
                sx = float(rng.uniform(0.12 * W, 0.22 * W))
                sy = float(rng.uniform(0.12 * H, 0.22 * H))
                amp = float(rng.uniform(0.10, 0.28) * hill_scale * relief_scale)
                hill_masses.append((cx, cy, sx, sy, amp))
            ground += gaussian_field(hill_masses)

            # Elongated hill belts / low ridges: broad and low, adding southeastern-hill texture.
            ridge_chain_count = int(rng.choice([0, 0, 1]))
            for _ in range(ridge_chain_count):
                cx = float(rng.uniform(0.14 * W, 0.86 * W))
                cy = float(rng.uniform(0.16 * H, 0.84 * H))
                length = float(rng.uniform(0.22, 0.40) * max(H, W))
                width = float(rng.uniform(0.05, 0.09) * min(H, W))
                local_theta = float(theta + rng.uniform(-0.75, 0.75))
                dx = xx - cx
                dy = yy - cy
                xp = dx * np.cos(local_theta) + dy * np.sin(local_theta)
                yp = -dx * np.sin(local_theta) + dy * np.cos(local_theta)
                ridge_amp = float(rng.uniform(0.03, 0.07) * hill_scale * relief_scale)
                ridge = ridge_amp * np.exp(-((xp ** 2) / (2.0 * length * length) + (yp ** 2) / (2.0 * width * width)))
                shoulder = (0.55 * ridge_amp) * np.exp(-((xp ** 2) / (2.0 * (length * 1.12) ** 2) + (yp ** 2) / (2.0 * (width * 1.9) ** 2)))
                ground += (ridge + shoulder).astype(np.float32)

            # Landmark hills: very few, wide bases, mostly on the periphery.
            mountain_count = int(rng.choice([0, 0, 0, 1]))
            outer_centers = []
            peak_points = []
            peak_min_dist = 0.34 * min(H, W)
            attempts = 0
            while len(outer_centers) < mountain_count and attempts < 96:
                attempts += 1
                edge_band = int(rng.choice([0, 1, 2, 3]))
                if edge_band == 0:  # northwest / north
                    cx = float(rng.uniform(0.08 * W, 0.28 * W)); cy = float(rng.uniform(0.08 * H, 0.24 * H))
                elif edge_band == 1:  # northeast background
                    cx = float(rng.uniform(0.72 * W, 0.92 * W)); cy = float(rng.uniform(0.10 * H, 0.28 * H))
                elif edge_band == 2:  # east / southeast background
                    cx = float(rng.uniform(0.76 * W, 0.94 * W)); cy = float(rng.uniform(0.36 * H, 0.84 * H))
                else:  # southwest / west background
                    cx = float(rng.uniform(0.06 * W, 0.22 * W)); cy = float(rng.uniform(0.58 * H, 0.90 * H))
                if any((cx - px) ** 2 + (cy - py) ** 2 < peak_min_dist * peak_min_dist for px, py in peak_points):
                    continue
                sx = float(rng.uniform(0.11 * W, 0.18 * W))
                sy = float(rng.uniform(0.11 * H, 0.18 * H))
                amp = float(rng.uniform(0.55, 1.05) * hill_scale * relief_scale)
                outer_centers.append((cx, cy, sx, sy, amp))
                peak_points.append((cx, cy))
            outer_mountain_count = len(outer_centers)
            for cx, cy, sx, sy, amp in outer_centers:
                dx = xx - cx
                dy = yy - cy
                local_theta = float(theta + rng.uniform(-0.35, 0.35))
                xp = dx * np.cos(local_theta) + dy * np.sin(local_theta)
                yp = -dx * np.sin(local_theta) + dy * np.cos(local_theta)
                core = amp * np.exp(-((xp ** 2) / (2.0 * (sx * 0.68) ** 2) + (yp ** 2) / (2.0 * (sy * 0.68) ** 2)))
                apron = (0.70 * amp) * np.exp(-((xp ** 2) / (2.0 * (sx * 1.40) ** 2) + (yp ** 2) / (2.0 * (sy * 1.40) ** 2)))
                shoulder = (0.30 * amp) * np.exp(-((xp ** 2) / (2.0 * (sx * 2.1) ** 2) + (yp ** 2) / (2.0 * (sy * 1.8) ** 2)))
                ground += (core + apron + shoulder).astype(np.float32)

            # If two landmark hills exist, connect them with a broad saddle / ridge rather than a spiky chain.
            if len(outer_centers) >= 2:
                for i in range(len(outer_centers) - 1):
                    ax, ay, *_ = outer_centers[i]
                    bx, by, *_ = outer_centers[i + 1]
                    seg_len = max(float(np.hypot(ax - bx, ay - by)), 1.0)
                    nseg = max(4, int(seg_len / max(1.0, 0.12 * min(H, W))))
                    ridge_amp = float(rng.uniform(0.03, 0.07) * hill_scale * relief_scale)
                    ridge_s_long = float(rng.uniform(0.08, 0.12) * min(H, W))
                    ridge_s_short = float(rng.uniform(0.05, 0.08) * min(H, W))
                    local_theta = float(np.arctan2(by - ay, bx - ax))
                    for t in np.linspace(0.20, 0.80, nseg):
                        cx = float((1.0 - t) * ax + t * bx)
                        cy = float((1.0 - t) * ay + t * by)
                        dx = xx - cx
                        dy = yy - cy
                        xp = dx * np.cos(local_theta) + dy * np.sin(local_theta)
                        yp = -dx * np.sin(local_theta) + dy * np.cos(local_theta)
                        rr = ridge_amp * np.exp(-((xp ** 2) / (2.0 * ridge_s_long * ridge_s_long) + (yp ** 2) / (2.0 * ridge_s_short * ridge_s_short)))
                        ground += rr.astype(np.float32)

            # Gentle terrain shaping. Keep ridge texture secondary, avoid dense peak fields.
            warp = fbm_2d(H, W, rng, base_grid=max(180, min(H, W) // 7), octaves=2)
            warp = domain_warp(warp, rng, strength=8.0)
            warp = (warp - warp.min()) / (warp.max() - warp.min() + 1e-9)
            ground += ((warp - 0.5) * rng.uniform(0.8, 1.6)).astype(np.float32)
            ground = domain_warp(ground, rng, strength=8.0)
            ground = carve_valleys(ground, iterations=3, strength=0.08)

            # Low/mid frequency relief: broad slopes and gentle rolling hills, not needle peaks.
            low_freq = fbm_2d(H, W, rng, base_grid=max(300, min(H, W) // 4), octaves=3)
            low_freq = (low_freq - low_freq.min()) / (low_freq.max() - low_freq.min() + 1e-9)
            ground += ((low_freq - 0.5) * rng.uniform(1.2, 2.0)).astype(np.float32)

            mid_freq = fbm_2d(H, W, rng, base_grid=max(180, min(H, W) // 6), octaves=3)
            mid_freq = domain_warp(mid_freq, rng, strength=6.0)
            mid_freq = (mid_freq - mid_freq.min()) / (mid_freq.max() - mid_freq.min() + 1e-9)
            ground += ((mid_freq - 0.5) * rng.uniform(0.5, 1.1)).astype(np.float32)

            detail = fbm_2d(H, W, rng, base_grid=max(520, min(H, W) // 2), octaves=2)
            detail = (detail - detail.min()) / (detail.max() - detail.min() + 1e-9)
            # Keep only a weak amount of high-frequency detail so hills remain broad and traversable.
            ground += ((detail - 0.5) * rng.uniform(0.08, 0.18)).astype(np.float32)

            # Multi-stage smoothing: first suppress needle-like peaks, then soften overly steep crowns.
            smooth_r_small = max(5, min(H, W) // 120)
            smooth_r_large = max(10, min(H, W) // 72)
            ground = box_mean(ground.astype(np.float32), smooth_r_small, smooth_r_small).astype(np.float32)
            smooth_ground = box_mean(ground.astype(np.float32), smooth_r_large, smooth_r_large).astype(np.float32)
            gy_tmp, gx_tmp = np.gradient(ground)
            slope_tmp = np.sqrt(gx_tmp * gx_tmp + gy_tmp * gy_tmp)
            slope_tmp_n = (slope_tmp - slope_tmp.min()) / (slope_tmp.max() - slope_tmp.min() + 1e-9)
            steep_alpha = np.clip((slope_tmp_n - 0.42) / 0.28, 0.0, 1.0).astype(np.float32)
            ground = ((1.0 - 0.65 * steep_alpha) * ground + (0.65 * steep_alpha) * smooth_ground).astype(np.float32)

            # 再做一次“减山化”：给未来主城区/通道区域一个宽缓低地背景，
            # 同时把中高海拔做非线性压缩，避免整张图到处都是山。
            x_n = (xx / max(1.0, W - 1)).astype(np.float32)
            y_n = (yy / max(1.0, H - 1)).astype(np.float32)
            develop_belt = (
                0.55 * np.exp(-(((y_n - (0.66 - 0.04 * x_n)) ** 2) / (2.0 * 0.12 * 0.12)))
                + 0.45 * np.exp(-(((y_n - (0.58 + 0.03 * np.sin(2.2 * np.pi * x_n))) ** 2) / (2.0 * 0.11 * 0.11)))
                + 0.26 * np.exp(-(((x_n - 0.50) ** 2) / (2.0 * 0.22 * 0.22) + ((y_n - 0.63) ** 2) / (2.0 * 0.18 * 0.18)))
            ).astype(np.float32)
            develop_belt = (develop_belt - develop_belt.min()) / (develop_belt.max() - develop_belt.min() + 1e-9)
            ground -= (develop_belt * rng.uniform(0.16, 0.24) * hill_scale * (1.0 + 0.12 * density_morph_bias)).astype(np.float32)

            ground = np.clip(ground, 0.0, None).astype(np.float32)
            gmax = float(np.max(ground))
            if gmax > 1e-6:
                ground_n = (ground / gmax).astype(np.float32)
                # 幂次>1 会压低大面积中高海拔，只保留少量真正的山峰。
                ground = ((ground_n ** 1.55) * gmax).astype(np.float32)
                ground = (0.72 * ground + 0.28 * box_mean(ground.astype(np.float32), max(12, min(H, W) // 80), max(12, min(H, W) // 80))).astype(np.float32)
                gmax = float(np.max(ground))
                target_max = float(rng.choice([72.0, 90.0, 108.0, 120.0], p=[0.30, 0.36, 0.22, 0.12]))
                ground *= np.float32(target_max / max(gmax, 1e-6))

            gy, gx = np.gradient(ground)
            slope = np.sqrt(gx * gx + gy * gy)
            slope_n = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
            flatness = 1.0 - slope_n
            elevation_n = (ground - ground.min()) / (ground.max() - ground.min() + 1e-9)
        else:
            base = fbm_2d(H, W, rng, base_grid=max(32, min(H, W) // 10), octaves=3)
            base = (base - base.min()) / (base.max() - base.min() + 1e-9)
            ground = (base * 3.5).astype(np.float32)
            slope_n = np.zeros((H, W), dtype=np.float32)
            flatness = np.ones((H, W), dtype=np.float32)
            elevation_n = (ground - ground.min()) / (ground.max() - ground.min() + 1e-9)
            inner_hill_count = 0
            outer_mountain_count = 0

        local_flat = box_mean(flatness, max(18, min(H, W) // 40), max(18, min(H, W) // 40))
        ground_mid = box_mean(ground.astype(np.float32), max(22, min(H, W) // 34), max(22, min(H, W) // 34))
        local_relief = box_mean(np.abs(ground.astype(np.float32) - ground_mid).astype(np.float32), max(10, min(H, W) // 72), max(10, min(H, W) // 72))
        local_relief = (local_relief - local_relief.min()) / (local_relief.max() - local_relief.min() + 1e-9)
        if use_hills:
            medium_pref = 1.0 - np.abs(elevation_n - 0.24) / 0.24
            medium_pref = np.clip(medium_pref, 0.0, 1.0).astype(np.float32)
            lowland_pref = 1.0 - np.clip((elevation_n - 0.20) / 0.34, 0.0, 1.0)
            lowland_pref = np.clip(lowland_pref, 0.0, 1.0).astype(np.float32)
            valley_signal = box_mean((elevation_n < 0.34).astype(np.float32), max(26, min(H, W) // 28), max(26, min(H, W) // 28))
            terrain_suitability = (
                0.22 * flatness.astype(np.float32)
                + 0.21 * local_flat.astype(np.float32)
                + 0.22 * medium_pref.astype(np.float32)
                + 0.16 * lowland_pref.astype(np.float32)
                + 0.15 * valley_signal.astype(np.float32)
                + 0.08 * (1.0 - np.abs(elevation_n - 0.36) / 0.36).clip(0.0, 1.0).astype(np.float32)
                - 0.13 * slope_n.astype(np.float32)
                - 0.08 * local_relief.astype(np.float32)
            ).astype(np.float32)
        else:
            medium_pref = 1.0 - np.abs(elevation_n - 0.35) / 0.35
            medium_pref = np.clip(medium_pref, 0.0, 1.0).astype(np.float32)
            terrain_suitability = (
                0.42 * flatness.astype(np.float32)
                + 0.33 * local_flat.astype(np.float32)
                + 0.17 * medium_pref.astype(np.float32)
                - 0.28 * slope_n.astype(np.float32)
                - 0.16 * local_relief.astype(np.float32)
            ).astype(np.float32)
        terrain_suitability += 0.04 * (fbm_2d(H, W, rng, base_grid=max(180, min(H, W) // 8), octaves=2) - 0.5).astype(np.float32)
        terrain_suitability = (terrain_suitability - terrain_suitability.min()) / (terrain_suitability.max() - terrain_suitability.min() + 1e-9)

        if use_hills:
            # Large hill-city morphology: broad, connected urbanized belt with multiple developable directions,
            # but without a single overwhelmingly best corner.
            x_n = (xx / max(1.0, W - 1)).astype(np.float32)
            y_n = (yy / max(1.0, H - 1)).astype(np.float32)
            west_lobe = np.exp(-(((x_n - 0.24) ** 2) / (2.0 * 0.18 * 0.18) + ((y_n - 0.70) ** 2) / (2.0 * 0.15 * 0.15))).astype(np.float32)
            central_lobe = np.exp(-(((x_n - 0.50) ** 2) / (2.0 * 0.20 * 0.20) + ((y_n - 0.66) ** 2) / (2.0 * 0.16 * 0.16))).astype(np.float32)
            east_lobe = np.exp(-(((x_n - 0.76) ** 2) / (2.0 * 0.18 * 0.18) + ((y_n - 0.68) ** 2) / (2.0 * 0.15 * 0.15))).astype(np.float32)
            north_lobe = np.exp(-(((x_n - 0.52) ** 2) / (2.0 * 0.22 * 0.22) + ((y_n - 0.54) ** 2) / (2.0 * 0.15 * 0.15))).astype(np.float32)
            south_lobe = np.exp(-(((x_n - 0.56) ** 2) / (2.0 * 0.22 * 0.22) + ((y_n - 0.78) ** 2) / (2.0 * 0.13 * 0.13))).astype(np.float32)
            main_axis = np.exp(-(((y_n - (0.69 - 0.03 * x_n)) ** 2) / (2.0 * 0.12 * 0.12))).astype(np.float32)
            central_axis = np.exp(-(((y_n - (0.61 + 0.025 * np.sin(2.4 * np.pi * x_n))) ** 2) / (2.0 * 0.13 * 0.13))).astype(np.float32)
            urban_backbone = (
                0.18 * west_lobe +
                0.20 * central_lobe +
                0.18 * east_lobe +
                0.15 * north_lobe +
                0.07 * south_lobe +
                0.12 * main_axis +
                0.10 * central_axis
            ).astype(np.float32)
            urban_backbone = (urban_backbone - urban_backbone.min()) / (urban_backbone.max() - urban_backbone.min() + 1e-9)
            terrain_suitability = ((1.0 - urban_backbone_mix) * terrain_suitability + urban_backbone_mix * urban_backbone).astype(np.float32)
            terrain_suitability = (terrain_suitability - terrain_suitability.min()) / (terrain_suitability.max() - terrain_suitability.min() + 1e-9)
        else:
            urban_backbone = np.ones((H, W), dtype=np.float32)

        center_target = int((rng.integers(8, 12) if use_hills else rng.integers(5, 9)) + settlement_delta)
        center_target = max(4 if not use_hills else 6, center_target)
        center_min_dist = (0.09 * settlement_min_dist_scale if use_hills else 0.12 * settlement_min_dist_scale) * min(H, W)
        settlement_centers = select_centers(terrain_suitability, center_target, min_dist=center_min_dist, margin_ratio=0.05 if use_hills else 0.08)
        if not settlement_centers:
            settlement_centers = [(float(W * 0.5), float(H * 0.5))]

        settlement_field = np.zeros((H, W), dtype=np.float32)
        for i, (cx, cy) in enumerate(settlement_centers):
            if use_hills:
                sx = float(rng.uniform(0.07 * W, 0.13 * W))
                sy = float(rng.uniform(0.06 * H, 0.11 * H))
            else:
                sx = float(rng.uniform(0.06 * W, 0.14 * W))
                sy = float(rng.uniform(0.06 * H, 0.14 * H))
            amp = float(rng.uniform(0.82, 1.08) if i < 3 else rng.uniform(0.64, 0.92))
            settlement_field += (amp * np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))).astype(np.float32)
        if use_hills:
            settlement_field += (0.72 * urban_backbone).astype(np.float32)
            settlement_field += (0.16 * box_mean(urban_backbone, max(28, min(H, W) // 30), max(28, min(H, W) // 30))).astype(np.float32)
        # Soft corridors between nearby centers, more like a connected hill city than isolated islands.
        if use_hills and len(settlement_centers) >= 2:
            for i, (ax, ay) in enumerate(settlement_centers):
                dists = []
                for j, (bx, by) in enumerate(settlement_centers):
                    if i == j:
                        continue
                    dists.append((float(np.hypot(ax - bx, ay - by)), bx, by))
                dists.sort(key=lambda t: t[0])
                for _, bx, by in dists[:3]:
                    cx = 0.5 * (ax + bx)
                    cy = 0.5 * (ay + by)
                    dx = xx - cx
                    dy = yy - cy
                    t = float(np.arctan2(by - ay, bx - ax))
                    xp = dx * np.cos(t) + dy * np.sin(t)
                    yp = -dx * np.sin(t) + dy * np.cos(t)
                    long_s = float(rng.uniform(0.11, 0.18) * min(H, W))
                    short_s = float(rng.uniform(0.035, 0.07) * min(H, W))
                    corridor = 0.22 * np.exp(-((xp ** 2) / (2.0 * long_s * long_s) + (yp ** 2) / (2.0 * short_s * short_s)))
                    settlement_field += corridor.astype(np.float32)
        settlement_field = (settlement_field - settlement_field.min()) / (settlement_field.max() - settlement_field.min() + 1e-9)

        road_mask = np.zeros((H, W), dtype=bool)
        protected_corridor_field = np.zeros((H, W), dtype=np.float32)
        protected_corridor_mask = np.zeros((H, W), dtype=bool)

        anchor = settlement_centers[0]
        for c in settlement_centers[1:]:
            mid = (float(c[0]), float(anchor[1])) if rng.random() < 0.5 else (float(anchor[0]), float(c[1]))
            road_w = 7 if use_hills else 6
            draw_road(road_mask, anchor, mid, width=road_w)
            draw_road(road_mask, mid, c, width=road_w)
            if use_hills:
                draw_corridor_field(protected_corridor_field, anchor, mid, width=road_w + 16, amp=1.00)
                draw_corridor_field(protected_corridor_field, mid, c, width=road_w + 16, amp=1.00)

        if use_hills:
            # 保护通道不再直接画成硬直线禁建带，而是生成更宽、更柔和的低阻 corridor field，
            # 再结合平坦度/坡度筛选，减少横平竖直的“人工切割感”。
            centers_sorted_x = sorted(settlement_centers, key=lambda t: t[0])
            if len(centers_sorted_x) >= 2:
                left = centers_sorted_x[0]
                right = centers_sorted_x[-1]
                draw_corridor_field(protected_corridor_field, (0.0, left[1]), left, width=24.0, amp=0.95)
                draw_corridor_field(protected_corridor_field, left, right, width=22.0, amp=1.10)
                draw_corridor_field(protected_corridor_field, right, (float(W - 1), right[1]), width=24.0, amp=0.95)
            centers_sorted_y = sorted(settlement_centers, key=lambda t: t[1])
            if len(centers_sorted_y) >= 2 and rng.random() < 0.7:
                low = centers_sorted_y[0]
                high = centers_sorted_y[-1]
                draw_corridor_field(protected_corridor_field, (low[0], 0.0), low, width=18.0, amp=0.78)
                draw_corridor_field(protected_corridor_field, low, high, width=16.0, amp=0.86)
                draw_corridor_field(protected_corridor_field, high, (high[0], float(H - 1)), width=18.0, amp=0.78)

        for cx, cy in settlement_centers:
            span_x = int(rng.integers(max(54, W // 32), max(150, W // 14)))
            span_y = int(rng.integers(max(38, H // 40), max(110, H // 16)))
            draw_road(road_mask, (cx - span_x, cy), (cx + span_x, cy), width=4)
            draw_road(road_mask, (cx, cy - span_y), (cx, cy + span_y), width=4)
            if rng.random() < (0.62 if use_hills else 0.78):
                offset_x = int(rng.integers(max(10, W // 140), max(30, W // 64)))
                draw_road(road_mask, (cx - span_x, cy + offset_x), (cx + span_x, cy + offset_x), width=3)
            if rng.random() < (0.62 if use_hills else 0.78):
                offset_y = int(rng.integers(max(10, H // 140), max(28, H // 64)))
                draw_road(road_mask, (cx + offset_y, cy - span_y), (cx + offset_y, cy + span_y), width=3)

        if use_hills:
            corridor_pref = protected_corridor_field.copy()
            corridor_pref = (corridor_pref - corridor_pref.min()) / (corridor_pref.max() - corridor_pref.min() + 1e-9)
            # 只把最强的一小部分 corridor 当作“硬性保留通道”，避免整片 developable belt
            # 被切成一个椭圆形禁建区，产生“摊大饼”观感。
            hard_corridor_mask = (
                (corridor_pref >= 0.62) &
                (local_flat >= 0.36) &
                (slope_n <= 0.36) &
                (local_relief <= 0.42)
            )
            road_halo = box_mean(road_mask.astype(np.float32), max(4, min(H, W) // 180), max(4, min(H, W) // 180)) > 0.10
            protected_corridor_mask = hard_corridor_mask | road_mask | road_halo
        else:
            corridor_pref = np.zeros((H, W), dtype=np.float32)

        # Natural open space should come from terrain: ridge tops, steeper hill crowns, and locally rough zones.
        open_mask = np.zeros((H, W), dtype=bool)
        if use_hills:
            ridge_core = (elevation_n >= 0.52) & (local_relief >= 0.28) & (flatness <= 0.66)
            ridge_band = box_mean(ridge_core.astype(np.float32), max(10, min(H, W) // 84), max(10, min(H, W) // 84)) > 0.16
            hilltops = (elevation_n >= 0.64) & (flatness <= 0.60)
            hilltops = box_mean(hilltops.astype(np.float32), max(8, min(H, W) // 96), max(8, min(H, W) // 96)) > 0.10
            steep_hills = (slope_n >= 0.55) & (elevation_n >= 0.38)
            steep_hills = box_mean(steep_hills.astype(np.float32), max(7, min(H, W) // 110), max(7, min(H, W) // 110)) > 0.12
            open_mask |= ridge_band | hilltops | steep_hills
            # 通道只保留很窄的骨架，不再整片挖空。
            open_mask |= (protected_corridor_mask & (corridor_pref >= 0.78))
        else:
            for cx, cy in settlement_centers:
                if rng.random() < 0.75:
                    rx = float(rng.uniform(0.02 * W, 0.05 * W))
                    ry = float(rng.uniform(0.02 * H, 0.05 * H))
                    open_mask |= (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0)
            for _ in range(int(rng.integers(2, 5))):
                cx = float(rng.uniform(0.08 * W, 0.92 * W))
                cy = float(rng.uniform(0.08 * H, 0.92 * H))
                rx = float(rng.uniform(0.03 * W, 0.07 * W))
                ry = float(rng.uniform(0.03 * H, 0.07 * H))
                open_mask |= (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0)

        road_bonus = box_mean(road_mask.astype(np.float32), max(16, min(H, W) // 60), max(16, min(H, W) // 60))
        city_prob = (0.62 * settlement_field.astype(np.float32)
                     + 0.20 * road_bonus.astype(np.float32)
                     + (0.18 * urban_backbone.astype(np.float32) if use_hills else 0.0)).astype(np.float32)
        city_prob = (city_prob - city_prob.min()) / (city_prob.max() - city_prob.min() + 1e-9)

        candidate_mask = (~road_mask) & (~open_mask)
        if use_hills:
            # 仅排除“硬通道骨架”，允许城市在 corridor 两侧继续扩展，减少单一椭圆城市团块。
            candidate_mask &= (~(protected_corridor_mask & (corridor_pref >= 0.76)))
            candidate_mask &= (flatness >= 0.16)
            candidate_mask &= (slope_n <= 0.56)
            candidate_mask &= (local_relief <= 0.62)
            candidate_mask &= (elevation_n <= 0.72)

        total_cells = H * W
        target_cells = int(round(float(city_density) * total_cells))
        if use_hills:
            candidate_count = int(np.count_nonzero(candidate_mask))
            # 对 hill_city 来说，density=0.24 应更接近“可建设区域基本已城市化”，
            # 剩余空间主要被道路、退界、小广场占掉，而不是大片空地。
            desired_fill_ratio = np.clip(0.18 + 1.45 * city_density, 0.28, 0.62)
            target_cells = max(target_cells, int(round(desired_fill_ratio * candidate_count)))
        target_cells = max(0, min(total_cells, target_cells))

        occupied = np.zeros((H, W), dtype=bool)
        if target_cells > 0 and np.any(candidate_mask):
            base_rank = (
                0.34 * city_prob.astype(np.float32)
                + 0.24 * terrain_suitability.astype(np.float32)
                + 0.14 * road_bonus.astype(np.float32)
                + (0.10 * corridor_pref.astype(np.float32) if use_hills else 0.0)
                + (0.10 * local_flat.astype(np.float32) if use_hills else 0.0)
                + (0.08 * flatness.astype(np.float32) if use_hills else 0.0)
                - (0.10 * slope_n.astype(np.float32) if use_hills else 0.0)
                - (0.10 * local_relief.astype(np.float32) if use_hills else 0.0)
                - (0.06 * np.maximum(elevation_n - 0.64, 0.0).astype(np.float32) if use_hills else 0.0)
            ).astype(np.float32)
            base_rank += 0.08 * box_mean(base_rank.astype(np.float32), max(20, min(H, W) // 42), max(20, min(H, W) // 42)).astype(np.float32)
            base_rank[~candidate_mask] = -1e9

            cand_scores = base_rank[candidate_mask]
            cand_count = int(cand_scores.size)
            choose_count = min(target_cells, cand_count)
            if choose_count > 0:
                q = float(np.quantile(cand_scores, max(0.0, 1.0 - choose_count / max(1, cand_count))))
                occupied = candidate_mask & (base_rank >= q)

                # 再做一次“团簇化”重排：优先保留成片、近路、近走廊、地形平稳区域，
                # 避免碎片化椒盐分布，让城市更像沿山谷/道路生长的团块。
                cluster_r = max(10, min(H, W) // 96)
                cluster_density = box_mean(occupied.astype(np.float32), cluster_r, cluster_r)
                blended_rank = (
                    0.58 * base_rank.astype(np.float32)
                    + 0.26 * cluster_density.astype(np.float32)
                    + 0.08 * road_bonus.astype(np.float32)
                    + (0.08 * corridor_pref.astype(np.float32) if use_hills else 0.0)
                ).astype(np.float32)
                blended_rank[~candidate_mask] = -1e9
                blended_scores = blended_rank[candidate_mask]
                q2 = float(np.quantile(blended_scores, max(0.0, 1.0 - choose_count / max(1, blended_scores.size))))
                occupied = candidate_mask & (blended_rank >= q2)

                # 去除过小碎块，同时保留大团块边缘，避免细碎孤岛障碍。
                occ_local = box_mean(occupied.astype(np.float32), 2, 2)
                occ_near = box_mean(occupied.astype(np.float32), max(5, min(H, W) // 240), max(5, min(H, W) // 240))
                occupied &= (occ_local >= 0.14) | (occ_near >= 0.30)

                # 若碎块清理后数量下降，则按 blended_rank 从高到低补回，保证密度仍接近目标值。
                occ_now = int(np.count_nonzero(occupied))
                if occ_now < choose_count:
                    need = choose_count - occ_now
                    residual_mask = candidate_mask & (~occupied)
                    if np.any(residual_mask):
                        ry, rx = np.where(residual_mask)
                        residual_scores = blended_rank[ry, rx]
                        order = np.argsort(residual_scores)[::-1]
                        pick = order[:need]
                        occupied[ry[pick], rx[pick]] = True

        build_score = (
            0.38 * city_prob.astype(np.float32)
            + 0.30 * terrain_suitability.astype(np.float32)
            + 0.18 * road_bonus.astype(np.float32)
            + (0.10 * flatness.astype(np.float32) if use_hills else 0.0)
            - (0.12 * slope_n.astype(np.float32) if use_hills else 0.0)
            - (0.08 * local_relief.astype(np.float32) if use_hills else 0.0)
        ).astype(np.float32)
        build_score[road_mask] -= 10.0
        build_score[open_mask] -= 10.0
        if use_hills:
            build_score[protected_corridor_mask & (corridor_pref >= 0.80)] -= 12.0
            build_score[(corridor_pref >= 0.48) & (~road_mask)] -= 2.0

        height_metric = ground.copy()
        local_int = build_score
        occ = np.zeros((H, W), dtype=bool)
        remaining = occupied.copy()
        ys, xs = np.where(remaining)
        order = np.argsort((local_int[ys, xs] + 0.10 * flatness[ys, xs]).astype(np.float32))[::-1]
        ys, xs = ys[order], xs[order]
        max_local_int = float(np.max(local_int) + 1e-9)

        footprint_density = box_mean(occupied.astype(np.float32), max(14, min(H, W) // 72), max(14, min(H, W) // 72))
        blocked_buffer = np.zeros((H, W), dtype=bool)

        for y, x in zip(ys.tolist(), xs.tolist()):
            if not remaining[y, x] or blocked_buffer[y, x]:
                continue
            li = float(local_int[y, x])
            core_signal = li / max_local_int
            zone_r = rng.random()
            local_flatness = float(flatness[y, x])
            local_slope = float(slope_n[y, x]) if use_hills else 0.0
            local_cov = float(footprint_density[y, x])

            # Increase inter-building spacing for hill_city/city maps: a density like 0.24 should
            # feel closer to "most buildable land has been urbanized", with extra land consumed by
            # roads / setbacks / small open spaces, rather than wall-to-wall footprints.
            if core_signal > 0.84 and zone_r < 0.10 and local_flatness > 0.72 and local_slope < 0.18:
                bw = int(rng.integers(8, 12))
                bh = int(rng.integers(8, 12))
                h = q3(rng.uniform(72.0, 108.0) if use_hills else rng.uniform(96.0, 132.0))
                gap = int(rng.integers(11, 16))
            elif core_signal > 0.66 and zone_r < 0.30 and local_flatness > 0.54:
                bw = int(rng.integers(7, 10))
                bh = int(rng.integers(7, 10))
                h = q3(rng.uniform(24.0, 54.0) if use_hills else rng.uniform(30.0, 78.0))
                gap = int(rng.integers(9, 13))
            elif local_cov > 0.22 and zone_r < 0.55:
                bw = int(rng.integers(5, 7))
                bh = int(rng.integers(5, 7))
                h = q3(rng.uniform(12.0, 24.0))
                gap = int(rng.integers(7, 10))
            elif zone_r < 0.38:
                bw = int(rng.integers(5, 8))
                bh = int(rng.integers(5, 8))
                h = q3(rng.uniform(15.0, 27.0))
                gap = int(rng.integers(7, 10))
            else:
                bw = int(rng.integers(4, 6))
                bh = int(rng.integers(4, 6))
                h = q3(rng.uniform(9.0, 18.0))
                gap = int(rng.integers(5, 8))

            x0 = max(0, x - bw // 2)
            y0 = max(0, y - bh // 2)
            x1 = min(W, x0 + bw)
            y1 = min(H, y0 + bh)
            mask = remaining[y0:y1, x0:x1]
            if use_hills:
                region_flat = flatness[y0:y1, x0:x1]
                region_slope = slope_n[y0:y1, x0:x1]
                mask = mask & (region_flat >= 0.14)
                if not np.any(mask):
                    continue
                mean_region_slope = float(np.mean(region_slope[mask]))
                if mean_region_slope > 0.54:
                    continue
                if mean_region_slope > 0.34:
                    h = min(h, 15.0)
                elif mean_region_slope > 0.24:
                    h = min(h, 24.0)
            if not np.any(mask):
                continue
            occ[y0:y1, x0:x1] |= mask
            height_metric[y0:y1, x0:x1][mask] = ground[y0:y1, x0:x1][mask] + h
            remaining[y0:y1, x0:x1][mask] = False

            bx0 = max(0, x0 - gap)
            by0 = max(0, y0 - gap)
            bx1 = min(W, x1 + gap)
            by1 = min(H, y1 + gap)
            blocked_buffer[by0:by1, bx0:bx1] |= True
            blocked_buffer[y0:y1, x0:x1][mask] = False

        missing = occupied & (~occ)
        if np.any(missing):
            # Fill remaining target cells with small parcels instead of single pixels, so the city keeps
            # visible spacing while still reaching the requested density.
            fallback_h = q3(12.0 if use_hills else 15.0)
            fill_idx = np.argsort(local_int[missing].reshape(-1))[::-1]
            miss_y, miss_x = np.where(missing)
            fallback_gap = 5 if use_hills else 4
            for k in fill_idx.tolist():
                y = int(miss_y[k]); x = int(miss_x[k])
                if blocked_buffer[y, x]:
                    continue
                fw = int(rng.integers(2, 4))
                fh = int(rng.integers(2, 4))
                x0 = max(0, x - fw // 2)
                y0 = max(0, y - fh // 2)
                x1 = min(W, x0 + fw)
                y1 = min(H, y0 + fh)
                patch = missing[y0:y1, x0:x1] & (~blocked_buffer[y0:y1, x0:x1])
                if not np.any(patch):
                    continue
                occ[y0:y1, x0:x1] |= patch
                height_metric[y0:y1, x0:x1][patch] = ground[y0:y1, x0:x1][patch] + fallback_h
                blocked_buffer[max(0, y0-fallback_gap):min(H, y1+fallback_gap), max(0, x0-fallback_gap):min(W, x1+fallback_gap)] = True
                blocked_buffer[y0:y1, x0:x1][patch] = False
            still_missing = occupied & (~occ)
            if np.any(still_missing):
                occ[still_missing] = True
                height_metric[still_missing] = ground[still_missing] + fallback_h

        threat = occ.astype(np.float32) * 2.4
        occ_u8 = occ.astype(np.uint8)
        I = np.pad(occ_u8, ((1, 0), (1, 0)), mode='constant', constant_values=0).cumsum(axis=0).cumsum(axis=1)

        def dilate_box(radius: int) -> np.ndarray:
            y0 = np.clip(np.arange(H) - radius, 0, H)
            y1 = np.clip(np.arange(H) + radius + 1, 0, H)
            x0 = np.clip(np.arange(W) - radius, 0, W)
            x1 = np.clip(np.arange(W) + radius + 1, 0, W)
            win = I[y1[:, None], x1[None, :]] - I[y0[:, None], x1[None, :]] - I[y1[:, None], x0[None, :]] + I[y0[:, None], x0[None, :]]
            return win > 0

        prev = occ.copy()
        for r in range(1, 7):
            cur = dilate_box(r)
            ring = cur & (~prev)
            threat[ring] += np.float32(1.2 * np.exp(-r / 4.0))
            prev = cur
        threat[road_mask & (~occ)] *= 0.55
        if use_hills:
            threat[protected_corridor_mask & (~occ)] *= 0.78
            threat += 0.45 * slope_n.astype(np.float32)
        threat += 0.03 * fbm_2d(H, W, rng, base_grid=max(24, min(H, W) // 14), octaves=3).astype(np.float32)
        env = GridEnv(
            occ,
            threat,
            resolution=5.0,
            height=height_metric.astype(np.float32),
            z_min=0.0,
            z_max=float(np.max(height_metric)) + 100.0,
            min_clearance=6.0,
            default_cruise_altitude=24.0,
            z_resolution=3.0,
        )

        b_heights = (height_metric - ground)[occ]
        low = float(np.mean(b_heights <= 18.0)) if b_heights.size else 0.0
        mid = float(np.mean((b_heights > 18.0) & (b_heights <= 27.0))) if b_heights.size else 0.0
        high = float(np.mean((b_heights > 27.0) & (b_heights <= 100.0))) if b_heights.size else 0.0
        supertall = float(np.mean(b_heights > 100.0)) if b_heights.size else 0.0
        meta = {
            'type': 'hill_city_map' if use_hills else 'city_map',
            'terrain_type': terrain_type,
            'xy_resolution_m': 5.0,
            'z_resolution_m': 3.0,
            'seed': int(seed),
            'H': int(H),
            'W': int(W),
            'city_density': float(city_density),
            'density_target': float(city_density),
            'density_actual': float(np.mean(occ)),
            'map_size': {'H': int(H), 'W': int(W), 'tag': size_tag},
            'z_min': float(env.z_min),
            'z_max': float(env.z_max),
            'min_clearance': float(env.min_clearance),
            'default_cruise_altitude': float(env.default_cruise_altitude),
            'lowrise_ratio_occ': low,
            'midrise_ratio_occ': mid,
            'highrise_ratio_occ': high,
            'supertall_ratio_occ': supertall,
            'settlement_center_count': int(len(settlement_centers)),
            'morphology_seed': int(morphology_seed),
            'density_key': int(density_key),
            'density_morph_bias': float(density_morph_bias),
            'density_affects_morphology': True,
        }
        if use_hills:
            meta['hill_scale'] = float(hill_scale)
            meta['mean_slope_occ'] = float(np.mean(slope_n[occ])) if np.any(occ) else 0.0
            meta['urban_hill_count'] = int(inner_hill_count)
            meta['outer_mountain_count'] = int(outer_mountain_count)
            meta['max_ground_height'] = float(np.max(ground))
            meta['mean_ground_height'] = float(np.mean(ground))
        return env, height_metric.astype(np.float32), meta

    def save_npz(self, path: str, height: np.ndarray | None = None, meta: dict | None = None):
        if meta is None:
            meta = {}
        meta = dict(meta)
        meta.setdefault('z_min', float(self.z_min))
        meta.setdefault('z_max', float(self.z_max))
        meta.setdefault('min_clearance', float(self.min_clearance))
        meta.setdefault('default_cruise_altitude', float(self.default_cruise_altitude))
        meta.setdefault('xy_resolution_m', float(self.resolution))
        meta.setdefault('z_resolution_m', float(self.z_resolution))
        meta_str = json.dumps(meta, ensure_ascii=False)
        payload = {
            'occupancy': self.occupancy.astype(np.uint8),
            'threat': self.threat.astype(np.float32),
            'meta': np.array(meta_str),
            'height': (self.height if height is None else height).astype(np.float32),
        }
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: str) -> tuple["GridEnv", np.ndarray | None, dict]:
        data = np.load(path, allow_pickle=True)
        occ = data['occupancy'].astype(np.uint8).astype(bool)
        threat = data['threat'].astype(np.float32)
        height = data['height'].astype(np.float32) if 'height' in data else None
        meta = {}
        if 'meta' in data:
            try:
                meta = json.loads(str(data['meta']))
            except Exception:
                meta = {}
        env = GridEnv(
            occ,
            threat,
            resolution=float(meta.get('xy_resolution_m', 1.0)),
            height=height,
            z_min=float(meta.get('z_min', 0.0)),
            z_max=float(meta.get('z_max', np.max(height) + 64.0 if height is not None else 64.0)),
            min_clearance=float(meta.get('min_clearance', 6.0)),
            default_cruise_altitude=float(meta.get('default_cruise_altitude', 12.0)),
            z_resolution=float(meta.get('z_resolution_m', 1.0)),
        )
        return env, height, meta


def _density_key(city_density: float) -> int:
    return int(round(float(city_density) * 1000.0))


def _mix_seed_with_density(seed: int, density_key: int, salt: int = 0) -> int:
    seed_u = int(seed) & 0xFFFFFFFF
    dens_u = int(density_key) & 0xFFFFFFFF
    mixed = (seed_u * 1664525 + 1013904223 + dens_u * 2246822519 + int(salt) * 3266489917) & 0xFFFFFFFF
    mixed ^= (mixed >> 16)
    mixed = (mixed * 2246822519) & 0xFFFFFFFF
    mixed ^= (mixed >> 13)
    mixed = (mixed * 3266489917) & 0xFFFFFFFF
    mixed ^= (mixed >> 16)
    return int(mixed)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def _value_noise_2d(H: int, W: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    gh = max(2, H // grid + 2)
    gw = max(2, W // grid + 2)
    g = rng.random((gh, gw)).astype(np.float32)
    yy = np.linspace(0, gh - 2, H, dtype=np.float32)
    xx = np.linspace(0, gw - 2, W, dtype=np.float32)
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    ty = _smoothstep(yy - y0)
    tx = _smoothstep(xx - x0)
    y0v = y0[:, None]
    y1v = y1[:, None]
    x0v = x0[None, :]
    x1v = x1[None, :]
    v00 = g[y0v, x0v]
    v01 = g[y0v, x1v]
    v10 = g[y1v, x0v]
    v11 = g[y1v, x1v]
    a = v00 * (1 - tx)[None, :] + v01 * tx[None, :]
    b = v10 * (1 - tx)[None, :] + v11 * tx[None, :]
    return (a * (1 - ty)[:, None] + b * ty[:, None]).astype(np.float32)


def fbm_2d(H: int, W: int, rng: np.random.Generator, base_grid: int = 48, octaves: int = 5, gain: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    grid = float(base_grid)
    norm = 0.0
    for _ in range(int(octaves)):
        out += amp * _value_noise_2d(H, W, max(2, int(round(grid))), rng)
        norm += amp
        amp *= gain
        grid /= lacunarity
        grid = max(2.0, grid)
    return (out / max(1e-9, norm)).astype(np.float32)


def ridged_fbm_2d(H: int, W: int, rng: np.random.Generator, base_grid: int = 56, octaves: int = 5) -> np.ndarray:
    x = fbm_2d(H, W, rng, base_grid=base_grid, octaves=octaves)
    y = 1.0 - np.abs(2.0 * x - 1.0)
    y = y * y
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    return y.astype(np.float32)


def domain_warp(height: np.ndarray, rng: np.random.Generator, strength: float = 20.0) -> np.ndarray:
    H, W = height.shape
    wx = fbm_2d(H, W, rng, base_grid=max(8, W // 8), octaves=4)
    wy = fbm_2d(H, W, rng, base_grid=max(8, H // 8), octaves=4)
    wx = (wx * 2.0 - 1.0) * strength
    wy = (wy * 2.0 - 1.0) * strength
    yy, xx = np.mgrid[0:H, 0:W]
    xs = np.clip(np.rint(xx + wx), 0, W - 1).astype(np.int32)
    ys = np.clip(np.rint(yy + wy), 0, H - 1).astype(np.int32)
    out = height[ys, xs]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out.astype(np.float32)


def carve_valleys(height: np.ndarray, iterations: int = 6, strength: float = 0.25) -> np.ndarray:
    h = height.astype(np.float32).copy()
    for _ in range(int(iterations)):
        up = np.roll(h, 1, axis=0)
        dn = np.roll(h, -1, axis=0)
        lf = np.roll(h, 1, axis=1)
        rg = np.roll(h, -1, axis=1)
        avg4 = 0.25 * (up + dn + lf + rg)
        h = h * (1.0 - strength) + np.minimum(h, avg4) * strength
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    return h.astype(np.float32)


def minecraft_terrace(height: np.ndarray, levels: int = 24, jitter: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    h = height.astype(np.float32)
    if rng is not None and jitter > 0:
        h = h + rng.normal(0.0, jitter, size=h.shape).astype(np.float32)
    h = np.clip(h, 0.0, 1.0)
    h = np.floor(h * float(levels)) / max(1.0, float(levels))
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    return h.astype(np.float32)
