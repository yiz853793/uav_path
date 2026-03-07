# UAV Path Planning on 2D Grid (RRT* + MOEA/D)

本项目用于**二维栅格地图**上的无人机路径规划（连续坐标表示 + 栅格碰撞/威胁查询），包含：

- **RRT\***：单目标（找一条可行路径），用于对比和 sanity check
- **MOEA/D**：多目标优化（路径长度 / 威胁 / 能量），输出 Pareto 解集
- **山地地形生成**：可保存/加载 `.npz`（occupancy/threat/height/meta）
- **批量 benchmark**：自动跑多个地形种子并输出汇总日志 + 可视化

> 地图尺寸三档：
> - **S**：160×200
> - **M**：480×600（S 的 3×）
> - **L**：1600×2000（S 的 10×）

---

## 1. 环境安装

建议 Python 3.9+（3.8 亦可）。

```bash
pip install -r requirements.txt
```

依赖：`numpy`、`matplotlib`。

---

## 2. 入口与参数总览

| 入口 | 用途 |
|------|------|
| `python main.py` | 单次：生成地形 + RRT* + MOEA/D，输出图与 log |
| `python -m experiments.gen_terrain` | 单张地形生成并保存 npz |
| `python -m experiments.gen_dataset` | 批量生成多张地形（按 seed 范围） |
| `python -m experiments.plan_from_terrain` | 从已有 npz 做单次规划（RRT* 或 MOEA/D） |
| `python -m experiments.run_benchmark` | 批量：对多张地形跑 RRT*+MOEA/D，汇总 + 图 |

以下各节给出**每个入口的完整参数表**与**推荐命令示例**。

---

## 3. main.py（单次生成+规划）

### 3.1 参数说明

**地形与地图**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--terrain_seed` | int | 35 | 地形随机种子 |
| `--planner_seed` | int | 0 | 规划器（RRT*/MOEA/D）随机种子 |
| `--size` | str | small | 地图预设：small / medium / large |
| `--H` | int | 0 | 覆盖高度（0 表示用 --size 预设） |
| `--W` | int | 0 | 覆盖宽度（0 表示用 --size 预设） |
| `--terrace_levels` | int | 28 | 地形阶梯层数 |
| `--inflate` | int | 1 | 障碍膨胀格数（安全距离） |
| `--start` | float float | 无 | 起点 (x,y)，不传则自动 |
| `--goal` | float float | 无 | 终点 (x,y)，不传则自动 |
| `--terrain_dir` | str | terrains | 地形 npz 保存根目录 |
| `--out_root` | str | outputs | 结果与日志根目录 |

**RRT\***

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--rrt_iter` | int | 50000 | RRT* 采样迭代次数（S 5万 / M 10万 / L 20万 会按尺寸覆盖默认） |

**MOEA/D 核心**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--moead_gen` | int | 80 | 进化代数 |
| `--moead_pop` | int | 60 | 种群大小 |
| `--K` | int | 30 | 路径点数量（编码维度） |
| `--moead_T` | int | 10 | 邻域大小 T |

**A* 初始化（MOEA/D 种子）**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--init_astar_ratio` | float | 0.25 | 初始种群中来自 A* 骨架的比例 |
| `--init_astar_threat_weight` | float | 0.0 | A* 中威胁权重（0 表示忽略） |
| `--init_astar_jitter_sigma` | float | 1.5 | A* 路径点高斯扰动标准差 |
| `--init_astar_max_paths` | int | 5 | 最多生成的 A* 多样化路径条数 |
| `--init_astar_penalty_step` | float | 2.5 | 每找到一条 A* 后对经过格子加的惩罚，促下一条走不同走廊 |
| `--init_astar_max_expansions` | int | 无 | A* 扩展步数上限，不传则按地图面积自动（大图会放大） |

**Debug**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--debug` | flag | 关 | 开启 MOEA/D 调试日志（写文件并实时打屏） |
| `--debug_log` | str | 无 | 调试日志路径，不传且 --debug 时为 out_dir/moead_debug.log |
| `--debug_every` | int | 1 | 每多少代打印一条日志 |
| `--debug_level` | 1/2/3 | 2 | 1=简要 2=含耗时 3=含碰撞统计 |
| `--no_debug_console` | flag | 关 | 与 --debug 同用时仅写文件、不实时打屏 |

### 3.2 推荐命令示例（main.py）

**小图快速验证（默认）**

```bash
python main.py --size small
```

**中图、适中代数与种群**

```bash
python main.py --size medium --moead_gen 80 --moead_pop 60 --K 30
```

**大图、提高可行解比例与收敛（推荐）**

```bash
python main.py --size large --moead_gen 100 --moead_pop 80 --K 60 --init_astar_ratio 0.4 --init_astar_max_paths 8 --debug
```

**大图、不打屏只写 debug 文件**

```bash
python main.py --size large --moead_gen 100 --moead_pop 80 --K 60 --init_astar_ratio 0.4 --debug --no_debug_console
```

**大图、A* 扩展上限手动放大**

```bash
python main.py --size large --init_astar_max_expansions 5000000 --moead_gen 100 --K 60 --init_astar_ratio 0.4
```

---

## 4. experiments.gen_terrain（单张地形）

### 4.1 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--out_dir` | str | terrains | 地形保存根目录，实际写入 out_dir/<size_tag>/ |
| `--seed` | int | 0 | 地形随机种子 |
| `--size` | str | small | small / medium / large |
| `--H` | int | 0 | 覆盖高度 |
| `--W` | int | 0 | 覆盖宽度 |
| `--terrace_levels` | int | 28 | 阶梯层数 |
| `--obstacle_height` | float | 0.78 | 障碍高度阈值 |

### 4.2 推荐命令示例

```bash
# 小图
python -m experiments.gen_terrain --size small --seed 7 --out_dir terrains

# 中图 / 大图
python -m experiments.gen_terrain --size medium --seed 7
python -m experiments.gen_terrain --size large --seed 7
```

---

## 5. experiments.gen_dataset（批量地形）

### 5.1 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--out_dir` | str | terrains | 地形保存根目录，按 size 分桶到 out_dir/S|M|L/ |
| `--seed_from` | int | 0 | 起始 seed（含） |
| `--seed_to` | int | 50 | 结束 seed（不含） |
| `--size` | str | small | small / medium / large |
| `--H` | int | 0 | 覆盖高度 |
| `--W` | int | 0 | 覆盖宽度 |
| `--terrace_levels` | int | 28 | 阶梯层数 |
| `--obstacle_height` | float | 0.78 | 障碍高度阈值 |

### 5.2 推荐命令示例

**小图 20 张**

```bash
python -m experiments.gen_dataset --size small --seed_from 0 --seed_to 100 --out_dir terrains --obstacle_height 0.5
```

**大图 20 张（适中）**

```bash
python -m experiments.gen_dataset --size large --seed_from 0 --seed_to 20 --out_dir terrains
```

**大图 50 张（全量）**

```bash
python -m experiments.gen_dataset --size large --seed_from 0 --seed_to 50 --out_dir terrains
```

---

## 6. experiments.plan_from_terrain（单地形规划）

### 6.1 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--terrain` | str | 必填 | npz 地形文件路径 |
| `--planner` | str | rrt | rrt / moead |
| `--inflate` | int | 1 | 障碍膨胀格数 |
| `--seed` | int | 0 | 规划器种子 |
| `--rrt_iter` | int | 4000 | RRT* 迭代次数（仅 planner=rrt） |
| `--moead_gen` | int | 80 | MOEA/D 代数（仅 planner=moead） |
| `--moead_pop` | int | 60 | MOEA/D 种群（仅 planner=moead） |
| `--K` | int | 30 | 路径点数（仅 planner=moead） |

### 6.2 推荐命令示例

```bash
# 仅 RRT*
python -m experiments.plan_from_terrain --terrain terrains/S/mountain_seed0007.npz --planner rrt --seed 0

# 仅 MOEA/D
python -m experiments.plan_from_terrain --terrain terrains/S/mountain_seed0007.npz --planner moead --moead_gen 80 --K 30
```

---

## 7. experiments.run_benchmark（批量 benchmark）

### 7.1 参数说明

**输入与输出**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--terrain_dir` | str | terrains | 地形目录，内含 mountain_seedXXXX.npz |
| `--out_root` | str | outputs | 结果根目录，按 size_tag/seedXXXX 分桶 |
| `--seed_from` | int | 0 | 起始 seed（含） |
| `--seed_to` | int | 50 | 结束 seed（不含），与 terrain_dir 配合 |
| `--glob` | str | 空 | 若指定则忽略 terrain_dir/seed 范围，用 glob 匹配文件，如 terrains/*/mountain_seed*.npz |
| `--summary_log` | str | 空 | 汇总日志路径，默认 out_root/benchmark_summary.log |
| `--start` | float float | 无 | 起点，不传则按地图尺寸自动 |
| `--goal` | float float | 无 | 终点，不传则按地图尺寸自动 |

**规划与膨胀**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--planner_seed` | int | 0 | 规划器种子 |
| `--inflate` | int | 1 | 障碍膨胀格数 |
| `--rrt_iter` | int | 4000 | RRT* 迭代次数（大图建议 10万～20万） |

**MOEA/D 核心**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--moead_gen` | int | 80 | 进化代数 |
| `--moead_pop` | int | 60 | 种群大小 |
| `--K` | int | 30 | 路径点数量 |
| `--moead_T` | int | 10 | 邻域大小 T |

**A* 初始化**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--init_astar_ratio` | float | 0.25 | 初始种群中 A* 骨架占比 |
| `--init_astar_threat_weight` | float | 0.0 | A* 威胁权重 |
| `--init_astar_jitter_sigma` | float | 1.5 | A* 路径点扰动标准差 |
| `--init_astar_max_paths` | int | 5 | 最多 A* 路径条数 |
| `--init_astar_penalty_step` | float | 2.5 | A* 重复路径惩罚步长 |

**MOEA/D 性能与调试**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--moead_eval_step` | float | 0.5 | 评估采样步长（越大越快、越粗糙） |
| `--moead_smooth_step` | float | 0.5 | 捷径平滑碰撞检测步长 |
| `--moead_debug` | flag | 关 | 每个地形写 moead_debug.log |
| `--moead_debug_every` | int | 1 | 每 N 代写一行 |
| `--moead_debug_level` | int | 2 | 1=简要 2=耗时 3=碰撞统计 |

### 7.2 推荐命令示例

**小图、少量种子快速跑**

```bash
python -m experiments.run_benchmark --terrain_dir terrains/S --seed_from 0 --seed_to 10 --out_root outputs --rrt_iter 10000 --moead_gen 80 --moead_pop 60 --K 30
```

**中图、适中规模**

```bash
python -m experiments.run_benchmark --terrain_dir terrains/M --seed_from 0 --seed_to 10 --out_root outputs --rrt_iter 20000 --moead_gen 80 --moead_pop 60 --K 60 --moead_T 10
```

**大图、适中规模（推荐）**

```bash
python -m experiments.run_benchmark --terrain_dir terrains/L --seed_from 0 --seed_to 20 --out_root outputs --rrt_iter 40000 --moead_gen 100 --moead_pop 80 --K 60 --init_astar_ratio 0.4 --init_astar_max_paths 8
```

**大图、提速（粗评估步长）**

```bash
python -m experiments.run_benchmark --terrain_dir terrains/L --seed_from 0 --seed_to 10 --rrt_iter 200000 --moead_gen 80 --moead_pop 80 --K 60 --init_astar_ratio 0.4 --moead_eval_step 1.0 --moead_smooth_step 1.0 --out_root outputs
```

**多尺寸混跑（glob）**

```bash
python -m experiments.run_benchmark --glob "terrains/*/mountain_seed*.npz" --seed_from 0 --seed_to 5 --out_root outputs --rrt_iter 50000 --moead_gen 60 --moead_pop 60 --K 30
```

**大图、带 debug 日志**

```bash
python -m experiments.run_benchmark --terrain_dir terrains/L --seed_from 0 --seed_to 5 --rrt_iter 200000 --moead_gen 100 --moead_pop 80 --K 60 --init_astar_ratio 0.4 --moead_debug --moead_debug_every 1 --moead_debug_level 2 --moead_eval_step 1.0 --out_root outputs
```

---

## 8. 输出文件说明（outputs/）

单次 main 或 run_benchmark 后，典型结构：

```
outputs/
  S|M|L/
    seedXXXX/
      paths_*.png           # 禁飞区 + Start/Goal + RRT* + 4 条 MOEA/D 代表解
      pareto3d_*.png        # Pareto 3D 散点（main 生成）
      pareto_points_*.json  # Pareto 点集 + 代表解索引（run_benchmark）
      metrics_*.json        # 运行时间、成功与否、archive 大小、map_size 等
      moead_debug.log       # 仅当开启 debug 时
  benchmark_summary.log     # 仅 run_benchmark：整批汇总
```

- **paths_*.png**：红色为禁飞区，叠加起点、终点、RRT* 与 MOEA/D 代表路径。
- **metrics_*.json**：含 `map_size`（H/W/tag）、耗时、archive 大小、A* 相关参数等。
- **moead_debug.log**：每代 feasible/archive、耗时分解（eval/smooth/repair/neigh）等。

---

## 9. 地图尺寸与 Start/Goal

- 栅格中 `H` 为行数、`W` 为列数，坐标 `(x,y)` 满足 `x∈[0,W)`, `y∈[0,H)`。
- 不传 `--start` / `--goal` 时：`start=(5,5)`，`goal=(W-60, H-70)`（有最小间距兜底）。
- 自定义尺寸（非 S/M/L）时建议显式指定起点、终点。

---

## 10. 常见问题（FAQ）

### Q1: 大图（L）很慢？

- L 为 1600×2000，碰撞与威胁采样次数多；主要瓶颈是 **eval**（路径评估）。
- 建议：`run_benchmark` 使用 `--moead_eval_step 1.0` 或 `2.0`、`--moead_smooth_step 1.0` 提速（略损精度）。
- main.py 下大图已自动提高 RRT* 迭代（L 约 20 万）；run_benchmark 需手动给 `--rrt_iter 200000`。

### Q2: 大图 feasible 为 0？

- 已做两点改进：A* 扩展上限按地图面积放大；若平滑后路径不可行则回退为仅 densify 的 A* 路径。
- 若仍为 0：增大 `--init_astar_ratio`（如 0.4）、`--init_astar_max_paths`（如 8）；main 可试 `--init_astar_max_expansions 5000000`。

### Q3: K 增大后 feasible 下降？

- K 越大约束越多（段数、转角），可行域更窄，正常。大图建议 K=60～90，不必一味 300；若用大 K 可提高 `init_astar_ratio` 或略减 jitter。

### Q4: benchmark_summary.log 里如何看地图大小？

- run_benchmark 会在 summary 中输出单尺寸的 `Map size: H=..., W=... (tag=...)`，混跑时会列出各尺寸数量。

---

## 11. 目录结构

```
.
├── main.py
├── readme.md
├── requirements.txt
├── experiments/
│   ├── gen_terrain.py      # 单张地形
│   ├── gen_dataset.py      # 批量地形
│   ├── plan_from_terrain.py # 单地形规划
│   └── run_benchmark.py    # 批量 benchmark
├── src/
│   ├── algorithms/         # a_star, rrt_star, moead
│   ├── env/               # GridEnv, collision
│   ├── models/             # objectives, constraints, evaluator, path
│   └── viz/                # plot
├── terrains/               # 地形 npz（按 S/M/L 分桶）
└── outputs/                # 结果与日志
```

---

## 12. License

课程/毕设用途：可自由修改、扩展，并将核心实验配置写入论文以便复现。

python -m experiments.run_benchmark \
  --terrain_dir terrains/L \
  --seed_from 0 --seed_to 100 \
  --out_root outputs \
  --rrt_iter 6000 \
  --moead_gen 80 \
  --moead_pop 70 \
  --K 50 \
  --init_astar_ratio 0.45 \
  --init_astar_max_paths 8

