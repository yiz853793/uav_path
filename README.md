# UAV Path Planning with MOEA/D on 2.5D Terrain Grids

本项目面向 **2.5D 地形高度场上的无人机路径规划**，核心目标是在复杂场景中同时优化多项路径指标，并输出一组具有代表性的 Pareto 路径解。项目以 **MOEA/D** 为主线算法，同时提供 **RRT\*** 和 **PRM** 作为采样式基线，用于对比规划质量、稳定性与运行效率。

与只追求单一路径代价最小的规划器不同，本项目将路径规划建模为 **多目标优化问题**：在满足碰撞安全、离地高度、路径可行性等约束的前提下，同时考虑路径长度、威胁暴露和转向代价，最终得到一组互相权衡的候选路径。

---

## 1. 项目特点

- 以 **MOEA/D** 为核心规划器，输出 Pareto 前沿解集，而不是单一路径。
- 支持三类场景：`mountain`、`city`、`hill_city`。
- 场景数据统一包含：
  - `occupancy`：障碍占据图
  - `threat`：威胁场
  - `height`：地形或建筑高度
  - `meta`：地形类型、尺寸、密度等元信息
- 提供完整实验链路：
  - 地形生成
  - 单次规划
  - 批量 benchmark
  - 多 `planner_seed` 复现实验
  - baseline / full / ablation 套件
  - 可视化与结果汇总
- MOEA/D 实现不是“朴素版本”，而是结合了适合本问题的初始化、搜索强化与早停机制。

---

## 2. MOEA/D 在本项目中的定位

本项目的核心研究对象是 **MOEA/D（Multi-Objective Evolutionary Algorithm based on Decomposition）**。

在这里，MOEA/D 不只是一个通用多目标优化器，而是被专门适配到了无人机路径规划问题中。项目的主线是：

1. 将一条路径表示为可评估的候选个体。
2. 用多个目标函数衡量路径质量。
3. 用 MOEA/D 将多目标问题分解为一组子问题并协同优化。
4. 从最终档案中提取代表性路径，例如：
   - `min_f1`：路径长度最优
   - `min_f2`：威胁代价最优
   - `min_f3`：转向代价最优
   - `compromise`：综合折中解

### 2.1 本项目中 MOEA/D 重点优化的内容

相较于只做基础分解和邻域更新的标准 MOEA/D，本项目中的实现进一步加入了多项适配路径规划的机制：

- **A\* 引导初始化**：用 A\* 生成部分高质量 backbone，改善初始种群质量。
- **分层/分带初始化**：增强初始解的多样性，避免种群过早集中。
- **极值方向权重偏置**：在权重向量分布上更关注目标极值区域。
- **额外极值方向 offspring 分配**：对潜力更高的目标方向分配更多搜索资源。
- **方向性局部搜索**：周期性对精英个体做局部增强。
- **活跃子问题采样**：不必每代均匀处理全部子问题，而是优先处理更有价值的部分。
- **MTOE 早停机制**：在满足统计条件时提前终止，减少无效迭代。
- **Basin Shadow / escape 机制**：用于缓解停滞，提高逃离局部 basin 的能力。

因此，这个仓库中的 MOEA/D 更适合被理解为：

> 一个面向 UAV 多目标路径规划问题、经过工程化增强的 MOEA/D 实验平台。

---

## 3. 优化目标与问题形式

项目将路径规划视为带约束的多目标优化问题。当前实验主线主要围绕以下三个目标展开：

- **f1：路径长度 / 飞行距离**
- **f2：威胁代价 / 风险暴露**
- **f3：转向代价 / 路径平稳性相关指标**

同时，路径需要满足若干可行性要求，例如：

- 不与障碍物碰撞
- 满足一定安全间隔或膨胀障碍约束
- 满足路径离地高度与可飞行空域约束
- 满足平滑性相关限制

这使得项目不只是“找一条能走的路”，而是：

> 在复杂约束下，寻找一组在长度、风险与平稳性之间具有不同权衡关系的可行路径。

---

## 4. 支持的场景

### `mountain`
山地地形，支持三档尺寸：

- `small` → `160 × 200`
- `medium` → `480 × 600`
- `large` → `1600 × 2000`

### `city`
纯城市场景，固定为：

- `1600 × 2000`

### `hill_city`
山地城市混合场景，固定为：

- `1600 × 2000`

`hill_city` 是当前更适合作为主实验场景的配置，因为它同时包含城市障碍、地形起伏与飞行走廊约束，更接近低空复杂环境下的 UAV 规划任务。

---

## 5. 目录结构

```text
uav_path_planning_mtoe/
├─ README.md
├─ requirements.txt
├─ main.py
├─ experiments/                  # 只保留用户直接运行的脚本
│  ├─ gen_terrain.py             # 单张地形生成入口
│  ├─ gen_dataset.py             # 批量地形生成入口
│  ├─ plan_from_terrain.py       # 从已有 .npz 执行一次规划
│  ├─ run_benchmark.py           # 基线 benchmark 主入口
│  ├─ run_suite.py               # baseline/full 套件入口
│  ├─ run_ablation_suite.py      # 消融实验入口
│  ├─ render_from_vis_json.py    # 从 visdata 渲染图
│  └─ collect_results.py         # 汇总结果
├─ src/
│  ├─ experiment/                # experiments 的内部支撑逻辑
│  │  ├─ terrain_generation.py
│  │  ├─ benchmark_shared.py
│  │  ├─ benchmark_parser.py
│  │  ├─ bench_core.py
│  │  ├─ run_reproducibility.py
│  │  ├─ suite_runner.py
│  │  ├─ suite_variants.py
│  │  ├─ vis_data.py
│  │  └─ common_io.py
│  ├─ algorithms/
│  │  ├─ a_star.py
│  │  ├─ prm.py
│  │  ├─ rrt_star.py
│  │  └─ moead.py                # 核心 MOEA/D 实现
│  ├─ env/
│  │  ├─ grid_env.py
│  │  └─ collision.py
│  ├─ models/
│  │  ├─ path.py
│  │  ├─ objectives.py
│  │  ├─ constraints.py
│  │  └─ evaluator.py
│  └─ viz/
│     └─ plot.py
└─ test/
   ├─ __main__.py
   ├─ terrain_preview.py
   └─ analyze_prm_from_npz.py
```

---

## 6. 环境安装

依赖较轻，核心依赖如下：

- `numpy`
- `matplotlib`
- `plotly`

安装方式：

```bash
pip install -r requirements.txt
```

---

## 7. 快速开始

## 7.1 生成一张地形

### mountain

```bash
python -m experiments.gen_terrain --terrain_type mountain --size small --seed 7
```

### city

```bash
python -m experiments.gen_terrain --terrain_type city --city_density 0.24 --seed 0
```

### hill_city

```bash
python -m experiments.gen_terrain --terrain_type hill_city --city_density 0.24 --seed 0
```

生成后的地形会保存为 `.npz` 文件，供后续规划脚本直接读取。

---

## 7.2 批量生成地形数据集

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

常见输出示例：

```text
terrains/hill_city_0.24/hill_city_seed0000.npz
terrains/hill_city_0.24/hill_city_seed0001.npz
...
```

---

## 7.3 运行一次单场景规划

如果只是想快速验证算法是否正常，可以先跑单个 terrain：

```bash
python -m experiments.plan_from_terrain \
  --terrain terrains/hill_city_0.24/hill_city_seed0000.npz \
  --planner_seed 0 \
  --out_root outputs \
  --rrt_iter 6000 \
  --prm_samples 12000 \
  --prm_k 48 \
  --prm_max_edge_len 250 \
  --moead_pop 160 \
  --moead_min_gen 80 \
  --moead_max_gen 500
```

这个入口会同时跑：

- RRT\*
- PRM
- MOEA/D

并输出数值结果与可视化所需数据。

---

## 7.4 运行批量 benchmark

这是更通用的主实验入口，适合按 terrain seed 批量跑：

```bash
python -m experiments.run_benchmark \
  --terrain_type hill_city \
  --city_density 0.24 \
  --seed_from 0 \
  --seed_to 20 \
  --planner_seed 0 \
  --out_root outputs \
  --inflate 1 \
  --rrt_iter 6000 \
  --prm_samples 12000 \
  --prm_k 48 \
  --prm_max_edge_len 250 \
  --prm_threat_weight 0.0 \
  --moead_pop 160 \
  --moead_min_gen 80 \
  --moead_max_gen 500 \
  --mtoe_tol_fun 1e-5 \
  --mtoe_confidence 0.995 \
  --K 30 \
  --moead_T 16
```


适用场景：

- 先固定一个 `planner_seed`
- 扫一批 terrain seed
- 看不同地形上的整体表现

---

## 7.5 运行基线/完整套件

当前对外保留的是 baseline/full/ablation 入口；多 planner seed 的复现逻辑已经下沉到 `src/experiment/`，由套件脚本直接调用。

```bash
python -m experiments.run_suite \
  --out_root outputs/repro \
  --repeat_count 5 \
  --planner_seed_base 0 \
  --progress_every 1 \
  --suite paper \
  -- \
  --terrain_type hill_city \
  --city_density 0.24 \
  --seed_from 0 \
  --seed_to 20 \
  --inflate 1 \
  --rrt_iter 6000 \
  --prm_samples 12000 \
  --prm_k 48 \
  --prm_max_edge_len 250 \
  --prm_threat_weight 0.0 \
  --moead_pop 160 \
  --moead_min_gen 80 \
  --moead_max_gen 500 \
  --mtoe_tol_fun 1e-5 \
  --mtoe_confidence 0.995 \
  --K 30 \
  --moead_T 16
```

这个入口尤其适合：

- 做 baseline 与 full 方法对比
- 计算均值、标准差、成功率与停止行为
- 分析同一实验配置在不同随机种子下的稳定性

---

## 7.6 运行 baseline / full / ablation 套件

如果你已经确定了主参数配置，希望一口气跑整套论文实验，可以使用套件入口。

### 主套件

```bash
python -m experiments.run_suite \
  --out_root outputs/suite_hillcity024_s0to19_r5 \
  --no_timestamp \
  --repeat_count 5 \
  --planner_seed_base 0 \
  --progress_every 1 \
  --suite smoke \
  -- \
  --terrain_type hill_city \
  --city_density 0.24 \
  --seed_from 0 \
  --seed_to 20 \
  --inflate 1 \
  --rrt_iter 8000 \
  --prm_samples 12000 \
  --prm_k 48 \
  --prm_max_edge_len 250 \
  --prm_threat_weight 0.0 \
  --moead_pop 160 \
  --moead_min_gen 80 \
  --moead_max_gen 1500 \
  --mtoe_tol_fun 1e-5 \
  --mtoe_confidence 0.995 \
  --K 30 \
  --moead_T 16
```

### 消融套件

```bash
python -m experiments.run_ablation_suite \
  --out_root outputs/ablation_hillcity024_s0to19_r5 \
  --no_timestamp \
  --repeat_count 5 \
  --planner_seed_base 0 \
  --progress_every 1 \
  --suite smoke \
  -- \
  --terrain_type hill_city \
  --city_density 0.24 \
  --seed_from 0 \
  --seed_to 20 \
  --inflate 1 \
  --moead_pop 160 \
  --moead_min_gen 80 \
  --moead_max_gen 1500 \
  --mtoe_tol_fun 1e-5 \
  --mtoe_confidence 0.995 \
  --K 30 \
  --moead_T 16
```

当前套件中常见变体包括：

- `baseline`
- `full`
- `wo_init`
- `wo_search`
- `wo_mtoe`
- `base_plus_init`
- `base_plus_init_search`

这些变体定义位于 `experiments/suite_variants.py`。

---

## 8. 推荐的 MOEA/D 主实验参数思路

如果你的重点是验证 MOEA/D 的改进是否有效，可以优先围绕以下几类参数进行设计：

### 基础控制项

- `--moead_pop`
- `--moead_min_gen`
- `--moead_max_gen`
- `--K`
- `--moead_T`

### 初始化相关

- `--init_astar_ratio`
- `--init_astar_threat_weight`
- `--init_astar_max_paths`
- `--init_stratified_ratio`
- `--init_global_random_ratio`

### 搜索强化相关

- `--weight_extreme_bias`
- `--extreme_offspring_ratio`
- `--local_search_interval`
- `--local_search_elite_k`
- `--local_search_attempts_per_obj`
- `--active_subproblem_ratio`

### 终止与稳定性相关

- `--mtoe_tol_fun`
- `--mtoe_confidence`
- `--disable_mtoe_stop`
- `--basin_shadow_enable`
- `--basin_escape_injections`

一个很自然的实验逻辑是：

1. 先定义 `baseline`。
2. 再定义当前使用的 `full`。
3. 然后做模块级消融：
   - 去掉初始化增强
   - 去掉搜索强化
   - 去掉 MTOE 早停
4. 最后用 reproducibility 或 suite 进行统计对比。

---

## 9. 输出文件说明

当前项目默认输出已经相对精简，核心保留以下几类文件。

### case 级输出

- `metrics_*.json`：单次运行的主要数值结果
- `visdata_*.json`：后续可视化所需数据

### reproducibility 级输出

- `reproducibility_cases.csv`：所有 case 的汇总行
- `reproducibility_overall.csv`：总体统计
- `reproducibility_by_terrain.csv`：按 terrain seed 聚合统计
- `seedXXXX/reproducibility_summary.csv`：单个 terrain seed 下多 planner seed 的汇总

### suite 级输出

- `suite_manifest.json`：套件元信息与变体列表

如果你额外打开某些开关，还可以导出更细的调试与中间结果。

---

## 10. 可视化与结果汇总

### 从 `visdata_*.json` 渲染图

```bash
python -m experiments.render_from_vis_json --vis_json path/to/visdata_xxx.json
```

### 汇总已有结果

```bash
python -m experiments.collect_results --root outputs
```

### 结果分析

当前推荐先通过 `collect_results.py` 汇总结果，再配合 `render_from_vis_json.py` 生成路径图和 Pareto 可视化。

---

## 11. 基线算法说明

虽然本项目突出 MOEA/D，但仍然保留了两个重要基线：

### RRT\*
- 采样树方法
- 更适合快速得到单一路径
- 可作为单目标采样式规划基线

### PRM
- 概率路图方法
- 当前版本使用 **3D node PRM**，允许在障碍物上方合法空域采样节点
- 适合作为另一类 sample-based baseline

因此，这个项目不是“只有 MOEA/D 的单算法仓库”，而是：

> 以 MOEA/D 为核心、并配有 RRT\* / PRM 基线的 UAV 多目标路径规划实验框架。

---

## 12. 开发建议

如果你准备继续扩展项目，推荐优先从以下几个方向入手：

1. **继续围绕 MOEA/D 做研究性改进**
   - 更好的初始化
   - 更高效的邻域更新
   - 更稳定的早停策略
   - 更适合路径规划的局部搜索算子

2. **增强结果分析链路**
   - 自动生成论文表格
   - 自动绘制 Pareto 曲线和 representative path 图
   - 自动比较 baseline / full / ablation

3. **扩展场景与约束**
   - 更真实的 no-fly zone
   - 动态威胁场
   - 能耗、速度、爬升约束
   - 多机协同或时变规划

---

## 13. 适用人群

这个仓库特别适合以下场景：

- 做 **无人机路径规划** 本科毕设 / 课程项目
- 做 **多目标优化** 与 **进化算法** 相关实验
- 研究 **MOEA/D 在路径规划中的适配与增强**
- 需要一个包含地形生成、批量实验、复现实验、消融套件的实验平台

---

## 14. 一句话总结

如果只用一句话概括这个项目，可以写成：

> 一个面向 2.5D 复杂地形 UAV 路径规划任务、以 MOEA/D 为核心并支持完整实验链路的多目标优化研究平台。
