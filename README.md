<<<<<<< HEAD
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
=======
# UAV Path Planning on 2.5D Terrain Grid
## RRT* + 3D-node PRM + MOEA/D on mountain / city / hill_city occupancy-threat-height maps

本项目用于 **2.5D 地形高度场** 上的无人机路径规划。地图统一包含：

- `occupancy`：建筑/障碍占据图
- `threat`：威胁场
- `height`：地表或建筑顶面高度
- `meta`：地形类型、尺寸、密度等元信息

当前支持三类地形：

- **mountain**：山地地形，支持 `S / M / L` 三档尺寸
- **city**：纯城市场景，固定为 **1600 × 2000**
- **hill_city**：山地城市混合场景，固定为 **1600 × 2000**，用于更接近真实 UAV 城市低空规划的测试

规划器包括：

- **RRT\***：单目标采样树方法
- **PRM**：概率路图基线，当前版本已改为 **3D node PRM**，允许在障碍物上方的合法空域采样节点
- **MOEA/D**：多目标优化，输出 Pareto 解集

同时提供以下实验脚本：

- 单张地形生成：`experiments/gen_terrain.py`
- 批量地形生成：`experiments/gen_dataset.py`
- 从 `.npz` 单次规划：`experiments/plan_from_terrain.py`
- 批量 benchmark：`experiments/run_benchmark.py`
- 从 `visdata_*.json` 后处理渲染：`experiments/render_from_vis_json.py`
- 结果汇总：`experiments/collect_results.py`
- 测试入口：`python test ...`
- PRM 单地形分析：`python -m test.analyze_prm_from_npz ...`

---

# 1. 安装
>>>>>>> origin/feature/3d

```bash
pip install -r requirements.txt
```

---

<<<<<<< HEAD
## 7. 快速开始

## 7.1 生成一张地形

### mountain
=======
# 2. 目录结构

```text
uav_path_planning_3d_debug/
├─ experiments/          # 数据生成、规划、benchmark、可视化脚本
├─ src/
│  ├─ algorithms/        # A*, PRM, RRT*, MOEA/D
│  ├─ env/               # 地形与碰撞环境
│  ├─ models/            # 路径、目标、约束、评估器
│  └─ viz/               # 基础可视化
├─ test/                 # 测试与调试脚本
│  ├─ __main__.py        # 支持直接 python test ...
│  ├─ terrain_preview.py # 原 test.py 的预览入口
│  └─ analyze_prm_from_npz.py  # 读取 terrain npz 并分析 PRM
├─ terrains/             # 生成的数据集（运行后产生）
├─ outputs/              # benchmark 输出（运行后产生）
└─ README.md
```

说明：

- 根目录下已不再依赖旧的 `test.py`
- 地形预览统一通过 `python test ...` 使用
- 需要作为模块运行的测试脚本仍使用 `python -m test.xxx ...`

---

# 3. 地形生成

## 3.1 mountain

山地支持三档尺寸：

- `small` -> `160 × 200`
- `medium` -> `480 × 600`
- `large` -> `1600 × 2000`

单张生成：
>>>>>>> origin/feature/3d

```bash
python -m experiments.gen_terrain --terrain_type mountain --size small --seed 7
```

<<<<<<< HEAD
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
=======
批量生成：

```bash
python -m experiments.gen_dataset --terrain_type mountain --size medium --seed_from 0 --seed_to 20
```

输出目录示例：

```text
terrains/S/mountain_seed0007.npz
terrains/M/mountain_seed0012.npz
```

---

## 3.2 city

城市场景固定使用：

- `H = 1600`
- `W = 2000`
- 约 `8 km × 10 km`（若按 5m 分辨率理解）
- 建筑高度通常为 **3m 的整数倍**

`city_density` 在 `city` 模式下更接近“目标建筑覆盖率参数”。输出目录按密度分桶：

```text
terrains/city_0.24/city_seed0000.npz
```

单张生成：

```bash
python -m experiments.gen_terrain --terrain_type city --seed 0 --city_density 0.24
```

批量生成：

```bash
python -m experiments.gen_dataset --terrain_type city --city_density 0.24 --seed_from 0 --seed_to 20
```

---

## 3.3 hill_city

`hill_city` 是当前推荐用于 UAV 路径规划实验的混合场景。其目标不是生成“纯随机山地 + 随机建筑”，而是生成：

- 山地地貌
- 中央或沿谷地发展的主建成区
- 可用于规划的主路网 / 走廊骨架
- 随 `city_density` 变化的开发强度

固定使用：

- `H = 1600`
- `W = 2000`

单张生成：

```bash
python -m experiments.gen_terrain --terrain_type hill_city --seed 0 --city_density 0.24
```

批量生成：
>>>>>>> origin/feature/3d

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

<<<<<<< HEAD
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
=======
输出目录示例：

```text
terrains/hill_city_0.24/hill_city_seed0000.npz
terrains/hill_city_0.12/hill_city_seed0000.npz
```

### 关于 `hill_city` 中的 density 说明

这是本项目里最容易混淆的地方。

在 `hill_city` 模式下，图上会同时显示两个量：

- `param_density`：命令行输入的 `--city_density`
- `actual_occupancy`：最终 `env.occupancy` 的实际占据比例

例如：

```text
param_density=0.2400, actual_occupancy=0.7916
```

这里的含义是：

- **`param_density` 不是“全图建筑占比”**
- 它更像是 **城市开发强度参数（urban development intensity）**
- `actual_occupancy` 才是最终生成后实际统计得到的占据比例

因此在 `hill_city` 下，推荐这样理解不同密度：

- `0.24`：高开发度 hill city
- `0.12`：中开发度 hill city
- `0.06`：低开发度 hill city

---

# 4. 地形预览与测试入口

## 4.1 直接现场生成并预览

现在推荐使用测试入口：

```bash
python test --terrain_type hill_city --seed 0 --city_density 0.24
```

等价的模块方式仍然可以用：

```bash
python -m test.terrain_preview --terrain_type hill_city --seed 0 --city_density 0.24
```

## 4.2 从已有 `.npz` 读取并预览

```bash
python test --terrain terrains/hill_city_0.12/hill_city_seed0000.npz
```

这两种方式的区别是：

- 指定 `--terrain`：读取已有文件
- 不指定 `--terrain`：按命令行参数现场重新生成

## 4.3 预览图中的标题含义

2D 预览标题现在显示：

```text
shape=1600x2000, param_density=0.1200, actual_occupancy=0.5431, z_max=108.00
```

各字段含义：

- `shape`：地图尺寸
- `param_density`：输入参数密度
- `actual_occupancy`：最终占据比例
- `z_max`：地图最大高度

---

# 5. 单次规划

## 5.1 从已有 `.npz` 做单次规划

```bash
python -m experiments.plan_from_terrain   --terrain terrains/hill_city_0.24/hill_city_seed0000.npz   --planner all   --seed 0
```

或：

```bash
python -m experiments.plan_from_terrain   --terrain terrains/S/mountain_seed0000.npz   --planner moead   --seed 0
```

说明：

- 若不手动指定起终点，程序会自动生成默认 `start / goal`
- 当前默认逻辑会把起终点抬到局部地面上方一段固定高度
- 生成的 `visdata_*.json` 会显式保存 `start` 与 `goal`
- PRM 当前走的是 **3D 节点版本**，在城市/山体上方的合法空域也可以布点

## 5.2 `run_benchmark` 参数精简

`experiments/run_benchmark.py` 现在优先推荐使用 **preset + 短参数** 的方式。

### 5.2.1 新增 preset

```bash
-P auto|quick|balanced|quality
```

含义：

- `auto`：默认值；`mountain/small` 自动走 `quick`，`city / hill_city / medium / large` 自动走 `balanced`
- `quick`：快速摸底
- `balanced`：推荐默认 benchmark 档
- `quality`：更高预算、更重结果质量

preset 会统一覆盖一组常用参数（如 `inflate / rrt_iter / prm_samples / prm_k / prm_max_edge_len / moead_pop / moead_min_gen / moead_max_gen / moead_T / active_subproblem_ratio / archive_soft_limit`），但**你手动传入的显式参数优先级更高**。

### 5.2.2 常用短参数

- `-t / --terrain` -> `--terrain_type`
- `-d / --density` -> `--city_density`
- `-n` -> `--num_terrains`
- `-o` -> `--out_root`
- `--pseed` -> `--planner_seed`
- `--pop` -> `--moead_pop`
- `--gmin` -> `--moead_min_gen`
- `--gmax` -> `--moead_max_gen`
- `--prm_n` -> `--prm_samples`
- `--prm_edge` -> `--prm_max_edge_len`
- `--active_ratio` -> `--active_subproblem_ratio`
- `--arch_soft` -> `--archive_soft_limit`

### 5.2.3 调整后的默认思路

- `seed_to` 默认从 `50` 改为 `10`
- `inflate` 默认从 `5.0` 改为 `1.0`
- benchmark 更推荐通过 `-P/--profile` 管理预算，而不是在命令行里堆很多长参数

---

# 6. 批量 benchmark

## 6.1 mountain benchmark

```bash
python -m experiments.run_benchmark   -t mountain -s small -n 10 -o outputs
```

## 6.2 city benchmark

```bash
python -m experiments.run_benchmark   -t city -d 0.24 -n 10 -o outputs
```

## 6.3 hill_city benchmark

```bash
python -m experiments.run_benchmark   -t hill_city -d 0.24 -n 10 -o outputs
```

> 默认 `-P auto` 会在 `city / hill_city` 上自动切到 `balanced` 档。

## 6.4 reproducibility benchmark（按 seed / run 组织）

如果你要对**同一个场景 seed 重复跑多次**，推荐使用：

```bash
python -m experiments.run_reproducibility   --repeat_count 7   --out_root outputs_repro_hill_city   --   -t hill_city -d 0.24 -n 20 --pseed 0 -P balanced --moead_debug
```

它会把结果组织成：

```text
outputs_repro_hill_city/
  seed0000/
    run00/
    run01/
    ...
    run06/
  seed0001/
    run00/
    ...
  ...
```

也就是：

- 外层 `seed0000/` 表示 terrain seed
- 内层 `run00/` 表示这个 seed 的第 1 次重复运行

每个 `runXX/` 目录里都会直接保存这一轮 `run_benchmark` 的原始输出，例如：

- `benchmark_summary.csv`
- `benchmark_summary.log`
- `metrics_*.json`
- `visdata_*.json`
- `moead_debug.log`
- `mtoe_debug.log`
- `moead_debug.csv`
- `mtoe.csv`
- `mtoe_debug.csv`

此外，在 `out_root` 下还会额外生成两个总表：

- `reproducibility_summary.csv`：一行对应一次 `seedXXXX/runYY`
- `reproducibility_cases.csv`：一行对应一个 `metrics_*.json` case，便于后续统计 mean/std

注意：

- `run_reproducibility` 会自动为每个 `seed + run` 单独调用一次 `run_benchmark`
- 你在 `--` 后面传的参数，基本都和 `run_benchmark` 一样
- 输出目录相关参数（如 `--out_root` / `--summary_csv`）会由脚本自己接管，不需要再传

也可以直接用 `--glob` 指定任意模式：

```bash
python -m experiments.run_benchmark   --glob 'terrains/hill_city_0.12/hill_city_seed*.npz'   -o outputs
```

benchmark 输出文件：

- `metrics_*.json`：数值统计（不包含 debug 诊断）
- `visdata_*.json`：后处理渲染输入
- `mtoe_debug_*.log`：仅在开启调试时输出的 MTOE / MOEA/D 调试日志

---

# 7. PRM 调试分析

如果你要单独分析某一张 terrain `.npz` 上 PRM 为什么成功或失败，推荐使用：

```bash
python -m test.analyze_prm_from_npz   --terrain_npz terrains/hill_city_0.24/hill_city_seed0000.npz   --out_dir prm_debug_seed0   --prm_samples 12000   --prm_k 48   --prm_max_edge_len 250   --seed 0   --save_occ_height
```

输出通常包括：

- `prm_debug_*.npz`
- `prm_debug_*.json`
- `prm_debug_*.png`

其中：

- `json` 便于快速看 `found / start_goal_connected / component` 等统计
- `png` 便于直接观察节点、边和连通分量
- `npz` 便于后续自己写脚本做更细的分析

---

# 8. 后处理渲染

从 benchmark 生成的 `visdata_*.json` 渲染：

```bash
python -m experiments.render_from_vis_json --input outputs/hill_city_0.24 --mode all --project_root .
```

---

# 9. 推荐工作流

## 9.1 hill_city

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 3
python -m experiments.run_benchmark -t hill_city -d 0.24 -n 3 -o outputs --pseed 2 -P balanced --moead_debug
python -m experiments.run_reproducibility   --repeat_count 7   --out_root outputs_repro   --   -t hill_city -d 0.24 -n 20 --pseed 0 -P balanced --moead_debug
python -m experiments.render_from_vis_json --input outputs/hill_city_0.24 --mode all --project_root .
```

如果 `balanced` 还不够，再按需加少量覆盖参数，例如：

```bash
python -m experiments.run_benchmark   -t hill_city -d 0.24 -n 3 -o outputs   -P quality --pseed 2 --gmax 800 --pop 192 --active_ratio 0.90 --moead_debug
```

## 9.2 快速调试单张地形上的 PRM

```bash
python -m test.analyze_prm_from_npz   --terrain_npz terrains/hill_city_0.24/hill_city_seed0000.npz   --out_dir prm_debug_seed0   --prm_samples 12000   --prm_k 48   --prm_max_edge_len 250   --seed 0   --save_occ_height
```

---

# 10. 当前建议

如果你的目标是做 **UAV 路径规划 benchmark**，当前更推荐使用 `hill_city`，因为它兼顾：

- 山地约束
- 城市障碍
- 主通行带/主建成区
- 可控的开发强度变化

另外，若要分析 PRM 的行为，不要只看最终 `found / not found`，更建议配合：

- `python -m test.analyze_prm_from_npz ...`
- 输出的 `prm_debug_*.json`
- 输出的 `prm_debug_*.png`

这样更容易定位问题到底出在：

- 采样不合理
- 连通分量断裂
- 起终点接入失败
- 还是边碰撞检查过严

推荐先从下面这条命令开始：

```bash
python -m experiments.run_benchmark   -t hill_city -d 0.24 -n 3 -o outputs   -P quality --pseed 2 --disable_mtoe_stop --moead_debug
```

如果要进一步精调，再从这个基础上少量追加覆盖参数，例如 `--gmax / --pop / --prm_n / --prm_edge / --active_ratio / --arch_soft`。
>>>>>>> origin/feature/3d
