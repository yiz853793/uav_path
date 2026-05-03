# UAV Path Planning with MOEA/D

这是一个面向 2.5D 复杂地形的无人机路径规划代码库。当前版本只保留代码和必要的项目入口，覆盖地图生成、路线规划、实验运行、数据收集分析和可视化绘图。

## 保留内容

```text
uav_path_planning_mtoe_slim_3/
|-- README.md
|-- requirements.txt
|-- main.py
|-- experiments/                  # 用户直接运行的实验入口
|   |-- gen_terrain.py             # 单张地图生成
|   |-- gen_dataset.py             # 批量地图生成
|   |-- plan_from_terrain.py       # 从已有 .npz 地图执行一次规划
|   |-- run_benchmark.py           # 批量 benchmark
|   |-- run_reproducibility.py     # 多 planner seed 复现实验
|   |-- run_suite.py               # baseline/full 套件
|   |-- run_ablation_suite.py      # 消融实验套件
|   |-- collect_results.py         # 收集和汇总实验结果
|   |-- render_from_vis_json.py    # 地图、路线、Pareto 等可视化
|   `-- plot_results.py            # 实验结果统计绘图
|-- diagram_viz/                  # 流程图、结构图等非实验数据图
|   |-- create_insert_fig3_ppt_style.py
|   `-- adjust_fig3_2_left_callouts.py
|-- src/
|   |-- algorithms/                # A*, PRM, RRT*, MOEA/D
|   |-- env/                       # 栅格环境和碰撞检测
|   |-- experiment/                # 精简后的实验内部逻辑
|   |-- models/                    # 路径、目标、约束、评估器
|   `-- viz/                       # 基础地图绘图工具
`-- test/                         # 辅助检查和预览脚本
```

`src/experiment/` 只保留以下功能模块：

- `terrain.py`：生成单个/批量地形。
- `path_planning.py`：运行一个/批量运行路径规划，并包含多 seed 复现实验支撑逻辑。
- `baseline_experiment.py`：baseline/full 实验套件。
- `ablation_experiment.py`：消融实验套件。
- `data_analysis.py`：处理和汇总实验数据。
- `data_plotting.py`：根据处理后的数据绘制实验统计图。
- `terrain_result_plotting.py`：根据规划结果绘制地形、路径和 Pareto 图。

已删除内容包括：运行输出、地形数据、论文文档、Word 临时文件、缓存文件、论文正文改写脚本和中间抽取文本。

## 环境安装

```bash
pip install -r requirements.txt
```

核心依赖：

- numpy
- matplotlib
- plotly

## 常用命令

生成单张地图：

```bash
python -m experiments.gen_terrain --terrain_type hill_city --city_density 0.24 --seed 0
```

批量生成地图：

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

从已有地图执行一次规划：

```bash
python -m experiments.plan_from_terrain --terrain terrains/hill_city_0.24/hill_city_seed0000.npz --planner_seed 0 --out_root outputs
```

运行 benchmark：

```bash
python -m experiments.run_benchmark --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20 --planner_seed 0 --out_root outputs
```

运行 baseline/full 套件：

```bash
python -m experiments.run_suite --out_root outputs/suite_hillcity024 --repeat_count 5 --planner_seed_base 0 --suite smoke -- --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

运行消融实验：

```bash
python -m experiments.run_ablation_suite --out_root outputs/ablation_hillcity024 --repeat_count 5 --planner_seed_base 0 --suite smoke -- --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

收集已有结果：

```bash
python -m experiments.collect_results --root outputs
```

从 `visdata_*.json` 渲染地图、路径和 Pareto 图：

```bash
python -m experiments.render_from_vis_json --input path/to/visdata_xxx.json --mode static
```

绘制实验结果统计图：

```bash
python -m experiments.plot_results --in_root outputs --out_dir outputs/_figures
```

## 可视化分类

数据可视化代码保留在 `experiments/` 和 `src/viz/` 中，主要处理地图、规划路线、Pareto 前沿、benchmark 结果和消融结果。

流程图和结构图代码统一放在 `diagram_viz/` 中，和实验结果绘图分开管理。

## 说明

仓库目前只保留代码。`outputs/`、`terrains/`、`paper_assets/` 等目录会在运行脚本时按需重新生成。
