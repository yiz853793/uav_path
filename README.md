# UAV Path Planning with MOEA/D

这是一个面向 2.5D 复杂地形的无人机路径规划代码库。当前版本只保留代码和必要入口，覆盖地形生成、路径规划、基线实验、消融实验、结果收集分析和可视化绘图。

## 目录结构

```text
uav_path_planning_mtoe_slim_3/
|-- README.md
|-- requirements.txt
|-- main.py
|-- experiments/                  # 命令行入口
|   |-- gen_terrain.py             # 生成单张地形
|   |-- gen_dataset.py             # 批量生成地形
|   |-- plan_from_terrain.py       # 从已有 .npz 执行单次规划
|   |-- run_benchmark.py           # 批量路径规划 benchmark
|   |-- run_suite.py               # baseline/full 套件
|   |-- run_ablation_suite.py      # 消融实验套件
|   |-- collect_results.py         # 汇总、分析和 Wilcoxon 显著性检验
|   |-- plot_results.py            # 根据汇总数据绘图
|   `-- render_from_vis_json.py    # 根据规划结果绘制地形/路径/Pareto 图
|-- src/
|   |-- algorithms/                # A*, PRM, RRT*, Improved RRT*, MOEA/D, NSGA-III
|   |-- env/                       # 栅格环境和碰撞检测
|   |-- experiment/                # 精简后的实验内部逻辑
|   |-- models/                    # 路径、目标、约束、评估器
|   `-- viz/                       # 基础绘图工具
`-- test/                         # 辅助检查和预览脚本
```

`src/experiment/` 只保留 7 个功能模块：

| 文件 | 功能 |
|---|---|
| `terrain.py` | 生成单个/批量地形。 |
| `path_planning.py` | 运行单次/批量路径规划，并支撑多 seed 复现实验。 |
| `baseline_experiment.py` | baseline/full 实验套件。 |
| `ablation_experiment.py` | 消融实验套件。 |
| `data_analysis.py` | 处理和汇总实验数据。 |
| `data_plotting.py` | 根据处理后的数据绘制统计图。 |
| `terrain_result_plotting.py` | 根据规划结果绘制地形、路径和 Pareto 图。 |

## 安装

```bash
pip install -r requirements.txt
```

核心依赖：

| 依赖 | 用途 |
|---|---|
| `numpy` | 地形、路径、目标函数等数值计算。 |
| `matplotlib` | 静态图绘制。 |
| `plotly` | 交互式路径和 Pareto 可视化。 |

## 快速开始

生成单张地形：

```bash
python -m experiments.gen_terrain --terrain_type hill_city --city_density 0.24 --seed 0
```

批量生成地形：

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

从已有地形执行一次规划：

```bash
python -m experiments.plan_from_terrain --terrain terrains/hill_city_0.24/hill_city_seed0000.npz --planner_seed 0 --out_root outputs
```

运行批量 benchmark：

```bash
python -m experiments.run_benchmark --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20 --planner_seed 0 --out_root outputs
```

运行 baseline/full 套件：

```bash
python -m experiments.run_suite --out_root outputs/suite_hillcity024_fair_sota --repeat_count 30 --planner_seed_base 0 --suite paper -- --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20 --profile quality --inflate 1.0 --rrt_iter 8000 --improved_rrt_iter 8000 --prm_samples 16000 --prm_k 64 --prm_max_edge_len 280 --prm_threat_weight 0.0 --improved_rrt_threat_weight 0.0 --moead_min_gen 80 --moead_max_gen 1600 --moead_pop 160 --moead_T 16 --nsga3_max_gen 1600 --nsga3_pop 160 --nsga3_ref_dirs 160 --K 30 --weight_extreme_bias 0.20
```

说明：该命令用于论文复现实验，避免对 PRM/RRT* 使用过低预算；PRM 与 Improved RRT* 的 threat 权重保持一致，MOEA/D 与 NSGA-III 使用相同 `pop/gen/K` 预算和同一组评价指标；NSGA-III 作为当前已接入的强基线/SOTA 类对照。若后续接入 Informed RRT*，应沿用本命令中的相同地图、seed、预算和评价设置。

轻量 smoke 检查（只用于验证流程，不用于论文统计）：

```bash
python -m experiments.run_suite --out_root outputs/_smoke_nsga3_baseline --suite smoke --planner_seeds 0 --no_timestamp --progress_every 1 -- --glob outputs/_codex_nsga3_smoke/terrains/S/mountain_seed0003.npz --profile quick --inflate 0 --rrt_iter 200 --improved_rrt_iter 200 --prm_samples 400 --prm_k 8 --prm_max_edge_len 60 --prm_threat_weight 0.0 --improved_rrt_threat_weight 0.0 --moead_min_gen 2 --moead_max_gen 2 --moead_pop 8 --moead_T 3 --nsga3_max_gen 2 --nsga3_pop 8 --nsga3_ref_dirs 8 --K 8 --moead_eval_step 2.0 --moead_smooth_step 2.0 --start_goal_z_offset 20.0
```

运行消融实验：
```bash
python -m experiments.run_ablation_suite --out_root outputs/ablation_hillcity024 --repeat_count 5 --planner_seed_base 0 --suite smoke -- --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

汇总已有结果：

```bash
python -m experiments.collect_results --in_root outputs/suite_hillcity024_fair_sota --out_dir outputs/suite_hillcity024_fair_sota/_collected --exclude_invalid --group_by size,variant,method --wilcoxon_baselines PRM,NSGA-III-compromise --wilcoxon_methods MOEAD_compromise --wilcoxon_metrics f1,f2,f3 --wilcoxon_group_by size_tag,variant --wilcoxon_alternative two-sided
```

绘制实验统计图：

```bash
python -m experiments.plot_results --mode collected --in_dir outputs/suite_hillcity024_fair_sota/_collected --out_dir outputs/suite_hillcity024_fair_sota/_figures
```

从 `visdata_*.json` 渲染地形、路径和 Pareto 图：

```bash
python -m experiments.render_from_vis_json --input path/to/visdata_xxx.json --mode static
```

## 参数表

### 地形生成参数

适用于 `experiments.gen_terrain` 和 `experiments.gen_dataset`。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--out_dir` | `terrains` | 地形输出根目录。 |
| `--seed` | `0` | 单张地形随机种子，仅 `gen_terrain` 使用。 |
| `--seed_from` | `0` | 批量生成起始 seed，包含，仅 `gen_dataset` 使用。 |
| `--seed_to` | `50` | 批量生成结束 seed，不包含，仅 `gen_dataset` 使用。 |
| `--terrain_type` | `mountain` | 地形类型：`mountain`、`city`、`hill_city`。 |
| `--size` | `small` | 山地尺寸：`small`、`medium`、`large`。仅 `mountain` 使用。 |
| `--terrace_levels` | `28` | 山地台阶/层级数量。 |
| `--obstacle_height` | `0.78` | 山地障碍高度参数。 |
| `--city_density` | `0.24` | 城市场景或山地城市场景的开发强度参数。 |
| `--hill_scale` | `26.0` | `hill_city` 的山地起伏尺度。 |

### 单次规划参数

`experiments.plan_from_terrain` 复用 `run_benchmark` 的规划参数，并额外提供单场景输入/输出参数。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--terrain` | 必填 | 要读取的 `.npz` 地形文件。 |
| `--out_root`, `-o` | `outputs` | 输出根目录。 |
| `--single_case_out_dir` | 空 | 指定后直接把该 case 输出写入此目录。 |
| `--planner_seed`, `--seed` | `0` | 规划器随机种子。 |
| `--out_json` | 空 | 兼容参数；指定后会额外复制一份生成的 `visdata_*.json` 到该路径。 |
| 其他规划参数 | 见下表 | 可继续传入 `--profile`、`--inflate`、`--rrt_iter`、`--prm_samples`、`--moead_pop` 等 benchmark 参数。 |

### Benchmark 路径规划参数

适用于 `experiments.run_benchmark`，也可转发给 `plan_from_terrain` 和 suite 脚本。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--terrain_dir` | `terrains` | `.npz` 地形数据根目录。 |
| `--out_root`, `-o` | `outputs` | benchmark 输出根目录。 |
| `--terrain_type`, `-t` | `mountain` | 地形类型。 |
| `--city_density`, `-d` | `0.24` | `city/hill_city` 的密度目录标签。 |
| `--size`, `-s` | `small` | `mountain` 尺寸。 |
| `--profile`, `-P` | `auto` | 预算预设：`auto`、`quick`、`balanced`、`quality`。显式传参会覆盖预设。 |
| `--seed_from` | `0` | terrain seed 起点，包含。 |
| `--seed_to` | `10` | terrain seed 终点，不包含。 |
| `--num_terrains`, `-n` | 空 | 不使用 `--glob` 时，等价于 `seed_to = seed_from + num_terrains`。 |
| `--glob` | 空 | 直接用 glob 匹配地形文件。 |
| `--planner_seed`, `--pseed` | `0` | 单个规划器 seed。 |
| `--planner_seeds` | 空 | 多个规划器 seed；设置后覆盖 `--planner_seed`。 |
| `--inflate`, `-i` | `1.0` | 障碍膨胀半径，单位米。 |
| `--start` | 空 | 手动起点，格式：`x_m y_m`。 |
| `--goal` | 空 | 手动终点，格式：`x_m y_m`。 |
| `--start_goal_z_offset` | `15.0` | 默认起终点离地高度偏移，单位米。 |

### 基线算法参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--rrt_iter`, `--rrt` | `4000` | RRT* 迭代次数。 |
| `--improved_rrt_iter`, `--irrt_iter` | `0` | Improved RRT* 迭代次数；`<=0` 时复用 `--rrt_iter`。 |
| `--improved_rrt_threat_weight` | `0.0` | Improved RRT* 威胁代价权重；公平对比时建议与 PRM threat 权重保持一致。 |
| `--prm_samples`, `--prm_n` | `1200` | PRM 采样点数量。 |
| `--prm_k` | `12` | PRM 每个节点连接的近邻数。 |
| `--prm_max_edge_len`, `--prm_edge` | `30.0` | PRM 最大边长，单位米。 |
| `--prm_threat_weight` | `0.0` | PRM 威胁代价权重；公平对比时建议与 Improved RRT* threat 权重保持一致。 |

### MOEA/D 核心参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--moead_pop`, `--pop` | `60` | MOEA/D 种群规模。 |
| `--K` | `30` | 路径控制点数量。 |
| `--moead_T`, `-T` | `10` | MOEA/D 邻域大小。 |
| `--moead_min_gen`, `--gmin` | `20` | 早停前最少迭代代数。 |
| `--moead_max_gen`, `--gmax` | 空 | 最大迭代代数；不传时内部补为 `80`。 |
| `--archive_size` | `0` | Pareto 档案硬上限；`<=0` 时使用软上限。 |
| `--archive_soft_limit`, `--arch_soft` | `320` | Pareto 档案软上限。 |
| `--archive_grid_bins` | `0` | 档案截断时的目标空间网格数，`0` 表示自动。 |
| `--archive_keep_extremes` | `1` | 档案截断时是否保护目标极值解。 |
| `--active_subproblem_ratio`, `--active_ratio` | `1.0` | 每代激活的子问题比例。 |

### NSGA-III 基线参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--nsga3_max_gen` | 空 | NSGA-III 最大迭代代数；不传时复用 `--moead_max_gen`。 |
| `--nsga3_pop` | 空 | NSGA-III 种群规模；不传时复用 `--moead_pop`。 |
| `--nsga3_ref_dirs` | `0` | 参考方向数量；`<=0` 时复用 `--nsga3_pop`。 |
| `--nsga3_crossover_prob` | `0.90` | 交叉概率。 |
| `--nsga3_mutation_prob` | `0.25` | 路径控制点变异概率。 |
| `--nsga3_mutation_sigma` | `2.5` | 变异标准差，单位为栅格。 |
| `--skip_nsga3` | `False` | 跳过 NSGA-III 基线。 |

### MOEA/D 增强与早停参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--init_astar_ratio` | `0.25` | A* 引导初始化占比。 |
| `--init_astar_threat_weight` | `0.0` | A* 初始化时的威胁权重。 |
| `--init_astar_jitter_sigma` | `1.5` | A* 路径点扰动标准差。 |
| `--init_astar_max_paths` | `5` | 最多生成多少条 A* backbone。 |
| `--init_astar_penalty_step` | `2.5` | 多条 A* 路径生成时的访问惩罚步长。 |
| `--init_stratified_ratio` | `0.60` | 分层走廊初始化占比。 |
| `--init_global_random_ratio` | `0.15` | 全局随机初始化占比。 |
| `--weight_extreme_bias` | `0.20` | 权重向目标极值方向偏置的比例。 |
| `--extreme_offspring_ratio` | `0.20` | 分配给极值方向的额外 offspring 比例。 |
| `--local_search_interval` | `10` | 定向局部搜索间隔；`0` 表示关闭。 |
| `--local_search_elite_k` | `3` | 每个目标参与局部搜索的精英数量。 |
| `--local_search_attempts_per_obj` | `2` | 每个目标每次触发的局部搜索尝试次数。 |
| `--mtoe_tol_fun` | `1e-5` | MTOE 早停容差。 |
| `--mtoe_confidence` | `0.995` | MTOE 置信度。 |
| `--mtoe_window` | `10` | MTOE 滚动窗口长度。 |
| `--disable_mtoe_stop` | `False` | 关闭真实早停，仅保留 shadow 记录。 |
| `--basin_shadow_enable` | `1` | 是否启用 basin shadow 监控。 |
| `--basin_escape_injections` | `0` | basin 监控触发 escape 时额外注入数量。 |

### 约束与调试参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--max_turn_deg` | `90.0` | 最大硬转角限制。 |
| `--soft_turn_deg` | `60.0` | 软转角偏好限制。 |
| `--max_pitch_deg` | `35.0` | 最大硬俯仰角限制。 |
| `--soft_pitch_deg` | `25.0` | 软俯仰角偏好限制。 |
| `--desired_clearance_margin` | `2.0` | 期望离地裕度。 |
| `--tau_soft` | `25.0` | 最大允许软约束加权违背。 |
| `--moead_eval_step` | `2.5` | 评估采样步长，单位米。越大越快但越粗。 |
| `--moead_smooth_step` | `2.5` | 平滑/碰撞检查采样步长，单位米。 |
| `--moead_debug` | `False` | 输出 MOEA/D debug log 和 MTOE 统计。 |
| `--moead_debug_every` | `1` | 每隔多少代记录一次 debug。 |
| `--moead_debug_level` | `2` | debug 级别：`1` 粗略，`2` 时间拆分，`3` 碰撞 profiling。 |

### 复现实验与套件参数

适用于 `run_suite.py` 和 `run_ablation_suite.py`。这些脚本用 `--` 分隔自身参数和转发给 benchmark 的参数。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--out_root` | 必填 | 套件或复现实验输出根目录。 |
| `--planner_seed_from` | 空 | planner seed 起点，包含。 |
| `--planner_seed_to` | 空 | planner seed 终点，不包含。 |
| `--planner_seeds` | 空 | 显式指定 planner seed 列表。 |
| `--repeat_count` | 空 | 从 `planner_seed_base` 开始重复运行次数。 |
| `--planner_seed_base` | `0` | `repeat_count` 模式下的起始 planner seed。 |
| `--progress_every` | `1` | 每隔多少个 case 打印进度。 |
| `--suite` | `paper` | 套件规模：`paper`、`all`、`smoke`。 |
| `--full_escape_injections` | `0` | full 配置中的 escape 注入数量。 |
| `--no_timestamp` | `False` | 不在输出目录下额外创建时间戳子目录。 |
| `--dry_run` | `False` | 只打印将要转发的参数，不实际运行。 |

示例：

```bash
python -m experiments.run_suite --out_root outputs/suite_hillcity024_fair_sota --repeat_count 30 --planner_seed_base 0 --suite paper -- --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20 --profile quality --inflate 1.0 --rrt_iter 8000 --improved_rrt_iter 8000 --prm_samples 16000 --prm_k 64 --prm_max_edge_len 280 --prm_threat_weight 0.0 --improved_rrt_threat_weight 0.0 --moead_min_gen 80 --moead_max_gen 1600 --moead_pop 160 --moead_T 16 --nsga3_max_gen 1600 --nsga3_pop 160 --nsga3_ref_dirs 160 --K 30 --weight_extreme_bias 0.20
```

### 数据处理与绘图参数

`collect_results.py`：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--in_root` | `outputs` | 搜索 benchmark 输出的根目录。 |
| `--pattern` | `**/metrics_*.json` | 在 `in_root` 下匹配 metrics 文件的 glob。 |
| `--out_dir` | `<in_root>/_collected` | 汇总结果输出目录。 |
| `--size` | 空 | 按 size tag 过滤，如 `S M L`。 |
| `--terrain_seed` | 空 | 按 terrain seed 过滤。 |
| `--planner_seed` | 空 | 按 planner seed 过滤。 |
| `--group_by` | `size,method` | 汇总分组字段，如 `size,method,moead_cfg`。 |
| `--exclude_invalid` | `False` | 是否排除 invalid case。 |
| `--no_paired` | `False` | 是否跳过 paired improvement 表。 |
| `--no_wilcoxon` | `False` | 是否跳过 Wilcoxon 秩和检验。 |
| `--wilcoxon_baselines` | `PRM,NSGA-III-compromise` | 作为对照组的方法名，逗号分隔。 |
| `--wilcoxon_methods` | `MOEAD_compromise` | 待检验的方法名，逗号分隔。 |
| `--wilcoxon_metrics` | `f1,f2,f3` | 检验指标。 |
| `--wilcoxon_group_by` | `size_tag,variant` | Wilcoxon 分组字段。 |
| `--wilcoxon_alternative` | `two-sided` | 备择假设：`two-sided`、`less`、`greater`。 |
| `--wilcoxon_alpha` | `0.05` | 显著性水平。 |

`plot_results.py`：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--mode` | `auto` | `collected` 读取 `results_long.csv`；`reproducibility` 读取 `reproducibility_cases.csv`。 |
| `--in_root` | `outputs` | 根目录；auto/reproducibility 模式使用。 |
| `--in_dir` | 空 | 包含 `results_long.csv` 的目录；collected 模式使用。 |
| `--in_csv` | 空 | 显式指定 `results_long.csv`。 |
| `--out_dir` | 自动 | 图像输出目录。 |
| `--kind` | `auto` | reproducibility 模式绘图类型：`auto`、`method`、`ablation`、`suite_variant`。 |
| `--methods` | 常用四类 | collected 模式下绘制的方法名，逗号分隔；空值表示全部。 |
| `--exclude_invalid` | `False` | collected 模式下是否排除 invalid case。 |
| `--dpi` | `220` | 输出图片 DPI。 |
| `--no_summary` | `False` | 不输出 `plot_summary.csv`。 |

`render_from_vis_json.py`：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--input` | 必填 | 单个 `visdata_*.json` 或包含 visdata 的目录。 |
| `--mode` | `all` | 输出模式：`static`、`interactive`、`all`。 |
| `--project_root` | `.` | 查找 terrain 文件时使用的项目根目录。 |
| `--out_dir` | 空 | 输出目录；不传时写在 visdata 旁边。 |

## 输出文件

| 文件/目录 | 说明 |
|---|---|
| `terrains/` | 地形生成结果，包含 `.npz`。 |
| `outputs/` | 规划、benchmark、suite 默认输出根目录。 |
| `metrics_*.json` | 单个 case 的数值结果。 |
| `visdata_*.json` | 后处理可视化所需的路径、Pareto 和元数据。 |
| `reproducibility_cases.csv` | 复现实验 case 级汇总表。 |
| `reproducibility_overall.csv` | 复现实验总体统计。 |
| `results_wide.csv` / `results_long.csv` | `collect_results.py` 生成的宽表和长表。 |
| `summary_grouped.csv` | 按指定字段聚合后的统计表。 |
| `wilcoxon_rank_sum.csv` | `collect_results.py` 生成的 Wilcoxon 秩和检验表，包含 p 值和显著性标记。 |

## 可视化分类

数据可视化代码位于 `experiments/plot_results.py`、`experiments/render_from_vis_json.py` 和 `src/experiment/*plotting.py`，主要处理地图、路径、Pareto 前沿、benchmark 结果和消融结果。

## 说明

仓库目前只保留可复现实验代码、必要配置和轻量入口脚本。`outputs/`、`terrains/`、`diagram_viz/`、`workspace_archive/`、`paper_assets/` 等目录属于本地生成结果或论文工作区，不纳入 Git 版本管理。
