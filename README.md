# UAV Path Planning on 2.5D Terrain Grid
## RRT* + PRM + MOEA/D on mountain / city / hill_city occupancy-threat-height maps

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
- **PRM**：概率路图基线
- **MOEA/D**：多目标优化，输出 Pareto 解集

同时提供以下实验脚本：

- 单张地形生成：`experiments/gen_terrain.py`
- 批量地形生成：`experiments/gen_dataset.py`
- 从 `.npz` 单次规划：`experiments/plan_from_terrain.py`
- 批量 benchmark：`experiments/run_benchmark.py`
- 从 `visdata_*.json` 后处理渲染：`experiments/render_from_vis_json.py`
- 结果汇总：`experiments/collect_results.py`

---

# 1. 安装

```bash
pip install -r requirements.txt
```

---

# 2. 目录结构

```text
uav_path_planning_3d_debug/
├─ experiments/          # 数据生成、规划、benchmark、可视化脚本
├─ src/
│  ├─ algorithms/        # A*, PRM, RRT*, MOEA/D
│  ├─ env/               # 地形与碰撞环境
│  ├─ models/            # 路径、目标、约束、评估器
│  └─ viz/               # 基础可视化
├─ terrains/             # 生成的数据集（运行后产生）
├─ outputs/              # benchmark 输出（运行后产生）
└─ README.md
```

---

# 3. 地形生成

## 3.1 mountain

山地支持三档尺寸：

- `small` -> `160 × 200`
- `medium` -> `480 × 600`
- `large` -> `1600 × 2000`

单张生成：

```bash
python -m experiments.gen_terrain --terrain_type mountain --size small --seed 7
```

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

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
```

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

也就是说，`hill_city` 中的密度变化更偏向：

- 同一座城市在不同开发强度下的变化

而不是：

- 城市面积线性缩小

这对 UAV 路径规划 benchmark 更有利，因为不同密度下地图语义更加稳定，便于做 controlled comparison。

---

# 4. 地形预览与可视化

## 4.1 直接现场生成并预览

```bash
python -m experiments.gen_terrain --terrain_type hill_city --seed 0 --city_density 0.24
```

## 4.2 从已有 `.npz` 读取并预览

```bash
python -m experiments.gen_terrain --terrain terrains/hill_city_0.12/hill_city_seed0000.npz
```

这两种方式的区别是：

- 指定 `--terrain`：读取已有文件
- 不指定 `--terrain`：按命令行参数现场重新生成

如果你先用 `gen_dataset.py` 生成了 `0.12` 的数据，后面又直接运行不带 `--terrain` 的 `gen_terrain.py`，那么它会按当前命令行参数重新生成一张图，而不是读取之前的数据集文件。

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

从已有 `.npz` 做单次规划：

```bash
python -m experiments.plan_from_terrain \
  --terrain terrains/hill_city_0.24/hill_city_seed0000.npz \
  --planner all \
  --seed 0
```

或：

```bash
python -m experiments.plan_from_terrain \
  --terrain terrains/S/mountain_seed0000.npz \
  --planner moead \
  --seed 0
```

说明：

- 若不手动指定起终点，程序会自动生成默认 `start / goal`
- 当前默认逻辑会把起终点抬到局部地面上方一段固定高度
- 生成的 `visdata_*.json` 会显式保存 `start` 与 `goal`

---

# 6. 批量 benchmark

## 6.1 mountain benchmark

```bash
python -m experiments.run_benchmark \
  --terrain_type mountain \
  --size small \
  --seed_from 0 --seed_to 10 \
  --out_root outputs
```

默认读取：

```text
terrains/S/mountain_seedXXXX.npz
```

## 6.2 city benchmark

```bash
python -m experiments.run_benchmark \
  --terrain_type city \
  --city_density 0.24 \
  --seed_from 0 --seed_to 10 \
  --out_root outputs
```

默认读取：

```text
terrains/city_0.24/city_seedXXXX.npz
```

## 6.3 hill_city benchmark

```bash
python -m experiments.run_benchmark \
  --terrain_type hill_city \
  --city_density 0.24 \
  --seed_from 0 --seed_to 10 \
  --out_root outputs
```

默认读取：

```text
terrains/hill_city_0.24/hill_city_seedXXXX.npz
```

也可以直接用 `--glob` 指定任意模式：

```bash
python -m experiments.run_benchmark \
  --glob 'terrains/hill_city_0.12/hill_city_seed*.npz' \
  --out_root outputs
```

benchmark 输出两类文件：

- `metrics_*.json`：数值统计
- `visdata_*.json`：后处理渲染输入

`visdata_*.json` 当前会保存：

- `terrain_file`
- `start`
- `goal`
- `rrt / prm` 路径
- `MOEA/D` 代表解路径
- `archive_points`
- `meta`

---

# 7. 后处理渲染

从 benchmark 生成的 `visdata_*.json` 渲染：

```bash
python -m experiments.render_from_vis_json --input outputs/hill_city_0.24 --mode all --project_root .
```

或：

```bash
python -m experiments.render_from_vis_json --input outputs/S --mode all --project_root .
```

脚本会自动读取 `visdata_*.json` 中的：

- 地形文件路径
- 起点终点
- 各规划器路径
- Pareto 点集

并渲染为：

- 2D 路径图
- 3D 路径预览图
- 3D Pareto 图
- 交互式 HTML（若对应模式启用）

---

# 8. 推荐工作流

## 8.1 mountain

```bash
python -m experiments.gen_dataset --terrain_type mountain --size small --seed_from 0 --seed_to 20
python -m experiments.run_benchmark --terrain_type mountain --size small --seed_from 0 --seed_to 20 --out_root outputs
python -m experiments.render_from_vis_json --input outputs/S --mode all --project_root .
```

## 8.2 city

```bash
python -m experiments.gen_dataset --terrain_type city --city_density 0.24 --seed_from 0 --seed_to 20
python -m experiments.run_benchmark --terrain_type city --city_density 0.24 --seed_from 0 --seed_to 20 --out_root outputs
python -m experiments.render_from_vis_json --input outputs/city_0.24 --mode all --project_root .
```

## 8.3 hill_city

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20
python -m experiments.run_benchmark --terrain_type hill_city --city_density 0.24 --seed_from 0 --seed_to 20 --out_root outputs
python -m experiments.render_from_vis_json --input outputs/hill_city_0.24 --mode all --project_root .
```

如果你要测试更低开发强度：

```bash
python -m experiments.gen_dataset --terrain_type hill_city --city_density 0.12 --seed_from 0 --seed_to 20
python -m experiments.run_benchmark --terrain_type hill_city --city_density 0.12 --seed_from 0 --seed_to 20 --out_root outputs
python -m experiments.render_from_vis_json --input outputs/hill_city_0.12 --mode all --project_root .
```

---

# 9. 当前建议

如果你的目标是做 **UAV 路径规划 benchmark**，当前更推荐使用 `hill_city`，因为它兼顾：

- 山地约束
- 城市障碍
- 主通行带/主建成区
- 可控的开发强度变化

而且：

- `0.24` 与 `0.12` 的差异主要体现在 **建成区变稠 / 变稀**
- 不会因为密度下降就把整座城市缩成一个很小的斑块
- 因此更适合作为不同复杂度场景下的对比实验底图
