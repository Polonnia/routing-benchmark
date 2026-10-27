# 可靠路径实验框架

用于在交通网络上复现和对比可靠路径算法的通用实验框架。

算法名称、路网数据、起点终点、链路波动参数和出行时间预算均通过命令行参数指定。新增算法只需在 `src/algorithms.py` 中注册，无需修改实验入口。

## 安装

```bash
cd fma_repro
pip install -r requirements.txt
```

要求 Python 3.10+，仅依赖 NumPy。

## 快速开始

```bash
# 全部内置网络、全部已注册算法、每个网络 50 个随机 OD 对
python run_experiments.py

# 单个算法、单个网络
python run_experiments.py --algorithm fma --network Anaheim

# 指定单个起终点
python run_experiments.py --algorithm fma --network Anaheim \
    --origin 1 --destination 416 --kappa 0.15 --t-factor 1.05
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--algorithm`, `--algorithms` | `fma let` | 算法名称，可指定多个，用空格分隔。当前可用：`fma`、`let`，以及在 `src/algorithms.py` 中注册的其他算法。 |
| `--network`, `--networks` | `Anaheim Winnipeg Chicago_Sketch` | 内置网络名称，可指定多个。与 `--network-file` 互斥。 |
| `--network-file`, `--network-files` | — | 自定义网络 CSV 文件，可指定多个；列格式与内置数据一致。与 `--network` 互斥。 |
| `--origin` | — | 起点节点 ID（从 1 开始），必须与 `--destination` 同时使用。 |
| `--destination` | — | 终点节点 ID（从 1 开始）。 |
| `--n-od` | `50` | 每个网络随机生成的连通 OD 对数量。指定 `--origin/--destination` 时忽略。 |
| `--min-hops` | `5` | 随机 OD 对的 LET 路径少于该链路数时丢弃。 |
| `--kappa`, `--kappas` | `0.15 0.25 0.50` | 链路标准差离散参数，可指定多个：`sigma_ij = Uniform(0, kappa) * mean_ij`。 |
| `--t-factor`, `--t-factors` | `1.0 1.05 1.1` | 出行时间预算系数，可指定多个：`T = factor * t_LET`。 |
| `--iterations` | `100` | 最大策略改进迭代次数（FMA 使用）。 |
| `--max-tries` | `100` | 每次迭代最大策略变化尝试次数（FMA 使用）。 |
| `--mode` | `random` | FMA 策略改进模式：`random` 或 `greedy`。 |
| `--seed` | `808` | 随机种子，用于 OD 采样、链路波动和算法随机性。 |
| `--output` | `results` | 输出目录。 |
| `--data-dir` | 项目根目录 | 内置网络 `data/` 目录所在位置。 |
| `--quiet` | 关闭 | 不输出实验进度。 |

`--origin` 与 `--destination` 必须同时提供。指定后，会在每个选中的网络上评估同一个 OD 对；节点 ID 必须存在于对应网络中。

## 示例

```bash
# 多算法、多网络、多组 kappa 和预算
python run_experiments.py \
    --algorithm fma let \
    --network Anaheim Winnipeg \
    --n-od 20 \
    --kappa 0.15 0.25 0.5 \
    --t-factor 1.0 1.05 1.1

# 只运行一个算法
python run_experiments.py --algorithm fma --network Chicago_Sketch

# 使用自定义网络文件
python run_experiments.py \
    --algorithm fma \
    --network-file data/Anaheim/Anaheim_network.csv \
    --n-od 10

# 预算扫描（类似 Fig. 7）
python run_experiments.py \
    --algorithm fma --network Anaheim \
    --t-factor 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3

# 使用贪心策略改进
python run_experiments.py --algorithm fma --mode greedy --network Anaheim
```

## 输出文件

输出目录包含：

| 文件 | 内容 |
|---|---|
| `per_run.csv` | 每个网络 × kappa × T 系数 × OD × 算法一行。 |
| `aggregated.csv` | 按网络 × kappa × T 系数 × 算法聚合后的均值。 |
| `results.md` | 便于阅读的下界结果汇总。 |
| `config.json` | 本次实验的完整命令行配置。 |
| `environment.json` | Python、NumPy 和平台信息。 |

结果中的 **下界** 是算法保证的到点概率 `1 - g`，不计算经验概率。

## 路网文件格式

CSV 文件，包含表头和四列：

```text
From,To,Volume,Cost
```

`Cost` 为链路的平均通行时间，用作 `M1`。内置网络文件位于 `data/` 目录；任何符合该格式的文件都可以通过 `--network-file` 传入。

## 添加新算法

在 `src/algorithms.py` 中添加算法函数，并使用 `register` 注册（`register` 和 `AlgorithmResult` 已在该模块中定义）：

```python
@register("my_algorithm")
def run_my_algorithm(
    network, model, origin, destination, T, precomputed, seed, options
):
    # network: src.network.Network
    # model:   src.moments.MomentModel（每条链路的 M1–M4）
    # precomputed: (dist_to_destination, next_link)，由 Dijkstra 预计算
    # options: ExperimentConfig（iterations、mode、seed 等）
    ...
    return AlgorithmResult(
        path_links=...,
        path_nodes=...,
        lower_bound=...,   # 1 - g
        g=...,
        condition=...,
        runtime_s=...,
    )
```

注册后即可直接运行：

```bash
python run_experiments.py --algorithm my_algorithm --network Anaheim
```
