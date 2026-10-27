# Reliable Routing Experiments

A reproducible runner for reliable-routing algorithms on transportation
networks.  Algorithms are selected by name; networks, OD pairs, variance
levels, and travel-time budgets are command-line parameters.  New algorithms
can be added without changing the runner.

## Install

```bash
cd fma_repro
pip install -r requirements.txt
```

Requires Python 3.10+ and NumPy.

## Quick start

```bash
# All built-in networks, all registered algorithms, 50 random OD pairs
python run_experiments.py

# One algorithm on one network
python run_experiments.py --algorithm fma --network Anaheim

# One explicit origin-destination pair
python run_experiments.py --algorithm fma --network Anaheim \
    --origin 1 --destination 416 --kappa 0.15 --t-factor 1.05
```

## Command-line parameters

| Parameter | Default | Description |
|---|---|---|
| `--algorithm`, `--algorithms` | `fma let` | Algorithm name(s), space-separated. Available: `fma`, `let`, plus any name registered in `src/algorithms.py`. |
| `--network`, `--networks` | `Anaheim Winnipeg Chicago_Sketch` | Built-in network name(s), space-separated. Mutually exclusive with `--network-file`. |
| `--network-file`, `--network-files` | — | Custom network CSV file(s); same columns as the built-in data. Mutually exclusive with `--network`. |
| `--origin` | — | Origin node ID from the network file (1-based). Must be used together with `--destination`. |
| `--destination` | — | Destination node ID from the network file (1-based). |
| `--n-od` | `50` | Number of random connected OD pairs per network. Ignored when `--origin/--destination` are given. |
| `--min-hops` | `5` | Reject random OD pairs whose LET path has fewer than this many links. |
| `--kappa`, `--kappas` | `0.15 0.25 0.50` | Link standard-deviation dispersion parameter(s): `sigma_ij = Uniform(0, kappa) * mean_ij`. |
| `--t-factor`, `--t-factors` | `1.0 1.05 1.1` | Travel-time budget factor(s): `T = factor * t_LET`. |
| `--iterations` | `100` | Maximum policy-improvement iterations (used by FMA). |
| `--max-tries` | `100` | Maximum policy-variation attempts per iteration (used by FMA). |
| `--mode` | `random` | FMA policy-improvement mode: `random` or `greedy`. |
| `--seed` | `808` | Random seed for OD sampling, link variances, and algorithm randomness. |
| `--output` | `results` | Output directory. |
| `--data-dir` | project directory | Directory containing `data/` for built-in networks. |
| `--quiet` | off | Suppress progress output. |

`--origin` and `--destination` must be supplied together.  If they are
provided, the same OD pair is evaluated on every selected network; the node
IDs must exist in each network.

## Examples

```bash
# Multiple algorithms, networks, kappa values, and budgets
python run_experiments.py \
    --algorithm fma let \
    --network Anaheim Winnipeg \
    --n-od 20 \
    --kappa 0.15 0.25 0.5 \
    --t-factor 1.0 1.05 1.1

# Only one algorithm
python run_experiments.py --algorithm fma --network Chicago_Sketch

# Custom network file
python run_experiments.py \
    --algorithm fma \
    --network-file data/Anaheim/Anaheim_network.csv \
    --n-od 10

# Budget sweep (Fig. 7 style)
python run_experiments.py \
    --algorithm fma --network Anaheim \
    --t-factor 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3

# Greedy policy improvement instead of random variation
python run_experiments.py --algorithm fma --mode greedy --network Anaheim
```

## Output

The output directory contains:

| File | Content |
|---|---|
| `per_run.csv` | One row per network × kappa × T-factor × OD × algorithm. |
| `aggregated.csv` | Averages grouped by network × kappa × T-factor × algorithm. |
| `results.md` | Human-readable lower-bound summary. |
| `config.json` | Exact command-line configuration. |
| `environment.json` | Python, NumPy, and platform information. |

The reported **lower bound** is the guaranteed on-time-arrival probability
`1 - g`; no empirical probability is computed.

## Network file format

CSV with a header and four columns:

```text
From,To,Volume,Cost
```

`Cost` is the link's mean travel time and is used as `M1`.  The built-in files
are under `data/`; any file with this format can be passed with
`--network-file`.

## Adding a new algorithm

Add a runner to `src/algorithms.py` and decorate it with `register` (both
`register` and `AlgorithmResult` are already defined in that module):

```python
@register("my_algorithm")
def run_my_algorithm(
    network, model, origin, destination, T, precomputed, seed, options
):
    # network: src.network.Network
    # model:   src.moments.MomentModel (M1..M4 per link)
    # precomputed: (dist_to_destination, next_link) from Dijkstra
    # options: ExperimentConfig (iterations, mode, seed, ...)
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

After registration, the name is automatically available:

```bash
python run_experiments.py --algorithm my_algorithm --network Anaheim
```
