"""Generic experiment driver.

The driver is algorithm-agnostic: algorithms are looked up in
``src.algorithms.ALGORITHMS``.  A run is fully described by

* the algorithm name(s),
* the network name or network CSV file,
* the OD pair(s) (explicit origin/destination or random sampling),
* the link-moment dispersion kappa, and
* the travel-time budget factor T / t_LET.

New algorithms only need to be registered in ``src.algorithms``.
"""

from __future__ import annotations

import csv
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .algorithms import available_algorithms, get_algorithm
from .moments import MomentModel
from .network import Network

NETWORK_FILES = {
    "Anaheim": "data/Anaheim/Anaheim_network.csv",
    "Winnipeg": "data/Winnipeg/Winnipeg_network.csv",
    "Chicago_Sketch": "data/Chicago_Sketch/Chicago_Sketch_network.csv",
}
NETWORK_LABELS = {
    "Anaheim": "Anaheim",
    "Winnipeg": "Winnipeg",
    "Chicago_Sketch": "Chicago Sketch",
}


@dataclass
class ExperimentConfig:
    algorithms: list[str] = field(default_factory=lambda: ["fma", "let"])
    networks: list[str] = field(default_factory=lambda: list(NETWORK_FILES))
    network_files: list[str] = field(default_factory=list)
    od_pairs: list[tuple[int, int]] = field(default_factory=list)  # original IDs
    kappas: list[float] = field(default_factory=lambda: [0.15, 0.25, 0.50])
    t_factors: list[float] = field(default_factory=lambda: [1.0, 1.05, 1.10])
    n_od: int = 50
    min_hops: int = 5
    iterations: int = 100
    max_tries: int = 100
    mode: str = "random"
    seed: int = 808


# ----------------------------------------------------------------------
# Data loading and OD sampling
# ----------------------------------------------------------------------
def resolve_network_specs(
    config: ExperimentConfig, data_dir: Path
) -> list[tuple[str, Path]]:
    """Return ``(display_name, path)`` pairs for built-in or custom networks."""
    specs: list[tuple[str, Path]] = []
    if config.network_files:
        for raw in config.network_files:
            path = Path(raw)
            if not path.is_absolute() and not path.exists():
                path = data_dir / path
            if not path.exists():
                raise FileNotFoundError(f"missing network file: {path}")
            specs.append((path.stem, path))
        return specs

    for name in config.networks:
        if name not in NETWORK_FILES:
            raise ValueError(
                f"unknown network {name!r}; built-ins: "
                f"{', '.join(NETWORK_FILES)} (or use --network-file)"
            )
        path = data_dir / NETWORK_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"missing network file: {path}")
        specs.append((NETWORK_LABELS[name], path))
    return specs


def generate_od_pairs(
    network: Network,
    n_pairs: int,
    min_hops: int,
    rng: np.random.Generator,
    max_attempts: int = 500_000,
) -> list[tuple[int, int]]:
    """Sample connected OD pairs whose LET path has at least ``min_hops`` links."""
    eligible = [
        i
        for i in range(network.n_nodes)
        if network.out_links[i] and network.in_links[i]
    ]
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        o = int(eligible[int(rng.integers(0, len(eligible)))])
        d = int(eligible[int(rng.integers(0, len(eligible)))])
        if o == d or (o, d) in seen:
            continue
        if d not in cache:
            cache[d] = network.dijkstra_to(d)
        dist, next_link = cache[d]
        if not np.isfinite(dist[o]):
            continue
        if network.hops_to_destination(o, next_link) < min_hops:
            continue
        seen.add((o, d))
        pairs.append((o, d))
    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"could only sample {len(pairs)}/{n_pairs} OD pairs with "
            f"min_hops={min_hops}"
        )
    return pairs


def _explicit_od_pairs(
    network: Network, od_pairs: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    converted: list[tuple[int, int]] = []
    for origin, destination in od_pairs:
        try:
            o = network.internal_id(int(origin))
            d = network.internal_id(int(destination))
        except KeyError as exc:
            raise ValueError(
                f"node not found in network {network.name}: {exc}"
            ) from exc
        dist, _ = network.dijkstra_to(d)
        if not np.isfinite(dist[o]):
            raise ValueError(
                f"origin {origin} cannot reach destination {destination} "
                f"in network {network.name}"
            )
        converted.append((o, d))
    return converted


def _od_pairs_for_network(
    network: Network,
    config: ExperimentConfig,
    net_idx: int,
) -> list[tuple[int, int]]:
    if config.od_pairs:
        return _explicit_od_pairs(network, config.od_pairs)
    rng = np.random.default_rng(config.seed + 1000 + net_idx)
    return generate_od_pairs(network, config.n_od, config.min_hops, rng)


# ----------------------------------------------------------------------
# One algorithm run
# ----------------------------------------------------------------------
def _run_one(
    network: Network,
    model: MomentModel,
    origin: int,
    destination: int,
    t_factor: float,
    precomputed: tuple[np.ndarray, np.ndarray],
    algorithm: str,
    seed: int,
    config: ExperimentConfig,
) -> dict[str, object]:
    dist, _policy = precomputed
    t_let = float(dist[origin])
    T = t_factor * t_let
    result = get_algorithm(algorithm)(
        network,
        model,
        origin,
        destination,
        T,
        precomputed,
        seed,
        config,
    )
    return {
        "network": network.name,
        "origin": network.original_id(origin),
        "destination": network.original_id(destination),
        "kappa": float(model.kappa),
        "t_factor": float(t_factor),
        "t_let": t_let,
        "T": T,
        "method": algorithm,
        "lower_bound": result.lower_bound,
        "delay_bound_g": result.g,
        "condition": result.condition,
        "time_s": result.runtime_s,
        "improvements": result.improvements,
        "evaluations": result.evaluations,
        "path_len": len(result.path_links),
        "path_nodes": " ".join(
            str(network.original_id(x)) for x in result.path_nodes
        ),
        "initial_lower_bound": result.metadata.get(
            "initial_lower_bound", float("nan")
        ),
    }


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def run_sweep(
    config: ExperimentConfig,
    data_dir: Path,
    verbose: bool = True,
) -> list[dict[str, object]]:
    unknown = [a for a in config.algorithms if a not in available_algorithms()]
    if unknown:
        raise ValueError(
            f"unknown algorithm(s): {', '.join(unknown)}; "
            f"available: {', '.join(available_algorithms())}"
        )

    rows: list[dict[str, object]] = []
    for net_idx, (display_name, path) in enumerate(
        resolve_network_specs(config, data_dir)
    ):
        network = Network.from_csv(path, name=display_name)
        od_pairs = _od_pairs_for_network(network, config, net_idx)
        if verbose:
            print(
                f"[{network.name}] {network.n_nodes} nodes, "
                f"{network.n_links} links, {len(od_pairs)} OD pair(s)"
            )

        for k_idx, kappa in enumerate(config.kappas):
            moment_rng = np.random.default_rng(
                config.seed + 100 * net_idx + k_idx
            )
            model = MomentModel.gaussian(network, kappa, moment_rng)
            pre_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            for tf_idx, tf in enumerate(config.t_factors):
                t_group = time.perf_counter()
                for od_idx, (o, d) in enumerate(od_pairs):
                    if d not in pre_cache:
                        pre_cache[d] = network.dijkstra_to(d)
                    pre = pre_cache[d]
                    for alg_idx, algorithm in enumerate(config.algorithms):
                        run_seed = (
                            config.seed
                            + 1_000_003 * (net_idx + 1)
                            + 10_007 * (k_idx + 1)
                            + 1_009 * (tf_idx + 1)
                            + 97 * (od_idx + 1)
                            + 7_919 * (alg_idx + 1)
                        )
                        rows.append(
                            _run_one(
                                network,
                                model,
                                o,
                                d,
                                tf,
                                pre,
                                algorithm,
                                run_seed,
                                config,
                            )
                        )
                if verbose:
                    group = [
                        r
                        for r in rows
                        if r["network"] == network.name
                        and abs(float(r["kappa"]) - kappa) < 1e-12
                        and abs(float(r["t_factor"]) - tf) < 1e-12
                    ]
                    parts = []
                    for algorithm in config.algorithms:
                        vals = [
                            float(r["lower_bound"])
                            for r in group
                            if r["method"] == algorithm
                        ]
                        if vals:
                            parts.append(
                                f"{algorithm}={statistics.fmean(vals):.4f}"
                            )
                    print(
                        f"  kappa={kappa:.2f} T={tf:.2f}*t_LET: "
                        f"{' '.join(parts)} "
                        f"({time.perf_counter() - t_group:.1f}s)"
                    )
    return rows


# ----------------------------------------------------------------------
# Aggregation and output
# ----------------------------------------------------------------------
def aggregate(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["network"],
            float(row["kappa"]),
            float(row["t_factor"]),
            row["method"],
        )
        groups[key].append(row)

    out: list[dict[str, object]] = []
    for (network, kappa, tf, method), items in sorted(groups.items()):
        lb = np.asarray([float(r["lower_bound"]) for r in items])
        times = np.asarray([float(r["time_s"]) for r in items])
        imp = np.asarray([float(r["improvements"]) for r in items])
        lengths = np.asarray([float(r["path_len"]) for r in items])
        conds = Counter(str(r["condition"]) for r in items)
        out.append(
            {
                "network": network,
                "kappa": kappa,
                "t_factor": tf,
                "method": method,
                "n": len(items),
                "lower_bound_mean": float(lb.mean()),
                "lower_bound_std": float(lb.std(ddof=1)) if len(lb) > 1 else 0.0,
                "lower_bound_median": float(np.median(lb)),
                "time_mean_s": float(times.mean()),
                "improvements_mean": float(imp.mean()),
                "path_len_mean": float(lengths.mean()),
                "condition_mode": conds.most_common(1)[0][0],
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(x: object, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if np.isnan(v) else f"{v:.{nd}f}"


def write_results_markdown(
    path: Path, aggregates: Sequence[dict[str, object]]
) -> None:
    lines = [
        "# Experiment results",
        "",
        "Averages over OD pairs.  `lower bound` is the algorithm's guaranteed "
        "on-time-arrival probability (1 - g); no empirical probability is "
        "computed.",
        "",
        "| Network | kappa | T / t_LET | Algorithm | Lower bound | "
        "Time (s) | Improvements | Path len |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in sorted(
        aggregates,
        key=lambda x: (
            str(x["network"]),
            float(x["kappa"]),
            float(x["t_factor"]),
            str(x["method"]),
        ),
    ):
        lb = f"{_fmt(r['lower_bound_mean'])} ± {_fmt(r['lower_bound_std'])}"
        lines.append(
            f"| {r['network']} | {float(r['kappa']):.2f} | "
            f"{float(r['t_factor']):.2f} | {r['method']} | {lb} | "
            f"{_fmt(r['time_mean_s'], 4)} | "
            f"{_fmt(r['improvements_mean'], 2)} | "
            f"{_fmt(r['path_len_mean'], 1)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def environment_info() -> dict[str, object]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
