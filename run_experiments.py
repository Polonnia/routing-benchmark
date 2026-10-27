"""Run routing experiments for one or more algorithms and networks.

Examples
--------
Random 50 OD pairs on all built-in networks::

    python run_experiments.py

One algorithm on one network::

    python run_experiments.py --algorithm fma --network Anaheim

A single explicit OD pair::

    python run_experiments.py --algorithm fma --network Anaheim \
        --origin 1 --destination 416 --kappa 0.15 --t-factor 1.05

A custom network CSV::

    python run_experiments.py --algorithm fma \
        --network-file path/to/my_network.csv --n-od 20

Multiple algorithms and budgets::

    python run_experiments.py --algorithm fma let \
        --network Winnipeg --kappa 0.15 0.5 --t-factor 1.0 1.05 1.1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.algorithms import available_algorithms  # noqa: E402
from src.experiment import (  # noqa: E402
    NETWORK_FILES,
    ExperimentConfig,
    aggregate,
    environment_info,
    run_sweep,
    write_csv,
    write_results_markdown,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run reliable-routing experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--algorithm",
        "--algorithms",
        dest="algorithms",
        nargs="+",
        choices=available_algorithms(),
        default=["fma", "let"],
        help="algorithm name(s) registered in src/algorithms.py",
    )

    network_group = p.add_mutually_exclusive_group()
    network_group.add_argument(
        "--network",
        "--networks",
        dest="networks",
        nargs="+",
        choices=list(NETWORK_FILES),
        default=list(NETWORK_FILES),
        help="built-in network name(s)",
    )
    network_group.add_argument(
        "--network-file",
        "--network-files",
        dest="network_files",
        nargs="+",
        default=None,
        help="custom network CSV file(s) with From,To,Volume,Cost columns",
    )

    p.add_argument("--origin", type=int, help="origin node ID (1-based)")
    p.add_argument("--destination", type=int, help="destination node ID (1-based)")
    p.add_argument(
        "--n-od",
        type=int,
        default=50,
        help="number of random OD pairs per network (ignored with --origin/--destination)",
    )
    p.add_argument("--min-hops", type=int, default=5, help="reject shorter OD paths")
    p.add_argument(
        "--kappa",
        "--kappas",
        dest="kappas",
        nargs="+",
        type=float,
        default=[0.15, 0.25, 0.50],
        help="link standard-deviation dispersion parameter(s)",
    )
    p.add_argument(
        "--t-factor",
        "--t-factors",
        dest="t_factors",
        nargs="+",
        type=float,
        default=[1.0, 1.05, 1.10],
        help="travel-time budget factor(s), T = factor * t_LET",
    )
    p.add_argument("--iterations", type=int, default=100, help="GPI iterations")
    p.add_argument("--max-tries", type=int, default=100, help="policy-variation tries")
    p.add_argument(
        "--mode",
        choices=["random", "greedy"],
        default="random",
        help="policy-improvement mode for FMA",
    )
    p.add_argument("--seed", type=int, default=808, help="random seed")
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "results",
        help="output directory",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR,
        help="directory containing data/ for built-in networks",
    )
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (args.origin is None) != (args.destination is None):
        raise SystemExit("--origin and --destination must be provided together")

    od_pairs = (
        [(args.origin, args.destination)]
        if args.origin is not None
        else []
    )
    config = ExperimentConfig(
        algorithms=list(args.algorithms),
        networks=list(args.networks),
        network_files=list(args.network_files or []),
        od_pairs=od_pairs,
        kappas=list(args.kappas),
        t_factors=list(args.t_factors),
        n_od=args.n_od,
        min_hops=args.min_hops,
        iterations=args.iterations,
        max_tries=args.max_tries,
        mode=args.mode,
        seed=args.seed,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (out / "environment.json").write_text(
        json.dumps(environment_info(), indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print(
        f"algorithms={config.algorithms}  "
        f"networks={'files' if config.network_files else config.networks}  "
        f"OD={'explicit' if config.od_pairs else f'{config.n_od} random'}  "
        f"kappa={config.kappas}  T={config.t_factors}*t_LET"
    )
    print("=" * 78)

    rows = run_sweep(config, Path(args.data_dir), verbose=not args.quiet)
    aggregates = aggregate(rows)
    write_csv(out / "per_run.csv", rows)
    write_csv(out / "aggregated.csv", aggregates)
    write_results_markdown(out / "results.md", aggregates)

    print(f"Wrote {out / 'per_run.csv'}")
    print(f"Wrote {out / 'aggregated.csv'}")
    print(f"Wrote {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
