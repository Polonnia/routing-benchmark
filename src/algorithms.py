"""Algorithm registry.

New routing algorithms can be added without touching the experiment driver:
implement a runner with the signature below and register it with
``@register("name")``.  The experiment layer passes the network, moment model,
origin/destination, budget, precomputed shortest-path data, a random seed, and
the experiment options object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .fma import FMASolver
from .moments import MomentModel, evaluate_path
from .network import Network


@dataclass
class AlgorithmResult:
    path_links: list[int]
    path_nodes: list[int]
    lower_bound: float
    g: float
    condition: int | str
    runtime_s: float
    improvements: int = 0
    evaluations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


AlgorithmRunner = Callable[
    [
        Network,
        MomentModel,
        int,
        int,
        float,
        tuple[np.ndarray, np.ndarray],
        int,
        Any,
    ],
    AlgorithmResult,
]

ALGORITHMS: dict[str, AlgorithmRunner] = {}


def register(name: str) -> Callable[[AlgorithmRunner], AlgorithmRunner]:
    """Register an algorithm runner under ``name``."""

    def decorator(fn: AlgorithmRunner) -> AlgorithmRunner:
        if name in ALGORITHMS:
            raise ValueError(f"algorithm already registered: {name}")
        ALGORITHMS[name] = fn
        return fn

    return decorator


def available_algorithms() -> list[str]:
    return sorted(ALGORITHMS)


def get_algorithm(name: str) -> AlgorithmRunner:
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown algorithm {name!r}; available: {', '.join(available_algorithms())}"
        ) from exc


# ----------------------------------------------------------------------
# Built-in algorithms
# ----------------------------------------------------------------------
@register("fma")
def run_fma(
    network: Network,
    model: MomentModel,
    origin: int,
    destination: int,
    T: float,
    precomputed: tuple[np.ndarray, np.ndarray],
    seed: int,
    options: Any,
) -> AlgorithmResult:
    """Fourth Moment Approach with generalized policy iteration."""
    solver = FMASolver(
        network,
        model,
        max_policy_iterations=getattr(options, "iterations", 100),
        max_variation_tries=getattr(options, "max_tries", 100),
        seed=seed,
        mode=getattr(options, "mode", "random"),
    )
    result = solver.solve(origin, destination, T, precomputed=precomputed)
    return AlgorithmResult(
        path_links=result.path_links,
        path_nodes=result.path_nodes,
        lower_bound=result.sota_lower,
        g=result.g,
        condition=result.condition,
        runtime_s=result.runtime_s,
        improvements=result.improvements,
        evaluations=result.evaluations,
        metadata={
            "initial_lower_bound": result.sota_lower_initial,
            "initial_g": result.g_initial,
        },
    )


@register("let")
def run_let(
    network: Network,
    model: MomentModel,
    origin: int,
    destination: int,
    T: float,
    precomputed: tuple[np.ndarray, np.ndarray],
    seed: int,
    options: Any,
) -> AlgorithmResult:
    """Least-expected-time path, used as the classic baseline."""
    del seed, options  # deterministic
    _dist, policy = precomputed
    t0 = time.perf_counter()
    nodes, links, ok = network.follow_policy_links(origin, policy)
    if not ok or not nodes or nodes[-1] != destination:
        raise RuntimeError("LET policy did not produce a valid path")
    bound, _moments = evaluate_path(model, links, T)
    runtime = time.perf_counter() - t0
    return AlgorithmResult(
        path_links=links,
        path_nodes=nodes,
        lower_bound=bound.sota_lower,
        g=bound.g,
        condition=bound.condition,
        runtime_s=runtime,
        improvements=0,
        evaluations=1,
    )
