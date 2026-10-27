"""Fourth Moment Approach (FMA) with generalized policy iteration (GPI).

The implementation follows Algorithms 1 and 2 of the paper:

* policy evaluation (PE): compute the moments of the policy-induced path with
  the extended Bellman equations, convert them to moments of ``Z = G(o) - T``,
  and evaluate the tight delay-probability bound;
* policy improvement (PI): vary the policy at a node along the current
  origin-destination path (Eq. 22), re-evaluate, and keep the variation only
  if it lowers the delay bound.

Two improvement operators are provided:

``mode="random"``
    Draw a random path node and a random outgoing link, exactly as in the
    authors' released code (Algorithm 2).  This is the paper-faithful default.
``mode="greedy"``
    Evaluate every single-node policy variation along the current path and
    accept the best improving one (best-improvement GPI).  This removes the
    variance of random sampling and is useful as an upper baseline for what
    the FMA policy-improvement step can achieve.

A deterministic policy is stored as an array mapping every node to the index of
its next link (``-1`` at the destination), which mirrors the paper's
``(i) = ij`` notation and supports parallel links.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .moments import BoundResult, MomentModel, evaluate_path
from .network import Network


@dataclass
class PolicyEvaluation:
    g: float
    sota_lower: float
    condition: int | str
    nodes: list[int]
    links: list[int]
    moments: tuple[float, float, float, float]
    bound: BoundResult


@dataclass
class GPIResult:
    network: str
    origin: int
    destination: int
    T: float
    t_let: float
    g_initial: float
    sota_lower_initial: float
    g: float
    sota_lower: float
    condition: int | str
    path_nodes: list[int]
    path_links: list[int]
    iterations: int
    improvements: int
    evaluations: int
    runtime_s: float
    mode: str = "random"
    improvement_history: list[float] = field(default_factory=list)


@dataclass
class _Variation:
    links: list[int]
    nodes: list[int]
    varied_node: int
    new_link: int


class FMASolver:
    """FMA/GPI solver for one network and one moment model."""

    def __init__(
        self,
        network: Network,
        moments: MomentModel,
        max_policy_iterations: int = 100,
        max_variation_tries: int = 100,
        seed: int = 0,
        mode: str = "random",
        improve_tol: float = 1e-12,
    ) -> None:
        if mode not in {"random", "greedy"}:
            raise ValueError("mode must be 'random' or 'greedy'")
        self.network = network
        self.moments = moments
        self.K = int(max_policy_iterations)
        self.max_tries = int(max_variation_tries)
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.improve_tol = float(improve_tol)

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------
    def evaluate_links(
        self, link_ids: Sequence[int], T: float, destination: int
    ) -> PolicyEvaluation:
        link_ids = list(link_ids)
        nodes = self.network.node_path(link_ids)
        if not nodes or nodes[-1] != destination:
            raise ValueError("policy path does not terminate at the destination")
        bound, Mo = evaluate_path(self.moments, link_ids, T)
        return PolicyEvaluation(
            g=bound.g,
            sota_lower=bound.sota_lower,
            condition=bound.condition,
            nodes=nodes,
            links=link_ids,
            moments=Mo,
            bound=bound,
        )

    # ------------------------------------------------------------------
    # Policy variation (Eq. 22)
    # ------------------------------------------------------------------
    def _make_variation(
        self,
        idx: int,
        lid: int,
        path_nodes: list[int],
        path_links: list[int],
        policy_links: np.ndarray,
        destination: int,
        reachable: np.ndarray,
    ) -> _Variation | None:
        """Re-route the path at position ``idx`` through link ``lid``.

        The suffix from the new next node follows the *existing* policy, so the
        candidate differs from the current policy at exactly one node.
        """
        node = path_nodes[idx]
        link = self.network.links[lid]
        if link.u != node:
            return None
        new_next = link.v
        if new_next == node or not reachable[new_next]:
            return None

        suffix_nodes, suffix_links, ok = self.network.follow_policy_links(
            new_next, policy_links
        )
        if not ok or not suffix_nodes or suffix_nodes[-1] != destination:
            return None

        prefix_nodes = path_nodes[: idx + 1]
        if set(prefix_nodes).intersection(suffix_nodes):
            return None  # would create a cycle

        return _Variation(
            links=path_links[:idx] + [lid] + suffix_links,
            nodes=prefix_nodes + suffix_nodes[1:],
            varied_node=node,
            new_link=lid,
        )

    def _random_candidate(
        self,
        path_nodes: list[int],
        path_links: list[int],
        policy_links: np.ndarray,
        explored: set[tuple[int, ...]],
        destination: int,
        reachable: np.ndarray,
    ) -> _Variation | None:
        """Paper/released-code variation: random path node + random out-link."""
        n_positions = len(path_nodes) - 1  # exclude destination
        if n_positions <= 0:
            return None
        for _ in range(self.max_tries):
            idx = int(self.rng.integers(0, n_positions))
            node = path_nodes[idx]
            out = self.network.out_links[node]
            if not out:
                continue
            lid = int(out[int(self.rng.integers(0, len(out)))])
            if lid == int(policy_links[node]):
                continue
            var = self._make_variation(
                idx,
                lid,
                path_nodes,
                path_links,
                policy_links,
                destination,
                reachable,
            )
            if var is None:
                continue
            key = tuple(var.links)
            if key in explored:
                continue
            explored.add(key)
            return var
        return None

    # ------------------------------------------------------------------
    # Main GPI loop
    # ------------------------------------------------------------------
    def solve(
        self,
        origin: int,
        destination: int,
        T: float,
        precomputed: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> GPIResult:
        """Run FMA from ``origin`` to ``destination`` with budget ``T``.

        ``precomputed`` may contain the ``(dist_to_dest, next_link)`` pair from
        :meth:`Network.dijkstra_to`; the experiment layer caches it per
        destination to avoid repeating the shortest-path search.
        """
        t_start = time.perf_counter()

        if precomputed is None:
            dist_to_dest, policy_links = self.network.dijkstra_to(destination)
        else:
            dist_to_dest, policy_links = precomputed
        if not np.isfinite(dist_to_dest[origin]):
            raise ValueError("origin cannot reach destination")
        t_let = float(dist_to_dest[origin])
        reachable = np.isfinite(dist_to_dest)

        init_nodes, init_links, ok = self.network.follow_policy_links(
            origin, policy_links
        )
        if not ok or not init_nodes or init_nodes[-1] != destination:
            raise RuntimeError("LET policy is not a valid origin-destination path")

        initial_eval = self.evaluate_links(init_links, T, destination)
        best_eval = initial_eval
        best_policy = policy_links.copy()
        explored: set[tuple[int, ...]] = {tuple(init_links)}
        history = [initial_eval.g]
        improvements = 0
        evaluations = 1
        iterations = 0

        if self.mode == "random":
            for _ in range(self.K):
                var = self._random_candidate(
                    best_eval.nodes,
                    best_eval.links,
                    best_policy,
                    explored,
                    destination,
                    reachable,
                )
                if var is None:
                    break
                iterations += 1
                ev = self.evaluate_links(var.links, T, destination)
                evaluations += 1
                if ev.g < best_eval.g:
                    best_eval = ev
                    best_policy[var.varied_node] = var.new_link
                    improvements += 1
                    history.append(ev.g)
        else:
            # Best-improvement sweep over every (path node, outgoing link).
            for _ in range(self.K):
                iterations += 1
                best_var: _Variation | None = None
                best_ev: PolicyEvaluation | None = None
                for idx in range(len(best_eval.nodes) - 1):
                    node = best_eval.nodes[idx]
                    for lid in self.network.out_links[node]:
                        if lid == int(best_policy[node]):
                            continue
                        var = self._make_variation(
                            idx,
                            lid,
                            best_eval.nodes,
                            best_eval.links,
                            best_policy,
                            destination,
                            reachable,
                        )
                        if var is None:
                            continue
                        key = tuple(var.links)
                        if key in explored:
                            continue
                        explored.add(key)
                        ev = self.evaluate_links(var.links, T, destination)
                        evaluations += 1
                        if (
                            ev.g < best_eval.g - self.improve_tol
                            and (best_ev is None or ev.g < best_ev.g)
                        ):
                            best_var, best_ev = var, ev
                if best_var is None or best_ev is None:
                    break
                best_eval = best_ev
                best_policy[best_var.varied_node] = best_var.new_link
                improvements += 1
                history.append(best_ev.g)

        runtime = time.perf_counter() - t_start
        return GPIResult(
            network=self.network.name,
            origin=origin,
            destination=destination,
            T=float(T),
            t_let=t_let,
            g_initial=initial_eval.g,
            sota_lower_initial=initial_eval.sota_lower,
            g=best_eval.g,
            sota_lower=best_eval.sota_lower,
            condition=best_eval.condition,
            path_nodes=best_eval.nodes,
            path_links=best_eval.links,
            iterations=iterations,
            improvements=improvements,
            evaluations=evaluations,
            runtime_s=runtime,
            mode=self.mode,
            improvement_history=history,
        )
