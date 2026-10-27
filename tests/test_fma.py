"""Unit tests for the FMA reproduction.

Run from the project root with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fma import FMASolver  # noqa: E402
from src.moments import (  # noqa: E402
    MomentModel,
    delay_probability_bound,
    evaluate_path,
    z_moments,
)
from src.network import Network  # noqa: E402
from src.simulate import monte_carlo_probability  # noqa: E402


SIMPLE_EDGES = [
    (1, 2, 5.0),
    (1, 3, 4.0),
    (2, 4, 1.0),
    (2, 5, 5.0),
    (3, 2, 2.0),
    (3, 4, 2.0),
    (4, 6, 6.0),
    (5, 4, 4.0),
    (5, 6, 4.0),
]
SIMPLE_M1 = [5.0, 4.0, 1.0, 5.0, 2.0, 2.0, 6.0, 4.0, 4.0]
SIMPLE_M2 = [30.0, 19.0, 3.0, 28.0, 6.0, 7.5, 38.0, 19.5, 20.0]
SIMPLE_M3 = [200.0, 100.0, 7.0, 170.0, 20.0, 29.0, 252.0, 106.0, 112.0]
SIMPLE_M4 = [1450.0, 571.0, 25.0, 1102.0, 76.0, 136.75, 1740.0, 628.75, 688.0]


def simple_model() -> tuple[Network, MomentModel]:
    net = Network.from_edges("Simple", SIMPLE_EDGES)
    model = MomentModel(
        M1=SIMPLE_M1, M2=SIMPLE_M2, M3=SIMPLE_M3, M4=SIMPLE_M4
    )
    return net, model


class TestTableIII(unittest.TestCase):
    def test_moments_and_bounds(self) -> None:
        net, model = simple_model()
        T = 1.15 * 12.0
        cases = {
            (1, 2, 5, 4, 6): (6.2, 51.94, 489.43, 5138.02, 1.0000),
            (1, 2, 5, 6): (0.2, 12.04, 7.21, 434.88, 0.8620),
            (1, 2, 4, 6): (-1.8, 12.24, -54.43, 428.46, 0.6758),
            (1, 3, 4, 6): (-1.8, 11.74, -51.74, 392.49, 0.6712),
        }
        for nodes, target in cases.items():
            internal = [net.internal_id(n) for n in nodes]
            links = [
                net.link_between(u, v)
                for u, v in zip(internal, internal[1:])
            ]
            bound, Mo = evaluate_path(model, links, T)
            Z = z_moments(Mo, T)
            self.assertAlmostEqual(Z.M1, target[0], delta=1e-6)
            self.assertAlmostEqual(Z.M2, target[1], delta=1e-3)
            # Tab. III prints moments to 2-3 decimals.
            self.assertAlmostEqual(Z.M3, target[2], delta=1e-2)
            self.assertAlmostEqual(Z.M4, target[3], delta=1e-2)
            self.assertAlmostEqual(bound.g, target[4], delta=1e-3)


class TestGaussianBound(unittest.TestCase):
    def test_magic_value_at_mean(self) -> None:
        # N(10, 1): g = 1 - (2 sqrt(3) - 3)/3, SOTA lower = 0.1547.
        bound = delay_probability_bound((10.0, 101.0, 1030.0, 10603.0), 10.0)
        self.assertEqual(bound.condition, 1)
        self.assertAlmostEqual(bound.g, 0.8452994616207485, places=12)
        self.assertAlmostEqual(bound.sota_lower, 0.1547005383792515, places=12)

    def test_lower_bound_is_valid(self) -> None:
        model = MomentModel(
            M1=np.array([10.0]),
            M2=np.array([101.0]),
            M3=np.array([1030.0]),
            M4=np.array([10603.0]),
            sigma=np.array([1.0]),
        )
        rng = np.random.default_rng(0)
        for T in (9.0, 10.0, 11.0, 13.0):
            bound, _ = evaluate_path(model, [0], T)
            actual = monte_carlo_probability(model, [0], T, 200_000, rng)
            self.assertLessEqual(bound.sota_lower, actual + 1e-6)

    def test_degenerate_zero_variance(self) -> None:
        bound = delay_probability_bound((5.0, 25.0, 125.0, 625.0), 5.0)
        self.assertEqual(bound.condition, "degenerate")
        self.assertEqual(bound.sota_lower, 0.0)


class TestDijkstra(unittest.TestCase):
    def test_shortest_paths(self) -> None:
        net = Network.from_edges(
            "Tiny",
            [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 3.0), (3, 4, 1.0)],
        )
        d = net.internal_id(4)
        dist, next_link = net.dijkstra_to(d)
        self.assertAlmostEqual(dist[net.internal_id(1)], 3.0)
        nodes, links, ok = net.follow_policy_links(net.internal_id(1), next_link)
        self.assertTrue(ok)
        self.assertEqual([net.original_id(x) for x in nodes], [1, 2, 3, 4])
        self.assertEqual(len(links), 3)

    def test_unreachable(self) -> None:
        net = Network.from_edges("Tiny", [(1, 2, 1.0), (3, 4, 1.0)])
        dist, _ = net.dijkstra_to(net.internal_id(4))
        self.assertFalse(np.isfinite(dist[net.internal_id(1)]))


def risky_safe_network() -> tuple[Network, MomentModel]:
    """A short but variable direct link vs. a long but reliable detour.

    t_LET = 10 (direct), T = 10.5.  The direct link's bound is 0.2036 while the
    detour's is 0.2467, so FMA must switch to the detour even though its mean
    is larger.
    """
    edges = [(1, 11, 10.0)]
    edges += [(i, i + 1, 1.04) for i in range(1, 11)]
    mu = np.array([10.0] + [1.04] * 10)
    sigma = np.array([3.0] + [0.1] * 10)
    var = sigma**2
    model = MomentModel(
        M1=mu,
        M2=mu**2 + var,
        M3=mu**3 + 3.0 * mu * var,
        M4=mu**4 + 6.0 * mu**2 * var + 3.0 * var**2,
        sigma=sigma,
    )
    return Network.from_edges("RiskySafe", edges), model


class TestGPI(unittest.TestCase):
    def test_detour_improvement(self) -> None:
        net, model = risky_safe_network()
        o, d = net.internal_id(1), net.internal_id(11)
        for mode in ("random", "greedy"):
            solver = FMASolver(net, model, seed=1, mode=mode)
            result = solver.solve(o, d, 10.5)
            self.assertEqual(result.improvements, 1)
            self.assertAlmostEqual(result.sota_lower_initial, 0.2036, places=3)
            self.assertAlmostEqual(result.sota_lower, 0.2467, places=3)
            self.assertEqual(len(result.path_links), 10)
            self.assertEqual(len(set(result.path_nodes)), len(result.path_nodes))

    def test_output_path_is_acyclic(self) -> None:
        net, model = risky_safe_network()
        o, d = net.internal_id(1), net.internal_id(11)
        result = FMASolver(net, model, seed=3).solve(o, d, 10.5)
        nodes = result.path_nodes
        self.assertEqual(nodes[0], o)
        self.assertEqual(nodes[-1], d)
        self.assertEqual(len(set(nodes)), len(nodes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
