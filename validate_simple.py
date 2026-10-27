"""Validate the implementation against the paper's simple network (Tab. III).

The network, link moments and target values are transcribed from Fig. 3 and
Table III of Guo et al., "Navigation With Time Limits in Transportation
Networks: A Fourth Moment Approach", IEEE T-ITS 2022.

It also checks the bound against a Gaussian single-link case, for which the
on-time-arrival probability is known analytically.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.moments import MomentModel, delay_probability_bound, evaluate_path, z_moments
from src.network import Network

# Fig. 3: links in the same order as the link-moment table.
EDGES = [
    (1, 2, 5.0),   # e12
    (1, 3, 4.0),   # e13
    (2, 4, 1.0),   # e24
    (2, 5, 5.0),   # e25
    (3, 2, 2.0),   # e32
    (3, 4, 2.0),   # e34
    (4, 6, 6.0),   # e46
    (5, 4, 4.0),   # e54
    (5, 6, 4.0),   # e56
]
M1 = [5.0, 4.0, 1.0, 5.0, 2.0, 2.0, 6.0, 4.0, 4.0]
M2 = [30.0, 19.0, 3.0, 28.0, 6.0, 7.5, 38.0, 19.5, 20.0]
M3 = [200.0, 100.0, 7.0, 170.0, 20.0, 29.0, 252.0, 106.0, 112.0]
M4 = [1450.0, 571.0, 25.0, 1102.0, 76.0, 136.75, 1740.0, 628.75, 688.0]

# Table III: node sequence -> (M1(Z), M2(Z), M3(Z), M4(Z), g)
TABLE_III = {
    (1, 2, 5, 4, 6): (6.2, 51.94, 489.43, 5138.02, 1.0000),
    (1, 2, 5, 6): (0.2, 12.04, 7.21, 434.88, 0.8620),
    (1, 2, 4, 6): (-1.8, 12.24, -54.43, 428.46, 0.6758),
    (1, 3, 4, 6): (-1.8, 11.74, -51.74, 392.49, 0.6712),
}
T_LET = 12.0
T = 1.15 * T_LET


def path_links(net: Network, nodes: tuple[int, ...]) -> list[int]:
    ids = [net.internal_id(n) for n in nodes]
    return [net.link_between(u, v) for u, v in zip(ids, ids[1:])]


def check_table_iii() -> bool:
    net = Network.from_edges("Simple", EDGES)
    model = MomentModel(M1=M1, M2=M2, M3=M3, M4=M4)
    print(f"Simple network: {net.n_nodes} nodes, {net.n_links} links, T = {T:.2f}")
    print("paper values vs. reproduced values")
    print("-" * 96)
    header = (
        f"{'policy':<16}{'M1(Z)':>10}{'M2(Z)':>10}{'M3(Z)':>11}"
        f"{'M4(Z)':>12}{'g (paper)':>11}{'g (ours)':>10}{'cond':>6}"
    )
    print(header)
    ok = True
    for nodes, target in TABLE_III.items():
        links = path_links(net, nodes)
        bound, Mo = evaluate_path(model, links, T)
        Z = z_moments(Mo, T)
        moment_err = max(
            abs(Z.M1 - target[0]),
            abs(Z.M2 - target[1]),
            abs(Z.M3 - target[2]),
            abs(Z.M4 - target[3]),
        )
        g_err = abs(bound.g - target[4])
        # Moments in Tab. III are printed to 2-3 decimals.
        ok &= moment_err < 1e-2 and g_err < 1e-3
        print(
            f"{str(nodes):<16}{Z.M1:>10.3f}{Z.M2:>10.3f}{Z.M3:>11.3f}"
            f"{Z.M4:>12.3f}{target[4]:>11.4f}{bound.g:>10.4f}"
            f"{str(bound.condition):>6}"
        )
    print("-" * 96)
    print("Table III reproduction:", "PASS" if ok else "FAIL")
    return ok


def gaussian_mc(mu: float, sigma: float, T: float, n: int = 2_000_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    samples = rng.normal(mu, sigma, n)
    return float(np.mean(samples <= T))


def check_gaussian_bound() -> bool:
    print("\nGaussian single-link checks (mean 10, std 1)")
    print("-" * 72)
    model = MomentModel(
        M1=np.array([10.0]),
        M2=np.array([101.0]),
        M3=np.array([1030.0]),   # 1000 + 3*10*1
        M4=np.array([10603.0]),  # 10000 + 600 + 3
        sigma=np.array([1.0]),
    )
    ok = True
    print(f"{'T':>5}{'z':>8}{'bound SOTA':>12}{'MC SOTA':>10}{'cond':>6}  valid")
    for T in (8.0, 9.0, 10.0, 11.0, 13.0):
        bound, _ = evaluate_path(model, [0], T)
        actual = gaussian_mc(10.0, 1.0, T)
        valid = bound.sota_lower <= actual + 1e-9
        ok &= valid
        print(
            f"{T:>5.1f}{(T - 10.0):>8.1f}{bound.sota_lower:>12.4f}"
            f"{actual:>10.4f}{str(bound.condition):>6}  {valid}"
        )
    # The magic value at T = t_LET for a Gaussian: g = 1 - (2*sqrt(3)-3)/3.
    at_mean = delay_probability_bound((10.0, 101.0, 1030.0, 10603.0), 10.0)
    expected_g = 1.0 - (2.0 * math.sqrt(3.0) - 3.0) / 3.0
    expected_sota = 1.0 - expected_g
    magic_ok = (
        abs(at_mean.g - expected_g) < 1e-12
        and abs(at_mean.sota_lower - expected_sota) < 1e-12
    )
    print(
        f"\nGaussian bound at T=mean: g={at_mean.g:.6f} (paper 0.85), "
        f"SOTA lower={at_mean.sota_lower:.6f} (paper 0.1547) -> "
        f"{'PASS' if magic_ok else 'FAIL'}"
    )
    print("-" * 72)
    return ok and magic_ok


def main() -> int:
    ok1 = check_table_iii()
    ok2 = check_gaussian_bound()
    print("\nOverall:", "PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
