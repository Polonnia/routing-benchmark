"""Extended Bellman equations, Z-conversion, and the fourth-moment bound.

Notation follows the paper:

* ``M_k(i)`` -- k-th raw moment of the travel time from node ``i`` to the
  destination under the current deterministic policy.
* ``M_k^ij`` -- k-th raw moment of link ``ij``'s travel time.
* ``G(o)``   -- travel time from the origin under the policy.
* ``Z = G(o) - T`` -- the bridge random variable.

For a deterministic policy the EBE reduces to a path computation, so the
moment recursion below is exact and requires no convolution.  The moments of a
path are accumulated backwards from the destination in O(path length).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .network import Network

SQRT3 = float(np.sqrt(3.0))


# ----------------------------------------------------------------------
# Link- and path-moment models
# ----------------------------------------------------------------------
@dataclass
class MomentModel:
    """First four raw moments of every link's travel time."""

    M1: np.ndarray
    M2: np.ndarray
    M3: np.ndarray
    M4: np.ndarray
    sigma: np.ndarray | None = None  # standard deviations, if Gaussian
    kappa: float = float("nan")      # spread parameter used to generate sigma

    def __post_init__(self) -> None:
        self.M1 = np.asarray(self.M1, dtype=np.float64).reshape(-1)
        self.M2 = np.asarray(self.M2, dtype=np.float64).reshape(-1)
        self.M3 = np.asarray(self.M3, dtype=np.float64).reshape(-1)
        self.M4 = np.asarray(self.M4, dtype=np.float64).reshape(-1)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=np.float64).reshape(-1)

    @classmethod
    def gaussian(
        cls, network: Network, kappa: float, rng: np.random.Generator
    ) -> "MomentModel":
        """Moment model used in the paper for Anaheim/Winnipeg/Chicago.

        The standard deviation of link ``ij`` is drawn as
        ``sigma_ij = Uniform(0, kappa) * M1_ij``; travel times are then assumed
        Gaussian with that mean and standard deviation, giving

            M2 = mu^2 + sigma^2
            M3 = mu^3 + 3 mu sigma^2
            M4 = mu^4 + 6 mu^2 sigma^2 + 3 sigma^4.
        """
        mu = network.mean_costs
        sigma = kappa * mu * rng.random(network.n_links)
        var = sigma**2
        return cls(
            M1=mu,
            M2=mu**2 + var,
            M3=mu**3 + 3.0 * mu * var,
            M4=mu**4 + 6.0 * mu**2 * var + 3.0 * var**2,
            sigma=sigma,
            kappa=float(kappa),
        )

    def moments_from_arrays(
        self, link_ids: Sequence[int]
    ) -> tuple[float, float, float, float]:
        """Exact first four moments of a fixed link path (EBE, backwards)."""
        M1 = M2 = M3 = M4 = 0.0
        for lid in reversed(link_ids):
            a1 = float(self.M1[lid])
            a2 = float(self.M2[lid])
            a3 = float(self.M3[lid])
            a4 = float(self.M4[lid])
            # Equations (3)-(6): node i -> link ij -> node j.
            new_M4 = a4 + M4 + 4.0 * a3 * M1 + 6.0 * a2 * M2 + 4.0 * a1 * M3
            new_M3 = a3 + M3 + 3.0 * a2 * M1 + 3.0 * a1 * M2
            new_M2 = a2 + M2 + 2.0 * a1 * M1
            new_M1 = a1 + M1
            M1, M2, M3, M4 = new_M1, new_M2, new_M3, new_M4
        return M1, M2, M3, M4

    def path_variances(self, link_ids: Sequence[int]) -> np.ndarray:
        """Per-link variances implied by the first two moments."""
        ids = np.asarray(list(link_ids), dtype=np.int64)
        var = self.M2[ids] - self.M1[ids] ** 2
        return np.maximum(var, 0.0)


# ----------------------------------------------------------------------
# Z-conversion (Eqs. 8-11)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ZMoments:
    M1: float
    M2: float
    M3: float
    M4: float


def z_moments(Mo: Sequence[float], T: float) -> ZMoments:
    """Raw moments of ``Z = G(o) - T`` (Eqs. 8-11)."""
    M1, M2, M3, M4 = (float(x) for x in Mo)
    return ZMoments(
        M1=M1 - T,
        M2=M2 - 2.0 * T * M1 + T**2,
        M3=M3 - 3.0 * T * M2 + 3.0 * T**2 * M1 - T**3,
        M4=M4 - 4.0 * T * M3 + 6.0 * T**2 * M2 - 4.0 * T**3 * M1 + T**4,
    )


# ----------------------------------------------------------------------
# FMA delay-probability bound (Eqs. 12-21)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BoundResult:
    g: float                 # tight upper bound on P[G >= T]
    sota_lower: float        # 1 - g, the lower bound on on-time arrival
    condition: int | str     # which case of the paper was used
    alpha: float | None = None
    v_min: float | None = None
    v_star: float | None = None
    sup_f: float | None = None

    @property
    def delay_bound(self) -> float:
        return self.g


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 1.0
    return float(min(1.0, max(0.0, x)))


def _max_f_over_v(M1: float, M2: float, M4: float) -> tuple[float, float]:
    """Maximise ``f(v) = -2 M1/v + 3 M2/v^2 - M4/v^4`` over ``v > 0``.

    The stationary points solve Eq. (14):
    ``-M1 v^3 + 3 M2 v^2 - 2 M4 = 0``.
    Newton's method is unnecessary here -- the cubic can be solved exactly with
    the companion matrix (``numpy.roots``), which is both faster and robust.
    A numerical fallback is used if no positive real root is found.
    """
    roots = np.roots([-M1, 3.0 * M2, 0.0, -2.0 * M4])
    candidates = [
        float(r.real)
        for r in roots
        if abs(r.imag) <= 1e-8 * max(1.0, abs(r.real)) and r.real > 0.0
    ]
    candidates = [v for v in candidates if np.isfinite(v) and v > 0.0]
    if candidates:
        values = [
            -2.0 * M1 / v + 3.0 * M2 / v**2 - M4 / v**4 for v in candidates
        ]
        j = int(np.argmax(values))
        return float(values[j]), candidates[j]

    # Numerical fallback: coarse log grid followed by golden-section refinement.
    scale = max(abs(M1), np.sqrt(max(M2, 0.0)), max(M4, 0.0) ** 0.25, 1e-9)
    grid = np.logspace(np.log10(scale * 1e-6), np.log10(scale * 1e6), 2000)
    vals = -2.0 * M1 / grid + 3.0 * M2 / grid**2 - M4 / grid**4
    i = int(np.argmax(vals))
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        a = hi - phi * (hi - lo)
        b = lo + phi * (hi - lo)
        fa = -2.0 * M1 / a + 3.0 * M2 / a**2 - M4 / a**4
        fb = -2.0 * M1 / b + 3.0 * M2 / b**2 - M4 / b**4
        if fa > fb:
            hi = b
        else:
            lo = a
    v = 0.5 * (lo + hi)
    return float(-2.0 * M1 / v + 3.0 * M2 / v**2 - M4 / v**4), float(v)


def delay_probability_bound(
    Mo: Sequence[float], T: float, tol: float | None = None
) -> BoundResult:
    """Tight upper bound on ``P[G(o) >= T]`` from the first four moments.

    Implements Conditions 1-5 (Eqs. 17-21) after the Z-conversion
    (Eqs. 8-11).  ``Mo`` are the first four raw moments of ``G(o)``.
    """
    Z = z_moments(Mo, T)
    M1, M2, M3, M4 = Z.M1, Z.M2, Z.M3, Z.M4

    if tol is None:
        tol = 1e-9 * max(1.0, abs(T), abs(M1))
    var_z = M2 - M1**2
    if not np.isfinite(M2) or M2 <= 1e-15 or var_z <= -1e-12:
        # Degenerate (zero-variance) case; cannot be reached with kappa > 0.
        return BoundResult(g=1.0, sota_lower=0.0, condition="degenerate")

    ratio_4_2 = M4 / M2**2

    # Conditions 1 and 2: mean of Z is zero.
    if abs(M1) <= tol:
        if ratio_4_2 >= (3.0 * SQRT3 - 3.0) / 2.0:
            g = 1.0 - (2.0 * SQRT3 - 3.0) * M2**2 / M4
            return BoundResult(g=_clip01(g), sota_lower=_clip01(1.0 - g), condition=1)
        g = 0.5 + np.sqrt(max(0.0, 0.25 - 1.0 / (3.0 + ratio_4_2)))
        return BoundResult(g=_clip01(g), sota_lower=_clip01(1.0 - g), condition=2)

    # Condition 3 (Eq. 19).
    if ratio_4_2 >= M2 / M1**2:
        g = 1.0 - M1**2 / M2 if M1 < 0.0 else 1.0
        return BoundResult(g=_clip01(g), sota_lower=_clip01(1.0 - g), condition=3)

    # Conditions 4 and 5 require the auxiliary variables of Eqs. (15)-(16).
    denom = M2 - M1**2
    alpha_sq = (M4 - M2**2) / denom
    alpha = float(np.sqrt(max(0.0, alpha_sq)))
    v_min = float(
        np.sqrt((SQRT3 - 1.0) * M2 + (7.0 - 4.0 * SQRT3) / 4.0 * M1**2)
        + (2.0 - SQRT3) / 2.0 * M1
    )
    threshold = (SQRT3 - 1.0) * v_min / 2.0

    if alpha >= threshold:
        # Condition 4 (Eq. 20).
        sup_f, v_star = _max_f_over_v(M1, M2, M4)
        g = 1.0 - (4.0 / 9.0) * (2.0 * SQRT3 - 3.0) * sup_f
        return BoundResult(
            g=_clip01(g),
            sota_lower=_clip01(1.0 - g),
            condition=4,
            alpha=alpha,
            v_min=v_min,
            v_star=v_star,
            sup_f=sup_f,
        )

    # Condition 5 (Eq. 21).
    radicand = 4.0 * M2 + alpha**2 + 4.0 * M1 * alpha
    denom5 = np.sqrt(max(0.0, radicand))
    if denom5 <= 1e-15:
        g = 0.5
    else:
        g = 0.5 * (1.0 + (alpha + 2.0 * M1) / denom5)
    return BoundResult(
        g=_clip01(g),
        sota_lower=_clip01(1.0 - g),
        condition=5,
        alpha=alpha,
        v_min=v_min,
    )


def evaluate_path(
    model: MomentModel, link_ids: Sequence[int], T: float
) -> tuple[BoundResult, tuple[float, float, float, float]]:
    """Evaluate the bound of a fixed path. Returns ``(bound, path moments)``."""
    Mo = model.moments_from_arrays(link_ids)
    return delay_probability_bound(Mo, T), Mo
