"""Monte Carlo evaluation of a path's empirical on-time-arrival probability.

The lower bound produced by FMA should never exceed this simulated value; the
paper reports it in parentheses next to the bound (Table VI).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .moments import MomentModel


def monte_carlo_probability(
    model: MomentModel,
    link_ids: Sequence[int],
    T: float,
    n_samples: int = 3000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate ``P(sum of link travel times <= T)`` by sampling.

    Travel times are sampled from independent Gaussians whose first two
    moments match the link model.  Negative samples are kept (they have
    negligible mass at the standard deviations used in the paper) so that the
    simulation exactly matches the moment model.
    """
    if n_samples <= 0:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    ids = np.asarray(list(link_ids), dtype=np.int64)
    if ids.size == 0:
        return 1.0 if T >= 0.0 else 0.0

    mu = model.M1[ids]
    if model.sigma is not None:
        sd = model.sigma[ids]
    else:
        sd = np.sqrt(np.maximum(model.M2[ids] - mu**2, 0.0))

    # Keep each batch around 2M floats to bound memory use.
    chunk = max(1, 2_000_000 // max(1, ids.size))
    hits = 0
    drawn = 0
    while drawn < n_samples:
        m = min(chunk, n_samples - drawn)
        samples = rng.normal(loc=mu, scale=sd, size=(m, ids.size))
        totals = samples.sum(axis=1)
        hits += int(np.count_nonzero(totals <= T))
        drawn += m
    return hits / n_samples
