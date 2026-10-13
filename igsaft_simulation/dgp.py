from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class DGPConfig:
    n: int = 10_000
    p: int = 10
    case: int = 1
    censor_rate: float = 0.20
    beta_true: float = 1.0
    interaction_scale: float = 0.4
    censor_width: float = 110.0
    seed: int = 2025


def _case_coefficients(rng, p, case):
    if case == 1:
        theta, phi = np.ones(p), np.zeros(p)
        phi[rng.choice(p, int(round(0.30 * p)), replace=False)] = 0.2
    elif case == 2:
        theta, phi = np.ones(p), np.zeros(p)
        block, order = int(round(0.20 * p)), rng.permutation(p)
        phi[order[:block]] = 0.2
        phi[order[block:2 * block]] = 0.4
        phi[order[2 * block:3 * block]] = 0.6
    elif case == 3:
        theta = rng.normal(1.0, 1.0, p)
        phi = rng.normal(0.2, np.sqrt(0.2), p)
    elif case == 4:
        theta, phi = rng.normal(1.0, 1.0, p), np.zeros(p)
        invalid = rng.choice(p, int(round(0.70 * p)), replace=False)
        phi[invalid] = 0.5 * theta[invalid]
    else:
        raise ValueError("case must be 1, 2, 3, or 4")
    return theta, phi


def simulate_data(config):
    rng = np.random.default_rng(config.seed)
    n, p = config.n, config.p
    Z = rng.normal(size=(n, p))
    theta, phi = _case_coefficients(rng, p, config.case)
    errors = rng.multivariate_normal(
        [0.0, 0.0], [[0.4, 0.2], [0.2, 0.4]], size=n
    )
    epsilon, nu = errors.T
    pairs = list(combinations(range(p), 2))
    H = np.column_stack([Z[:, j] * Z[:, k] for j, k in pairs])
    active_count = len(pairs) if p == 10 else int(round(0.40 * len(pairs)))
    active = np.sort(rng.choice(len(pairs), active_count, replace=False))
    gamma = np.zeros(len(pairs))
    gamma[active] = (
        rng.normal(size=active_count) * config.interaction_scale * n ** (-0.25)
    )
    D = Z @ theta + H @ gamma + nu
    T = config.beta_true * D + Z @ phi + epsilon
    u = rng.uniform(size=n)
    lower = float(np.quantile(T - config.censor_width * u, 1 - config.censor_rate))
    C = lower + config.censor_width * u
    delta = (T <= C).astype(int)
    return {
        "Y": np.minimum(T, C),
        "D": D,
        "Z": Z,
        "delta": delta,
        "T": T,
        "active_interactions": active,
        "observed_censoring": float(np.mean(1 - delta)),
    }

