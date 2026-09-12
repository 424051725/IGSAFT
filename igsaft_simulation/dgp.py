"""Data-generating process used in the simulation section."""

from dataclasses import dataclass

import numpy as np

from .basis import interaction_basis, interaction_indices
from .config import DGPConfig


@dataclass(frozen=True)
class SimulatedData:
    observed_time: np.ndarray
    exposure: np.ndarray
    instruments: np.ndarray
    event: np.ndarray
    failure_time: np.ndarray
    censoring_time: np.ndarray
    active_interactions: np.ndarray
    theta: np.ndarray
    phi: np.ndarray


def _main_effects(config: DGPConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate (theta, phi) for Cases 1--4 in the paper."""

    p = config.p
    if config.scenario == 1:
        theta = np.ones(p)
        phi = np.zeros(p)
        invalid = rng.choice(p, size=int(round(0.30 * p)), replace=False)
        phi[invalid] = 0.2
    elif config.scenario == 2:
        theta = np.ones(p)
        phi = np.zeros(p)
        group_size = int(round(0.20 * p))
        shuffled = rng.permutation(p)
        phi[shuffled[:group_size]] = 0.2
        phi[shuffled[group_size : 2 * group_size]] = 0.4
        phi[shuffled[2 * group_size : 3 * group_size]] = 0.6
    elif config.scenario == 3:
        theta = rng.normal(1.0, 1.0, size=p)
        phi = rng.normal(0.2, 0.2, size=p)
    else:
        theta = rng.normal(1.0, 1.0, size=p)
        phi = np.zeros(p)
        invalid = rng.choice(p, size=int(round(0.70 * p)), replace=False)
        phi[invalid] = 0.5 * theta[invalid]
    return theta, phi


def _calibrated_uniform_censoring(
    failure_time: np.ndarray,
    target_rate: float,
    width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose the uniform interval location to attain the requested sample rate.

    A single vector of uniform draws is held fixed while the interval is shifted.
    This makes calibration deterministic for a given seed and avoids an unbounded
    stochastic search.
    """

    uniforms = rng.uniform(size=failure_time.size)
    base = width * uniforms

    def rate(lower: float) -> float:
        return float(np.mean(failure_time > lower + base))

    lower_lo = float(np.min(failure_time - base) - width)
    lower_hi = float(np.max(failure_time - base) + width)
    for _ in range(80):
        midpoint = 0.5 * (lower_lo + lower_hi)
        if rate(midpoint) > target_rate:
            lower_lo = midpoint
        else:
            lower_hi = midpoint
    return lower_hi + base


def simulate_data(config: DGPConfig) -> SimulatedData:
    """Generate one censored data set from a selected paper scenario."""

    config.validate()
    rng = np.random.default_rng(config.seed)
    z = rng.normal(size=(config.n, config.p))
    theta, phi = _main_effects(config, rng)

    error_covariance = np.array([[0.4, 0.2], [0.2, 0.4]])
    errors = rng.multivariate_normal(np.zeros(2), error_covariance, size=config.n)
    outcome_error, exposure_error = errors[:, 0], errors[:, 1]

    pair_count = len(interaction_indices(config.p, 2))
    active_count = max(1, int(round(config.active_interaction_fraction * pair_count)))
    active = np.sort(rng.choice(pair_count, size=active_count, replace=False))
    interaction_coefficients = np.zeros(pair_count)
    strength = config.interaction_scale / config.n**0.25
    interaction_coefficients[active] = rng.normal(size=active_count) * strength

    interaction_effect = interaction_basis(z, 2) @ interaction_coefficients
    exposure = z @ theta + interaction_effect + exposure_error
    failure_time = config.beta_true * exposure + z @ phi + outcome_error
    censoring_time = _calibrated_uniform_censoring(
        failure_time, config.censor_rate, config.censoring_width, rng
    )
    event = (failure_time <= censoring_time).astype(int)
    observed_time = np.minimum(failure_time, censoring_time)

    return SimulatedData(
        observed_time=observed_time,
        exposure=exposure,
        instruments=z,
        event=event,
        failure_time=failure_time,
        censoring_time=censoring_time,
        active_interactions=active,
        theta=theta,
        phi=phi,
    )
