"""Marginal Kaplan--Meier nuisance functions for cross-fitted AIPCW moments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AIPCWNuisance:
    """AIPCW nuisance curves estimated on one auxiliary fold."""

    time_grid: np.ndarray
    survival_after: np.ndarray
    conditional_mean: np.ndarray
    integral: np.ndarray
    epsilon: float


def fit_aipcw_nuisance(
    observed_time: np.ndarray,
    event: np.ndarray,
    complete_data_moment: np.ndarray,
    epsilon: float = 1e-8,
) -> AIPCWNuisance:
    """Estimate marginal ``G`` and ``xi`` using an auxiliary fold only."""

    observed_time = np.asarray(observed_time, dtype=float)
    event = np.asarray(event, dtype=float)
    complete_data_moment = np.asarray(complete_data_moment, dtype=float)
    if complete_data_moment.ndim != 2:
        raise ValueError("complete_data_moment must be a two-dimensional array")
    if not (observed_time.size == event.size == complete_data_moment.shape[0]):
        raise ValueError("time, event, and moment arrays must have the same number of rows")

    order = np.argsort(observed_time)
    sorted_time = observed_time[order]
    sorted_event = event[order]
    sorted_moment = complete_data_moment[order]
    time_grid, group_starts = np.unique(sorted_time, return_index=True)

    risk_set = np.arange(sorted_time.size, 0, -1)[group_starts]
    censoring_events = np.add.reduceat(1.0 - sorted_event, group_starts)
    survival_after = np.cumprod(
        1.0 - censoring_events / np.maximum(risk_set, epsilon)
    )
    survival_left = np.concatenate(([1.0], survival_after[:-1]))

    sorted_groups = np.searchsorted(time_grid, sorted_time, side="left")
    ipcw_weights = sorted_event / np.clip(
        survival_left[sorted_groups], epsilon, 1.0
    )
    weighted_moment = ipcw_weights[:, None] * sorted_moment
    tail_numerator = np.cumsum(weighted_moment[::-1], axis=0)[::-1]
    tail_denominator = np.cumsum(ipcw_weights[::-1])[::-1]
    conditional_mean = tail_numerator[group_starts] / np.maximum(
        tail_denominator[group_starts, None], epsilon
    )

    increments = np.zeros_like(conditional_mean)
    increments[1:] = conditional_mean[1:] - conditional_mean[:-1]
    integral = np.cumsum(
        increments / np.maximum(survival_left[:, None], epsilon), axis=0
    )
    return AIPCWNuisance(
        time_grid=time_grid,
        survival_after=survival_after,
        conditional_mean=conditional_mean,
        integral=integral,
        epsilon=epsilon,
    )


def apply_aipcw_nuisance(
    nuisance: AIPCWNuisance,
    observed_time: np.ndarray,
    event: np.ndarray,
    complete_data_moment: np.ndarray,
) -> np.ndarray:
    """Evaluate an auxiliary-fold AIPCW nuisance fit on another fold."""

    observed_time = np.asarray(observed_time, dtype=float)
    event = np.asarray(event, dtype=float)
    complete_data_moment = np.asarray(complete_data_moment, dtype=float)
    if complete_data_moment.ndim != 2:
        raise ValueError("complete_data_moment must be a two-dimensional array")
    if not (observed_time.size == event.size == complete_data_moment.shape[0]):
        raise ValueError("time, event, and moment arrays must have the same number of rows")
    if complete_data_moment.shape[1] != nuisance.conditional_mean.shape[1]:
        raise ValueError("evaluation moments do not match the fitted nuisance dimension")

    # Evaluate the left-continuous censoring survival G(Y-).
    left_positions = np.searchsorted(nuisance.time_grid, observed_time, side="left")
    survival_at_time = np.ones(observed_time.size)
    has_previous_time = left_positions > 0
    previous_positions = np.clip(left_positions - 1, 0, nuisance.time_grid.size - 1)
    survival_at_time[has_previous_time] = nuisance.survival_after[
        previous_positions[has_previous_time]
    ]
    survival_at_time = np.clip(survival_at_time, nuisance.epsilon, 1.0)

    # Use nearest-boundary extrapolation outside the auxiliary time grid.
    xi_positions = np.clip(left_positions, 0, nuisance.time_grid.size - 1)
    xi_at_time = nuisance.conditional_mean[xi_positions]

    # Include augmentation jumps at auxiliary-fold times no greater than Y.
    integral_positions = np.searchsorted(
        nuisance.time_grid, observed_time, side="right"
    ) - 1
    integral_at_time = np.zeros_like(complete_data_moment)
    has_integral_term = integral_positions >= 0
    clipped_integral_positions = np.clip(
        integral_positions, 0, nuisance.time_grid.size - 1
    )
    integral_at_time[has_integral_term] = nuisance.integral[
        clipped_integral_positions[has_integral_term]
    ]

    return (
        event[:, None] / survival_at_time[:, None]
        * (complete_data_moment - xi_at_time)
        + nuisance.conditional_mean[0]
        + integral_at_time
    )
