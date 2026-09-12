"""Interaction selection, cross-fitted AIPCW moments, and GEL estimation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.linear_model import LassoCV

from .basis import interaction_basis, lower_order_design, residualize
from .censoring import apply_aipcw_nuisance, fit_aipcw_nuisance
from .config import EstimatorConfig

GELMethod = Literal["CUE", "EL", "ET"]


@dataclass(frozen=True)
class GELResult:
    method: str
    beta_hat: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    objective: float
    overid_statistic: float
    overid_df: int
    overid_pvalue: float
    moment_count: int
    selected_interaction_count: int


@dataclass(frozen=True)
class PreparedMoments:
    """Cross-fitted moment components shared by all three GEL criteria."""

    outcome_component: np.ndarray
    exposure_component: np.ndarray
    selected_interaction_count: int


def select_interactions(
    z: np.ndarray,
    exposure: np.ndarray,
    config: EstimatorConfig,
    true_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Select exposure-relevant pairwise interactions by adaptive LASSO."""

    all_interactions = interaction_basis(z, 2)
    if config.selection == "all":
        return np.arange(all_interactions.shape[1])
    if config.selection == "truth":
        if true_indices is None:
            raise ValueError("truth selection requires true interaction indices")
        return np.asarray(true_indices, dtype=int)

    nuisance_design = np.column_stack([np.ones(z.shape[0]), z])
    residual_exposure = residualize(exposure, nuisance_design)
    residual_interactions = residualize(all_interactions, nuisance_design)

    initial = np.linalg.lstsq(residual_interactions, residual_exposure, rcond=None)[0]
    adaptive_weights = 1.0 / (np.abs(initial) + 1e-6)
    adaptive_design = residual_interactions / adaptive_weights
    lasso = LassoCV(
        cv=config.lasso_folds,
        alphas=100,
        fit_intercept=False,
        random_state=0,
        n_jobs=config.lasso_jobs,
    )
    lasso.fit(adaptive_design, residual_exposure)
    coefficients = lasso.coef_ / adaptive_weights
    selected = np.flatnonzero(np.abs(coefficients) > 1e-6)
    if selected.size == 0:
        raise RuntimeError(
            "Adaptive LASSO selected no interactions. Try another seed, a larger n, "
            "or --selection all for a diagnostic run."
        )
    return selected


def _one_fold_complete_data_components(
    observed_train: np.ndarray,
    exposure_train: np.ndarray,
    z_train: np.ndarray,
    observed_test: np.ndarray,
    exposure_test: np.ndarray,
    z_test: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct train and test complete-data components from train nuisances."""

    design_train = lower_order_design(z_train, 2)
    outcome_coef = np.linalg.lstsq(design_train, observed_train, rcond=None)[0]
    exposure_coef = np.linalg.lstsq(design_train, exposure_train, rcond=None)[0]
    design_test = lower_order_design(z_test, 2)

    outcome_residual_train = observed_train - design_train @ outcome_coef
    exposure_residual_train = exposure_train - design_train @ exposure_coef
    outcome_residual_test = observed_test - design_test @ outcome_coef
    exposure_residual_test = exposure_test - design_test @ exposure_coef
    h_train = interaction_basis(z_train, 2, center=z_train.mean(axis=0))[:, selected]
    h_test = interaction_basis(z_test, 2, center=z_train.mean(axis=0))[:, selected]
    return (
        h_train * outcome_residual_train[:, None],
        h_train * exposure_residual_train[:, None],
        h_test * outcome_residual_test[:, None],
        h_test * exposure_residual_test[:, None],
    )


def _cross_fitted_components(
    observed_time: np.ndarray,
    exposure: np.ndarray,
    z: np.ndarray,
    event: np.ndarray,
    selected: np.ndarray,
    config: EstimatorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.split_seed)
    shuffled = rng.permutation(observed_time.size)
    fold_a, fold_b = np.array_split(shuffled, 2)
    transformed_outcome: list[np.ndarray] = []
    transformed_exposure: list[np.ndarray] = []

    for test, train in ((fold_a, fold_b), (fold_b, fold_a)):
        (
            outcome_train,
            exposure_train,
            outcome_test,
            exposure_test,
        ) = _one_fold_complete_data_components(
            observed_time[train], exposure[train], z[train],
            observed_time[test], exposure[test], z[test], selected,
        )
        outcome_nuisance = fit_aipcw_nuisance(
            observed_time[train], event[train], outcome_train, config.numerical_epsilon
        )
        exposure_nuisance = fit_aipcw_nuisance(
            observed_time[train], event[train], exposure_train, config.numerical_epsilon
        )
        transformed_outcome.append(
            apply_aipcw_nuisance(
                outcome_nuisance, observed_time[test], event[test], outcome_test
            )
        )
        transformed_exposure.append(
            apply_aipcw_nuisance(
                exposure_nuisance, observed_time[test], event[test], exposure_test
            )
        )
    return np.vstack(transformed_outcome), np.vstack(transformed_exposure)


def _rho(name: GELMethod, values: np.ndarray) -> np.ndarray:
    if name == "CUE":
        return -(values + 0.5 * values**2)
    if name == "EL":
        return np.log(np.clip(1.0 - values, 5e-3, None))
    return 1.0 - np.exp(np.clip(values, -30.0, 30.0))


def _rho_derivatives(name: GELMethod, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if name == "CUE":
        return -(1.0 + values), -np.ones_like(values)
    if name == "EL":
        denominator = np.clip(1.0 - values, 5e-3, None)
        return -1.0 / denominator, -1.0 / denominator**2
    exponential = np.exp(np.clip(values, -30.0, 30.0))
    return -exponential, -exponential


def _objective(beta: float, outcome: np.ndarray, exposure: np.ndarray, method: GELMethod) -> float:
    moments = outcome - beta * exposure
    mean_moment = moments.mean(axis=0)
    covariance = moments.T @ moments / moments.shape[0]
    multiplier = -np.linalg.pinv(covariance) @ mean_moment
    values = moments @ multiplier
    objective = _rho(method, values)
    return float(np.mean(objective)) if np.all(np.isfinite(objective)) else np.inf


def _standard_error(moments: np.ndarray, derivative: np.ndarray, method: GELMethod) -> float:
    n = moments.shape[0]
    covariance = moments.T @ moments / n
    covariance_inverse = np.linalg.pinv(covariance)
    covariance_d1 = (derivative.T @ moments + moments.T @ derivative) / n
    covariance_d2 = 2.0 * derivative.T @ derivative / n
    inverse_d1 = -covariance_inverse @ covariance_d1 @ covariance_inverse
    inverse_d2 = (
        -2.0 * inverse_d1 @ covariance_d1 @ covariance_inverse
        - covariance_inverse @ covariance_d2 @ covariance_inverse
    )

    mean_moment = moments.mean(axis=0)
    mean_derivative = derivative.mean(axis=0)
    multiplier = -mean_moment @ covariance_inverse
    multiplier_d1 = -mean_derivative @ covariance_inverse - mean_moment @ inverse_d1
    multiplier_d2 = -2.0 * mean_derivative @ inverse_d1 - mean_moment @ inverse_d2

    index = moments @ multiplier
    rho_prime, rho_double_prime = _rho_derivatives(method, index)
    index_d1 = derivative @ multiplier + moments @ multiplier_d1
    index_d2 = 2.0 * derivative @ multiplier_d1 + moments @ multiplier_d2
    hessian = np.mean(rho_double_prime * index_d1**2 + rho_prime * index_d2)

    weights = rho_prime / rho_prime.sum()
    gradient = derivative.T @ weights
    variance = (gradient.T @ covariance_inverse @ gradient) / hessian**2
    return float(np.sqrt(max(float(variance), 0.0) / n))


def prepare_igsaft_moments(
    observed_time: np.ndarray,
    exposure: np.ndarray,
    z: np.ndarray,
    event: np.ndarray,
    config: EstimatorConfig,
    true_interactions: np.ndarray | None = None,
) -> PreparedMoments:
    """Select interactions and build components once for reuse across methods."""

    config.validate()
    selected = select_interactions(z, exposure, config, true_interactions)
    outcome_component, exposure_component = _cross_fitted_components(
        observed_time, exposure, z, event, selected, config
    )
    return PreparedMoments(outcome_component, exposure_component, selected.size)


def fit_prepared_igsaft(
    prepared: PreparedMoments,
    sample_size: int,
    method: GELMethod,
    config: EstimatorConfig,
) -> GELResult:
    """Fit one GEL criterion to already-prepared cross-fitted moments."""

    outcome_component = prepared.outcome_component
    exposure_component = prepared.exposure_component
    result = minimize_scalar(
        _objective,
        args=(outcome_component, exposure_component, method),
        bounds=(config.beta_lower, config.beta_upper),
        method="bounded",
        options={"xatol": config.optimizer_tolerance},
    )
    beta_hat = float(result.x)
    moments = outcome_component - beta_hat * exposure_component
    derivative = -exposure_component
    standard_error = _standard_error(moments, derivative, method)
    overid_df = int(moments.shape[1] - 1)
    overid_statistic = abs(2.0 * sample_size * float(result.fun))
    overid_pvalue = float(chi2.sf(overid_statistic, overid_df)) if overid_df > 0 else np.nan
    return GELResult(
        method=method,
        beta_hat=beta_hat,
        standard_error=standard_error,
        ci_lower=beta_hat - 1.96 * standard_error,
        ci_upper=beta_hat + 1.96 * standard_error,
        objective=float(result.fun),
        overid_statistic=overid_statistic,
        overid_df=overid_df,
        overid_pvalue=overid_pvalue,
        moment_count=moments.shape[1],
        selected_interaction_count=prepared.selected_interaction_count,
    )


def fit_igsaft(
    observed_time: np.ndarray,
    exposure: np.ndarray,
    z: np.ndarray,
    event: np.ndarray,
    method: GELMethod,
    config: EstimatorConfig,
    true_interactions: np.ndarray | None = None,
) -> GELResult:
    """Convenience wrapper for fitting a single GEL variant."""

    prepared = prepare_igsaft_moments(
        observed_time, exposure, z, event, config, true_interactions
    )
    return fit_prepared_igsaft(prepared, observed_time.size, method, config)
