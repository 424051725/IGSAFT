"""Diagnostics and the conventional Weibull AFT benchmark."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .basis import interaction_basis


@dataclass(frozen=True)
class FTestResult:
    statistic: float
    numerator_df: int
    denominator_df: int
    pvalue: float


@dataclass(frozen=True)
class AFTResult:
    beta_hat: float
    standard_error: float
    ci_lower: float
    ci_upper: float


def interaction_relevance_test(exposure: np.ndarray, z: np.ndarray) -> FTestResult:
    """HC1-robust joint F test for the centered interaction block."""

    interactions = interaction_basis(z, 2, center=z.mean(axis=0))
    design = sm.add_constant(np.column_stack([z, interactions]), has_constant="add")
    fit = sm.OLS(exposure, design).fit(cov_type="HC1")
    restriction = np.zeros((interactions.shape[1], design.shape[1]))
    restriction[:, 1 + z.shape[1] :] = np.eye(interactions.shape[1])
    test = fit.f_test(restriction)
    return FTestResult(
        statistic=float(test.fvalue),
        numerator_df=int(test.df_num),
        denominator_df=int(test.df_denom),
        pvalue=float(test.pvalue),
    )


def fit_weibull_aft(
    observed_log_time: np.ndarray,
    event: np.ndarray,
    exposure: np.ndarray,
) -> AFTResult:
    """Fit the paper's naive AFT benchmark, ignoring unmeasured confounding."""

    try:
        from lifelines import WeibullAFTFitter
    except ImportError as error:
        raise ImportError(
            "The optional AFT benchmark requires lifelines. Install all packages "
            "with: python -m pip install -r requirements.txt"
        ) from error

    data = pd.DataFrame(
        {
            "duration": np.exp(np.clip(observed_log_time, 0.001, 100.0)),
            "event": event,
            "exposure": exposure,
        }
    )
    model = WeibullAFTFitter()
    model.fit(data, duration_col="duration", event_col="event", formula="exposure")
    beta_hat = float(model.params_["lambda_"]["exposure"])
    standard_error = float(model.standard_errors_["lambda_"]["exposure"])
    return AFTResult(
        beta_hat=beta_hat,
        standard_error=standard_error,
        ci_lower=beta_hat - 1.96 * standard_error,
        ci_upper=beta_hat + 1.96 * standard_error,
    )
