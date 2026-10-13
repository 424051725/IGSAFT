from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize


@dataclass(frozen=True)
class AFTResult:
    beta_hat: float
    standard_error: float


def interaction_f_statistic(D, Z, interactions):
    X = np.column_stack((Z, interactions))
    fit = sm.OLS(D, sm.add_constant(X)).fit(cov_type="HC1")
    q, p = interactions.shape[1], Z.shape[1]
    restriction = np.zeros((q, X.shape[1] + 1))
    restriction[:, p + 1:] = np.eye(q)
    return float(np.asarray(fit.f_test(restriction).fvalue).squeeze())


def fit_weibull_aft(Y, delta, D):
    Y, delta, D = map(lambda x: np.asarray(x, float), (Y, delta, D))
    X = np.column_stack((np.ones(len(D)), D))
    observed = delta == 1
    initial = np.linalg.lstsq(X[observed], Y[observed], rcond=None)[0]
    sd = max(float(np.std(Y[observed] - X[observed] @ initial)), 1e-3)

    def objective(params):
        coef, log_shape = params[:2], params[2]
        shape = np.exp(log_shape)
        z = shape * (Y - X @ coef)
        hazard = np.exp(np.clip(z, -50.0, 50.0))
        loglik = np.sum(delta * (log_shape - Y + z) - hazard)
        grad_coef = shape * (X.T @ (delta - hazard))
        grad_shape = np.sum(hazard * z - delta * (1 + z))
        return -loglik, np.r_[grad_coef, grad_shape]

    fit = minimize(
        objective, np.r_[initial, np.clip(np.log(1 / sd), -3, 3)], jac=True,
        method="L-BFGS-B", bounds=((-100, 100), (-100, 100), (-5, 5)),
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    shape, coef = np.exp(fit.x[2]), fit.x[:2]
    z = shape * (Y - X @ coef)
    hazard = np.exp(np.clip(z, -50.0, 50.0))
    h_bb = shape**2 * (X.T @ (hazard[:, None] * X))
    h_ba = shape * (X.T @ (delta - hazard * (1 + z)))
    h_aa = np.sum(hazard * (z**2 + z) - delta * z)
    hessian = np.block(
        [[h_bb, h_ba[:, None]], [h_ba[None, :], np.array([[h_aa]])]]
    )
    variance = float(np.linalg.pinv(hessian)[1, 1])
    if variance <= 0 or not np.isfinite(variance):
        raise RuntimeError("invalid AFT variance")
    return AFTResult(float(coef[1]), float(np.sqrt(variance)))
