from dataclasses import dataclass
from itertools import combinations
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.linear_model import Lasso, LassoCV


EL_MARGIN = 0.05
ET_BOUND = 5.0


@dataclass(frozen=True)
class EstimationResult:
    method: str
    beta_hat: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    overid_statistic: float
    overid_df: int
    overid_pvalue: float
    selected_interactions: int


def pair_indices(p):
    return list(combinations(range(p), 2))


def interaction_matrix(Z, mean=None):
    Z = np.asarray(Z, float)
    centered = Z if mean is None else Z - np.asarray(mean)[None, :]
    return np.column_stack(
        [centered[:, j] * centered[:, k] for j, k in pair_indices(Z.shape[1])]
    )


def select_interactions(Z, D, method="lasso", active=None, alpha_scale=0.5):
    q = len(pair_indices(Z.shape[1]))
    if method == "all":
        return np.arange(q)
    if method == "truth":
        if active is None:
            raise ValueError("truth selection requires active interaction indices")
        return np.asarray(active, int)
    if method != "lasso":
        raise ValueError("selection must be lasso, truth, or all")
    V = np.column_stack((np.ones(len(Z)), Z))
    H = interaction_matrix(Z)
    rD = D - V @ (np.linalg.pinv(V) @ D)
    rH = H - V @ (np.linalg.pinv(V) @ H)
    initial = np.linalg.pinv(rH) @ rD
    weights = 1 / (np.abs(initial) + 1e-6)
    design = rH / weights[None, :]
    jobs = int(os.environ.get("IGSAFT_LASSO_JOBS", "1"))
    cv = LassoCV(cv=5, alphas=100, fit_intercept=False, random_state=0, n_jobs=jobs)
    cv.fit(design, rD)
    fit = Lasso(
        alpha=float(cv.alpha_) * alpha_scale,
        fit_intercept=False,
        max_iter=10_000,
    )
    fit.fit(design, rD)
    selected = np.flatnonzero(np.abs(fit.coef_ / weights) > 1e-6)
    if not len(selected):
        raise RuntimeError("adaptive lasso selected no interactions")
    return selected


def three_way_split(n, seed, selection_fraction=0.20):
    order = np.random.default_rng(seed).permutation(n)
    n_select = int(round(selection_fraction * n))
    remaining = order[n_select:]
    cut = len(remaining) // 2
    return order[:n_select], remaining[:cut], remaining[cut:]


class _CoxPrediction:
    def __init__(self, model, columns, shift):
        self.model, self.columns, self.shift = model, columns, shift
        baseline = model.baseline_cumulative_hazard_.iloc[:, 0]
        self.times = baseline.index.to_numpy(float) - shift
        self.cumulative_hazard = baseline.to_numpy(float)

    def risk(self, X):
        return self.model.predict_partial_hazard(
            pd.DataFrame(X, columns=self.columns)
        ).to_numpy(float)

    def baseline_at(self, times):
        positions = np.searchsorted(self.times, np.asarray(times), side="right") - 1
        result = np.zeros(np.shape(times), float)
        valid = positions >= 0
        result[valid] = self.cumulative_hazard[positions[valid]]
        return result


def _fit_cox(Y, event, Z, D):
    columns = [f"Z{j}" for j in range(Z.shape[1])] + ["D"]
    shift = max(0.0, 1.0 - float(np.min(Y)))
    frame = pd.DataFrame(np.column_stack((Z, D)), columns=columns)
    frame["time"] = np.asarray(Y) + shift
    frame["event"] = np.asarray(event, int)
    model = CoxPHFitter(baseline_estimation_method="breslow")
    model.fit(
        frame, duration_col="time", event_col="event",
        formula=" + ".join(columns), show_progress=False,
    )
    return _CoxPrediction(model, columns, shift)


def _tail_quantities(failure, censoring, Y, X, grid, eps=1e-8):
    grid = np.asarray(grid, float)
    if not len(grid):
        raise RuntimeError("nuisance training fold has no observed failures")
    risk_t, risk_c = failure.risk(X), censoring.risk(X)
    Ht, Hc = failure.baseline_at(grid), censoring.baseline_at(grid)
    G_y = np.clip(np.exp(-risk_c * censoring.baseline_at(Y)), eps, 1.0)
    mu_y, mu0, A = (np.empty(len(Y)) for _ in range(3))
    dt, Ht_left = np.diff(grid), np.r_[0.0, Ht[:-1]]
    for start in range(0, len(Y), 256):
        stop = min(start + 256, len(Y))
        rt, rc, yy = risk_t[start:stop], risk_c[start:stop], Y[start:stop]
        residual = np.zeros((stop - start, len(grid)))
        for j in range(len(grid) - 2, -1, -1):
            jump = np.exp(-rt * (Ht[j] - Ht_left[j]))
            residual[:, j] = jump * (dt[j] + residual[:, j + 1])
        mu_grid = grid[None, :] + residual
        mu0_chunk = mu_grid[:, 0]
        next_position = np.searchsorted(grid, yy, side="right")
        mu_chunk = yy.copy()
        valid = next_position < len(grid)
        rows = np.flatnonzero(valid)
        cols = next_position[valid]
        mu_chunk[valid] += grid[cols] - yy[valid] + residual[rows, cols]
        increments = np.empty_like(mu_grid)
        increments[:, 0] = mu_grid[:, 0] - mu0_chunk
        if len(grid) > 1:
            increments[:, 1:] = np.diff(mu_grid, axis=1)
        G_grid = np.clip(np.exp(-rc[:, None] * Hc[None, :]), eps, 1.0)
        upto = np.searchsorted(grid, yy, side="right")
        sums = np.cumsum(increments / G_grid, axis=1)
        A_chunk = np.zeros(stop - start)
        valid = upto > 0
        rows = np.flatnonzero(valid)
        A_chunk[valid] = sums[rows, upto[valid] - 1]
        mu_y[start:stop], mu0[start:stop], A[start:stop] = mu_chunk, mu0_chunk, A_chunk
    return G_y, mu_y, mu0, A


def _fit_fold(Y, D, Z, delta, train, evaluate, selected):
    mean = Z[train].mean(axis=0)
    I = interaction_matrix(Z[evaluate], mean)[:, selected]
    V_train = np.column_stack((np.ones(len(train)), Z[train]))
    V_eval = np.column_stack((np.ones(len(evaluate)), Z[evaluate]))
    theta = np.linalg.pinv(V_train) @ Y[train]
    omega = np.linalg.pinv(V_train) @ D[train]
    censoring = _fit_cox(Y[train], 1 - delta[train], Z[train], D[train])
    failure = _fit_cox(Y[train], delta[train], Z[train], D[train])
    grid = np.unique(Y[train][delta[train] == 1])
    X_eval = np.column_stack((Z[evaluate], D[evaluate]))
    G, mu, mu0, A = _tail_quantities(failure, censoring, Y[evaluate], X_eval, grid)
    return {
        "I": I,
        "G": G,
        "mu": mu,
        "mu0": mu0,
        "A": A,
        "Vtheta": V_eval @ theta,
        "Vomega": V_eval @ omega,
        "D": D[evaluate],
        "Y": Y[evaluate],
        "delta": delta[evaluate],
    }


def prepare_moments(Y, D, Z, delta, split_seed=1, selection="lasso",
                    active=None, alpha_scale=0.5):
    Y, D, Z, delta = map(np.asarray, (Y, D, Z, delta))
    selection_idx, fold1, fold2 = three_way_split(len(Y), split_seed)
    selected = select_interactions(
        Z[selection_idx], D[selection_idx], selection, active, alpha_scale
    )
    artifacts = (
        _fit_fold(Y, D, Z, delta, fold2, fold1, selected),
        _fit_fold(Y, D, Z, delta, fold1, fold2, selected),
    )
    pooled = {
        key: np.concatenate([artifact[key] for artifact in artifacts], axis=0)
        for key in artifacts[0]
    }
    pooled["selected"] = selected
    pooled["inference_indices"] = np.concatenate((fold1, fold2))
    return pooled


def _moments(beta, data, eps=1e-8):
    common = (
        data["delta"] / np.clip(data["G"], eps, None) * (data["Y"] - data["mu"])
        + data["mu0"] - data["Vtheta"]
        - beta * (data["D"] - data["Vomega"]) + data["A"]
    )
    return data["I"] * common[:, None], -data["I"] * (
        data["D"] - data["Vomega"]
    )[:, None]


def _rho(name, values):
    if name == "CUE":
        return -(values + 0.5 * values**2)
    if name == "EL":
        return np.log(np.clip(1 - values, EL_MARGIN, None))
    if name == "ET":
        return 1 - np.exp(np.clip(values, -ET_BOUND, ET_BOUND))
    raise ValueError("method must be CUE, EL, or ET")


def _rho_derivatives(name, values):
    if name == "CUE":
        return -(1 + values), -np.ones_like(values)
    if name == "EL":
        denominator = np.clip(1 - values, EL_MARGIN, None)
        return -1 / denominator, -1 / denominator**2
    exponential = np.exp(np.clip(values, -ET_BOUND, ET_BOUND))
    return -exponential, -exponential


def _objective(beta, data, method):
    moments, _ = _moments(beta, data)
    omega = moments.T @ moments / len(moments)
    multiplier = -np.linalg.pinv(omega) @ moments.mean(axis=0)
    value = _rho(method, moments @ multiplier)
    return float(np.mean(value)) if np.isfinite(value).all() else np.inf


def _standard_error(moments, derivative, method):
    n = len(moments)
    omega = moments.T @ moments / n
    inverse = np.linalg.pinv(omega)
    omega_1 = (derivative.T @ moments + moments.T @ derivative) / n
    omega_2 = 2 * derivative.T @ derivative / n
    inverse_1 = -inverse @ omega_1 @ inverse
    inverse_2 = -2 * inverse_1 @ omega_1 @ inverse - inverse @ omega_2 @ inverse
    mean, derivative_mean = moments.mean(axis=0), derivative.mean(axis=0)
    multiplier = -mean @ inverse
    multiplier_1 = -derivative_mean @ inverse - mean @ inverse_1
    multiplier_2 = -2 * derivative_mean @ inverse_1 - mean @ inverse_2
    index = moments @ multiplier
    rho_1, rho_2 = _rho_derivatives(method, index)
    score_1 = derivative @ multiplier + moments @ multiplier_1
    score_2 = 2 * derivative @ multiplier_1 + moments @ multiplier_2
    curvature = np.mean(rho_2 * score_1**2 + rho_1 * score_2)
    weights = rho_1 / rho_1.sum()
    loading = inverse @ (derivative.T @ weights)
    influence = -(moments @ loading) / curvature
    return float(np.sqrt(np.sum(influence**2)) / n)


def fit_prepared(data, method="CUE", beta_bounds=(-100, 100)):
    method = method.upper()
    fit = minimize_scalar(
        lambda beta: _objective(beta, data, method),
        bounds=beta_bounds, method="bounded", options={"xatol": 1e-3},
    )
    beta, objective = float(fit.x), float(fit.fun)
    moments, derivative = _moments(beta, data)
    se = _standard_error(moments, derivative, method)
    statistic = abs(2 * len(moments) * objective)
    df = moments.shape[1] - 1
    return EstimationResult(
        method, beta, se, beta - 1.96 * se, beta + 1.96 * se,
        statistic, df, float(chi2.sf(statistic, df)) if df > 0 else np.nan,
        moments.shape[1],
    )


def fit_igsaft(Y, D, Z, delta, methods=("CUE", "EL", "ET"), **kwargs):
    prepared = prepare_moments(Y, D, Z, delta, **kwargs)
    return [fit_prepared(prepared, method) for method in methods], prepared

