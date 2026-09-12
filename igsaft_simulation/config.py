"""User-facing configuration for one simulation run."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DGPConfig:
    """Parameters of the data-generating process described in the paper."""

    n: int = 10_000
    p: int = 10
    beta_true: float = 1.0
    scenario: int = 1
    censor_rate: float = 0.20
    interaction_scale: float = 4.0
    active_interaction_fraction: float = 1.0
    censoring_width: float = 1_000.0
    seed: int = 240

    def validate(self) -> None:
        if self.n < 10:
            raise ValueError("n must be at least 10")
        if self.p < 2:
            raise ValueError("p must be at least 2")
        if self.scenario not in {1, 2, 3, 4}:
            raise ValueError("scenario must be one of 1, 2, 3, or 4")
        if not 0.0 < self.censor_rate < 1.0:
            raise ValueError("censor_rate must lie strictly between 0 and 1")
        if not 0.0 < self.active_interaction_fraction <= 1.0:
            raise ValueError("active_interaction_fraction must lie in (0, 1]")
        if self.censoring_width <= 0:
            raise ValueError("censoring_width must be positive")


@dataclass(frozen=True)
class EstimatorConfig:
    """Tuning parameters for interaction selection and GEL estimation."""

    interaction_order: int = 2
    selection: str = "lasso"
    beta_lower: float = -5.0
    beta_upper: float = 5.0
    optimizer_tolerance: float = 1e-3
    numerical_epsilon: float = 1e-8
    lasso_folds: int = 5
    lasso_jobs: int = 1
    split_seed: int = 1

    def validate(self) -> None:
        if self.interaction_order != 2:
            raise ValueError("This public example implements the paper's second-order moments (q=2)")
        if self.selection not in {"lasso", "truth", "all"}:
            raise ValueError("selection must be 'lasso', 'truth', or 'all'")
        if self.beta_lower >= self.beta_upper:
            raise ValueError("beta_lower must be smaller than beta_upper")
        if self.lasso_folds < 2:
            raise ValueError("lasso_folds must be at least 2")

