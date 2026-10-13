"""Simulation code for the Cox-based iGSAFT estimator."""

from .dgp import DGPConfig, simulate_data
from .estimator import EstimationResult, fit_igsaft

__all__ = ["DGPConfig", "EstimationResult", "fit_igsaft", "simulate_data"]

