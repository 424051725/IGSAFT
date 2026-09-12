"""Interaction bases and small linear-algebra helpers."""

from itertools import combinations

import numpy as np


def interaction_indices(p: int, order: int = 2) -> list[tuple[int, ...]]:
    return list(combinations(range(p), order))


def interaction_basis(
    z: np.ndarray,
    order: int = 2,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Return all distinct products of ``order`` columns of ``z``."""

    z_used = z if center is None else z - np.asarray(center).reshape(1, -1)
    indices = interaction_indices(z.shape[1], order)
    basis = np.ones((z.shape[0], len(indices)))
    for column, index_tuple in enumerate(indices):
        for index in index_tuple:
            basis[:, column] *= z_used[:, index]
    return basis


def lower_order_design(z: np.ndarray, order: int = 2) -> np.ndarray:
    """Design V_k containing an intercept and terms below order k."""

    parts = [np.ones((z.shape[0], 1))]
    for lower_order in range(1, order):
        parts.append(interaction_basis(z, lower_order))
    return np.column_stack(parts)


def residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residualize one or more columns without forming an n-by-n projection."""

    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ coefficients

