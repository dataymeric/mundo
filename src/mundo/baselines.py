"""Classical reconstructions."""

import numpy as np
from scipy.interpolate import CubicSpline


def _lin_weights(n: int, L: int) -> np.ndarray:
    """Weights (n * L, n) of the linear interpolation through the n window centers."""
    centers = (np.arange(n) + 0.5) * L
    tt = np.arange(n * L) + 0.5
    return np.stack([np.interp(tt, centers, np.eye(n)[j]) for j in range(n)], axis=1)


def lin_disagg(means: np.ndarray, L: int) -> np.ndarray:
    """Linear disaggregation.

    Args:
        means: (n, C, *S) observed means of n consecutive windows.
        L: Window length in steps.

    Returns:
        (n * L, C, *S) fine record.
    """
    n = len(means)
    flat = means.reshape(n, -1)
    out = (_lin_weights(n, L) @ flat).reshape(n, L, -1)
    out += (flat - out.mean(axis=1))[:, None]
    return out.reshape(n * L, *means.shape[1:]).astype(means.dtype)


def denton(means: np.ndarray, L: int) -> np.ndarray:
    r"""Denton-Cholette (Denton, 1971; Cholette, 1984).

    Solves

    .. math::

        \min_x \sum_t (x_t - x_{t-1})^2 \quad \text{subject to} \quad A x = y,

    with :math:`A` the block mean of each window, through its KKT linear system.

    Args:
        means: (n, C, *S) observed means of n consecutive windows.
        L: Window length in steps.

    Returns:
        (n * L, C, *S) fine record.
    """
    n = len(means)
    T = n * L
    D = np.eye(T, k=1)[:-1] - np.eye(T)[:-1]
    A = np.kron(np.eye(n), np.full((1, L), 1.0 / L))
    K = np.block([[D.T @ D, A.T], [A, np.zeros((n, n))]])
    flat = means.reshape(n, -1)
    rhs = np.concatenate([np.zeros((T, flat.shape[1])), flat])
    return np.linalg.solve(K, rhs)[:T].reshape(T, *means.shape[1:])


def cdf_t(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    r"""CDF-t (Michelangeli et al., 2009).

    Each value :math:`v` at a position takes the truth quantile at its rank among the
    reconstructions at that position, :math:`F_S^{-1}(F_L(v))`, with :math:`F_L`
    the distribution of the reconstructions being corrected, and :math:`F_S` that of
    the training truth.

    Args:
        pred: (N, P) reconstructions to correct, at P positions.
        truth: (M, P) training truth at the same positions.

    Returns:
        (N, P) corrected reconstructions.
    """
    levels_pred = np.arange(1, len(pred) + 1) / len(pred)
    levels_truth = np.arange(1, len(truth) + 1) / len(truth)
    out = np.empty_like(pred)
    for p in range(pred.shape[1]):
        u = np.interp(pred[:, p], np.sort(pred[:, p]), levels_pred)  # F_L(v)
        out[:, p] = np.interp(u, levels_truth, np.sort(truth[:, p]))  # F_S^-1
    return out


def lin_interp(first: np.ndarray, last: np.ndarray, L: int) -> np.ndarray:
    """Linear interpolation.

    Args:
        first: (C, *S) frame at t = 0.
        last: (C, *S) frame at t = L.
        L: Window length in steps.

    Returns:
        (L, C, *S) fine record.
    """
    return np.stack([first * (1 - t / L) + last * (t / L) for t in range(L)])


def natural_spline(anchors: np.ndarray, ts: np.ndarray, L: int) -> np.ndarray:
    """Natural cubic spline interpolation.

    Args:
        anchors: (k, C, *S) observed frames: the window's two ends and, when present,
            the far ends of the neighboring windows.
        ts: (k,) their times, in steps from the window's first frame.
        L: Window length in steps.

    Returns:
        (L, C, *S) fine record.
    """
    order = np.argsort(ts)
    return CubicSpline(ts[order], anchors[order], axis=0, bc_type="natural")(
        np.arange(L)
    )
