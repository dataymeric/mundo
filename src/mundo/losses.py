"""Training losses: proper scores of an ensemble of M draws against the truth."""

from collections.abc import Callable

import torch

VG_EPS = 1e-4  # smooths |dx|^p at 0, whose gradient is unbounded for p < 1


def _flat(t):
    return t.reshape(t.shape[0], -1)


def _pairs(draws):
    m = len(draws)
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def _rank(x: torch.Tensor, n_spatial: int | None) -> int:
    if n_spatial is not None:
        return n_spatial
    return x.dim() - 3 if x.dim() >= 4 else 1


def energy_score(draws: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    r"""Fair energy score of M draws.

    .. math::

        \mathrm{ES} = \frac{1}{M} \sum_{m} \lVert u_m - x \rVert_2
        - \frac{1}{2 M (M - 1)} \sum_{m \neq m'} \lVert u_m - u_{m'} \rVert_2,

    then averaged.

    Args:
        draws: The :math:`u_m`, M >= 2 tensors shaped like `x`.
        x: (B, ...) truth.

    Returns:
        Scalar loss.
    """
    assert len(draws) >= 2, "the energy score needs at least two draws"
    acc = sum((_flat(g) - _flat(x)).norm(dim=-1) for g in draws) / len(draws)
    pairs = _pairs(draws)
    spr = sum((_flat(draws[i]) - _flat(draws[j])).norm(dim=-1) for i, j in pairs)
    return (acc - 0.5 * spr / len(pairs)).mean()


def _increments(a: torch.Tensor, dim: int, h: int, p: float) -> torch.Tensor:
    lo, hi = [slice(None)] * a.dim(), [slice(None)] * a.dim()
    lo[dim], hi[dim] = slice(None, -h), slice(h, None)
    d = a[tuple(hi)] - a[tuple(lo)]
    return (d * d + VG_EPS).pow(0.5 * p)


def variogram_score(
    draws: list[torch.Tensor],
    x: torch.Tensor,
    p: float = 0.5,
    lags=(1, 2, 4),
    n_spatial: int | None = None,
    clamp: bool = False,
    reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    r"""Variogram score of M draws (Scheuerer and Hamill, 2015).

    Let :math:`\gamma_h^a(u) = \lvert u_{a+h} - u_a \rvert^p` be the order-:math:`p`
    difference of a field :math:`u` at cell :math:`a` and lag :math:`h`, smoothed as
    :math:`(\lvert u_{a+h} - u_a \rvert^2 + \varepsilon)^{p/2}` with
    :math:`\varepsilon` = `VG_EPS`. The score is

    .. math::

        \mathrm{VS} = \frac{1}{M (M - 1)} \sum_{m \neq m'}
        \left(\gamma_h^a(x) - \gamma_h^a(u_m)\right)
        \left(\gamma_h^a(x) - \gamma_h^a(u_{m'})\right),

    then averaged.

    Args:
        draws: The :math:`u_m`, M >= 2 tensors shaped like `x`.
        x: Truth.
        p: Variogram order.
        lags: Lags :math:`h`, equally weighted.
        n_spatial: Number of spatial axes, inferred from `x` when `None`.
        clamp: Floor each axis-and-lag term at 0.
        reduce: Cross-rank mean. Used only when ``clamp=True``.

    Returns:
        Scalar loss.
    """
    pairs = _pairs(draws)
    tot, n = 0.0, 0
    for dim in range(-1, -_rank(x, n_spatial) - 1, -1):
        for h in lags:
            if x.shape[dim] <= h:
                continue
            d_t = _increments(x, dim, h, p)
            d_g = [_increments(g, dim, h, p) for g in draws]
            t = sum(((d_t - d_g[i]) * (d_t - d_g[j])).mean() for i, j in pairs)
            if clamp:
                t = (t if reduce is None else reduce(t)).clamp_min(0.0)
            tot = tot + t
            n += len(pairs)
    return tot / max(n, 1)


def spectral_crps(
    draws: list[torch.Tensor], x: torch.Tensor, n_spatial: int | None = None
) -> torch.Tensor:
    r"""Spectral CRPS of M draws (Bonev et al., 2025).

    With :math:`s(\cdot)` the real or imaginary part of one coefficient of the
    orthonormal FFT over the spatial axes, the fair CRPS is

    .. math::

        \mathrm{CRPS}_{\mathrm{spectral}} =
        \frac{1}{M} \sum_{m} \lvert s(u_m) - s(x) \rvert
        - \frac{1}{2 M (M - 1)} \sum_{m \neq m'} \lvert s(u_m) - s(u_{m'}) \rvert,

    then averaged.

    Args:
        draws: The :math:`u_m`, M >= 2 tensors shaped like `x`.
        x: Truth.
        n_spatial: Number of spatial axes, inferred from `x` when `None`.

    Returns:
        Scalar loss.
    """
    r = _rank(x, n_spatial)
    dims = tuple(range(x.dim() - r, x.dim()))

    def s(t):
        f = torch.fft.fftn(t.float(), dim=dims, norm="ortho")
        return torch.cat([f.real, f.imag], dim=-1)

    S, X = [s(g) for g in draws], s(x)
    pairs = _pairs(S)
    acc = sum((g - X).abs() for g in S) / len(S)
    spr = sum((S[i] - S[j]).abs() for i, j in pairs) / len(pairs)
    return (acc - 0.5 * spr).mean()


def resolve_aux_weight(w: float, main: torch.Tensor, aux: torch.Tensor) -> float:
    r"""Weight of an auxiliary loss term.

    Args:
        w: Requested weight. Negative means relative to the batch, see Returns.
        main: Main loss on the batch.
        aux: Auxiliary loss on the same batch.

    Returns:
        `w` when ``w >= 0``, otherwise :math:`\lvert w \rvert \cdot \lvert
        \mathrm{main} \rvert / \lvert \mathrm{aux} \rvert`, so that ``w = -1`` makes the
        weighted auxiliary term as large as the main one.
    """
    if w >= 0:
        return float(w)
    return float(-w * main.detach().abs() / (aux.detach().abs() + 1e-12))
