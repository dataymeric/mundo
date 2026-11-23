"""Evaluation metrics, on an ensemble of M members against the truth."""

import math
from itertools import pairwise

import numpy as np
import torch


def crps(ens: torch.Tensor, x: torch.Tensor) -> float:
    r"""Fair CRPS of an ensemble (Ferro, 2014).

    .. math::

        \mathrm{CRPS} = \frac{1}{M} \sum_{m} \lvert u_m - x \rvert
        - \frac{1}{2 M (M - 1)} \sum_{m \neq m'} \lvert u_m - u_{m'} \rvert,

    then averaged over time steps, variables and grid points. With one member, it is
    the mean absolute error. The pair sum is taken in :math:`O(M \log M)` from the
    sorted members, :math:`\sum_{m \neq m'} \lvert u_m - u_{m'} \rvert = 2
    \sum_k u_{(k)} (2k - M + 1)`, with :math:`k` from 0.

    Args:
        ens: (M, ...) members :math:`u_1, \dots, u_M`.
        x: (...) truth.

    Returns:
        Scalar score.
    """
    M = ens.shape[0]
    ens, x = ens.float(), x.float()  # the weighted sum overflows in half precision
    a = (ens - x[None]).abs().mean()
    if M == 1:
        return float(a)
    xs = ens.sort(dim=0).values
    w = (2 * torch.arange(M, device=ens.device) - M + 1).to(xs.dtype)
    gini = (xs * w.view(M, *([1] * (ens.dim() - 1)))).sum(0).mean()
    return float(a - gini / (M * (M - 1)))


def coverage(ens: torch.Tensor, x: torch.Tensor, levels) -> dict:
    r"""Realized coverage of the ensemble's central intervals.

    .. math::

        \widehat{c}(\alpha) = \Bigl\langle \mathbf{1}\bigl\{
        \widehat{q}_{(1-\alpha)/2}(u_{1:M}) \le x \le
        \widehat{q}_{(1+\alpha)/2}(u_{1:M}) \bigr\} \Bigr\rangle,

    with :math:`\widehat{q}_p` the member quantile at the Weibull plotting position
    :math:`p (M + 1)`, interpolated linearly (Greenbury et al., 2026). A calibrated
    ensemble gives :math:`\widehat{c}(\alpha) \approx \alpha`.

    Args:
        ens: (M, ...) members.
        x: (...) truth.
        levels: Nominal levels :math:`\alpha`, in (0, 1).

    Returns:
        :math:`\widehat{c}(\alpha)` for each level, e.g., ``cov90`` for
        :math:`\alpha = 0.9`.
    """
    s, _ = torch.sort(ens, dim=0)
    M = s.shape[0]

    def q(p):
        pos = min(max(p * (M + 1) - 1, 0.0), float(M - 1))
        lo = int(pos)
        w = pos - lo
        return s[lo] * (1 - w) + s[min(lo + 1, M - 1)] * w

    out = {}
    for lv in levels:
        inside = (x >= q((1 - lv) / 2)) & (x <= q(1 - (1 - lv) / 2))
        out[f"cov{int(lv * 100)}"] = float(inside.float().mean())
    return out


def cov_calibration(cov: dict, levels) -> dict:
    r"""Coverage error over the levels :math:`\mathcal{A}`.

    .. math::

        \mathrm{cov.\ MAE} = \frac{1}{\lvert \mathcal{A} \rvert}
        \sum_{\alpha \in \mathcal{A}} \lvert \widehat{c}(\alpha) - \alpha \rvert,

        \mathrm{cov.\ bias} = \frac{1}{\lvert \mathcal{A} \rvert}
        \sum_{\alpha \in \mathcal{A}} \bigl( \widehat{c}(\alpha) - \alpha \bigr),

    returned as ``cov_mae`` and ``cov_bias``. The bias is negative for an overconfident
    ensemble.

    Args:
        cov: From `coverage`.
        levels: Nominal levels :math:`\alpha`.
    """
    d = [
        cov[f"cov{int(lv * 100)}"] - lv for lv in levels if f"cov{int(lv * 100)}" in cov
    ]
    if not d:
        return {}
    return {"cov_mae": float(np.mean(np.abs(d))), "cov_bias": float(np.mean(d))}


class SpreadSkill:
    r"""Spread-skill ratio (Rasp et al., 2024).

    .. math::

        \mathrm{SSR} = \sqrt{\frac{M + 1}{M}} \,
        \frac{\sqrt{\langle \operatorname{var}(u_{1:M}) \rangle}}
        {\sqrt{\langle (\bar u - x)^2 \rangle}}

    with the unbiased member variance, the member mean :math:`\bar u`, and
    :math:`\langle \cdot \rangle` the mean over all entries of every window. The
    factor makes a calibrated finite ensemble score 1.
    """

    def __init__(self, n_spatial: int | None = None):
        self.sv = self.nv = self.se = self.ne = 0.0
        self.n_spatial = n_spatial
        self.bv, self.be = torch.zeros(4), torch.zeros(4)

    def update(self, ens, x):
        """Add one window: members (M, ...) and truth (...)."""
        if ens.shape[0] < 2:
            return
        var, se = ens.var(dim=0), (ens.mean(dim=0) - x) ** 2
        self.sv += float(var.sum())
        self.nv += var.numel()
        self.se += float(se.sum())
        self.ne += se.numel()
        if self.n_spatial is not None:
            mean = ens.mean(dim=0)
            self.bv += spectral_bands(ens - mean, self.n_spatial).cpu() / (
                ens.shape[0] - 1
            )
            self.be += spectral_bands(mean - x, self.n_spatial).cpu()

    def summary(self, M: int) -> dict:
        nan = float("nan")
        bands = {k: nan for k in SSR_KEYS} if self.n_spatial is not None else {}
        if M < 2 or not self.nv:
            return {"spread_skill": nan, "spread": nan, "rmse": nan, **bands}
        spread, rmse = math.sqrt(self.sv / self.nv), math.sqrt(self.se / self.ne)
        for k, v, e in zip(SSR_KEYS, self.bv, self.be):
            if k in bands and float(e) > 0:
                bands[k] = math.sqrt((M + 1) / M * float(v) / float(e))
        return {
            "spread_skill": math.sqrt((M + 1) / M) * spread / (rmse + 1e-9),
            "spread": spread,
            "rmse": rmse,
            **bands,
        }


def consistency(op, ens: torch.Tensor, y: torch.Tensor) -> float:
    r"""Largest observation error of the members.

    .. math::

        \max_m \lVert A u_m - y \rVert_\infty.
    """
    return float((op.apply(ens) - y[None]).abs().max())


class VariogramTier:
    r"""Variogram ratios and scores, accumulated over windows.

    With :math:`\gamma_h(u)` the mean of :math:`\lvert u_{a+h} - u_a \rvert^p` over
    cells :math:`a` along one axis, each ratio is

    .. math::

        \frac{\sum_{\text{windows}} \frac{1}{M} \sum_m \gamma_h(u_m)}
        {\sum_{\text{windows}} \gamma_h(x)}.

    Args:
        n_spatial: Number of spatial axes.
        space: Spatial lags.
        time: Time lags.
        p: Variogram order.
    """

    def __init__(self, n_spatial: int, space, time, p=0.5):
        self.n_spatial, self.space, self.time, self.p = (
            n_spatial,
            tuple(space),
            tuple(time),
            p,
        )
        self.num, self.den = {}, {}
        self.vs, self.n_win = {"s": 0.0, "t": 0.0}, 0

    @staticmethod
    def _incr(a, dim, h, p):
        lo, hi = [slice(None)] * a.dim(), [slice(None)] * a.dim()
        lo[dim], hi[dim] = slice(None, -h), slice(h, None)
        return (a[tuple(hi)] - a[tuple(lo)]).abs().pow(p)

    def update(self, ens, x):
        """Accumulate one window: members (M, Lf, C, *S) and truth (Lf, C, *S)."""
        specs = [
            (-(d + 1), h, f"s{h}") for d in range(self.n_spatial) for h in self.space
        ]
        specs += [(-(self.n_spatial + 2), t, f"t{t}") for t in self.time]
        for dim, h, key in specs:
            if x.shape[dim] <= h:
                continue
            d_t = self._incr(x, dim, h, self.p)
            d_g = sum(self._incr(ens[m], dim, h, self.p) for m in range(ens.shape[0]))
            d_g = d_g / ens.shape[0]
            self.num[key] = self.num.get(key, 0.0) + float(d_g.mean())
            self.den[key] = self.den.get(key, 0.0) + float(d_t.mean())
            self.vs[key[0]] += float(((d_t - d_g) ** 2).mean())
        self.n_win += 1

    def summary(self) -> dict:
        out = {f"vgram_{k}": self.num[k] / (self.den[k] + 1e-12) for k in self.num}
        out.update({f"vs_{a}": v / max(self.n_win, 1) for a, v in self.vs.items()})
        return out


SPEC_KEYS = ("spec_large", "spec_mid", "spec_fine", "spec_finest")
SSR_KEYS = ("ssr_large", "ssr_mid", "ssr_fine", "ssr_finest")


def spectral_bands(x: torch.Tensor, n_spatial: int) -> torch.Tensor:
    r"""Spatial power spectrum of a field, summed over four wavenumber bands.

    Args:
        x: (..., *S) field.
        n_spatial: Number of spatial axes, 1 or 2.

    Returns:
        (4,) power per band, from large to fine scales.
    """
    dim = -n_spatial
    p = torch.fft.rfft(x.float(), dim=dim).abs().pow(2).movedim(dim, 0)
    p = p.reshape(p.shape[0], -1).sum(1)  # (K + 1,) power per wavenumber index
    K = p.shape[0] - 1
    edges = [1, max(2, round(0.03 * K)), round(0.125 * K), round(0.5 * K), K + 1]
    for i in range(1, 4):
        edges[i] = max(edges[i], edges[i - 1])
    return torch.stack([p[lo:hi].sum() for lo, hi in pairwise(edges)])


def band_ratios(gen: torch.Tensor, true: torch.Tensor) -> dict:
    """Ratio of generated to true power in each band of `spectral_bands`."""
    return {
        k: float("nan") if float(t) == 0 else float(g) / float(t)
        for k, g, t in zip(SPEC_KEYS, gen, true)
    }


def junction_scores(chains: np.ndarray, truth: np.ndarray, L: int) -> dict:
    r"""Cross-window continuity.

    Scores the junction between consecutive reconstructions, on stitched chains of
    windows, against the truth.

    Args:
        chains: (n, T, C, *S) stitched chains of windows.
        truth: (n, T, C, *S) true trajectories.
        L: Window length in steps.

    Returns:
        ``jump_ratio``, ``jump_seam``, ``jump_interior``: mean absolute step of the
        chains over that of the truth, over all steps, junctions and interiors.
        ``zcorr`` and ``zcorr_truth``: lag-1 autocorrelations of the steps of the
        chains and of the truth.
    """
    # the first window has no predecessor
    gs, ts = np.diff(chains, axis=1)[:, L - 1 :], np.diff(truth, axis=1)[:, L - 1 :]
    seam = np.zeros(gs.shape[1], bool)
    seam[::L] = True

    def ratio(a, b):
        return float(np.abs(a).mean() / (np.abs(b).mean() + 1e-9))

    def zcorr(a):
        u, v = a[:, :-1].ravel(), a[:, 1:].ravel()
        if u.std() == 0 or v.std() == 0:
            return float("nan")
        return float(np.corrcoef(u, v)[0, 1])

    return {
        "jump_ratio": ratio(gs, ts),
        "jump_seam": ratio(gs[:, seam], ts[:, seam]) if seam.any() else float("nan"),
        "jump_interior": ratio(gs[:, ~seam], ts[:, ~seam]),
        "zcorr": zcorr(gs),
        "zcorr_truth": zcorr(ts),
    }


JUNCTION_MODES = ("ctx_off", "ctx_on", "causal")
JUNCTION_KEYS = ("jump_ratio", "jump_seam", "jump_interior", "zcorr", "zcorr_truth")
ID_KEYS = ("tag", "arm", "face", "seed", "n_windows", "m_gen", "context")


def SCORE_KEYS(vg_space, vg_time, cov_levels) -> list[str]:
    return [
        *ID_KEYS,
        "crps",
        "consistency",
        "spread_skill",
        "spread",
        "rmse",
        *SSR_KEYS,
        *(f"cov{int(lv * 100)}" for lv in cov_levels),
        "cov_mae",
        "cov_bias",
        *(f"vgram_s{h}" for h in vg_space),
        *(f"vgram_t{t}" for t in vg_time),
        "vs_s",
        "vs_t",
        *SPEC_KEYS,
        *(f"junction_{m}_{k}" for m in JUNCTION_MODES for k in JUNCTION_KEYS),
        "crps_per_t",
        "crps_per_var",
    ]


def empty_scores(keys) -> dict:
    """Placeholder scores."""
    return {k: (None if k in ID_KEYS else float("nan")) for k in keys}
