r"""An arm sets how the backbone networks are trained (`loss`) and how they reconstruct
a fine record from its observation (`sample`).

- ``"recursive"``: recursive MUNDO. One network per level. Level :math:`r` reads its
  parents and draws a residual on :math:`A^\dagger` of them.
- ``"single_step"``: single-step MUNDO. One network; :math:`A^\dagger y + G(y)`,
  projected in the loss or after sampling.
- ``"FlowMatching"``: flow matching on the residual :math:`x - A^\dagger y`, projected
  after sampling.
- ``"FlowDPS"``: the ``"FlowMatching"`` prior on the window itself, with a consistency
  update at every solver step, either the exact projection (``"blend"``) or gradient
  steps (``"grad"``).
- ``"SimVP"``: deterministic video model, reading :math:`A^\dagger y` and trained with
  the MSE on the window, projected after sampling.
"""

import torch
from torch import nn

from mundo.losses import (
    energy_score,
    resolve_aux_weight,
    spectral_crps,
    variogram_score,
)
from mundo.operator import chain, coarsen, fine_len, level_lens, window_operator


def fm_schedule(steps: int, n_train: int = 1000, shift: float = 1.0):
    r"""Flow-matching schedule.

    Args:
        steps: Number of solver steps.
        n_train: Number of discrete timesteps the model is trained on.
        shift: Timestep shift of the sampling schedule, :math:`s \mapsto
            \mathrm{shift}\, s / (1 + (\mathrm{shift} - 1) s)`.

    Returns:
        sigmas: (steps + 1,) noise levels, from 1 down to a shifted
            ``1 / n_train``, then a final 0.
        timesteps: (steps,) the same levels times `n_train`, given to the model as
            its time input.
    """
    s = torch.linspace(n_train, 1, steps) / n_train
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, torch.zeros(1)]), s * n_train


def _rep(t, M):
    """Copy the whole batch M times along axis 0.

    (B, ...) -> (M * B, ...); draw m of sample b sits at index ``m * B + b``.
    """
    return t.repeat(M, *([1] * (t.dim() - 1)))


def _w(x, vw):
    """Multiply each variable of `x` (B, Lf, C, *S) by its weight in `vw` (C,).
    `x` is returned unchanged when `vw` is `None`.
    """
    return x if vw is None else x * vw.to(x).view(1, 1, -1, *([1] * (x.dim() - 3)))


class Arm:
    r"""Base of an arm.

    Args:
        face: ``"disagg"`` or ``"interp"``.
        L: Window length in steps.
        factors: Coarsening factor of each level, coarse to fine, multiplying to `L`.
        aux: Auxiliary loss added to the energy score: ``"none"``, ``"variogram"`` or
            ``"spectral"``.
        aux_w: Weight of the auxiliary loss. Negative means relative, i.e., ``|aux_w|``
            times the ratio of the two terms on the first batch (`resolve_aux_weight`).
        es_draws: Number of draws per window in the energy score.
        vg_p: Order of the variogram loss.
        vg_lags: Lags of the variogram loss.
        vg_clamp: Whether to floor each lag term of the variogram loss at 0.
        fm_loss: Training recipe of the flow arms, ``"cfm"`` or ``"sd3"`` (see
            `FlowMatching`).
        fm_steps: Number of solver steps of the flow arms at sampling.
        fm_shift: Timestep shift of their sampling schedule (`fm_schedule`).
        fm_train_steps: Number of discrete noise levels of the flow arms in training.
        dps_step: Step size of the FlowDPS gradient update.
        dps_iters: Number of gradient steps per FlowDPS solver step, used in
            ``dps="grad"`` only.
        noise: Where the engression networks draw their noise, as ``+``-joined
            tokens.

            - ``input``: ``C * L`` standard-normal channels appended to the surface
              of a network producing `L` frames, one per output value.
            - ``lg``: 8 channels drawn per site and 8 drawn per sample and broadcast
              over space.
            - ``mdi``: the backbone injects noise at depth.
            - ``none``: alone, makes the networks deterministic.
    """

    name = ""
    c_level_mult = 1  # level channels = mult * C
    n_cond_extra = 0  # cond columns the arm appends (FlowMatching: the timestep)

    def __init__(
        self,
        face,
        L,
        factors,
        aux="none",
        aux_w=0.0,
        es_draws=2,
        vg_p=0.5,
        vg_lags=(1, 2, 4),
        vg_clamp=False,
        fm_loss="cfm",
        fm_steps=25,
        fm_shift=1.0,
        fm_train_steps=1000,
        dps_step=3.0,
        dps_iters=3,
        noise="mdi",
    ):
        self.face, self.L = face, L
        tokens = str(noise).split("+")
        self.noise = frozenset(tokens)
        assert self.noise <= {"none", "input", "lg", "mdi"} and len(tokens) == len(
            self.noise
        ), f"unknown noise route {noise!r}"
        assert "none" not in self.noise or self.noise == {"none"}, noise
        self.noise_n: list[int] = []  # noise channels appended to each net's surface
        assert fm_loss in ("cfm", "sd3"), f"unknown flow schedule {fm_loss!r}"
        self.fm_loss = fm_loss
        self.fm_steps, self.fm_train_steps = fm_steps, fm_train_steps
        self.fm_shift = fm_shift
        self.dps_step, self.dps_iters = dps_step, dps_iters
        self.op = window_operator(face, L)
        self.ops = chain(face, factors, L)
        assert aux in ("none", "variogram", "spectral"), (
            f"unknown auxiliary loss {aux!r}"
        )
        self.aux = {
            "none": None,
            "variogram": lambda d, y: variogram_score(
                d, y, p=vg_p, lags=vg_lags, clamp=vg_clamp, reduce=self.reduce
            ),
            "spectral": spectral_crps,
        }[aux]
        self.aux_w, self.es_draws = aux_w, es_draws
        self.aux_weights: dict[int, float] = {}
        self.reduce = lambda t: t  # cross-rank mean of a batch score, set by train
        self.terms = None  # (energy score, weighted aux term) of the last batch scored
        self.vw = None  # per-variable loss scale (C,)
        self.gen_c = 0  # channels per level of the generated context, 0 without it

    def n_nets(self):
        return 1

    def _net(self, Net, C, c_surface, L_in, L_out, spatial, periodic, n_cond, bkw):
        """Build one backbone that reads `L_in` frames and writes `L_out`."""
        on_level = "input" in self.noise and getattr(Net, "LEVEL_NOISE", False)
        self.noise_n.append(
            (C * L_out if "input" in self.noise and not on_level else 0)
            + (16 if "lg" in self.noise else 0)
        )
        return Net(
            c_level=self.c_level_mult * C,
            c_surface=c_surface + self.noise_n[-1],
            c_out=C,
            L=L_in,
            L_out=L_out,
            spatial=spatial,
            periodic=periodic,
            n_cond=n_cond + self.n_cond_extra,
            **({"entry_noise": C} if on_level else {}),
            **bkw,
        )

    def nets(
        self, Net, C, c_surface, L, spatial, periodic, n_cond, bkw
    ) -> nn.ModuleList:
        """One network over the whole window."""
        self.noise_n = []
        Lf = fine_len(self.face, L)
        return nn.ModuleList(
            self._net(Net, C, c_surface, Lf, Lf, spatial, periodic, n_cond, bkw)
            for _ in range(self.n_nets())
        )

    def _noisy(self, surface, i, g):
        """Append the noise channels of net `i` to `surface`."""
        n = self.noise_n[i]
        if not n:
            return surface
        B, S = surface.shape[0], surface.shape[2:]
        lg = "lg" in self.noise
        parts = [surface]
        if n - 16 * lg:
            parts.append(
                torch.randn(B, n - 16 * lg, *S, generator=g, device=surface.device)
            )
        if lg:
            parts.append(torch.randn(B, 8, *S, generator=g, device=surface.device))
            parts.append(
                torch.randn(
                    B, 8, *([1] * len(S)), generator=g, device=surface.device
                ).expand(B, 8, *S)
            )
        return torch.cat(parts, dim=1)

    def _score(self, draws, y, level=0):
        draws, y = [_w(d, self.vw) for d in draws], _w(y, self.vw)
        es = energy_score(draws, y)
        if self.aux is None or not self.aux_w:
            return es
        v = self.aux(draws, y)
        if level not in self.aux_weights:
            self.aux_weights[level] = resolve_aux_weight(
                self.aux_w, self.reduce(es.detach()), self.reduce(v.detach())
            )
        aux = self.aux_weights[level] * v
        self.terms = (es.detach(), aux.detach())
        return es + aux

    def loss(self, nets, fine, surface, cond, g):
        raise NotImplementedError

    def sample(self, nets, obs, surface, cond, M, g):
        raise NotImplementedError


class SingleStep(Arm):
    r"""Single-step MUNDO.

    Each member is :math:`u = A^\dagger y + G(y)`, with :math:`G` reading the
    observation's frames and writing the whole window, projected so that
    :math:`A u = y`.

    Args:
        project: Where the draws are projected.

            - ``"sample"``: at sampling only.
            - ``"loss"``: in training as well.
    """

    name = "single_step"

    def __init__(self, *a, project="sample", **kw):
        super().__init__(*a, **kw)
        assert project in ("loss", "sample"), f"unknown project {project!r}"
        self.project = project

    def nets(
        self, Net, C, c_surface, L, spatial, periodic, n_cond, bkw
    ) -> nn.ModuleList:
        """One network reading the observation, writing the `Lf` frames."""
        self.noise_n = []
        n_obs, Lf = fine_len(self.face, 1), fine_len(self.face, L)
        L_in = Lf if self._reads_base(Net) else n_obs
        return nn.ModuleList(
            [self._net(Net, C, c_surface, L_in, Lf, spatial, periodic, n_cond, bkw)]
        )

    def _reads_base(self, net):
        r"""Whether `net` takes the interpolant :math:`A^\dagger y` as input.

        `True` on the ``"interp"`` face for a backbone with `READS_BASE`.
        Otherwise, `net` takes the observed frames :math:`y`.
        """
        return self.face == "interp" and getattr(net, "READS_BASE", False)

    def _draw(self, net, obs, surface, cond, g):
        base = self.op.right_inverse(obs)
        x = base if self._reads_base(net) else obs
        return base + net(x, self._noisy(surface, 0, g), cond, generator=g)

    def loss(self, nets, fine, surface, cond, g):
        obs = self.op.apply(fine)
        draws = [
            self._draw(nets[0], obs, surface, cond, g) for _ in range(self.es_draws)
        ]
        if self.project == "loss":
            draws = [self.op.project(d, obs) for d in draws]
        return self._score(draws, fine)

    def sample(self, nets, obs, surface, cond, M, g):
        obs = _rep(obs, M)
        w = self._draw(nets[0], obs, _rep(surface, M), _rep(cond, M), g)
        return self.op.project(w, obs)


class Recursive(Arm):
    r"""Recursive MUNDO.

    Levels are indexed coarse to fine, :math:`x^{(0)} = y` up to
    :math:`x^{(R)} = x`, and ``self.ops[r]`` is the coarsening :math:`A_r` with
    :math:`x^{(r)} = A_r x^{(r+1)}`, so that :math:`A = A_0 \circ \cdots \circ
    A_{R-1}` and

    .. math::

        p(x \mid y) = \int \prod_{r=0}^{R-1} p(x^{(r+1)} \mid x^{(r)})
        \, \mathrm{d}x^{(1)} \cdots \mathrm{d}x^{(R-1)}.

    ``nets[r]`` samples the factor of level :math:`r` with a null-space
    parameterization: from the parent :math:`x^{(r)}`, it writes the degrees of
    freedom of :math:`\ker A_r` lost between the two levels, and
    `Operator.complete` adds the residual they span to :math:`A_r^\dagger x^{(r)}`,
    so every level reproduces its parent exactly.
    """

    name = "recursive"

    def n_nets(self):
        return len(self.ops)

    def nets(
        self, Net, C, c_surface, L, spatial, periodic, n_cond, bkw
    ) -> nn.ModuleList:
        """One network per level."""
        self.noise_n = []
        parents = [fine_len(self.face, 1)] + level_lens(self.face, self.ops)[:-1]
        return nn.ModuleList(
            self._net(
                Net,
                C,
                c_surface,
                op.n_free(p) if self._reads_base(Net, op, p) else p,
                op.n_free(p),
                spatial,
                periodic,
                n_cond,
                bkw,
            )
            for p, op in zip(parents, self.ops)
        )

    def _reads_base(self, net, op, p):
        r"""Whether `net` takes the interpolant :math:`A_r^\dagger x^{(r)}` as input.

        `True` on the ``"interp"`` face for a backbone with `READS_BASE` when the level
        adds more frames than its `p` parents. Otherwise, `net` takes the parents.
        """
        return (
            self.face == "interp"
            and getattr(net, "READS_BASE", False)
            and op.n_free(p) > p
        )

    def _own_slot(self, surface, r):
        """Zero the generated context of every level but `r`.

        Level `r` conditions on the previous window at its own resolution only.

        Args:
            surface: (B, c_surface, *S) from `pack_surface`, whose last channels are
                the R generated slots of `gen_c` channels each, then their presence
                map.
            r: Level index, coarse to fine.
        """
        C, R = self.gen_c, len(self.ops)
        a = surface.shape[1] - 1 - R * C
        keep = torch.ones(surface.shape[1], dtype=surface.dtype, device=surface.device)
        keep[a : a + R * C] = 0
        keep[a + r * C : a + (r + 1) * C] = 1
        return surface * keep.view(1, -1, *([1] * (surface.dim() - 2)))

    def _children(self, nets, r, parent, surface, cond, g):
        r"""One draw of level `r`: refine `parent` by the factor of ``ops[r]``.

        Args:
            nets: One network per level.
            r: Level index, coarse to fine.
            parent: :math:`x^{(r)}`.
            surface: (B, c_surface, *S) context channels.
            cond: (B, n_cond) scalar conditioning.
            g: Torch generator of the network's noise.

        Returns:
            :math:`x^{(r+1)}`, which coarsens exactly to `parent`.
        """
        op = self.ops[r]
        if self.gen_c:
            surface = self._own_slot(surface, r)
        x = parent
        if self._reads_base(nets[r], op, parent.shape[1]):
            x = op.right_inverse(parent)
            x = x[:, torch.arange(x.shape[1], device=x.device) % op.k != 0]
        d = nets[r](x, self._noisy(surface, r, g), cond, generator=g)
        return op.complete(parent, d)

    def loss(self, nets, fine, surface, cond, g, levels=None):
        r"""Teacher-forced loss.

        Args:
            nets: One network per level.
            fine: (B, Lf, C, *S) true window.
            surface: (B, c_surface, *S) context channels.
            cond: (B, n_cond) scalar conditioning.
            g: Torch generator.
            levels: Level indices to train, all when `None`.

        Returns:
            Sum over the trained levels of the score of the draws of
            :math:`x^{(r+1)}`.
        """
        loss = 0.0
        for r, op in enumerate(self.ops):
            if levels is not None and r not in levels:
                continue
            truth = coarsen(fine, self.ops[r + 1 :])
            parent = op.apply(truth)
            draws = [
                self._children(nets, r, parent, surface, cond, g)
                for _ in range(self.es_draws)
            ]
            loss = loss + self._score(draws, truth, r)
        return loss

    def rollout(self, nets, obs, surface, cond, g):
        cur = obs
        for r in range(len(self.ops)):
            cur = self._children(nets, r, cur, surface, cond, g)
        return cur

    def sample(self, nets, obs, surface, cond, M, g):
        return self.rollout(nets, _rep(obs, M), _rep(surface, M), _rep(cond, M), g)


class FlowMatching(Arm):
    r"""Flow matching (Lipman et al., 2023).

    A flow samples the residual :math:`r = x - A^\dagger y` conditioned on
    :math:`A^\dagger y`. Samples are then projected onto the observation.

    Training under `fm_loss`:

    - ``"cfm"``: predicts the velocity :math:`\varepsilon - r` at a uniform
      :math:`\sigma`.
    - ``"sd3"``: predicts :math:`r` at a logit-normal :math:`\sigma`, with the
      squared error weighted by :math:`\min((1 - \sigma) / \sigma, 100)`.
    """

    name = "FlowMatching"
    c_level_mult = 2
    n_cond_extra = 1

    def _target(self, fine, obs):
        return fine - self.op.right_inverse(obs)

    def _state(self, obs, noisy):
        return torch.cat([self.op.right_inverse(obs), noisy], dim=2)

    def loss(self, nets, fine, surface, cond, g):
        obs = self.op.apply(fine)
        x0 = self._target(fine, obs)
        B, n = fine.shape[0], self.fm_train_steps
        sig_all, ts_all = (t.to(fine) for t in fm_schedule(n, n))
        if self.fm_loss == "cfm":
            u = torch.rand(B, generator=g, device=fine.device)
        else:
            u = torch.randn(B, generator=g, device=fine.device).sigmoid()
        idx = (u * n).long().clamp(0, n - 1)
        sgm = sig_all[idx].view(B, *([1] * (fine.dim() - 1)))
        eps = torch.randn(x0.shape, generator=g, device=fine.device)
        noisy = eps * sgm + x0 * (1 - sgm)
        c = torch.cat([cond, ts_all[idx][:, None]], dim=1)
        pred = nets[0](self._state(obs, noisy), surface, c, generator=g)
        if self.fm_loss == "cfm":
            return (_w(pred - (eps - x0), self.vw) ** 2).mean()
        snr = ((1 - sgm) / sgm.clamp_min(1e-4)).clamp(max=100)
        return (snr * _w(pred - x0, self.vw) ** 2).mean()

    def _euler(self, net, obs, surface, cond, M, g, consistent=None):
        obs, surface, cond = _rep(obs, M), _rep(surface, M), _rep(cond, M)
        Lf = fine_len(self.face, self.L)
        x = torch.randn(M, Lf, *obs.shape[2:], generator=g, device=obs.device)
        sig, ts = fm_schedule(self.fm_steps, self.fm_train_steps, self.fm_shift)
        sig = sig.to(x)
        for k, t in enumerate(ts):
            c = torch.cat([cond, torch.full((M, 1), float(t), device=x.device)], dim=1)
            x1 = net(self._state(obs, x), surface, c, generator=g)
            s_k, s_k1 = sig[k], sig[k + 1]
            if self.fm_loss == "cfm":
                x1 = x - s_k * x1
            if consistent is None:
                x = x + (s_k1 - s_k) * (x - x1) / s_k.clamp_min(1e-4)
                continue
            # the noise estimate, the consistency-blended data estimate, and
            # the re-noised state of the next step
            x0n = (x - (1 - s_k) * x1) / s_k.clamp_min(1e-4)
            x1 = (1 - s_k) * x1 + s_k * consistent(x1)
            fresh = torch.randn(x.shape, generator=g, device=x.device)
            noise = s_k1.sqrt() * x0n + (1 - s_k1).clamp_min(0).sqrt() * fresh
            x = (1 - s_k1) * x1 + s_k1 * noise
        return x, obs

    def sample(self, nets, obs, surface, cond, M, g):
        x, obs_m = self._euler(nets[0], obs, surface, cond, M, g)
        return self.op.project(self.op.right_inverse(obs_m) + x, obs_m)


class FlowDPS(FlowMatching):
    r"""FlowDPS (Kim et al., 2025).

    An unconditional flow on the window :math:`x`. The observation enters only at
    sampling, as a consistency update toward :math:`A u = y` at every solver step.

    Args:
        dps: The consistency update, ``"blend"`` for the exact projection or
            ``"grad"`` for `dps_iters` gradient steps of size `dps_step`.
    """

    name = "FlowDPS"
    c_level_mult = 1

    def __init__(self, *a, dps="blend", **kw):
        super().__init__(*a, **kw)
        assert dps in ("blend", "grad"), f"unknown dps {dps!r}"
        self.dps = dps

    def _target(self, fine, obs):
        return fine

    def _state(self, obs, noisy):
        return noisy

    def _consistent(self, obs):
        if self.dps == "blend":
            return lambda x1: self.op.project(x1, obs)

        def grad_steps(x1):
            z = x1.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                for _ in range(self.dps_iters):
                    r = self.op.range_residual(z, obs).flatten(1).norm(dim=1).sum()
                    z = (
                        (z - self.dps_step * torch.autograd.grad(r, z)[0])
                        .detach()
                        .requires_grad_(True)
                    )
            return z.detach()

        return grad_steps

    def sample(self, nets, obs, surface, cond, M, g):
        r"""Sample with a consistency update, then project onto :math:`A u = y`."""
        x, obs_m = self._euler(
            nets[0],
            obs,
            surface,
            cond,
            M,
            g,
            consistent=self._consistent(_rep(obs, M)),
        )
        return self.op.project(x, obs_m)


class SimVP(Arm):
    name = "SimVP"

    def loss(self, nets, fine, surface, cond, g):
        base = self.op.right_inverse(self.op.apply(fine))
        return (_w(nets[0](base, surface, cond) - fine, self.vw) ** 2).mean()

    def sample(self, nets, obs, surface, cond, M, g):
        obs = _rep(obs, M)
        base = self.op.right_inverse(obs)
        return self.op.project(nets[0](base, _rep(surface, M), _rep(cond, M)), obs)


ARMS = {
    "recursive": Recursive,
    "single_step": SingleStep,
    "FlowMatching": FlowMatching,
    "FlowDPS": FlowDPS,
    "SimVP": SimVP,
}


def build_arm(name, face, L, factors, project="sample", dps="blend", **kw) -> Arm:
    """Build the arm `name` for one face and window length.

    Args:
        name: A key of `ARMS`.
        face: ``"disagg"`` or ``"interp"``.
        L: Window length in steps.
        factors: Refinement factor of each level, coarse to fine.
        project: Passed to `SingleStep`.
        dps: Passed to `FlowDPS`.
        **kw: `Arm` options. `aux`, `aux_w` and `noise` are reset for the flow arms and
            `SimVP`, which have their own loss and noise.
    """
    if name == "single_step":
        kw["project"] = project
    if name == "FlowDPS":
        kw["dps"] = dps
    if name in ("FlowMatching", "FlowDPS", "SimVP"):
        kw.update(aux="none", aux_w=0.0, noise="none")
    return ARMS[name](face, L, factors, **kw)
