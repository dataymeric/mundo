"""Evaluate the checkpoints and the baselines on the test split, and write a JSON."""

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from mundo import baselines as classical
from mundo.checkpoint import (
    checkpoint_dir,
    fit_path,
    load_weights,
    read_fit,
    read_stats,
    trained_tags,
)
from mundo.config import CONF_DIR
from mundo.metrics import (
    JUNCTION_MODES,
    SCORE_KEYS,
    SpreadSkill,
    VariogramTier,
    band_ratios,
    consistency,
    cov_calibration,
    coverage,
    crps,
    empty_scores,
    junction_scores,
    spectral_bands,
)
from mundo.operator import window_operator
from mundo.systems import build_system
from mundo.train import batch_to, build, device_of, make_surface
from mundo.windows import Windows

BASELINES = {
    "disagg": ("linear", "denton_cholette", "cdf_t"),
    "interp": ("linear", "spline"),
}

log = logging.getLogger(__name__)


def found_checkpoints(cfg, system_name: str, face: str) -> list[tuple[str, int, Path]]:
    """Checkpoints to evaluate: every tag of ``eval.arms`` (all trained tags when it is
    empty) at every seed of ``eval.seeds``.

    Returns:
        ``(tag, seed, ckpt_dir)`` for each finished checkpoint.
    """
    tags = list(cfg.eval.arms) or trained_tags(cfg.out, system_name, face)
    out = []
    for tag in tags:
        for seed in cfg.eval.seeds:
            ckpt_dir = checkpoint_dir(cfg.out, system_name, face, tag, int(seed))
            if fit_path(ckpt_dir).exists():
                out.append((tag, int(seed), ckpt_dir))
            elif cfg.eval.arms:
                log.warning("no finished checkpoint at %s", ckpt_dir)
    if not out:
        log.warning(
            "no checkpoint to evaluate under %s/%s/%s", cfg.out, system_name, face
        )
    return out


def checkpoint_stats(found: list[tuple[str, int, Path]]) -> dict | None:
    """Normalization shared by the checkpoints of `found`, or `None` if it is empty.

    Raises:
        ValueError: If the checkpoints were normalized differently. Evaluate them
            separately with ``eval.arms``.
    """
    groups: dict[str, list[str]] = {}
    for tag, seed, d in found:
        groups.setdefault(json.dumps(read_stats(d), sort_keys=True), []).append(
            f"{tag}/seed{seed}"
        )
    if len(groups) > 1:
        lines = ["  " + ", ".join(g) for g in groups.values()]
        raise ValueError(
            "checkpoints normalized differently; evaluate each line apart with "
            "eval.arms:\n" + "\n".join(lines)
        )
    return json.loads(next(iter(groups))) if groups else None


def _cfg_of_checkpoint(cfg, ckpt_dir: Path):
    """The checkpoint's own training config with this run's eval, launch and out."""
    c = OmegaConf.create(read_fit(ckpt_dir)["cfg"])
    c.eval, c.launch, c.out = cfg.eval, cfg.launch, cfg.out
    return c


def _sample(
    arm, nets, spec, item, device, M, g, context_on: bool, prev_fine=None, mb=None
):
    """Draw members for one window.

    Args:
        arm, nets, spec: From `build`.
        item: A window, ``ds[i]`` of a `Windows` dataset; its observation is
            computed from its true record.
        device: Torch device.
        M: Number of members.
        g: Torch generator of the draws.
        context_on: Give the arm the observed context, when the checkpoint uses it.
        prev_fine: (M, Lf, C, *S) generated previous window of each member, each then
            conditioned on its own.
        mb: Members per forward pass, all `M` when `None`.

    Returns:
        (M, Lf, C, *S) members.
    """
    spec = {**spec, "observed": context_on and spec["observed"]}
    mb = M if mb is None else int(mb)
    out = []
    for lo in range(0, M, mb):
        prev = None if prev_fine is None else prev_fine[lo : lo + mb]
        B = 1 if prev is None else prev.shape[0]
        batch = batch_to(
            {k: v[None].expand(B, *v.shape) for k, v in item.items()}, device
        )
        surface = make_surface(batch, spec, device, False, 0.0, g, prev_fine=prev)
        obs = spec["op"].apply(batch["fine"])
        m = min(mb, M - lo) if prev is None else 1
        out.append(arm.sample(nets, obs, surface, batch["cond"], m, g))
    return torch.cat(out)


def windows_per_trajectory(ds) -> int:
    """Number of windows per trajectory of `ds`."""
    counts = {}
    for traj, _ in ds.index:
        counts[traj] = counts.get(traj, 0) + 1
    assert len(set(counts.values())) == 1, f"trajectories differ in length: {counts}"
    return next(iter(counts.values()))


def chain_length(cfg, ds) -> int:
    """``eval.chain_len``, or every window of a trajectory when it is 0."""
    return int(cfg.eval.chain_len) or windows_per_trajectory(ds)


def chain_starts(ds, n: int, k: int) -> list[int]:
    """Starts of up to `k` chains of `n` consecutive windows (all chains when `k` is
    0), spread evenly over `ds`. Within a trajectory, one unused window separates
    consecutive chains.
    """
    # eval windows are ordered by trajectory and start every L frames
    starts, i = [], 0
    while i + n <= len(ds):
        if ds.index[i][0] == ds.index[i + n - 1][0]:
            starts.append(i)
            i += n
            if i < len(ds) and ds.index[i][0] == ds.index[i - 1][0]:
                i += 1
        else:
            i += 1
    pick = np.linspace(
        0, len(starts) - 1, min(k or len(starts), len(starts)), dtype=int
    )
    return [starts[j] for j in pick]


def scored_windows(starts: list[int], n: int) -> list[int]:
    """Windows of the chains, except the first of each, which has no predecessor."""
    return [i0 + k for i0 in starts for k in range(1, n)]


def _prev_window(ds, i: int):
    """The window before ``ds[i]`` in its trajectory, or `None`."""
    traj, t0 = ds.index[i]
    return ds[i - 1] if i > 0 and ds.index[i - 1] == (traj, t0 - ds.L) else None


def _spaced_members(arm, nets, spec, ds, idxs, dev, M, g, context_on, mb):
    """``(i, members)`` for each window ``ds[i]`` of `idxs`. With generated context,
    the previous window is drawn first, and each member is conditioned on its own.
    """
    for i in idxs:
        prev = None
        if spec["generated"]:
            prev_window = _prev_window(ds, i)
            if prev_window is not None:
                prev = _sample(
                    arm, nets, spec, prev_window, dev, M, g, context_on, mb=mb
                )
        yield i, _sample(arm, nets, spec, ds[i], dev, M, g, context_on, prev, mb)


def _chain_members(arm, nets, spec, ds, starts, n, dev, M, g, context_on, mb):
    """``(i, members)`` of every scored window ``ds[i]`` of the chains at `starts`."""
    for i0 in starts:
        prev = None
        for k in range(n):
            if k == 0 and not spec["generated"]:
                continue  # unscored, and no later window depends on it
            ens = _sample(arm, nets, spec, ds[i0 + k], dev, M, g, context_on, prev, mb)
            prev = ens if spec["generated"] else None
            if k > 0:
                yield i0 + k, ens


def _save_members(
    cfg, face: str, name: str, seed: int, ds, idxs, members, stats
) -> None:
    """Write the windows `idxs` to ``<eval.save_members>/<face>/<name>_s<seed>.npz``.

    The file holds ``members`` (W, M, Lf, C, *S) and ``truth`` (W, Lf, C, *S), both
    normalized, ``windows`` (the `idxs`) and ``stats`` (JSON) to undo it.
    """
    path = Path(cfg.eval.save_members) / face / f"{name}_s{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        members=np.stack(members).astype(np.float32),
        truth=np.stack([ds[int(i)]["fine"].numpy() for i in idxs]),
        windows=np.asarray(idxs),
        stats=json.dumps(stats),
    )


def _checkpoint_scores(
    cfg, system, face, found, ds, starts, idxs, dev, n_spatial, keys
) -> list[dict]:
    """Scores of each checkpoint of `found` on the windows `idxs`, lone
    (``eval.windows=spaced``) or in the chains of `starts` (``chained``). Junction
    metrics always use the chains of `starts`.
    """
    n = chain_length(cfg, ds)
    chained = str(cfg.eval.windows) == "chained"
    results = []
    context_on = str(cfg.eval.context) == "on"
    for tag, seed, ckpt_dir in found:
        arm, nets, spec = build(_cfg_of_checkpoint(cfg, ckpt_dir), system)
        load_weights(
            ckpt_dir, nets, map_location=dev, last=str(cfg.eval.weights) == "last"
        )
        nets.to(dev).eval()
        M = 1 if arm.name == "SimVP" else int(cfg.eval.m_gen)
        g = torch.Generator(device=dev).manual_seed(int(cfg.eval.seed))
        scores = empty_scores(keys)
        scores.update(
            tag=tag,
            arm=arm.name,
            face=face,
            seed=seed,
            n_windows=len(idxs),
            m_gen=M,
            context=str(cfg.eval.context),
        )
        crps_windows, consistencies, coverages, crps_steps = [], [], {}, []
        crps_vars = {v: [] for v in system.vars}
        spread_skill = SpreadSkill(n_spatial)
        variograms = VariogramTier(
            n_spatial, cfg.eval.vg_space, cfg.eval.vg_time, float(cfg.eval.vg_p)
        )
        n_sp = int(cfg.eval.spec_members)
        spec_gen, spec_true = torch.zeros(4), torch.zeros(4)
        mb = int(cfg.eval.m_batch) or None  # 0: every member in one pass
        dumped = []
        with torch.no_grad():
            members = (
                _chain_members(
                    arm, nets, spec, ds, starts, n, dev, M, g, context_on, mb
                )
                if chained
                else _spaced_members(
                    arm, nets, spec, ds, idxs, dev, M, g, context_on, mb
                )
            )
            for i, ens in track(
                members,
                total=len(idxs),
                description=f"{tag} s{seed}",
            ):
                x = ds[i]["fine"].to(dev)
                if cfg.eval.save_members:
                    dumped.append(ens.cpu().numpy())
                consistencies.append(
                    consistency(spec["op"], ens, spec["op"].apply(x[None])[0])
                )
                # on interp both anchors are pinned: score frames 1..L-1
                ens_s, x_s = (ens[:, 1:-1], x[1:-1]) if face == "interp" else (ens, x)
                crps_windows.append(crps(ens_s, x_s))
                crps_steps.append(
                    [crps(ens_s[:, t], x_s[t]) for t in range(x_s.shape[0])]
                )
                for ci, v in enumerate(system.vars):
                    crps_vars[v].append(crps(ens_s[:, :, ci], x_s[:, ci]))
                if M >= 2:
                    for k, v in coverage(ens_s, x_s, cfg.eval.cov_levels).items():
                        coverages.setdefault(k, []).append(v)
                spread_skill.update(ens_s, x_s)
                variograms.update(ens_s, x_s)
                spec_gen += spectral_bands(ens[:n_sp], n_spatial).cpu() / min(n_sp, M)
                spec_true += spectral_bands(x, n_spatial).cpu()
        scores.update(
            crps=float(np.mean(crps_windows)),
            consistency=float(np.max(consistencies)),
            **spread_skill.summary(M),
            **{k: float(np.mean(v)) for k, v in coverages.items()},
            **variograms.summary(),
        )
        scores.update(
            cov_calibration({k: scores[k] for k in coverages}, cfg.eval.cov_levels)
        )
        scores.update(band_ratios(spec_gen, spec_true))
        scores["crps_per_t"] = np.array(crps_steps).mean(axis=0).tolist()
        scores["crps_per_var"] = {v: float(np.mean(c)) for v, c in crps_vars.items()}
        if cfg.eval.save_members:
            _save_members(cfg, face, tag, seed, ds, idxs, dumped, system.stats)
        if face == "disagg":
            scores.update(_junction(arm, nets, spec, ds, starts, n, dev, g))
        results.append(scores)
        shown = ("crps", "consistency", "spread_skill", "cov_mae", "vgram_t1")
        log.info(
            "%s s%d: %s", tag, seed, " ".join(f"{k} {scores[k]:.4f}" for k in shown)
        )
    return results


def _junction(arm, nets, spec, ds, starts, n, dev, g) -> dict:
    """Junction metrics of the chains of `n` windows at `starts`.

    Each window gets one draw, in three modes:

    - ``ctx_off``: without observed context.
    - ``ctx_on``: with it.
    - ``causal``: conditioned on the previous draw of the chain; only for a
      checkpoint trained on generated context.

    Returns:
        ``junction_<mode>_<key>`` for the keys of `junction_scores`.
    """
    out = {}
    for mode in JUNCTION_MODES:
        if mode == "causal" and not spec["generated"]:
            continue
        chains, truths = [], []
        with torch.no_grad():
            for i0 in track(starts, description=f"junction {mode}"):
                wins, prev = [], None
                for k in range(n):
                    e = _sample(
                        arm,
                        nets,
                        spec,
                        ds[i0 + k],
                        dev,
                        1,
                        g,
                        context_on=mode != "ctx_off",
                        prev_fine=prev,
                    )
                    wins.append(e[0].cpu().numpy())
                    prev = e if mode == "causal" else None
                chains.append(np.concatenate(wins))
                truths.append(
                    np.concatenate([ds[i0 + k]["fine"].numpy() for k in range(n)])
                )
        if chains:
            jr = junction_scores(np.stack(chains), np.stack(truths), spec["L"])
            out.update({f"junction_{mode}_{k}": v for k, v in jr.items()})
    return out


def _window_means(window) -> np.ndarray:
    """(3, C, *S) time means of the previous, current and next windows, the input of
    the disaggregation baselines.
    """
    flags, m = window["flags"].numpy(), window["fine"].numpy().mean(0)
    before = window["before"][0].numpy().mean(0) if flags[0] else m
    after = window["after"][0].numpy().mean(0) if flags[1] else m
    return np.stack([before, m, after])


def _baseline_scores(cfg, system, face, ds, idxs, L, keys) -> list[dict]:
    """Scores of each baseline of ``eval.baselines`` that applies to `face`.

    Each baseline runs on the previous, current and next windows, and is scored on the
    current one, as a single member. Its CRPS is its mean absolute error. No baseline
    is scored when ``eval.arms`` is set.
    """
    if cfg.eval.arms:
        return []
    want = [n for n in cfg.eval.baselines if n in BASELINES[face]]
    crps_windows = {n: [] for n in want}
    recons = {n: [] for n in want}
    crps_steps = {n: [] for n in want}
    consistencies = {n: [] for n in want}
    crps_vars = {n: {v: [] for v in system.vars} for n in want}
    variograms = {
        n: VariogramTier(
            len(system.spatial),
            cfg.eval.vg_space,
            cfg.eval.vg_time,
            float(cfg.eval.vg_p),
        )
        for n in want
    }
    n_spatial = len(system.spatial)
    spec_gen = {n: torch.zeros(4) for n in want}
    spec_true = {n: torch.zeros(4) for n in want}
    truth_cal = None
    if "cdf_t" in want:
        # the truth of eval.cdft_windows training windows, spread over the split
        cal = Windows(system, "train", L, face, (0, 0), sliding=False)
        picks = np.linspace(0, len(cal) - 1, int(cfg.eval.cdft_windows), dtype=int)
        truth_cal = np.stack([cal[int(k)]["fine"].numpy() for k in picks])
    dc_windows, dc_truths = [], []
    op = window_operator(face, L)

    def score(name, u_full, x_full, frames):
        """Score the reconstruction `u_full` of the true window `x_full`: consistency
        and spectra on the whole window, the other scores on `frames`.
        """
        u_full = torch.from_numpy(np.asarray(u_full, np.float32))[None]
        recons[name].append(u_full.numpy())
        x_full = torch.from_numpy(x_full)
        u, x = u_full[:, frames], x_full[frames]
        crps_windows[name].append(crps(u, x))
        crps_steps[name].append([crps(u[:, t], x[t]) for t in range(x.shape[0])])
        consistencies[name].append(consistency(op, u_full, op.apply(x_full[None])[0]))
        for ci, v in enumerate(system.vars):
            crps_vars[name][v].append(crps(u[:, :, ci], x[:, ci]))
        variograms[name].update(u, x)
        spec_gen[name] += spectral_bands(u_full, n_spatial)
        spec_true[name] += spectral_bands(x_full, n_spatial)

    for i in track(idxs, description="baselines"):
        window = ds[int(i)]
        x, flags = window["fine"].numpy(), window["flags"].numpy()
        if face == "disagg":
            means3 = _window_means(window)
            if "linear" in want:
                score(
                    "linear", classical.lin_disagg(means3, L)[L : 2 * L], x, slice(None)
                )
            if {"denton_cholette", "cdf_t"} & set(want):
                dc = classical.denton(means3, L)[L : 2 * L]
                if "denton_cholette" in want:
                    score("denton_cholette", dc, x, slice(None))
                if "cdf_t" in want:
                    dc_windows.append(dc)
                    dc_truths.append(x)
        else:
            anchors, ts = [x[0], x[-1]], [0.0, float(L)]
            if flags[0]:
                anchors, ts = (
                    [window["before"][0][0].numpy()] + anchors,
                    [-float(L)] + ts,
                )
            if flags[1]:
                anchors, ts = anchors + [window["after"][0][-1].numpy()], ts + [2.0 * L]
            if "linear" in want:
                lin = np.concatenate([classical.lin_interp(x[0], x[-1], L), x[-1:]])
                score("linear", lin, x, slice(1, L))
            if "spline" in want:
                sp = classical.natural_spline(np.stack(anchors), np.array(ts), L)
                score("spline", np.concatenate([sp, x[-1:]]), x, slice(1, L))
    if dc_windows:
        # every evaluated window's Denton-Cholette record, mapped together
        dc_all = np.stack(dc_windows)  # (n, L, C, *S)
        steps = [[t] for t in range(L)] if cfg.eval.cdft_per_step else [list(range(L))]
        mapped = np.empty_like(dc_all)
        for ts in steps:
            pred, truth = dc_all[:, ts], truth_cal[:, ts]
            mapped[:, ts] = classical.cdf_t(
                pred.reshape(-1, pred[0, 0].size), truth.reshape(-1, truth[0, 0].size)
            ).reshape(pred.shape)
        for u, x in zip(mapped, dc_truths):
            score("cdf_t", u, x, slice(None))
    results = []
    for n, v in crps_windows.items():
        scores = empty_scores(keys)
        scores.update(
            tag=n,
            arm=n,
            face=face,
            seed=0,
            n_windows=len(idxs),
            m_gen=1,
            context="on",
            crps=float(np.mean(v)),
            consistency=float(np.max(consistencies[n])),
            **variograms[n].summary(),
        )
        scores.update(band_ratios(spec_gen[n], spec_true[n]))
        scores["crps_per_t"] = np.array(crps_steps[n]).mean(axis=0).tolist()
        scores["crps_per_var"] = {
            var: float(np.mean(c)) for var, c in crps_vars[n].items()
        }
        results.append(scores)
        if cfg.eval.save_members:
            _save_members(cfg, face, n, 0, ds, idxs, recons[n], system.stats)
        log.info("%s: crps %.4f (M=1)", n, scores["crps"])
    return results


def evaluate(cfg) -> list[dict]:
    """Scores of the checkpoints, then of the baselines, on the test windows of
    ``eval.windows``:

    - ``spaced``: up to ``eval.max_windows`` lone windows spread evenly over the
      split.
    - ``chained``: ``eval.chains`` chains of ``eval.chain_len`` consecutive windows
      (every window of a trajectory when it is 0) spread evenly over the split; the
      first window of each is drawn when needed but never scored.

    Junction metrics always use the chains.

    Returns:
        One dict per model, all with the keys of `SCORE_KEYS`.
    """
    t0 = time.time()
    dev = device_of(cfg)
    face, L = str(cfg.system.face), int(cfg.system.L)
    found = found_checkpoints(cfg, str(cfg.system.name), face)
    system = build_system(cfg, checkpoint_stats(found))
    n_spatial = len(system.spatial)
    neighbors = tuple(int(m) for m in cfg.system.neighbors)
    ds = Windows(system, "test", L, face, neighbors, sliding=False)
    n = chain_length(cfg, ds)
    starts = chain_starts(ds, n, int(cfg.eval.chains))
    mode = str(cfg.eval.windows)
    assert mode in ("spaced", "chained"), f"unknown eval.windows {mode!r}"
    if mode == "chained":
        idxs = scored_windows(starts, n)
    else:
        n_win = min(len(ds), int(cfg.eval.max_windows))
        idxs = [int(i) for i in np.linspace(0, len(ds) - 1, n_win, dtype=int)]
    log.info(
        "%s %s: %s, %d chains, %d scored windows, context=%s",
        system.name,
        face,
        mode,
        len(starts),
        len(idxs),
        cfg.eval.context,
    )
    keys = SCORE_KEYS(cfg.eval.vg_space, cfg.eval.vg_time, cfg.eval.cov_levels)
    results = _checkpoint_scores(
        cfg, system, face, found, ds, starts, idxs, dev, n_spatial, keys
    )
    results += _baseline_scores(cfg, system, face, ds, idxs, L, keys)
    log.info("%d models scored (%.0fs)", len(results), time.time() - t0)
    return results


def scores_name(cfg) -> str:
    """``<face>_ctx<on|off>[_<arm>+<arm>...][_last].json``, the arms of
    ``eval.arms``, ``_last`` when ``eval.weights`` scores the final weights.
    """
    slug = "_" + "+".join(cfg.eval.arms) if cfg.eval.arms else ""
    last = "_last" if str(cfg.eval.weights) == "last" else ""
    return f"{cfg.system.face}_ctx{cfg.eval.context}{slug}{last}.json"


def write_scores(results: list[dict], path: Path) -> Path:
    """Write `results` to `path` as a JSON list and return `path`; NaN stays ``NaN``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, default=float))
    return path


@hydra.main(config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate and write ``<out>/eval/<system>/<scores_name>``."""
    p = write_scores(
        evaluate(cfg), Path(cfg.out) / "eval" / str(cfg.system.name) / scores_name(cfg)
    )
    log.info("wrote %s", p)


if __name__ == "__main__":
    main()
