"""Training of one arm on one system.

Stage 1 trains any arm with its own loss. For the recursive arm it trains one
network per level, each with its own optimizer, early stopping and best weights, as
one job per level or all of them in turn.
Stage 2, for the recursive arm only, fine-tunes the whole recursive chain end-to-end
on its own rollouts, starting from a stage-1 checkpoint.
"""

import contextlib
import copy
import json
import logging
import math
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from mundo import dist
from mundo.arms import build_arm
from mundo.backbones import BACKBONES
from mundo.checkpoint import (
    best_path,
    checkpoint_dir,
    checkpoint_path,
    fit_path,
    load_checkpoint,
    load_weights,
    read_stats,
    save_checkpoint,
    stats_path,
    write_fit,
    write_stats,
)
from mundo.config import CONF_DIR, checkpoint_tag
from mundo.context import c_surface, drop_groups, generated, observed, pack_surface
from mundo.operator import chain_factors, fine_len
from mundo.systems import build_system
from mundo.windows import Windows

log = logging.getLogger(__name__)


def device_of(cfg) -> str:
    d = str(cfg.launch.device)
    d = d if d != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    return (
        f"cuda:{dist.local_rank()}"
        if d.startswith("cuda") and dist.world_size() > 1
        else d
    )


def batch_to(batch: dict, device) -> dict:
    """Move every tensor of a `Windows` batch to `device`."""
    return {k: v.to(device) for k, v in batch.items()}


def build(cfg, system):
    """Build the arm and its untrained networks for `cfg` on `system`.

    Returns:
        arm: The `Arm` of ``arm.name``.
        nets: Its networks, coarse to fine.
        spec: What training and evaluation share about the run: window lengths,
            operators, statics, context channels and flags.

    Raises:
        AssertionError: A chain whose factors do not multiply to `L`; ``arm.level`` or
            stage 2 with another arm than ``recursive``; generated context outside
            disaggregation with ``recursive`` or ``single_step``.
    """
    face, L, stage = str(cfg.system.face), int(cfg.system.L), int(cfg.train.stage)
    C = len(system.vars)
    factors = list(cfg.arm.chain) if cfg.arm.chain else chain_factors(L)
    assert math.prod(factors) == L, f"chain {factors} does not multiply to L={L}"
    arm = build_arm(
        str(cfg.arm.name),
        face,
        L,
        factors,
        project=str(cfg.arm.project),
        dps=str(cfg.arm.dps),
        aux=str(cfg.train.ft_aux if stage == 2 else cfg.train.aux),
        aux_w=float(cfg.train.aux_w),
        es_draws=int(cfg.train.es_draws),
        vg_p=float(cfg.train.vg_p),
        vg_lags=tuple(int(x) for x in cfg.train.vg_lags),
        vg_clamp=bool(cfg.train.vg_clamp),
        fm_loss=str(cfg.arm.fm_loss),
        fm_steps=int(cfg.arm.fm_steps),
        fm_shift=float(cfg.arm.fm_shift),
        fm_train_steps=int(cfg.arm.fm_train_steps),
        dps_step=float(cfg.arm.dps_step),
        dps_iters=int(cfg.arm.dps_iters),
        noise=str(cfg.train.noise),
    )
    gen = bool(cfg.train.context_generated)
    assert not gen or (arm.name in ("recursive", "single_step") and face == "disagg"), (
        "generated context: disaggregation, recursive or single-step only"
    )
    arm.gen_c = C if gen else 0
    gen_ops = arm.ops if arm.name == "recursive" else [arm.op]
    assert cfg.arm.level is None or arm.name == "recursive", "arm.level: recursive only"
    assert stage == 1 or arm.name == "recursive", "stage 2: recursive only"
    statics = system.statics()
    static_maps = None if statics is None else torch.from_numpy(statics)
    k_neighbors = sum(int(m) for m in cfg.system.neighbors)
    n_obs = arm.op.apply(torch.zeros(1, fine_len(face, L), 1, *system.spatial)).shape[1]
    surface_channels = c_surface(
        C,
        n_obs,
        k_neighbors,
        0 if static_maps is None else static_maps.shape[0],
        len(gen_ops),
        gen,
        bool(cfg.train.context_obs),
    )
    n_cond = len(system.cond("train", system.trajectories("train")[0], 0))
    if cfg.train.var_weights and system.var_weights() is not None:
        arm.vw = torch.as_tensor(system.var_weights())
    bkw = dict(mdi="mdi" in arm.noise, **dict(cfg.backbone.args))
    nets = arm.nets(
        BACKBONES[str(cfg.backbone.name)],
        C,
        surface_channels,
        L,
        tuple(system.spatial),
        tuple(system.periodic),
        n_cond,
        bkw,
    )
    spec = {
        "C": C,
        "L": L,
        "Lf": fine_len(face, L),
        "face": face,
        "op": arm.op,
        "ops": gen_ops,
        "statics": static_maps,
        "n_cond": n_cond,
        "c_surface": surface_channels,
        "generated": gen,
        "observed": bool(cfg.train.context_observed),
        "own": bool(cfg.train.context_obs),
        "var_weights": system.var_weights(),
    }
    return arm, nets, spec


def make_surface(batch, spec, device, train: bool, drop: float, g, prev_fine=None):
    """Context channels of a batch, in the order of `pack_surface`.

    Args:
        batch: A `Windows` batch on `device`.
        spec: From `build`.
        device: Device of the statics.
        train: Apply context dropout with probability `drop`.
        drop: Context dropout probability.
        g: Torch generator of the dropout.
        prev_fine: (B, Lf, C, *S) previous window to use as generated context; the
            true one, ``batch["before"][:, 0]``, when `None`.

    Returns:
        (B, c_surface, *S).
    """
    before, after, flags = batch["before"], batch["after"], batch["flags"]
    obs_ctx, obs_flags = observed(spec["op"], before, after, flags)
    if not spec["observed"]:
        obs_ctx, obs_flags = torch.zeros_like(obs_ctx), torch.zeros_like(obs_flags)
    if train and drop > 0:
        groups = obs_ctx.reshape(
            obs_ctx.shape[0], obs_flags.shape[1], -1, *obs_ctx.shape[2:]
        )
        groups, obs_flags = drop_groups(groups, obs_flags, drop, g)
        obs_ctx = groups.flatten(1, 2)
    gen_ctx = gen_flag = None
    if spec["generated"]:
        prev = before[:, 0] if prev_fine is None else prev_fine
        gen_ctx = generated(spec["ops"], prev)
        gen_flag = flags[:, 0]  # a previous window exists or not, generated or true
        if train and drop > 0:
            gen_ctx, gen_flag = drop_groups(
                gen_ctx[:, None], gen_flag[:, None], drop, g
            )
            gen_ctx, gen_flag = gen_ctx[:, 0], gen_flag[:, 0]
    static_maps = None if spec["statics"] is None else spec["statics"].to(device)
    own = spec["op"].apply(batch["fine"]).flatten(1, 2) if spec["own"] else None
    return pack_surface(own, static_maps, obs_ctx, obs_flags, gen_ctx, gen_flag)


def stage_loss(cfg, arm, nets, batch, spec, dev, train: bool, g):
    """Loss of one batch for the current training stage.

    Args:
        train: Training mode, with context dropout.

    Returns:
        Scalar loss.
    """
    fine, cond = batch["fine"], batch["cond"]
    drop = float(cfg.train.drop)
    if int(cfg.train.stage) == 1:
        surface = make_surface(batch, spec, dev, train, drop, g)
        if arm.name == "recursive":
            levels = None if cfg.arm.level is None else {int(cfg.arm.level)}
            return arm.loss(nets, fine, surface, cond, g, levels=levels)
        return arm.loss(nets, fine, surface, cond, g)
    # stage 2: roll the chain end to end; the generated context is the member's own
    # previous window, rolled out from that window's observation with the current
    # window as its after-context
    obs = spec["op"].apply(fine)
    surface_prev = None
    if spec["generated"]:
        assert batch["before"].shape[1] == 1, (
            "stage 2 with generated context needs neighbors=[1,1]"
        )
        prev_batch = {
            "before": torch.zeros_like(batch["before"]),
            "after": fine[:, None],
            "flags": torch.stack(
                [
                    torch.zeros_like(batch["flags"][:, 0]),
                    torch.ones_like(batch["flags"][:, 0]),
                ],
                1,
            ),
        }
        surface_prev = make_surface(prev_batch, spec, dev, False, 0.0, g)
    draws = []
    for _ in range(int(cfg.train.es_draws)):
        prev = None
        if spec["generated"]:
            with torch.no_grad():
                prev = arm.rollout(
                    nets,
                    spec["op"].apply(batch["before"][:, 0]),
                    surface_prev,
                    cond,
                    g,
                )
        surface = make_surface(batch, spec, dev, train, drop, g, prev_fine=prev)
        draws.append(arm.rollout(nets, obs, surface, cond, g))
    return arm._score(draws, fine, level=-1)


def _save_training(state, nets, arm, g, path: Path, level=None):
    """Save `nets` (or level `level` alone) with everything needed to resume or
    extend the training.
    """
    save_checkpoint(
        path,
        nets,
        {
            "it": state["it"],
            "opt": state["opt"].state_dict(),
            "sched": None if state["sched"] is None else state["sched"].state_dict(),
            "losses_tail": state["losses"][-200:],
            "history": state["history"],
            "torch_rng": torch.get_rng_state(),
            "gen": g.get_state(),
            "aux_weights": dict(arm.aux_weights),
            "sig": state["sig"],
            "best_val": state["best_val"],
            "best_it": state["best_it"],
            "bad": state["bad"],
        },
        level,
    )


def train_steps(
    cfg,
    arm,
    nets,
    params,
    dl,
    val_dl,
    spec,
    dev,
    g,
    state,
    ckpt_path,
    best_file,
    steps,
    tb=None,
    level=None,
):
    """Run the optimizer from ``state["it"]`` to step `steps`, or until early
    stopping.

    Accumulates gradients over ``train.accum`` batches, clips them, validates every
    ``train.val_every`` steps (at the end of each pass over `dl` when it is 0) and
    checkpoints every ``train.ckpt_every`` steps to `ckpt_path`. A validation loss
    below the best by more than ``train.min_delta`` writes the weights to `best_file`;
    after ``train.patience`` validations in a row without one, training stops
    (``train.patience=0`` never stops).

    Args:
        params: The parameters being trained, all of `nets` or one level's.
        state: Mutated in place, with keys ``it``, ``losses``, ``history``, ``opt``,
            ``sched``, ``sig``, ``best_val``, ``best_it``, ``bad`` and, after an
            early stop, ``stopped_at``.
        steps: The last step of the stage.
        tb: TensorBoard writer of the losses and learning rate, or `None`.
        level: The level being trained, whose weights alone are checkpointed.

    Returns:
        The number of steps of the stage.
    """
    opt, sched = state["opt"], state["sched"]
    accum, micro, acc = int(cfg.train.accum), 0, 0.0
    patience, min_delta = int(cfg.train.patience), float(cfg.train.min_delta)
    val_every = int(cfg.train.val_every)
    ws, main = dist.world_size(), dist.is_main()
    sampler = getattr(dl, "sampler", None)
    sharded = isinstance(sampler, DistributedSampler)
    # the pass a resumed run was in, for the sampler's shuffle
    passes = state["it"] * accum // len(dl) if sharded else 0
    t0 = time.time()

    bar = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss]}"),
        disable=not main,
    )
    task = bar.add_task("train", total=steps, completed=state["it"], loss="")

    def validate():
        nets.eval()
        reduce, arm.reduce = arm.reduce, (lambda t: t)  # rank 0 validates alone
        gv = torch.Generator(device=dev).manual_seed(int(cfg.train.val_seed))
        tot, n = 0.0, 0
        vtask = bar.add_task("val", total=len(val_dl), loss="")
        with torch.no_grad():
            for vb in val_dl:
                tot += float(
                    stage_loss(cfg, arm, nets, batch_to(vb, dev), spec, dev, False, gv)
                )
                n += 1
                bar.advance(vtask)
        bar.remove_task(vtask)
        arm.reduce = reduce
        nets.train()
        return tot / max(n, 1)

    def check(it):
        """Validate after step `it`, track the best, and say whether to stop."""
        vl, train_loss = validate(), state["losses"][-1]
        state["history"].append({"it": it, "train": train_loss, "val": vl})
        if tb is not None:
            tb.add_scalar("loss/val", vl, it)
        if vl < state["best_val"] - min_delta:
            state.update(best_val=vl, best_it=it, bad=0)
            save_checkpoint(best_file, nets, {"it": it, "val": vl}, level)
        else:
            state["bad"] += 1
        log.info(
            "it %d train %.4f val %.4f best %.4f @%d (%.0fs)",
            it,
            train_loss,
            vl,
            state["best_val"],
            state["best_it"],
            time.time() - t0,
        )
        if patience and state["bad"] >= patience:
            state["stopped_at"] = it
            log.info(
                "early stop at it %d: no val gain in %d checks, best %.4f @%d",
                it,
                patience,
                state["best_val"],
                state["best_it"],
            )
            return True
        return False

    with bar:
        while state["it"] < steps:
            if sharded:
                sampler.set_epoch(passes)
            passes += 1
            checked = -1
            for batch in dl:
                it = state["it"]
                if it >= steps:
                    break
                bf16 = bool(cfg.train.bf16) and dev.startswith("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    loss = stage_loss(
                        cfg, arm, nets, batch_to(batch, dev), spec, dev, True, g
                    )
                (loss / accum).backward()
                acc += float(loss.detach()) / accum
                micro += 1
                if micro % accum:
                    continue
                if ws > 1:
                    grads = [p.grad for p in params if p.grad is not None]
                    flat = dist.all_reduce_mean(_flatten_dense_tensors(grads))
                    for gr, r in zip(grads, _unflatten_dense_tensors(flat, grads)):
                        gr.copy_(r)
                    acc = float(dist.all_reduce_mean(torch.tensor(acc, device=dev)))
                torch.nn.utils.clip_grad_norm_(params, float(cfg.train.clip))
                opt.step()
                if sched is not None:
                    sched.step()
                opt.zero_grad()
                state["losses"].append(acc)
                acc = 0.0
                train_loss = state["losses"][-1]
                bar.update(task, advance=1, loss=f"loss {train_loss:.4f}")
                if tb is not None:
                    tb.add_scalar("loss/train", train_loss, it)
                    tb.add_scalar("lr", opt.param_groups[0]["lr"], it)
                    if arm.terms is not None:  # this rank's last scored batch
                        tb.add_scalar("loss/es", float(arm.terms[0]), it)
                        tb.add_scalar("loss/aux", float(arm.terms[1]), it)
                state["it"] = it + 1
                if (val_every and it % val_every == 0) or it + 1 == steps:
                    checked = it
                    # rank 0 decides; every rank leaves on the same step
                    if dist.broadcast_flag(main and check(it)):
                        return steps
                ckpt = state["it"] % int(cfg.train.ckpt_every) == 0
                if main and ckpt and state["it"] < steps:
                    _save_training(state, nets, arm, g, ckpt_path, level)
            # the end of a pass over the training windows
            if (
                not val_every
                and state["it"] - 1 != checked
                and dist.broadcast_flag(main and check(state["it"] - 1))
            ):
                return steps
    return steps


def train(cfg) -> Path:
    """Train the checkpoint of `cfg`, or one of its levels, and return its directory.

    Skips a finished training, resumes an interrupted one from its checkpoint, and
    loads the stage-1 weights for stage 2. Writes ``checkpoint.pt`` (last weights),
    ``best.pt`` (lowest validation loss) and ``fit.json``, or their ``_level<k>``
    versions for a single level. Stage 1 of the recursive arm without ``arm.level``
    trains every level in turn, each as its own single-level job.

    Under ``torchrun``, every rank calls it: each trains on its share of
    ``train.batch``, the gradients are averaged over the ranks, and rank 0 alone
    validates and writes.

    Raises:
        ValueError: ``train.batch`` is not divisible by the number of ranks.
        AssertionError: SimVP on several ranks.
    """
    t0 = time.time()
    ws, main = dist.world_size(), dist.is_main()
    if not main:
        log.setLevel(logging.WARNING)
    assert ws == 1 or str(cfg.arm.name) != "SimVP", (
        "SimVP: BatchNorm and DropPath are not synchronized across ranks"
    )
    batch = dist.per_rank_batch(int(cfg.train.batch))
    seed, stage = int(cfg.seed), int(cfg.train.stage)
    face, L = str(cfg.system.face), int(cfg.system.L)
    tag = checkpoint_tag(cfg)
    ckpt_dir = checkpoint_dir(Path(cfg.out), str(cfg.system.name), face, tag, seed)
    if main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    level = None if cfg.arm.level is None else int(cfg.arm.level)
    suffix = "" if level is None else f"_level{level}"
    fit_file, ckpt_path = fit_path(ckpt_dir, level), checkpoint_path(ckpt_dir, level)
    best_file = best_path(ckpt_dir, level)
    dist.barrier()
    if fit_file.exists():
        log.info("%s already trained", ckpt_dir)
        return ckpt_dir
    if str(cfg.arm.name) == "recursive" and stage == 1 and level is None:
        # one independent training per level; the last one merges the fits
        for k in range(len(cfg.arm.chain or chain_factors(L))):
            c = copy.deepcopy(cfg)
            c.arm.level = k
            train(c)
        return ckpt_dir
    device = device_of(cfg)
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed + dist.rank())
    src = checkpoint_dir(
        Path(cfg.out), str(cfg.system.name), face, tag.replace("-ft-", "-"), seed
    )
    stats = read_stats(src) if stage == 2 or stats_path(src).exists() else None
    dist.barrier()  # every rank has looked for stats.json before rank 0 writes it
    system = build_system(cfg, stats)
    if main and not stats_path(ckpt_dir).exists():
        write_stats(ckpt_dir, system.stats)
    arm, nets, spec = build(cfg, system)
    if ws > 1:
        arm.reduce = dist.all_reduce_mean
    nets.to(device)
    if stage == 2:
        load_weights(src, nets, map_location=device)
    neighbors = tuple(int(m) for m in cfg.system.neighbors)
    ds = Windows(system, "train", L, face, neighbors, sliding=True, seed=seed)
    if int(cfg.train.windows):
        n_tr = min(int(cfg.train.windows), len(ds))
        t_idx = np.random.default_rng(seed).choice(len(ds), size=n_tr, replace=False)
        ds = torch.utils.data.Subset(ds, t_idx.tolist())
    val_ds = Windows(system, "val", L, face, neighbors, sliding=True)
    workers = int(cfg.train.workers)
    sampler = (
        DistributedSampler(ds, shuffle=True, seed=seed, drop_last=True)
        if ws > 1
        else None
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    n_val = min(int(cfg.train.val_windows), len(val_ds))
    v_idx = np.random.default_rng(0).choice(len(val_ds), size=n_val, replace=False)
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_ds, v_idx.tolist()),
        batch_size=int(cfg.train.batch),
        num_workers=min(4, workers),
    )
    params = list(nets.parameters() if level is None else nets[level].parameters())
    if stage == 2:
        steps = int(cfg.train.ft_steps)
    elif int(cfg.train.epochs):
        steps = int(cfg.train.epochs) * len(dl)
    else:
        steps = int(cfg.train.steps)
    opt = torch.optim.AdamW(
        params, lr=float(cfg.train.lr if stage == 1 else cfg.train.ft_lr)
    )
    sched = None
    if str(cfg.train.sched) == "cosine":
        wu = max(1, int(cfg.train.warmup))

        def lr_at(s):
            if s < wu:
                return (s + 1) / wu
            return 0.5 * (
                1 + math.cos(math.pi * min(1.0, (s - wu) / max(1, steps - wu)))
            )

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    sig = {
        "tag": tag,
        "face": face,
        "seed": seed,
        "steps": steps,
        "stage": stage,
        "level": level,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    sig["cfg"]["system"].pop("stats_path", None)
    state = {
        "it": 0,
        "losses": [],
        "history": [],
        "opt": opt,
        "sched": sched,
        "sig": sig,
        "best_val": math.inf,
        "best_it": None,
        "bad": 0,
    }
    resumed_at = None
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        nets.load_state_dict(ckpt["nets"], strict=level is None)
        train_state = ckpt["train"]
        assert train_state["sig"]["cfg"] == sig["cfg"], (
            f"{ckpt_path} was written for a different cell"
        )
        opt.load_state_dict(train_state["opt"])
        if sched is not None:
            sched.load_state_dict(train_state["sched"])
        state.update(
            it=int(train_state["it"]),
            losses=list(train_state["losses_tail"]),
            history=list(train_state["history"]),
            best_val=float(train_state["best_val"]),
            best_it=train_state["best_it"],
            bad=int(train_state["bad"]),
        )
        arm.aux_weights = {
            int(k): float(v) for k, v in train_state["aux_weights"].items()
        }
        resumed_at = state["it"]
        if main:
            torch.set_rng_state(train_state["torch_rng"].cpu())
            g.set_state(train_state["gen"].cpu())
        else:  # the other ranks' noise is not checkpointed
            g.manual_seed(seed + dist.rank() + resumed_at)
        log.info("resumed %s at it %d", ckpt_dir, resumed_at)
    log.info(
        "%s %s %s s%d: %d windows, %.2fM params, device=%s",
        system.name,
        face,
        tag,
        seed,
        len(ds),
        sum(p.numel() for p in params) / 1e6,
        device,
    )
    nets.train()
    # a resumed run drops the events logged after its checkpoint
    writer = (
        SummaryWriter(ckpt_dir / f"tb{suffix}", purge_step=resumed_at)
        if main
        else contextlib.nullcontext()
    )
    with writer as tb:
        train_steps(
            cfg,
            arm,
            nets,
            params,
            dl,
            val_dl,
            spec,
            device,
            g,
            state,
            ckpt_path,
            best_file,
            steps,
            tb=tb,
            level=level,
        )
    fit = {
        "tag": tag,
        "arm": arm.name,
        "face": face,
        "seed": seed,
        "stage": stage,
        "steps": steps,
        "level": level,
        "n_params": sum(p.numel() for p in params),
        "aux_weights": {str(k): v for k, v in arm.aux_weights.items()},
        "history": state["history"],
        "val_final": state["history"][-1]["val"] if state["history"] else None,
        "best_val": state["best_val"],
        "best_it": state["best_it"],
        "stopped_at": state.get("stopped_at"),
        "resumed_at": resumed_at,
        "world_size": ws,
        "wall_s": time.time() - t0,
        "cfg": sig["cfg"],
    }
    if main:
        _save_training(state, nets, arm, g, ckpt_path, level)
        write_fit(fit_file, fit)
    # the last level job to finish writes the checkpoint's own fit.json
    if (
        main
        and level is not None
        and all(fit_path(ckpt_dir, k).exists() for k in range(len(nets)))
    ):
        per_level = [
            json.loads(fit_path(ckpt_dir, k).read_text()) for k in range(len(nets))
        ]
        keep = ("n_params", "best_val", "best_it", "stopped_at", "wall_s")
        merged = {
            **fit,
            "level": None,
            "merged_from": len(nets),
            "n_params": sum(f["n_params"] for f in per_level),
            "levels": [{k: f[k] for k in keep} for f in per_level],
        }
        write_fit(fit_path(ckpt_dir), merged)
    dist.barrier()  # the files are on disk before any rank reads them
    log.info("wrote %s (%.0fs)", ckpt_dir, time.time() - t0)
    return ckpt_dir


@hydra.main(config_path=CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """``mundo-train``: train the model set by the Hydra config, on N devices under
    ``torchrun --nproc_per_node=N``.
    """
    if not dist.is_main():  # one train.log per run, rank 0's
        root = logging.getLogger()
        for h in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
            root.removeHandler(h)
            h.close()
    dist.init(device_of(cfg))
    try:
        train(cfg)
    finally:
        dist.finish()


if __name__ == "__main__":
    main()
