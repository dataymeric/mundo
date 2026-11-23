"""Typed config."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class SystemConf:
    name: str = "synthetic"
    face: str = "disagg"  # disagg | interp
    L: int = 8
    neighbors: list[int] = field(default_factory=lambda: [1, 1])
    args: dict[str, Any] = field(default_factory=dict)  # kwargs of the system class
    stats_path: str | None = None


@dataclass
class ArmConf:
    name: str = "recursive"  # recursive | single_step | FlowMatching | FlowDPS | SimVP
    chain: list[int] | None = None  # overrides chain_factors(L)
    level: int | None = None  # train only this level (recursive)
    project: str = "sample"  # single_step: loss | sample
    dps: str = "blend"  # FlowDPS: blend | grad
    fm_loss: str = MISSING  # flow training recipe: cfm | sd3
    fm_steps: int = MISSING  # flow solver steps
    fm_shift: float = MISSING  # flow sampling timestep shift
    fm_train_steps: int = MISSING  # flow training noise levels
    dps_step: float = MISSING  # FlowDPS gradient step size
    dps_iters: int = MISSING  # FlowDPS gradient steps per solver step


@dataclass
class BackboneConf:
    name: str = "conv"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConf:
    stage: int = 1
    steps: int = MISSING  # optimizer steps per network, unless `epochs` > 0
    epochs: int = 0  # stage 1 epochs; 0: use `steps`
    windows: int = 0  # random subset of training windows; 0: all
    batch: int = MISSING
    accum: int = 1
    workers: int = 8
    lr: float = MISSING
    sched: str = MISSING  # learning rate schedule: const | cosine
    warmup: int = MISSING  # learning rate warmup steps
    bf16: bool = False  # bfloat16 precision
    noise: str = MISSING  # ``+``-joined tokens: none | input | lg | mdi; see `Arm`
    es_draws: int = MISSING  # draws per window in the energy score
    aux: str = MISSING  # none | variogram | spectral
    aux_w: float = MISSING  # aux loss weight; < 0: relative, see `resolve_aux_weight`
    vg_p: float = MISSING  # variogram order
    vg_lags: list[int] = MISSING  # variogram lags in the training loss
    vg_clamp: bool = False  # floor each variogram lag term at 0
    clip: float = MISSING  # gradient-norm clip
    val_seed: int = MISSING  # noise seed of the validation draws
    var_weights: bool = False  # scale the loss per variable
    context_obs: bool = False  # window's own observation as context
    context_observed: bool = True  # neighboring windows' observations as context
    context_generated: bool = False  # previous window's generated members as context
    drop: float = MISSING  # context dropout
    val_every: int = MISSING  # steps between validations; 0: every epoch
    val_windows: int = MISSING
    patience: int = MISSING  # early-stopping: validations without a new best; 0: off
    min_delta: float = MISSING  # early-stopping improvement threshold
    ckpt_every: int = MISSING  # steps between checkpoints
    # stage 2: end-to-end fine-tuning of a recursive chain
    ft_steps: int = MISSING  # stage 2 optimizer steps
    ft_lr: float = MISSING  # stage 2 learning rate
    ft_aux: str = MISSING  # stage 2 auxiliary loss


@dataclass
class EvalConf:
    m_gen: int = MISSING  # ensemble members per window
    m_batch: int = MISSING  # members per forward pass; 0: all at once
    windows: str = MISSING  # how test windows are scored: spaced | chained
    max_windows: int = MISSING  # spaced: number of test windows, evenly spaced
    chains: int = MISSING  # chained: number of chains; 0: all
    chain_len: int = MISSING  # number of test windows per chain; 0: a whole trajectory
    seeds: list[int] = field(default_factory=lambda: [0])
    arms: list[str] = field(default_factory=list)  # tags; empty: every checkpoint found
    baselines: list[str] = field(
        default_factory=lambda: ["linear", "spline", "denton_cholette", "cdf_t"]
    )
    context: str = "on"  # observed context at eval: on | off
    weights: str = "best"  # best | last: the lowest-validation or the final weights
    cdft_windows: int = MISSING  # training windows giving the CDF-t truth distribution
    cdft_per_step: bool = False  # fit CDF-t per step of the window
    vg_space: list[int] = MISSING  # spatial lags of the variogram ratios
    vg_time: list[int] = MISSING  # time lags of the variogram ratios
    vg_p: float = MISSING  # variogram order
    cov_levels: list[float] = MISSING  # nominal levels of the central intervals
    spec_members: int = MISSING  # members per window used for the spectrum
    seed: int = MISSING  # noise seed of the sampling
    save_members: str | None = None  # dir to save generated members


@dataclass
class LaunchConf:
    device: str = "auto"


@dataclass
class Config:
    system: SystemConf = field(default_factory=SystemConf)
    arm: ArmConf = field(default_factory=ArmConf)
    backbone: BackboneConf = field(default_factory=BackboneConf)
    train: TrainConf = field(default_factory=TrainConf)
    eval: EvalConf = field(default_factory=EvalConf)
    launch: LaunchConf = field(default_factory=LaunchConf)
    seed: int = 0
    out: str = MISSING


CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


def _range(a: int, b: int) -> list[int]:
    """``${range:a,b}`` in the yamls: the integers from a to b - 1."""
    return list(range(int(a), int(b)))


ConfigStore.instance().store(name="base_config", node=Config)
OmegaConf.register_resolver("range", _range, replace=True)
OmegaConf.register_resolver("conf", lambda p: str(Path(CONF_DIR) / p), replace=True)


def checkpoint_tag(cfg) -> str:
    """Tag of the checkpoint of `cfg`.

    ``<arm>[-proj][-grad][-m<M>][-<aux>][-vw][-gen][-noctx][-c<chain>][-ft]-<backbone>``
    """
    a, t = cfg.arm, cfg.train
    parts = [a.name]
    if a.name == "single_step" and a.project == "loss":
        parts.append("proj")
    if a.name == "FlowDPS" and a.dps != "blend":
        parts.append(a.dps)
    if a.name in ("recursive", "single_step"):
        if t.es_draws > 2:
            parts.append(f"m{t.es_draws}")
        if t.aux != "none":
            parts.append(t.aux)
    if t.var_weights:
        parts.append("vw")
    if t.context_generated:
        parts.append("gen")
    if not t.context_observed:
        parts.append("noctx")
    if a.chain:
        parts.append("c" + "".join(str(x) for x in a.chain))
    if t.stage == 2:
        parts.append("ft")
    return "-".join(parts + [cfg.backbone.name])
