"""Checkpointing.

Save a model periodically during training. Each run writes to
``root/<system>/<face>/<tag>/seed<N>/``:

- ``checkpoint.pt``: weights and training state, saved every ``train.ckpt_every`` steps.
- ``best.pt``: weights at the best validation loss.
- ``fit.json``: loss history and the config of the networks.
- ``stats.json``: standardization mean and std, ``null`` if the system has none.
"""

import json
import os
from pathlib import Path

import torch
from torch import nn


def checkpoint_dir(root, system: str, face: str, tag: str, seed: int) -> Path:
    return Path(root) / system / face / tag / f"seed{seed}"


def checkpoint_path(ckpt_dir, level: int | None = None) -> Path:
    name = "checkpoint.pt" if level is None else f"checkpoint_level{level}.pt"
    return Path(ckpt_dir) / name


def best_path(ckpt_dir, level: int | None = None) -> Path:
    name = "best.pt" if level is None else f"best_level{level}.pt"
    return Path(ckpt_dir) / name


def fit_path(ckpt_dir, level: int | None = None) -> Path:
    name = "fit.json" if level is None else f"fit_level{level}.json"
    return Path(ckpt_dir) / name


def write_fit(path: Path, fit: dict) -> None:
    Path(path).write_text(json.dumps(fit, indent=1))


def read_fit(ckpt_dir) -> dict:
    return json.loads(fit_path(ckpt_dir).read_text())


def stats_path(ckpt_dir) -> Path:
    return Path(ckpt_dir) / "stats.json"


def write_stats(ckpt_dir, stats: dict | None) -> None:
    p = stats_path(ckpt_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")  # level jobs may write it at once
    tmp.write_text(json.dumps(stats, indent=1))
    tmp.replace(p)


def read_stats(ckpt_dir) -> dict | None:
    return json.loads(stats_path(ckpt_dir).read_text())


def trained_tags(root, system: str, face: str) -> list[str]:
    fits = (Path(root) / system / face).glob("*/seed*/fit.json")
    return sorted({p.parent.parent.name for p in fits})


def save_checkpoint(
    path: Path, nets: nn.ModuleList, train: dict, level: int | None = None
) -> None:
    weights = nets.state_dict()
    if level is not None:
        weights = {k: v for k, v in weights.items() if k.startswith(f"{level}.")}
    tmp = Path(path).with_suffix(".tmp")
    torch.save({"nets": weights, "train": train}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_weights(
    ckpt_dir, nets: nn.ModuleList, map_location="cpu", last: bool = False
) -> None:
    """Load the trained weights of a checkpoint.

    Args:
        ckpt_dir: The directory containing the checkpoint files.
        nets: A list of neural network modules to load the weights into.
        map_location: The device to map the loaded weights to. Defaults to ``"cpu"``.
        last: If `True`, load the last checkpoint. Otherwise, load the best checkpoint.

    Raises:
        FileNotFoundError: No whole-checkpoint file and a level without any file.
    """
    first, second = (
        (checkpoint_path, best_path) if last else (best_path, checkpoint_path)
    )
    for full in (first(ckpt_dir), second(ckpt_dir)):
        if full.exists():
            nets.load_state_dict(load_checkpoint(full, map_location)["nets"])
            return
    levels = [
        first(ckpt_dir, k) if first(ckpt_dir, k).exists() else second(ckpt_dir, k)
        for k in range(len(nets))
    ]
    missing = [p.name for p in levels if not p.exists()]
    if missing:
        raise FileNotFoundError(f"no checkpoint under {ckpt_dir}, missing {missing}")
    for k, (net, p) in enumerate(zip(nets, levels)):
        state = load_checkpoint(p, map_location)["nets"]
        net.load_state_dict(
            {n[len(f"{k}.") :]: v for n, v in state.items() if n.startswith(f"{k}.")}
        )
