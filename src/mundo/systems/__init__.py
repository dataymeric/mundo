import json
from pathlib import Path
from typing import Protocol

import numpy as np

from mundo.systems.autosim import Autosim
from mundo.systems.era5 import Era5
from mundo.systems.l96 import L96
from mundo.systems.synthetic import Synthetic


class System(Protocol):
    # read-only properties, so a system may set them as fixed-length tuples
    @property
    def name(self) -> str: ...

    @property
    def vars(self) -> tuple[str, ...]: ...

    @property
    def spatial(self) -> tuple[int, ...]:
        """S, the spatial shape."""

    @property
    def periodic(self) -> tuple[bool, ...]:
        """Whether each spatial axis wraps around."""

    @property
    def stats(self) -> dict | None:
        """The ``mean`` and ``std`` `read` normalizes with, or `None`."""

    def fitted_on(self) -> dict:
        """Settings the stats depend on: saved with them, checked at loading."""

    def trajectories(self, split: str) -> list[int]: ...

    def n_frames(self, split: str, traj: int) -> int: ...

    def read(self, split: str, traj: int, t0: int, n: int) -> np.ndarray:
        """(n, C, *S) float32, normalized."""

    def statics(self) -> np.ndarray | None:
        """(Cs, *S) float32, or `None`."""

    def cond(self, split: str, traj: int, t0: int) -> np.ndarray:
        """(d,) float32 scalar conditioning of frame `t0`."""

    def var_weights(self) -> np.ndarray | None:
        """(C,) loss scale per variable, or `None`."""


SYSTEMS = {"synthetic": Synthetic, "autosim": Autosim, "l96": L96, "era5": Era5}


def build_system(cfg, stats: dict | None = None) -> System:
    """Instantiate ``system.name`` with ``system.args`` as kwargs, normalized with
    `stats` when given, else with the file ``system.stats_path`` when set.

    Raises:
        FileNotFoundError: ``system.stats_path`` is set but the file does not exist.
        ValueError: The stats were fitted with other settings than this run's.
    """
    if stats is None and cfg.system.get("stats_path"):
        path = Path(cfg.system.stats_path)
        if not path.exists():
            raise FileNotFoundError(
                f"no stats at {path}: fit them with scripts/data/make_stats.py"
            )
        stats = json.loads(path.read_text())
    kw = dict(cfg.system.args or {})
    if stats is not None:
        kw["stats"] = stats
    system = SYSTEMS[str(cfg.system.name)](**kw)
    if stats is not None:
        run = system.fitted_on()
        fitted = {k: stats.get(k) for k in run}
        if fitted != run:
            diff = {k: (fitted[k], run[k]) for k in run if fitted[k] != run[k]}
            raise ValueError(
                f"stats fitted with other settings, (theirs, this run's): {diff}; "
                "fit them for this run with scripts/data/make_stats.py"
            )
    return system
