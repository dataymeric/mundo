"""AutoSim 2D simulations (Greenbury et al., 2026).

Files live at ``root/<dataset>/data/<split>/*.hdf5``, with split one of ``train``,
``valid``, ``test``. Each file holds several trajectories of one simulation regime:

- ``t0_fields/<name>``: scalar field, (n_traj, T, H, W).
- ``t1_fields/<name>``: vector field, (n_traj, T, H, W, n_components), optional.

Every field becomes channels in a fixed order, t0 then t1, names sorted, a vector field
giving one channel per component (``<name>0``, ``<name>1``, ...). A trajectory is one
row of one file, and all files of a split share T. Frames are standardized per channel
with a mean and std fitted on random train frames. There are no static maps and no
scalar conditioning; Gray-Scott wraps on both axes, the others on none, unless the
config says otherwise.
"""

import re
from pathlib import Path

import h5py
import numpy as np

SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}
FIELD_GROUPS = ("t0_fields", "t1_fields")


def _regime(fname: str) -> str:
    return re.sub(r"_\d+\.hdf5$|\.hdf5$", "", fname)


def periodic_of(dataset: str, periodic=None) -> tuple[bool, bool]:
    """Which spatial axes wrap: `periodic` when given, else Gray-Scott's torus and
    bounded domains for the other systems.
    """
    if periodic is not None:
        return tuple(bool(x) for x in periodic)
    return (True, True) if dataset.startswith("gray_scott") else (False, False)


class Autosim:
    name = "autosim"

    def __init__(
        self,
        root,
        dataset: str,
        periodic=None,
        n_stats_frames: int = 64,
        regime: str | None = None,
        stats=None,
    ):
        self.root, self.dataset, self.regime = Path(root), dataset, regime
        self.periodic = periodic_of(dataset, periodic)
        self._n_stats_frames = n_stats_frames
        self._files: dict[str, list[Path]] = {}
        self._index: dict[str, list[tuple[int, int]]] = {}  # split -> [(file, traj)]
        self._T: dict[str, int] = {}
        self._handles: dict[tuple[str, int], h5py.File] = {}
        with h5py.File(self._split_files("train")[0], "r") as file:
            self._chans = self._channels(file)
            self.spatial = tuple(file[self._chans[0][0]][self._chans[0][1]].shape[2:4])
        self.vars = tuple(
            f"{n}{i}" if nc > 1 else n for _, n, nc in self._chans for i in range(nc)
        )
        self.stats = stats

    # files
    def _split_files(self, split: str) -> list[Path]:
        if split not in self._files:
            files = sorted(
                (self.root / self.dataset / "data" / SPLIT_DIR[split]).glob("*.hdf5")
            )
            if self.regime is not None:
                files = [f for f in files if _regime(f.name) == self.regime]
            if not files:
                raise FileNotFoundError(
                    f"{self.dataset}/{SPLIT_DIR[split]}/{self.regime or '*'}"
                )
            self._files[split] = files
        return self._files[split]

    @staticmethod
    def _channels(file) -> list[tuple[str, str, int]]:
        """[(group, name, n_components)] in a fixed order."""
        out = []
        for grp in FIELD_GROUPS:
            for name in sorted(file[grp]) if grp in file else ():
                ds = file[grp][name]
                out.append((grp, name, 1 if ds.ndim == 4 else ds.shape[-1]))
        return out

    def _split_index(self, split: str) -> list[tuple[int, int]]:
        if split not in self._index:
            idx = []
            for fi, f in enumerate(self._split_files(split)):
                with h5py.File(f, "r") as file:
                    n_traj, T = file[self._chans[0][0]][self._chans[0][1]].shape[:2]
                self._T.setdefault(split, T)
                assert self._T[split] == T, "files of one split must share T"
                idx += [(fi, t) for t in range(n_traj)]
            self._index[split] = idx
        return self._index[split]

    def _handle(self, split: str, fi: int) -> h5py.File:
        key = (split, fi)
        if key not in self._handles:
            self._handles[key] = h5py.File(
                self._split_files(split)[fi], "r", libver="latest"
            )
        return self._handles[key]

    def _raw(self, file, traj: int, t0: int, n: int) -> np.ndarray:
        parts = []
        for grp, name, nc in self._chans:
            a = file[grp][name][traj, t0 : t0 + n]
            parts.append(a[..., None] if nc == 1 else a)
        return np.concatenate(parts, axis=-1).transpose(0, 3, 1, 2).astype(np.float32)

    # stats
    def fitted_on(self) -> dict:
        return {"dataset": self.dataset, "regime": self.regime, "vars": list(self.vars)}

    def fit_stats(self) -> dict:
        """Per-channel mean and std over `n_stats_frames` random train frames."""
        rng = np.random.default_rng(0)
        idx = self._split_index("train")
        acc = []
        for _ in range(self._n_stats_frames):
            fi, traj = idx[int(rng.integers(len(idx)))]
            with h5py.File(self._split_files("train")[fi], "r") as file:
                fr = self._raw(file, traj, int(rng.integers(self._T["train"])), 1)[0]
            acc.append(fr.reshape(fr.shape[0], -1).astype(np.float64))
        a = np.concatenate(acc, axis=1)
        return {
            **self.fitted_on(),
            "mean": a.mean(1).tolist(),
            "std": (a.std(1) + 1e-9).tolist(),
        }

    # System
    def trajectories(self, split):
        return list(range(len(self._split_index(split))))

    def n_frames(self, split, traj):
        self._split_index(split)
        return self._T[split]

    def read(self, split, traj, t0, n):
        fi, tj = self._split_index(split)[traj]
        assert self.stats is not None, "no normalization stats: set system.stats_path"
        stats = self.stats
        mean = np.array(stats["mean"], np.float32)[None, :, None, None]
        std = np.array(stats["std"], np.float32)[None, :, None, None]
        return (self._raw(self._handle(split, fi), tj, t0, n) - mean) / std

    def statics(self):
        return None

    def cond(self, split, traj, t0):
        return np.zeros(0, np.float32)

    def var_weights(self):
        return None

    # the open h5 handles are not shared with DataLoader workers
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_handles"}

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._handles = {}
