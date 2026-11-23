"""Two-scale Lorenz-96 (Lorenz, 2006).

One netCDF4 archive per forcing F, at ``root/L96_K<K>_J<J>_F<F>_TU1000.nc``, read
through `h5py`. Its dataset ``X`` holds K slow variables at every integration
step, (n_steps, K), and the J fast variables per slow variable are not read.

The archive is a single long trajectory: frames are taken every `stride` integration
steps and split in time into contiguous train, val and test blocks by `split_frac`.
Each frame is one channel ``X`` on a periodic ring of K points, so the field is (1, K).
Frames are standardized with one mean and std fitted on evenly spaced train frames.
There are no static maps and no scalar conditioning.
"""

from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "val", "test")


class L96:
    name = "l96"
    vars = ("X",)
    roll = False  # no random circular shift of training windows along the ring

    def __init__(self, root, forcing, stride, split_frac, n_stats_frames, stats=None):
        self.root, self.forcing, self.stride = Path(root), int(forcing), int(stride)
        paths = sorted(self.root.glob(f"L96_K*_J*_F{self.forcing}_TU1000.nc"))
        assert len(paths) == 1, (
            f"one archive for F={self.forcing} under {self.root}, found {paths}"
        )
        self.path = paths[0]
        with h5py.File(self.path, "r") as h:
            n_steps, K = h["X"].shape
        self.spatial, self.periodic = (K,), (True,)
        n = (n_steps - 1) // self.stride + 1
        n_train, n_val = int(n * float(split_frac[0])), int(n * float(split_frac[1]))
        self._range = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n),
        }
        self._n_stats_frames = int(n_stats_frames)
        self._handle = None
        self.stats = stats

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def _raw(self, file, frame0: int, n: int) -> np.ndarray:
        """(n, 1, K) frames every `stride` steps from absolute frame index `frame0`."""
        a = file["X"][
            frame0 * self.stride : (frame0 + n - 1) * self.stride + 1 : self.stride
        ]
        return a.astype(np.float32)[:, None]

    # stats
    def fitted_on(self) -> dict:
        return {"forcing": self.forcing, "stride": self.stride}

    def fit_stats(self) -> dict:
        """Mean and std of ``X`` over `n_stats_frames` frames spread over train."""
        lo, hi = self._range["train"]
        picks = np.linspace(lo, hi - 1, self._n_stats_frames, dtype=int)
        with h5py.File(self.path, "r") as h:
            a = np.stack([self._raw(h, int(f), 1)[0, 0] for f in picks]).astype(
                np.float64
            )
        return {
            **self.fitted_on(),
            "mean": [float(a.mean())],
            "std": [float(a.std() + 1e-9)],
        }

    # System
    def trajectories(self, split):
        return [0]

    def n_frames(self, split, traj):
        lo, hi = self._range[split]
        return hi - lo

    def read(self, split, traj, t0, n):
        lo, hi = self._range[split]
        assert t0 >= 0 and t0 + n <= hi - lo, (split, t0, n)
        assert self.stats is not None, "no normalization stats: set system.stats_path"
        s = self.stats
        return (
            self._raw(self._h5(), lo + t0, n) - np.float32(s["mean"][0])
        ) / np.float32(s["std"][0])

    def statics(self):
        return None

    def cond(self, split, traj, t0):
        return np.zeros(0, np.float32)

    def var_weights(self):
        return None

    # the open h5 handle is not shared with DataLoader workers
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_handle"}

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._handle = None
