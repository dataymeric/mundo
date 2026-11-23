"""Windows of a `System` as a torch `Dataset`."""

import numpy as np
import torch

from mundo.operator import fine_len


class Windows(torch.utils.data.Dataset):
    """Windows of `L` steps of a `System`, each with its neighboring windows.

    Neighbor `j` starts ``j * L`` steps before or after the window. A neighbor outside
    the trajectory is all zeros, with flag 0. Neighbors are returned as fine windows;
    `context.observed` applies the operator to them.

    An item is a dict of tensors:

    - ``fine``: (Lf, C, *S) the window, ``Lf = fine_len(face, L)``.
    - ``before``, ``after``: (n_before, Lf, C, *S) and (n_after, Lf, C, *S),
      nearest first.
    - ``flags``: (n_before + n_after,) 1 where the neighbor exists, before then after.
    - ``cond``: (d,) scalar conditioning of the window.
    - ``traj``, ``t0``: the trajectory and first frame of the window.

    Args:
        system: The `System` to read.
        split: ``"train"``, ``"val"`` or ``"test"``.
        L: Window length in steps.
        face: ``"disagg"`` or ``"interp"``.
        neighbors: (n_before, n_after).
        sliding: Start a window at every frame rather than every `L` frames. In
            training, periodic axes are also rolled randomly, unless the system sets
            ``roll = False``.
        seed: Seed of the roll.
    """

    def __init__(
        self,
        system,
        split: str,
        L: int,
        face: str,
        neighbors=(1, 1),
        sliding=True,
        seed=0,
    ):
        self.system, self.split, self.L, self.face = system, split, L, face
        self.Lf = fine_len(face, L)
        self.n_before, self.n_after = neighbors
        self.roll = sliding and split == "train" and getattr(system, "roll", True)
        self._rng = np.random.default_rng(seed)
        step = 1 if sliding else L
        self.index = [
            (traj, t0)
            for traj in system.trajectories(split)
            for t0 in range(0, system.n_frames(split, traj) - self.Lf + 1, step)
        ]

    def __len__(self):
        return len(self.index)

    def _read(self, traj, t0, T):
        if 0 <= t0 and t0 + self.Lf <= T:
            return self.system.read(self.split, traj, t0, self.Lf), 1.0
        shape = (self.Lf, len(self.system.vars), *self.system.spatial)
        return np.zeros(shape, np.float32), 0.0

    def __getitem__(self, i):
        traj, t0 = self.index[i]
        T = self.system.n_frames(self.split, traj)
        fine, _ = self._read(traj, t0, T)
        before = [
            self._read(traj, t0 - j * self.L, T) for j in range(1, self.n_before + 1)
        ]
        after = [
            self._read(traj, t0 + j * self.L, T) for j in range(1, self.n_after + 1)
        ]

        def stack(ws):
            return (
                np.stack([w for w, _ in ws])
                if ws
                else np.zeros((0, *fine.shape), np.float32)
            )

        item = {
            "fine": fine,
            "before": stack(before),
            "after": stack(after),
            "flags": np.array(
                [f for _, f in before] + [f for _, f in after], np.float32
            ),
            "cond": self.system.cond(self.split, traj, t0),
            "traj": np.int64(traj),
            "t0": np.int64(t0),
        }
        if self.roll:
            shifts = [
                int(self._rng.integers(n)) if p else 0
                for n, p in zip(self.system.spatial, self.system.periodic)
            ]
            for k in ("fine", "before", "after"):
                axes = tuple(range(item[k].ndim - len(shifts), item[k].ndim))
                item[k] = np.roll(item[k], shifts, axis=axes)
        return {k: torch.as_tensor(v) for k, v in item.items()}
