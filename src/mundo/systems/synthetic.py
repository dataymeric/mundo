"""A seeded AR(1) field with smooth spatial noise, for tests."""

import numpy as np

SPLIT_SEED = {"train": 0, "val": 1, "test": 2}


class Synthetic:
    name = "synthetic"
    stats = None

    def __init__(
        self,
        rank: int = 2,
        C: int = 2,
        n_frames: int = 192,
        size: int = 8,
        seed: int = 0,
    ):
        assert rank in (1, 2), rank
        self.vars = tuple(f"v{c}" for c in range(C))
        self.spatial = (size,) * rank
        self.periodic = (True,) * rank
        self._n, self._seed = n_frames, seed
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def fitted_on(self):
        return {}

    def trajectories(self, split):
        return [0, 1]

    def n_frames(self, split, traj):
        return self._n

    def _smooth(self, e: np.ndarray) -> np.ndarray:
        """Circular 3-point mean on every spatial axis, kept at unit variance."""
        for ax in range(1, e.ndim):
            e = (e + np.roll(e, 1, ax) + np.roll(e, -1, ax)) / np.sqrt(3.0)
        return e

    def _traj(self, split, traj) -> np.ndarray:
        key = (split, traj)
        if key not in self._cache:
            rng = np.random.default_rng(
                self._seed * 100 + SPLIT_SEED[split] * 10 + traj
            )
            shape = (len(self.vars), *self.spatial)
            x = np.zeros((self._n, *shape), np.float32)
            x[0] = self._smooth(rng.normal(size=shape))
            for t in range(1, self._n):
                x[t] = 0.9 * x[t - 1] + np.sqrt(1 - 0.81) * self._smooth(
                    rng.normal(size=shape)
                )
            self._cache[key] = x
        return self._cache[key]

    def read(self, split, traj, t0, n):
        assert 0 <= t0 and t0 + n <= self._n, (t0, n)
        return self._traj(split, traj)[t0 : t0 + n]

    def statics(self):
        return None

    def cond(self, split, traj, t0):
        return np.zeros(0, np.float32)

    def var_weights(self):
        return None
