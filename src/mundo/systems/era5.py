"""ERA5 (Hersbach et al., 2020).

Surface fields over Europe, one HDF5 file per year.
"""

from pathlib import Path

import cftime
import h5py
import hdf5plugin  # noqa: F401
import numpy as np


def _hours_since_epoch(time_ds) -> int:
    """Hours since 1970-01-01 of the first entry of a CF time dataset."""
    units = time_ds.attrs.get("units")
    if units is None:
        return int(time_ds[0]) // 3600

    def dec(s):
        return s.decode() if isinstance(s, bytes) else s

    d = cftime.num2date(
        time_ds[0],
        units=dec(units),
        calendar=dec(time_ds.attrs.get("calendar", "standard")),
    )
    return int(
        (np.datetime64(d.isoformat(), "s") - np.datetime64(0, "s"))
        // np.timedelta64(1, "h")
    )


class Era5:
    name = "era5"
    periodic = (False, False)

    def __init__(
        self,
        root,
        vars,
        box,
        years_train,
        years_val,
        years_test,
        n_stats_days,
        var_weights=None,
        stats=None,
    ):
        self.root = Path(root)
        self.vars, self.box = tuple(vars), tuple(int(b) for b in box)
        y0, y1, x0, x1 = self.box
        self.spatial = (y1 - y0, x1 - x0)
        self.years = {
            "train": list(years_train),
            "val": list(years_val),
            "test": list(years_test),
        }
        self._n_stats_days = n_stats_days
        self._shares = (
            None if var_weights is None else np.array(list(var_weights), np.float64)
        )
        assert self._shares is None or len(self._shares) == len(self.vars), (
            "one share per variable"
        )
        self._handles: dict[int, h5py.File] = {}
        self._T: dict[int, int] = {}
        self._h0: dict[int, int] = {}
        self.stats = stats

    # files
    def _file(self, year: int) -> Path:
        p = self.root / f"{year}.h5"
        assert p.exists(), f"missing {p}"
        return p

    def _handle(self, year: int) -> h5py.File:
        if year not in self._handles:
            self._handles[year] = h5py.File(self._file(year), "r", libver="latest")
        return self._handles[year]

    def _meta(self, year: int) -> tuple[int, int]:
        if year not in self._T:
            with h5py.File(self._file(year), "r") as h:
                T, H, W = h[self.vars[0]].shape
                assert self.box[1] <= H and self.box[3] <= W, (
                    f"box {self.box} outside {H}x{W}"
                )
                self._T[year] = T
                self._h0[year] = _hours_since_epoch(h["time"])
        return self._T[year], self._h0[year]

    def _raw(self, file, t0, n) -> np.ndarray:
        y0, y1, x0, x1 = self.box
        return np.stack(
            [file[v][t0 : t0 + n, y0:y1, x0:x1] for v in self.vars], axis=1
        ).astype(np.float32)

    # stats
    def fitted_on(self) -> dict:
        return {
            "vars": list(self.vars),
            "box": list(self.box),
            "years": list(self.years["train"]),
        }

    def fit_stats(self) -> dict:
        """Per-variable mean and std from `n_stats_days` random train days."""
        rng = np.random.default_rng(0)
        acc_m, acc_v = [], []
        for _ in range(self._n_stats_days):
            year = self.years["train"][int(rng.integers(len(self.years["train"])))]
            T, _ = self._meta(year)
            with h5py.File(self._file(year), "r") as h:
                w = self._raw(h, int(rng.integers(T // 24)) * 24, 24).astype(np.float64)
            acc_m.append(w.mean(axis=(0, 2, 3)))
            acc_v.append(w.var(axis=(0, 2, 3)))
        mean = np.mean(acc_m, axis=0)
        std = np.sqrt(np.mean(acc_v, axis=0) + np.var(acc_m, axis=0)) + 1e-9
        return {
            **self.fitted_on(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    # System
    def trajectories(self, split):
        return list(self.years[split])

    def n_frames(self, split, traj):
        return self._meta(traj)[0]

    def read(self, split, traj, t0, n):
        assert self.stats is not None, "no normalization stats: set system.stats_path"
        s = self.stats
        mean = np.array(s["mean"], np.float32)[None, :, None, None]
        std = np.array(s["std"], np.float32)[None, :, None, None]
        return (self._raw(self._handle(traj), t0, n) - mean) / std

    def statics(self):
        """(4, H, W) z-scored orography, land-sea mask, latitude and longitude ramps."""
        y0, y1, x0, x1 = self.box
        with h5py.File(self._file(self.years["train"][0]), "r") as file:
            z = file["geopotential_at_surface"][y0:y1, x0:x1].astype(np.float32)
            lsm = file["land_sea_mask"][y0:y1, x0:x1].astype(np.float32)
            lat = file["latitude"][y0:y1].astype(np.float32)
            lon = file["longitude"][x0:x1].astype(np.float32)
        z = (z - z.mean()) / (z.std() + 1e-9)
        latg = np.repeat(
            ((lat - lat.mean()) / (np.ptp(lat) / 2))[:, None], lon.size, axis=1
        )
        long_ = np.repeat(
            ((lon - lon.mean()) / (np.ptp(lon) / 2))[None, :], lat.size, axis=0
        )
        return np.stack([z, lsm, latg, long_]).astype(np.float32)

    def cond(self, split, traj, t0):
        """[hour of day, day of year] of frame `t0`."""
        h_abs = self._meta(traj)[1] + t0
        ts = np.datetime64("1970-01-01") + np.timedelta64(h_abs, "h")
        doy = (
            int((ts.astype("datetime64[D]") - ts.astype("datetime64[Y]")).astype(int))
            + 1
        )
        return np.array([h_abs % 24, doy], np.float32)

    def var_weights(self):
        """sqrt((1/C) / share) per variable, or `None` without shares."""
        if self._shares is None:
            return None
        return np.sqrt((1.0 / len(self.vars)) / self._shares).astype(np.float32)

    # the open h5 handles are not shared with DataLoader workers
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_handles"}

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._handles = {}
