"""ERA5 (Hersbach et al., 2020).

Hourly surface fields over Europe (25°W–45°E, 35°N–72°N) at 0.25°, downloaded from
WeatherBench 2 (Rasp et al., 2024): 10 m wind, 2 m temperature and dewpoint, mean
sea-level pressure, total column water and total precipitation, with the surface
geopotential and land-sea mask. One file per year, about 6.5 GB each. Needs the
``data`` extra.

Example:
    python scripts/data/make_era5.py --out <data dir> --years 1979 1980
"""

import argparse
from pathlib import Path

import dask
import hdf5plugin
import numpy as np
import xarray as xr
import zarr

ZARR = (
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
KEEP = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
    "total_column_water",
    "land_sea_mask",
    "geopotential_at_surface",
]
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -25, 45, 35, 72


def europe() -> xr.Dataset:
    ds = xr.open_zarr(ZARR, storage_options={"session_kwargs": {"trust_env": True}})
    ds = ds[KEEP]
    lat = (
        slice(LAT_MAX, LAT_MIN)
        if ds.latitude[0] > ds.latitude[-1]
        else slice(LAT_MIN, LAT_MAX)
    )
    east = ds.sel(longitude=slice(0, LON_MAX), latitude=lat)
    west = ds.sel(longitude=slice(360 + LON_MIN, 360), latitude=lat)
    ds = xr.concat([west, east], dim="longitude")
    return ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--years", nargs="+", type=int, default=list(range(1979, 2023)))
    args = ap.parse_args()
    dask.config.set(scheduler="threads", num_workers=16)
    zarr.config.set({"async.concurrency": 100})
    args.out.mkdir(parents=True, exist_ok=True)

    ds = europe()
    H, W = ds.sizes["latitude"], ds.sizes["longitude"]
    assert (H, W) == (149, 281), (H, W)
    blosc = hdf5plugin.Blosc2(cname="lz4", clevel=1)
    for year in args.years:
        path = args.out / f"{year}.h5"
        if path.exists():
            print(f"{path} exists, skipping")
            continue
        ds_year = ds.sel(time=ds.time.dt.year.isin([year]))
        enc = {
            v: {**blosc, "chunksizes": (1, H, W), "dtype": np.float32}
            for v in ds_year.data_vars
            if ds_year[v].ndim == 3
        }
        tmp = path.with_suffix(".h5.part")
        ds_year.to_netcdf(tmp, engine="h5netcdf", encoding=enc)
        tmp.rename(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
