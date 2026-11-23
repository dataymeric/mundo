r"""AutoSim 2D simulations (Greenbury et al., 2026).

Advection-diffusion, Gray-Scott, Navier-Stokes and Gross-Pitaevskii on 64 x 64 grids.

The ``_e`` sets make the dynamics livelier and the ``_s`` sets space the snapshots
further apart, changing only the parameters listed in ``SETS``.

AutoSim runs in its own environment, see ``docs/DATA.md``.

Example:
    python scripts/data/make_autosim.py gpe_wake200 --root <data dir> \
        --autosim <path to the autosim executable>
"""

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np
import torch

PR = "++simulator.parameters_range"
# dataset: (simulator, channel names in stored order, overrides)
SETS = {
    "advection_diffusion200": ("advection_diffusion", "c0", []),
    "advection_diffusion200_e2": (
        "advection_diffusion",
        "c0",
        [f"{PR}.mu=[1.0,4.0]", f"{PR}.nu=[0.00005,0.005]"],
    ),
    "advection_diffusion200_e4": (
        "advection_diffusion",
        "c0",
        [f"{PR}.mu=[2.0,8.0]", f"{PR}.nu=[0.000025,0.0025]"],
    ),
    "gray_scott200": ("gray_scott", "u,v", []),
    "gray_scott200_s2": (
        "gray_scott",
        "u,v",
        ["simulator.snapshot_dt=8.0", "simulator.T=2560.0"],
    ),
    "gray_scott200_s4": (
        "gray_scott",
        "u,v",
        ["simulator.snapshot_dt=16.0", "simulator.T=5120.0"],
    ),
    "conditioned_ns200": ("conditioned_navier_stokes_2d", "smoke,u,v", []),
    "conditioned_ns200_e2": (
        "conditioned_navier_stokes_2d",
        "smoke,u,v",
        [f"{PR}.buoyancy_y=[1.6,6.4]"],
    ),
    "conditioned_ns200_e4": (
        "conditioned_navier_stokes_2d",
        "smoke,u,v",
        [f"{PR}.buoyancy_y=[3.2,12.8]"],
    ),
    "gpe_wake200": ("gpe/laser_only_wake", "c0,c1,c2", []),
    "gpe_wake200_e2": (
        "gpe/laser_only_wake",
        "c0,c1,c2",
        [f"{PR}.spoon_speed=[1.5,3.0]", f"{PR}.g=[160.0,400.0]"],
    ),
}
SPLITS = ("train", "valid", "test")


def simulate(autosim: str, sim: str, overrides: list[str], gen: Path, seed: int):
    if all((gen / s / "data.pt").exists() for s in SPLITS):
        print(f"{gen} already holds all three splits, skipping the simulation")
        return
    cmd = [
        autosim,
        f"simulator=spatiotemporal/{sim}",
        f"seed={seed}",
        "dataset.n_train=200",
        "dataset.n_valid=20",
        "dataset.n_test=20",
        f"dataset.output_dir={gen}",
        "visualize.enabled=false",
        *overrides,
    ]
    subprocess.run(cmd, check=True)


def to_h5(gen: Path, root: Path, dataset: str, names: list[str]):
    for split in SPLITS:
        a = torch.load(gen / split / "data.pt", map_location="cpu", weights_only=True)[
            "data"
        ]
        if torch.is_complex(a):
            a = torch.cat([a.real, a.imag], dim=-1)
        a = a.numpy().astype(np.float32)  # (n_traj, T, H, W, C)
        assert a.ndim == 5 and a.shape[-1] == len(names), (a.shape, names)
        out = root / dataset / "data" / split
        out.mkdir(parents=True, exist_ok=True)
        with h5py.File(out / f"{dataset}_0.hdf5", "w") as h:
            for c, name in enumerate(names):
                h[f"t0_fields/{name}"] = a[..., c]
        print(f"{split}: {a.shape} -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", choices=sorted(SETS))
    ap.add_argument("--root", type=Path, required=True, help="datasets root")
    ap.add_argument(
        "--gen", type=Path, help="simulator output, default <root>/../gen/<dataset>"
    )
    ap.add_argument("--autosim", default="autosim", help="the autosim executable")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    sim, names, overrides = SETS[args.dataset]
    names = names.split(",")
    # the loader orders channels by sorted name, so stored order must already be sorted
    assert names == sorted(names), names
    gen = args.gen or args.root.parent / "gen" / args.dataset
    simulate(args.autosim, sim, overrides, gen, args.seed)
    to_h5(gen, args.root, args.dataset, names)


if __name__ == "__main__":
    main()
