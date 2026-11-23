# Data

The three systems read their data from files built by the scripts in `scripts/data/`. Each script writes exactly the layout its loader in `src/mundo/systems/` reads, under the directory you then pass to training and evaluation as `system.args.root`.

## Lorenz-96

Two-scale Lorenz-96 (Lorenz, 2006), integrated with RK4 as in [L96_demo](https://github.com/m2lines/L96_demo) (MIT, see `THIRD_PARTY_NOTICES.md`). One archive per forcing $F$, for $F \in \{10, 24, 32, 64\}$:

```bash
python scripts/data/make_l96.py --forcing <F> --out <data dir>
```

The defaults are $K = 36, J = 32, h = 1, b = c = 10$ and $dt = 0.001$. The system is integrated for 1000 time units after a 100-unit spin-up (about 150 MB per forcing).

> [!NOTE]
> The loader only finds files named `TU1000`, and its `stride` counts integration steps. With the default $dt$, `stride: 10` gives frames every 0.01 time units. Changing `--dt` changes what `stride` means.

## AutoSim

Four 2D systems from the [AutoSim package](https://github.com/alan-turing-institute/autosim) (Greenbury et al., 2026): advection-diffusion, Gray-Scott, buoyancy-driven Navier-Stokes and Gross-Pitaevskii. Each comes with 200 training, 20 validation, and 20 test trajectories from seed 2026:

```bash
python scripts/data/make_autosim.py <dataset> --root <data dir> --autosim <path to the autosim executable>
```

> [!IMPORTANT]
> **AutoSim is not a dependency of MUNDO.** It requires Python < 3.13, while MUNDO's environment may run a newer Python, so it lives in its own environment, and the script calls its `autosim` command by path:
>
> ```bash
> git clone https://github.com/alan-turing-institute/autosim
> cd autosim && git checkout e5310db && uv sync
> ```
>
> Then, pass `--autosim <clone>/.venv/bin/autosim`. The benchmark sets use the simulator's default parameters, here those of commit `e5310db`.


## ERA5

ERA5 (Hersbach et al., 2020): hourly surface fields over Europe (25°W–45°E, 35°N–72°N, 149 × 281 points at 0.25°), downloaded from [WeatherBench 2](https://weatherbench2.readthedocs.io) (Rasp et al., 2024). One file per year, about 6.5 GB each, 286 GB for 1979–2022:

```bash
uv sync --extra data
python scripts/data/make_era5.py --out <data dir>
```

> [!NOTE]
> `--years` restricts the download, and years already on disk are skipped.

## Normalization stats

Training and evaluation standardize each variable with a mean and standard deviation fitted once on the training split. We provide the stats of the available systems in `conf/stats/`, and each `conf/system/<name>.yaml` points `system.stats_path` to them. For another system, dataset or setting (training years, box, split, `n_stats_*`), fit them once with the same overrides as training:

```bash
python scripts/data/make_stats.py +experiment=era5 system.args.root=<data dir> system.stats_path=<file>
```

Then, pass the same `system.stats_path=<file>` to training. A stats file records the settings it was fitted with (Lorenz-96: forcing and stride; ERA5: variables, box and training years; AutoSim: dataset, regime and variables), and a run with other settings refuses it. Each checkpoint keeps a copy in its `stats.json`, which evaluation uses.
