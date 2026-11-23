<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/mundo-logo-dark.svg">
    <img alt="MUNDO" src="docs/logo/mundo-logo.svg" width="360">
  </picture>
</p>

# MUNDO: A Multiresolution Generative Framework for Temporal Downscaling

MUNDO (**M**ultiresolution **UN**ified **DO**wnscaling) is a generative framework for temporal downscaling. It generates
fine-scale trajectories of dynamical systems from sparse states (interpolation) or temporal aggregates (disaggregation).
Using reverse Markov learning, it trains one engression sampler per level of a hierarchy of resolutions: each level
generates only the degrees of freedom lost between two adjacent resolutions, and every sample matches the observation
exactly.

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dataymeric/mundo
cd mundo
uv sync
```

This installs PyTorch with CUDA 12.6, along with the [`NATTEN`](https://github.com/SHI-Labs/NATTEN) library.

Without a GPU, install the CPU build instead:

```bash
uv sync --no-default-groups --group cpu
```

> [!IMPORTANT]
> With the CPU build, run the commands below with `uv run --no-sync`, or activate the environment with
`source .venv/bin/activate`. A plain `uv run` syncs the environment back to its defaults and reinstalls the GPU build.

## Quick start

You can try MUNDO right after installing it. We provide a small synthetic system that generates its own data, and
training and evaluation take about a minute, even on a CPU.

```bash
uv run mundo-train +experiment=synthetic_smoke out=runs
uv run mundo-eval  +experiment=synthetic_smoke out=runs
```

The first command trains MUNDO on disaggregation, and the second evaluates it, alongside the classical baselines.
Results are written to `runs/eval/synthetic/disagg_ctxon.json`.

## Data

See [`docs/DATA.md`](docs/DATA.md) for details.

## Train

To train one model (`arm`) on one system, for one type of observation (`system.face`):

```bash
uv run mundo-train +experiment=l96 system.args.root=<data dir> out=runs \
    system.args.forcing=32 system.face=disagg arm=recursive

uv run mundo-train +experiment=autosim system.args.root=<data dir> out=runs \
    system.args.dataset=gpe_wake200 system.L=24 arm=FlowDPS

uv run mundo-train +experiment=era5 system.args.root=<data dir> out=runs \
    system.face=interp arm=single_step
```

The configuration is handled by [Hydra](https://hydra.cc/), so any setting can be changed from the command line. The
main ones are:

| Option             | Values                                                                                |
|--------------------|---------------------------------------------------------------------------------------|
| `+experiment`      | `l96`, `autosim`, `era5`, `synthetic_smoke`                                           |
| `system.face`      | `disagg` (time means), `interp` (sparse snapshots)                                    |
| `system.L`         | window length in steps                                                                |
| `arm`              | `recursive` (MUNDO), `single_step`, `FlowMatching`, `FlowDPS`, `SimVP`                |
| `train.steps`      | optimizer steps per network                                                           |
| `train.val_every`  | steps between validations. `0` validates at the end of each epoch                     |
| `train.stage`      | `1` (default), `2` for end-to-end fine-tuning of a trained recursive arm              |
| `seed`             | training seed                                                                         |
| `system.args.root` | directory containing the system's data (required, see [`docs/DATA.md`](docs/DATA.md)) |
| `out`              | output directory (required)                                                           |

`src/mundo/config.py` lists every option, and every default configuration lives in `conf/`.
Checkpoints go to `<out>/<system>/<face>/<tag>/seed<N>/`.

## Evaluate

To evaluate every trained checkpoint of one face of one system, along with the classical baselines, on the test split:

```bash
uv run mundo-eval +experiment=l96 system.args.root=<data dir> out=runs \
    system.args.forcing=32 system.face=disagg
```

The scores are written to `<out>/eval/<system>/<face>_ctx<on|off>.json`, one row per model, all with the same metrics.
Only seed 0 is scored by default: pass `eval.seeds=[0,1,2]` to score more seeds.

## License

MUNDO is released under the BSD-3-Clause license (see [`LICENSE`](LICENSE)). Vendored and adapted code is listed in [
`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
