r"""Two-scale Lorenz-96 (Lorenz, 2006).

.. math::

    \frac{dX_k}{dt} = X_{k-1} (X_{k+1} - X_{k-2}) - X_k + F
        - \frac{hc}{b} \sum_j Y_{j,k},

    \frac{dY_{j,k}}{dt} = -cb\, Y_{j+1,k} (Y_{j+2,k} - Y_{j-1,k}) - c\, Y_{j,k}
        + \frac{hc}{b} X_k,

integrated with RK4 as in L96_demo (https://github.com/m2lines/L96_demo), with
:math:`K = 36`, :math:`J = 32`, :math:`h = 1`, :math:`b = c = 10` and
:math:`\Delta t = 0.001` by default. After 100 time units of spin-up, 1000 time
units of :math:`X` are kept.

Example:
    python scripts/data/make_l96.py --forcing 32 --out <data dir>
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def tendencies(X, Y, F, h, b, c):
    # Y is flat, Y[k * J + j]: its rolls then cross from one X sector to the next
    K = X.size
    J = Y.size // K
    hcb = h * c / b
    Xdot = (
        np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2))
        - X
        + F
        - hcb * Y.reshape(K, J).sum(-1)
    )
    Ydot = (
        -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
        - c * Y
        + hcb * np.repeat(X, J)
    )
    return Xdot, Ydot


def rk4_step(X, Y, dt, *p):
    k1x, k1y = tendencies(X, Y, *p)
    k2x, k2y = tendencies(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y, *p)
    k3x, k3y = tendencies(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y, *p)
    k4x, k4y = tendencies(X + dt * k3x, Y + dt * k3y, *p)
    X = X + (dt / 6.0) * ((k1x + k4x) + 2.0 * (k2x + k3x))
    Y = Y + (dt / 6.0) * ((k1y + k4y) + 2.0 * (k2y + k3y))
    return X, Y


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--forcing", type=int, required=True, help="10, 24, 32 or 64")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=int, default=36)
    ap.add_argument("--J", type=int, default=32)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--spinup", type=int, default=100, help="time units")
    ap.add_argument("--tu", type=int, default=1000, help="time units stored")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = a.out / f"L96_K{a.K}_J{a.J}_F{a.forcing}_TU{a.tu}.nc"
    assert not path.exists(), f"{path} exists"

    rng = np.random.default_rng(a.seed)
    X, Y = a.b * rng.standard_normal(a.K), rng.standard_normal(a.J * a.K)
    p = (float(a.forcing), a.h, a.b, a.c)
    for _ in range(round(a.spinup / a.dt)):
        X, Y = rk4_step(X, Y, a.dt, *p)
    n = round(a.tu / a.dt)
    Xh = np.empty((n + 1, a.K), np.float32)
    Xh[0] = X
    for i in range(n):
        X, Y = rk4_step(X, Y, a.dt, *p)
        Xh[i + 1] = X
    assert np.isfinite(Xh).all()

    a.out.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["X"] = Xh
        f["time"] = a.spinup + a.dt * np.arange(n + 1)
        f.attrs.update(F=p[0], K=a.K, J=a.J, h=a.h, b=a.b, c=a.c, dt=a.dt, seed=a.seed)
    print(f"wrote {path}  X {Xh.shape}  mean {Xh.mean():.3f}  std {Xh.std():.3f}")


if __name__ == "__main__":
    main()
