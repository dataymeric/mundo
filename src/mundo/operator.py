r"""Observation operators :math:`A` and the per-level chain.

Averaging and subsampling, with their right inverses and the affine projection onto
:math:`\{x : A x = y\}`.
"""

import math

import torch

FACES = ("disagg", "interp")


def chain_factors(L: int) -> list[int]:
    """Refinement factor of each level of the recursion on an `L`-step window."""
    out, x, f = [], L, 2
    while x > 1:
        while x % f == 0:
            out.append(f)
            x //= f
        f += 1
    return out


class Operator:
    """A linear coarsening by factor `k` on the time axis of (B, Lf, C, *S)."""

    def __init__(self, k: int):
        assert k >= 2, k
        self.k = k

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def right_inverse(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def n_fine(self, n: int) -> int:
        raise NotImplementedError

    def n_free(self, n: int) -> int:
        r"""Number of free values among the children of `n` parents, i.e., the dimension
        of the null space of :math:`A`."""
        return self.n_fine(n) - n

    def complete(self, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        r"""Children of the parents `y` with free values `d`.

        Args:
            y: (B, n, C, *S) parents.
            d: (B, n_free(n), C, *S) free values.

        Returns:
            (B, n_fine(n), C, *S) children :math:`x`, with :math:`A x = y` exactly.
        """
        raise NotImplementedError

    def range_residual(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Range-space error :math:`A^\dagger (A z - y)`."""
        return self.right_inverse(self.apply(z) - y)

    def project(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Projection of `u` onto :math:`\{x : A x = y\}`.

        :math:`A^\dagger y + (I - A^\dagger A) u` with :math:`A^\dagger` = `right_inverse`.
        """
        return self.right_inverse(y) + u - self.right_inverse(self.apply(u))


class Avg(Operator):
    r"""Block mean over `k` consecutive frames.

    :math:`A^\dagger` repeats each mean `k` times.
    """

    def apply(self, x):
        B, Lf = x.shape[:2]
        assert Lf % self.k == 0, (Lf, self.k)
        return x.reshape(B, Lf // self.k, self.k, *x.shape[2:]).mean(dim=2)

    def right_inverse(self, y):
        return y.repeat_interleave(self.k, dim=1)

    def n_fine(self, n):
        return n * self.k

    def complete(self, y, d):
        r""":math:`y + d_1, \dots, y + d_{k-1}, y - \sum_i d_i` for each parent."""
        B, n = y.shape[:2]
        d = d.reshape(B, n, self.k - 1, *y.shape[2:])
        kids = torch.cat([d, -d.sum(dim=2, keepdim=True)], dim=2) + y[:, :, None]
        return kids.reshape(B, n * self.k, *y.shape[2:])


class Sub(Operator):
    r"""Keep every `k`-th frame, including the first and last.

    :math:`A^\dagger` is the piecewise-linear interpolant through the kept frames.
    """

    def apply(self, x):
        assert (x.shape[1] - 1) % self.k == 0, (x.shape[1], self.k)
        return x[:, :: self.k]

    def right_inverse(self, y):
        n, k = y.shape[1], self.k
        lam = torch.arange(k, device=y.device, dtype=y.dtype) / k
        lam = lam.view(1, 1, k, *([1] * (y.dim() - 2)))
        seg = y[:, :-1, None] * (1 - lam) + y[:, 1:, None] * lam
        seg = seg.reshape(y.shape[0], (n - 1) * k, *y.shape[2:])
        return torch.cat([seg, y[:, -1:]], dim=1)

    def n_fine(self, n):
        return (n - 1) * self.k + 1

    def range_residual(self, z, y):
        r""":math:`A z - y`, the error at the kept frames only."""
        return self.apply(z) - y

    def project(self, u, y):
        """`u` with its kept frames overwritten by `y`."""
        out = u.clone()
        out[:, :: self.k] = y
        return out

    def complete(self, y, d):
        r""":math:`y` at the kept frames, :math:`A^\dagger y + d` elsewhere."""
        out = self.right_inverse(y).clone()
        new = torch.arange(out.shape[1], device=y.device) % self.k != 0
        out[:, new] += d
        return out


def window_operator(face: str, L: int) -> Operator:
    r"""Observation operator :math:`A` of an `L`-step window."""
    assert face in FACES, face
    return Avg(L) if face == "disagg" else Sub(L)


def fine_len(face: str, L: int) -> int:
    """Frames of an `L`-step window: `L` on ``"disagg"``, ``L + 1`` on ``"interp"``."""
    return L if face == "disagg" else L + 1


def chain(face: str, factors: list[int], L: int | None = None) -> list[Operator]:
    """Operator of each level of the recursion, coarse to fine."""
    if L is not None:
        assert math.prod(factors) == L, (factors, L)
    cls = Avg if face == "disagg" else Sub
    return [cls(f) for f in factors]


def level_lens(face: str, ops: list[Operator]) -> list[int]:
    """Number of frames each level of `ops` produces, coarse to fine."""
    out, n = [], fine_len(face, 1)
    for op in ops:
        n = op.n_fine(n)
        out.append(n)
    return out


def coarsen(x: torch.Tensor, ops: list[Operator]) -> torch.Tensor:
    """Coarsen `x` through `ops`."""
    for op in reversed(ops):
        x = op.apply(x)
    return x
