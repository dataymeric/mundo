"""Spatial padding utilities shared by the backbones: wrap-around on periodic axes,
edge replication on the others.
"""

import torch


def _edge(x: torch.Tensor, dim: int, index: int, n: int) -> torch.Tensor:
    """Edge padding: repeat one slice of `x` `n` times along an axis.

    Args:
        x: Tensor to pad.
        dim: Axis to repeat along.
        index: Position of the repeated slice on that axis.
        n: Number of copies.

    Returns:
        View of `x`, with `n` on `dim`.
    """
    shape = [-1] * x.dim()
    shape[dim] = n
    return x.narrow(dim, index, 1).expand(*shape)


def pad_to(x: torch.Tensor, multiple: int, periodic):
    """Pad the trailing spatial axes of `x` at their end, up to a multiple.

    Args:
        x: (..., *S) tensor whose last ``len(periodic)`` axes are spatial.
        multiple: Each spatial size is rounded up to a multiple of it.
        periodic: One flag per spatial axis. A periodic axis continues with
            its first entries, the others repeat their last one.

    Returns:
        out: (..., *S_padded) padded tensor.
        crop: Function taking any (..., *S_padded) tensor back to (..., *S).
    """
    sizes = x.shape[-len(periodic) :]
    out = x
    for ax, (s, per) in enumerate(zip(sizes, periodic)):
        p = -(-s // multiple) * multiple - s
        if p == 0:
            continue
        dim = x.dim() - len(periodic) + ax
        tail = out.narrow(dim, 0, p) if per else _edge(out, dim, out.shape[dim] - 1, p)
        out = torch.cat([out, tail], dim=dim)

    def crop(y):
        return y[(..., *[slice(0, s) for s in sizes])]

    return out, crop


def conv_pad(x: torch.Tensor, k: int, periodic) -> torch.Tensor:
    """Pad ``k // 2`` on both sides of each spatial axis, so an unpadded conv
    of odd kernel `k` keeps the spatial size.

    Args:
        x: (..., *S) tensor whose last ``len(periodic)`` axes are spatial.
        k: Conv kernel size.
        periodic: One flag per spatial axis; a periodic axis wraps around,
            the others repeat their edge entries.

    Returns:
        (..., *(S + 2 * (k // 2))) padded tensor.
    """
    h = k // 2
    for ax, per in enumerate(periodic):
        dim = x.dim() - len(periodic) + ax
        if per:
            lo, hi = x.narrow(dim, x.shape[dim] - h, h), x.narrow(dim, 0, h)
        else:
            lo, hi = _edge(x, dim, 0, h), _edge(x, dim, x.shape[dim] - 1, h)
        x = torch.cat([lo, x, hi], dim=dim)
    return x
