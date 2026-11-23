"""Neighborhood-attention U-Net (Hassani et al., 2023)."""

import torch
from torch import nn

from mundo.backbones.geometry import pad_to


def clamp_kernel(kernel: int, spatial, stages: int) -> int:
    """Neighborhood-attention kernel size."""
    pad = 2**stages
    side = min(-(-s // pad) * pad for s in spatial) // pad
    k = min(kernel, side)
    return max(3, k - (k + 1) % 2)


class NABlock(nn.Module):
    """Pre-LN neighborhood-attention block."""

    def __init__(self, dim: int, kernel: int = 7):
        from natten import NeighborhoodAttention2D

        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = NeighborhoodAttention2D(dim, max(1, dim // 32), kernel)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.mlp(self.n2(x))


class NAT(nn.Module):
    """U-Net of neighborhood-attention blocks."""

    def __init__(
        self,
        c_in,
        c_out,
        width=42,
        stages=3,
        kernel=7,
        depth=2,
        mdi=False,
        n_noise=8,
        zero_head=True,
    ):
        super().__init__()
        mult = [1, 2, 4, 4][: stages + 1]
        dims = [max(32, 32 * round(width * m / 32)) for m in mult]
        self.stem = nn.Conv2d(c_in, dims[0], 3, padding=1)
        self.enc = nn.ModuleList(
            nn.ModuleList(NABlock(dims[s], kernel) for _ in range(depth))
            for s in range(stages)
        )
        self.down = nn.ModuleList(
            nn.Conv2d(dims[s], dims[s + 1], 2, stride=2) for s in range(stages)
        )
        self.mid = nn.ModuleList(NABlock(dims[stages], kernel) for _ in range(depth))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(dims[s + 1], dims[s], 2, stride=2)
            for s in reversed(range(stages))
        )
        self.dec = nn.ModuleList(
            nn.ModuleList(NABlock(dims[s], kernel) for _ in range(depth))
            for s in reversed(range(stages))
        )
        self.head = nn.Conv2d(dims[0], c_out, 3, padding=1)
        if zero_head:  # an untrained net writes a zero residual
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        inj = [dims[stages]] + [dims[s] for s in reversed(range(stages))]
        self.n_noise = n_noise
        self.noise_proj = (
            nn.ModuleList(nn.Conv2d(n_noise, d, 1) for d in inj) if mdi else None
        )

    @staticmethod
    def _run(blocks, x):
        x = x.permute(0, 2, 3, 1)
        for b in blocks:
            x = b(x)
        return x.permute(0, 3, 1, 2)

    def _inj(self, k, x, g):
        if self.noise_proj is None:
            return x
        eps = torch.randn(
            x.shape[0], self.n_noise, *x.shape[2:], generator=g, device=x.device
        )
        return x + self.noise_proj[k](eps)

    def forward(self, x, generator=None):
        x = self.stem(x)
        skips = []
        for blocks, down in zip(self.enc, self.down):
            x = self._run(blocks, x)
            skips.append(x)
            x = down(x)
        x = self._run(self.mid, self._inj(0, x, generator))
        for j, (up, blocks, skip) in enumerate(zip(self.up, self.dec, reversed(skips))):
            x = self._run(blocks, self._inj(j + 1, up(x) + skip, generator))
        return self.head(x)


class NATNet(nn.Module):
    def __init__(
        self,
        c_level,
        c_surface,
        c_out,
        L,
        spatial,
        periodic,
        n_cond=0,
        mdi=False,
        width=42,
        stages=3,
        kernel=7,
        depth=2,
        n_noise=8,
        L_out=None,
        zero_head=True,
    ):
        super().__init__()
        assert len(spatial) == 2, "nat is 2D"
        L_out = L if L_out is None else L_out
        k = clamp_kernel(kernel, spatial, stages)
        self.cfg = {
            "c_level": c_level,
            "c_surface": c_surface,
            "c_out": c_out,
            "L": L,
            "L_out": L_out,
            "spatial": tuple(spatial),
            "periodic": tuple(periodic),
            "n_cond": n_cond,
            "mdi": mdi,
            "width": width,
            "stages": stages,
            "kernel": k,
            "depth": depth,
            "n_noise": n_noise,
            "zero_head": zero_head,
        }
        self.body = NAT(
            c_level * L + c_surface + n_cond,
            c_out * L_out,
            width,
            stages,
            k,
            depth,
            mdi,
            n_noise,
            zero_head,
        )

    def forward(self, level, surface, cond, generator=None):
        """(B, L, c_level, H, W) * (B, c_surface, H, W) * (B, n_cond)
        -> (B, L_out, c_out, H, W).
        """
        B = level.shape[0]
        S = self.cfg["spatial"]
        # scales the fm step to [0, 1)
        cmaps = cond.reshape(B, cond.shape[1], 1, 1).expand(B, -1, *S) / 1000.0
        x, crop = pad_to(
            torch.cat([level.flatten(1, 2), surface, cmaps], dim=1),
            2 ** self.cfg["stages"],
            self.cfg["periodic"],
        )
        return crop(self.body(x, generator)).reshape(
            B, self.cfg["L_out"], self.cfg["c_out"], *S
        )
