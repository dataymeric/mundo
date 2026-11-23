"""Small residual convolutional network of any spatial rank."""

import torch
import torch.nn.functional as F
from torch import nn

from mundo.backbones.geometry import conv_pad

CONV = {1: nn.Conv1d, 2: nn.Conv2d}


class ConvNet(nn.Module):
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
        width=64,
        depth=4,
        kernel=5,
        n_noise=8,
        L_out=None,
    ):
        super().__init__()
        L_out = L if L_out is None else L_out
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
            "depth": depth,
            "kernel": kernel,
            "n_noise": n_noise,
        }
        Conv = CONV[len(spatial)]
        c_in = L * c_level + c_surface + n_cond
        self.convs = nn.ModuleList(
            Conv(c_in if i == 0 else width, width, kernel) for i in range(depth)
        )
        self.noise_proj = (
            nn.ModuleList(Conv(n_noise, width, 1) for _ in range(depth - 1))
            if mdi
            else None
        )
        self.head = Conv(width, L_out * c_out, kernel)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, level, surface, cond, generator=None):
        """(B, L, c_level, *S) * (B, c_surface, *S) * (B, n_cond)
        -> (B, L_out, c_out, *S).
        """
        B = level.shape[0]
        S, per, k = self.cfg["spatial"], self.cfg["periodic"], self.cfg["kernel"]
        # scales the fm step to [0, 1)
        cmaps = (
            cond.reshape(B, cond.shape[1], *([1] * len(S))).expand(B, -1, *S) / 1000.0
        )
        x = torch.cat([level.flatten(1, 2), surface, cmaps], dim=1)
        for i, conv in enumerate(self.convs):
            if i and self.noise_proj is not None:
                eps = torch.randn(
                    B, self.cfg["n_noise"], *S, device=x.device, generator=generator
                )
                x = x + self.noise_proj[i - 1](eps)
            x = F.gelu(conv(conv_pad(x, k, per)))
        return self.head(conv_pad(x, k, per)).reshape(
            B, self.cfg["L_out"], self.cfg["c_out"], *S
        )
