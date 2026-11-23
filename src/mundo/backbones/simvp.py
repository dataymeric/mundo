"""SimVP with the gSTA translator (Gao et al., 2022).

Adapted from OpenSTL (Apache License 2.0, Copyright 2022-2023 CAIRI AI Lab), see
THIRD_PARTY_NOTICES.md.
"""

import math

import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn

from mundo.backbones.geometry import pad_to


def _init_conv_fan_out(m):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class BasicConv2d(nn.Module):
    """Conv (or conv + PixelShuffle for 2x upsampling), GroupNorm, SiLU."""

    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, upsampling=False):
        super().__init__()
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in, 4 * c_out, kernel_size, padding=padding),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size, stride=stride, padding=padding
            )
        self.norm = nn.GroupNorm(2, c_out)
        self.act = nn.SiLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvSC(nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size=3, downsampling=False, upsampling=False
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        self.conv = BasicConv2d(
            c_in,
            c_out,
            kernel_size,
            stride,
            (kernel_size - stride + 1) // 2,
            upsampling,
        )

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MixMlp(nn.Module):
    """Per-pixel MLP with a 3x3 depthwise convolution between its two layers."""

    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)
        self.apply(_init_conv_fan_out)

    def forward(self, x):
        x = self.drop(self.act(self.dwconv(self.fc1(x))))
        return self.drop(self.fc2(x))


class AttentionModule(nn.Module):
    """Large-kernel attention (Guo et al., 2022)."""

    def __init__(self, dim, kernel_size=21, dilation=3):
        super().__init__()
        d_k = 2 * dilation - 1
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=(d_k - 1) // 2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            dd_k,
            padding=dilation * (dd_k - 1) // 2,
            groups=dim,
            dilation=dilation,
        )
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)

    def forward(self, x):
        f, g = self.conv1(self.conv_spatial(self.conv0(x))).chunk(2, dim=1)
        return torch.sigmoid(g) * f


class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size=21):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(dim, kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return x + self.proj_2(
            self.spatial_gating_unit(self.activation(self.proj_1(x)))
        )


class GASubBlock(nn.Module):
    """gSTA block."""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(dim, int(dim * mlp_ratio), drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))
        self.apply(_init_conv_fan_out)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1[:, None, None] * self.attn(self.norm1(x))
        )
        return x + self.drop_path(
            self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x))
        )


class MetaBlock(nn.Module):
    def __init__(self, c_in, c_out, mlp_ratio, drop, drop_path):
        super().__init__()
        self.block = GASubBlock(c_in, 21, mlp_ratio, drop, drop_path)
        self.reduction = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.reduction(self.block(x))


class Encoder(nn.Module):
    def __init__(self, c_in, c_hid, n, kernel_size):
        super().__init__()
        down = [False, True] * (n // 2)
        self.enc = nn.Sequential(
            ConvSC(c_in, c_hid, kernel_size, downsampling=down[0]),
            *(ConvSC(c_hid, c_hid, kernel_size, downsampling=d) for d in down[1:n]),
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        return self.enc[1:](enc1), enc1


class Decoder(nn.Module):
    def __init__(self, c_hid, c_out, n, kernel_size):
        super().__init__()
        up = ([False, True] * (n // 2))[:n][::-1]
        self.dec = nn.Sequential(
            *(ConvSC(c_hid, c_hid, kernel_size, upsampling=u) for u in up)
        )
        self.readout = nn.Conv2d(c_hid, c_out, 1)

    def forward(self, hid, enc1):
        hid = self.dec[:-1](hid)
        return self.readout(self.dec[-1](hid + enc1))


class MidMetaNet(nn.Module):
    """Temporal translator."""

    def __init__(self, c_in, c_hid, n, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__()
        dpr = torch.linspace(1e-2, drop_path, n).tolist()
        dims = [c_in] + [c_hid] * (n - 1) + [c_in]
        self.enc = nn.Sequential(
            *(
                MetaBlock(dims[i], dims[i + 1], mlp_ratio, drop, dpr[i])
                for i in range(n)
            )
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        return self.enc(x.reshape(B, T * C, H, W)).reshape(B, T, C, H, W)


class SimVP(nn.Module):
    """(B, T, C, H, W) -> (B, T, C, H, W)."""

    def __init__(
        self, T, C, hid_s=16, hid_t=256, n_s=4, n_t=4, mlp_ratio=8.0, kernel=3
    ):
        super().__init__()
        self.enc = Encoder(C, hid_s, n_s, kernel)
        self.dec = Decoder(hid_s, C, n_s, kernel)
        self.hid = MidMetaNet(T * hid_s, hid_t, n_t, mlp_ratio, 0.0, 0.0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        embed, skip = self.enc(x.reshape(B * T, C, H, W))
        hid = self.hid(embed.reshape(B, T, *embed.shape[1:]))
        return self.dec(hid.reshape(B * T, *embed.shape[1:]), skip).reshape(
            B, T, C, H, W
        )


class SimVPNet(SimVP):
    """C -> C video model."""

    def __init__(
        self,
        c_level,
        c_surface,
        c_out,
        L,
        spatial,
        periodic=(False, False),
        n_cond=0,
        mdi=False,
        hid_s=16,
        hid_t=256,
        n_s=4,
        n_t=4,
        mlp_ratio=8.0,
        kernel=3,
        L_out=None,
    ):
        spatial = tuple(spatial)
        assert c_out == c_level and len(spatial) == 2, (
            "SimVP is a 2D C -> C video model"
        )
        assert L_out in (None, L), "simvp writes the frames it reads"
        super().__init__(L, c_level, hid_s, hid_t, n_s, n_t, mlp_ratio, kernel)
        self.cfg = {
            "c_level": c_level,
            "c_surface": c_surface,
            "c_out": c_out,
            "L": L,
            "L_out": L,
            "spatial": spatial,
            "periodic": tuple(periodic),
            "n_cond": n_cond,
            "mdi": False,
            "hid_s": hid_s,
            "hid_t": hid_t,
            "n_s": n_s,
            "n_t": n_t,
            "mlp_ratio": mlp_ratio,
            "kernel": kernel,
        }

    def forward(self, level, surface=None, cond=None, generator=None):
        """(B, L, C, H, W) -> (B, L, C, H, W)."""
        x, crop = pad_to(level, 8, self.cfg["periodic"])
        return crop(super().forward(x.contiguous()))
