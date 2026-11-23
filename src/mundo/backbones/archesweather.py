# Copyright (c) 2026, ARCHES team @ INRIA. BSD-3-Clause, see THIRD_PARTY_NOTICES.md.
"""ArchesWeather conditional Swin backbone (Couairon et al., 2026)."""

import math

import numpy as np
import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.utils.checkpoint import checkpoint


class TimestepEmbedder(nn.Module):
    """Sinusoidal features of a scalar, mapped by a two-layer MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t, max_period=10000):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


def _pad3d(resolution, window):
    """Zero padding (left, right, top, bottom, front, back) to whole windows."""
    pads = []
    for n, w in zip(reversed(resolution), reversed(window)):
        p = -n % w
        pads += [p // 2, p - p // 2]
    return tuple(pads)


def _crop3d(x, resolution):
    """Undo `_pad3d` on the three trailing axes of (B, C, Pl, Lat, Lon)."""
    for d, n in zip((2, 3, 4), resolution):
        x = x.narrow(d, (x.shape[d] - n) // 2, n)
    return x


def _window_partition(x, window):
    """(B, Pl, Lat, Lon, C) -> (B * nLon, nPl * nLat, wPl, wLat, wLon, C)."""
    B, Pl, Lat, Lon, C = x.shape
    wp, wa, wo = window
    x = x.view(B, Pl // wp, wp, Lat // wa, wa, Lon // wo, wo, C)
    x = x.permute(0, 5, 1, 3, 2, 4, 6, 7)
    return x.reshape(-1, (Pl // wp) * (Lat // wa), wp, wa, wo, C)


def _window_reverse(w, window, Pl, Lat, Lon):
    """Inverse of `_window_partition`."""
    wp, wa, wo = window
    B = w.shape[0] // (Lon // wo)
    x = w.view(B, Lon // wo, Pl // wp, Lat // wa, wp, wa, wo, -1)
    return x.permute(0, 2, 4, 3, 5, 1, 6, 7).reshape(B, Pl, Lat, Lon, -1)


def _earth_position_index(window):
    """Bias table row of every (query, key) pair of a window."""
    wp, wa, wo = window
    q = torch.meshgrid(
        torch.arange(wp), torch.arange(wa), torch.arange(wo), indexing="ij"
    )
    k = torch.meshgrid(
        -torch.arange(wp) * wp, -torch.arange(wa) * wa, torch.arange(wo), indexing="ij"
    )
    q, k = torch.stack(q).flatten(1), torch.stack(k).flatten(1)
    d = (q[:, :, None] - k[:, None, :]).permute(1, 2, 0)
    d[:, :, 2] += wo - 1
    d[:, :, 1] *= 2 * wo - 1
    d[:, :, 0] *= (2 * wo - 1) * wa * wa
    return d.sum(-1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class EarthAttention3D(nn.Module):
    """3D window attention with earth position bias."""

    def __init__(self, dim, input_resolution, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        wp, wa, wo = window_size
        n_win = (input_resolution[0] // wp) * (input_resolution[1] // wa)
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(wp * wp * wa * wa * (2 * wo - 1), n_win, num_heads)
        )
        self.register_buffer("earth_position_index", _earth_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.earth_position_bias_table, std=0.02)

    def forward(self, x):
        """(B * nLon, nPl * nLat, N, C)."""
        B_, nW, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, nW, N, 3, self.num_heads, -1)
        q, k, v = qkv.permute(3, 0, 4, 1, 2, 5)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.earth_position_bias_table[self.earth_position_index.view(-1)]
        attn = attn + bias.view(N, N, nW, -1).permute(3, 2, 0, 1)
        x = (attn.softmax(-1) @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW, N, C)
        return self.proj(x)


class EarthSpecificBlock(nn.Module):
    """3D Transformer block."""

    def __init__(
        self, dim, input_resolution, num_heads, window_size, roll, drop_path, mlp_ratio
    ):
        super().__init__()
        self.input_resolution = tuple(input_resolution)
        self.window_size = tuple(window_size)
        self.shift_size = (1, 3, 5)
        self.roll = roll
        self.norm1 = nn.LayerNorm(dim)
        padding = _pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)
        pad_resolution = [
            n + padding[i] + padding[i + 1] for n, i in zip(input_resolution, (4, 2, 0))
        ]
        self.attn = EarthAttention3D(dim, pad_resolution, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, c):
        """(B, Pl * Lat * Lon, C) tokens * (B, 6 * C) modulation."""
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c[
            :, None
        ].chunk(6, dim=-1)
        shortcut = x
        x = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x.view(B, Pl, Lat, Lon, C).permute(0, 4, 1, 2, 3)
        x = self.pad(x).permute(0, 2, 3, 4, 1)
        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape
        if self.roll:
            x = torch.roll(x, shifts=[-s for s in self.shift_size], dims=(1, 2, 3))
        w = _window_partition(x, self.window_size).flatten(2, 4)
        w = self.attn(w).view(*w.shape[:2], *self.window_size, C)
        x = _window_reverse(w, self.window_size, Pl_pad, Lat_pad, Lon_pad)
        if self.roll:
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        x = _crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, L, C)
        x = shortcut + gate_msa * self.drop_path(x)
        mlp_in = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        return x + self.drop_path(gate_mlp * self.mlp(mlp_in))


class CondBasicLayer(nn.Module):
    """3D Transformer layer with `depth` blocks."""

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, drop_path, cond_dim
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            EarthSpecificBlock(
                dim,
                input_resolution,
                num_heads,
                window_size,
                bool(i % 2),
                drop_path[i],
                4.0,
            )
            for i in range(depth)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond_emb):
        c = self.adaLN_modulation(cond_emb)
        for blk in self.blocks:
            x = blk(x, c)
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = tuple(input_resolution)
        self.output_resolution = tuple(output_resolution)
        (_, lat, lon), (_, out_lat, out_lon) = input_resolution, output_resolution
        h, w = out_lat * 2 - lat, out_lon * 2 - lon
        self.pad = nn.ZeroPad3d((w // 2, w - w // 2, h // 2, h - h // 2, 0, 0))

    def forward(self, x):
        B, _, C = x.shape
        pl, lat, lon = self.input_resolution
        _, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, pl, lat, lon, C).permute(0, 4, 1, 2, 3)
        x = self.pad(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        return self.linear(self.norm(x.reshape(B, pl * out_lat * out_lon, 4 * C)))


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = tuple(input_resolution)
        self.output_resolution = tuple(output_resolution)

    def forward(self, x):
        B = x.shape[0]
        pl, lat, lon = self.input_resolution
        x = self.linear1(x).reshape(B, pl, lat, lon, 2, 2, -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, pl, lat * 2, lon * 2, -1)
        x = _crop3d(x.movedim(-1, 1), self.output_resolution).movedim(1, -1)
        return self.linear2(self.norm(x.reshape(B, -1, x.shape[-1])))


class ArchesWeatherCondBackbone(nn.Module):
    r"""Four-layer Swin U-Net on (B, C, Pl, Lat, Lon) tokens, conditioned by adaLN.

    Args:
        tensor_size: Number of tokens along (Pl, Lat, Lon).
        emb_dim: Token channels in the first layer. Defaults to 192.
        cond_dim: Size of the condition vector. Defaults to 256.
        num_heads: Number of attention heads in each of the four layers.
            Defaults to (6, 12, 12, 6).
        window_size: Attention window, in tokens along (Pl, Lat, Lon).
            Defaults to (2, 6, 12).
        droppath_coeff: Maximum drop-path rate, divided by `depth_multiplier`.
            Defaults to 0.2.
        depth_multiplier: Scale the number of blocks per layer, which is (2, 6, 6, 2)
            when set to 1. Defaults to 2.
        gradient_checkpointing: Recompute activations in the backward pass to save
            memory. Defaults to False.

    Returns:
        (B, 2 * emb_dim, Pl, Lat, Lon).
    """

    def __init__(
        self,
        tensor_size,
        emb_dim=192,
        cond_dim=256,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        droppath_coeff=0.2,
        depth_multiplier=2,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        d = depth_multiplier
        drop_path = np.linspace(0, droppath_coeff / d, 8 * d).tolist()
        full = tuple(tensor_size)
        half = (full[0], full[1] // 2, full[2] // 2)

        def layer(dim, res, depth, heads, dp):
            return CondBasicLayer(dim, res, depth, heads, window_size, dp, cond_dim)

        self.layer1 = layer(emb_dim, full, 2 * d, num_heads[0], drop_path[: 2 * d])
        self.downsample = Downsample(emb_dim, full, half)
        self.layer2 = layer(2 * emb_dim, half, 6 * d, num_heads[1], drop_path[2 * d :])
        self.layer3 = layer(2 * emb_dim, half, 6 * d, num_heads[2], drop_path[2 * d :])
        self.upsample = Upsample(2 * emb_dim, emb_dim, half, full)
        self.layer4 = layer(2 * emb_dim, full, 2 * d, num_heads[3], drop_path[: 2 * d])

    def _maybe_checkpoint(self, f, *a):
        if self.gradient_checkpointing:
            return checkpoint(f, *a, use_reentrant=False)
        return f(*a)

    def forward(self, x, cond_emb):
        B, _, *grid = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self._maybe_checkpoint(self.layer1, x, cond_emb)
        skip = x
        x = self._maybe_checkpoint(self.downsample, x)
        x = self._maybe_checkpoint(self.layer2, x, cond_emb)
        x = self._maybe_checkpoint(self.layer3, x, cond_emb)
        x = self._maybe_checkpoint(self.upsample, x)
        x = self._maybe_checkpoint(self.layer4, torch.cat([x, skip], dim=-1), cond_emb)
        return x.transpose(1, 2).reshape(B, -1, *grid)
