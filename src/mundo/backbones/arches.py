"""ArchesWeather Swin network (Couairon et al., 2026) with time on the level axis."""

import torch
from torch import nn

from mundo.backbones.arches_geometry import time_tokens
from mundo.backbones.archesweather import ArchesWeatherCondBackbone, TimestepEmbedder
from mundo.backbones.geometry import pad_to


class ArchesNet(nn.Module):
    LEVEL_NOISE = True  # noise goes in as extra channels (entry_noise)
    READS_BASE = True  # time is a token axis

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
        emb=96,
        grad_ckpt=True,
        cond_dim=32,
        n_noise=16,
        window=(2, 6, 12),
        L_out=None,
        entry_noise=0,
        zero_head=True,
    ):
        super().__init__()
        assert len(spatial) == 2, "arches is 2D"
        L_out = L if L_out is None else L_out
        assert emb % 12 == 0, (
            "arches runs 6 and 12 attention heads: emb must be a multiple of 12"
        )
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
            "emb": emb,
            "grad_ckpt": grad_ckpt,
            "cond_dim": cond_dim,
            "n_noise": n_noise,
            "window": tuple(window),
            "entry_noise": entry_noise,
            "zero_head": zero_head,
        }
        H_pad, W_pad = (-(-s // 8) * 8 for s in spatial)
        self._n = max(L, L_out)
        self._z, pad_after = time_tokens(self._n)
        self.z_pad = nn.ZeroPad3d((0, 0, 0, 0, 1, pad_after))
        self.level_proj = nn.Conv3d(c_level + entry_noise, emb, kernel_size=2, stride=2)
        self.surface_proj = nn.Conv2d(c_surface, emb, kernel_size=2, stride=2)
        self.backbone = ArchesWeatherCondBackbone(
            tensor_size=(self._z + 1, H_pad // 2, W_pad // 2),
            emb_dim=emb,
            cond_dim=cond_dim,
            window_size=tuple(window),
            depth_multiplier=1,
            gradient_checkpointing=grad_ckpt,
        )
        self.deconv = nn.Conv2d(emb, c_out * 4, kernel_size=3, padding=1, bias=False)
        if zero_head:  # an untrained net writes a zero residual
            nn.init.zeros_(self.deconv.weight)
        self.shuffle = nn.PixelShuffle(2)
        self.cond_emb = nn.ModuleList(TimestepEmbedder(cond_dim) for _ in range(n_cond))
        self.mdi_lat = nn.Conv3d(n_noise, emb, 1) if mdi else None
        self.mdi_dec = nn.Conv2d(n_noise, emb, 1) if mdi else None

    def _decode_index(self, L):
        """Return the time token that decodes each of the `L` output frames."""
        return torch.arange(1, L + 1) % self._z

    def forward(self, level, surface, cond, generator=None):
        """(B, L, c_level, H, W) * (B, c_surface, H, W) * (B, n_cond)
        -> (B, L_out, c_out, H, W).
        """
        B, L_out = level.shape[0], self.cfg["L_out"]
        per = self.cfg["periodic"]
        if level.shape[1] < self._n:
            assert self._n % level.shape[1] == 0, "frames read must divide the grid"
            level = level.repeat_interleave(self._n // level.shape[1], dim=1)
        if self.cfg["entry_noise"]:
            eps = torch.randn(
                B,
                self._n,
                self.cfg["entry_noise"],
                *level.shape[3:],
                generator=generator,
                device=level.device,
            )
            level = torch.cat([level, eps], dim=2)
        lv, crop = pad_to(level.movedim(1, 2), 8, per)
        sf, _ = pad_to(surface, 8, per)
        c = level.new_zeros(B, self.cfg["cond_dim"])
        for i, emb in enumerate(self.cond_emb):
            c = c + emb(cond[:, i])
        lv = self.level_proj(self.z_pad(lv))
        x = torch.cat([self.surface_proj(sf).unsqueeze(2), lv], dim=2)
        if self.mdi_lat is not None:
            eps = torch.randn(
                B,
                self.cfg["n_noise"],
                *x.shape[2:],
                generator=generator,
                device=x.device,
            )
            x = x + self.mdi_lat(eps)
        E = self.cfg["emb"]
        x = self.backbone(x, c)[:, :, 1:]  # drop the surface token
        x = x.reshape(B, E, 2, self._z, *x.shape[3:]).flatten(2, 3)
        x = x[:, :, 1 : L_out + 1].movedim(2, 1).flatten(0, 1)  # (B * L_out, E, h, w)
        if self.mdi_dec is not None:
            eps = torch.randn(
                x.shape[0],
                self.cfg["n_noise"],
                *x.shape[2:],
                generator=generator,
                device=x.device,
            )
            x = x + self.mdi_dec(eps)
        out = crop(self.shuffle(self.deconv(x)))
        return out.reshape(B, L_out, -1, *self.cfg["spatial"])
