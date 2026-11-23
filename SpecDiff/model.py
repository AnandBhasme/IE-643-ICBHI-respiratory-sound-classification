"""Conditional U-Net for diffusion directly on AST log-mel inputs.

Input / output layout: (B, 1, F=128, T=512). Conditioning: diffusion time,
lung-sound class, stethoscope, and cycle duration (fraction of the 5 s window).
Class and stethoscope each have an extra "null" index used for
classifier-free guidance.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 4
N_DEVICES = 4
DEVICES = ['AKGC417L', 'LittC2SE', 'Meditron', 'Litt3200']
CLASSES = ['normal', 'crackle', 'wheeze', 'both']


def sinusoidal_embedding(values, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=values.device) / half)
    args = values[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout * 2)
        self.norm2 = nn.GroupNorm(32, cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch, head_dim=64):
        super().__init__()
        self.heads = max(ch // head_dim, 1)
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        # .contiguous(): torch 2.0's fused attention kernels reject a non-contiguous last dimension for some batch sizes
        # (RuntimeError "query: last dimension must be contiguous"); the values are unchanged.
        out = F.scaled_dot_product_attention(q.transpose(-1, -2).contiguous(), k.transpose(-1, -2).contiguous(),
                                             v.transpose(-1, -2).contiguous())
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3))  # nearest; bf16-safe


class SpecUNet(nn.Module):
    def __init__(self, base=96, mults=(1, 2, 3, 4), num_res=2, attn_levels=(2, 3), dropout=0.1, patch=2):
        super().__init__()
        self.config = dict(base=base, mults=tuple(mults), num_res=num_res,
                           attn_levels=tuple(attn_levels), dropout=dropout, patch=patch)
        # 2x2 pixel-unshuffle stem: the U-Net works at (F/2, T/2) = 64 x 256,
        # which cuts activation memory and compute by about 4x.
        self.patch = patch
        emb_dim = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.dur_mlp = nn.Sequential(nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(N_CLASSES + 1, emb_dim)
        self.dev_emb = nn.Embedding(N_DEVICES + 1, emb_dim)

        self.in_conv = nn.Conv2d(patch * patch, base, 3, padding=1)
        self.down = nn.ModuleList()
        skip_chs = [base]
        ch = base
        for level, mult in enumerate(mults):
            for _ in range(num_res):
                block = nn.ModuleList([ResBlock(ch, base * mult, emb_dim, dropout)])
                ch = base * mult
                if level in attn_levels:
                    block.append(AttnBlock(ch))
                self.down.append(block)
                skip_chs.append(ch)
            if level != len(mults) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for i in range(num_res + 1):
                block = nn.ModuleList([ResBlock(ch + skip_chs.pop(), base * mult, emb_dim, dropout)])
                ch = base * mult
                if level in attn_levels:
                    block.append(AttnBlock(ch))
                if level != 0 and i == num_res:
                    block.append(Upsample(ch))
                self.up.append(block)

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(),
                                 zero_module(nn.Conv2d(ch, patch * patch, 3, padding=1)))

    def forward(self, x, t, cls, dev, dur):
        """t in [0, 1]; cls in 0..4 (4 = null); dev in 0..4 (4 = null); dur in (0, 1]."""
        emb = self.time_mlp(sinusoidal_embedding(t * 1000.0, self.base))
        emb = emb + self.dur_mlp(sinusoidal_embedding(dur * 1000.0, self.base))
        emb = emb + self.class_emb(cls) + self.dev_emb(dev)
        h = self.in_conv(F.pixel_unshuffle(x, self.patch))
        skips = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for block in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return F.pixel_shuffle(self.out(h), self.patch)
