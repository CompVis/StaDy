"""StaDy's two released architectures, training objective, and inference API.

Videos use BTHWC and images BHWC, with RGB values in [-1, 1]. Parameter names
match the released checkpoints. DINOv2-derived components below are Copyright
(c) Meta Platforms, Inc. and affiliates, Apache-2.0; see LICENSE-DINOv2 and
THIRD_PARTY_NOTICES.md (via cloneofsimo/minDinoV2).
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import partial, reduce
from types import SimpleNamespace

import diffusers
import einops
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.stateless import _reparametrize_module


# Positions and transformer building blocks

class Pos:

    def map(self, fn):
        return type(self)(**{f.name: fn(getattr(self, f.name)) for f in fields(self)})

    def __getitem__(self, index):
        return self.map(lambda x: x[index])

    def to(self, *args, **kwargs):
        return self.map(lambda x: x.to(*args, **kwargs))

    def mean(self, *args, **kwargs):
        return self.map(lambda x: x.mean(*args, **kwargs))

    def rearrange(self, pattern):
        return self.map(lambda x: rearrange(x, pattern))

    def reduce(self, pattern, **kwargs):
        return self.map(lambda x: einops.reduce(x, pattern, **kwargs))

    def repeat(self, pattern, **kwargs):
        return self.map(lambda x: repeat(x, pattern, **kwargs))

    @classmethod
    def cat(cls, positions, dim=0):
        return cls(
            **{f.name: torch.cat([getattr(p, f.name) for p in positions], dim) for f in fields(positions[0])}
        )

    @classmethod
    def stack(cls, positions, dim=0):
        return cls(
            **{
                f.name: torch.stack([getattr(p, f.name) for p in positions], dim)
                for f in fields(positions[0])
            }
        )


@dataclass
class Pos2D(Pos):
    y: torch.Tensor
    x: torch.Tensor

    @classmethod
    def make_grid(cls, b, h, w, *, device):
        (y_min, y_max, x_min, x_max) = bounding_box(h=h, w=w)
        h_pos = centers(y_min, y_max, num=h, dtype=torch.float32, device=device)
        w_pos = centers(x_min, x_max, num=w, dtype=torch.float32, device=device)
        grid: torch.Tensor = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
        return cls(y=grid[None, :, :, 0].expand(b, h, w), x=grid[None, :, :, 1].expand(b, h, w))


@dataclass
class Pos2DT(Pos):
    y: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor

    @classmethod
    def make_grid(cls, b, t, h, w, *, device):
        pos2d: Pos2D = Pos2D.make_grid(b=b, h=h, w=w, device=device)
        return cls(
            y=pos2d.y[:, None].expand(b, t, h, w),
            x=pos2d.x[:, None].expand(b, t, h, w),
            t=torch.arange(t, dtype=torch.int64, device=device)
            .float()[None, :, None, None]
            .expand(b, t, h, w),
        )


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, steps=num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj
    (y_min, y_max, x_min, x_max) = (-1.0, 1.0, -1.0, 1.0)
    if ar_adj > 1:
        (y_min, y_max) = (-1 / ar_adj, 1 / ar_adj)
    elif ar_adj < 1:
        (x_min, x_max) = (-ar_adj, ar_adj)
    return (y_min, y_max, x_min, x_max)


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def apply_wd(layer, wd=0.02):
    nn.init.normal_(layer.weight, std=wd)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return (q * scale_q.to(q.dtype), k * scale_k.to(k.dtype))


def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    (x, gate) = x.chunk(2, dim=-1)
    return x * F.silu(gate)


class LinearSwiGLU(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return linear_swiglu(x, self.weight, self.bias)


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0, d_out=None):
        super().__init__()
        d_out = d_out or d_model
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearSwiGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_out, bias=False)))
        if d_out != d_model:
            self.skip_proj = apply_wd(nn.Linear(d_model, d_out, bias=False))
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x, cond_norm=None, **kwargs):
        skip = self.skip_proj(x)
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class AdaRMSNorm(nn.Module):

    def __init__(self, features, cond_features, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        if len(x.shape) != len(cond.shape):
            assert len(cond.shape) == 2, f"x.shape={x.shape!r}, cond.shape={cond.shape!r}"
            while len(cond.shape) < len(x.shape):
                cond = cond.unsqueeze(1)
        return rms_norm(x, self.linear(cond) + 1, self.eps)


class RMSNorm(nn.Module):

    def __init__(self, shape, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class MappingNetwork(nn.Module):

    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList(
            [FeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class FourierFeatures(nn.Module):

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


@dataclass
class RoPEAxisFreqs:
    min_freq: float
    max_freq: float
    log_spacing: bool = True


RelativeRoPEAxisFreqs = partial(RoPEAxisFreqs, min_freq=math.pi, max_freq=10.0 * math.pi, log_spacing=True)
_default_theta = 100
AbsoluteTemporalRoPEAxisFreqs = lambda theta=_default_theta: RoPEAxisFreqs(
    min_freq=1.0, max_freq=1.0 / theta, log_spacing=True
)


class GenericRoPEBase(nn.Module):

    def __init__(self, d_head, n_heads, freqs_axes, d_axes):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        assert set(freqs_axes) == set(
            d_axes
        ), f"Axes {freqs_axes} and d_axes {d_axes} must have the same keys"
        assert (
            2 * sum(d_axes.values()) <= self.d_head
        ), f"Sum of d_axes d_axes={d_axes!r} * 2 must be less than or equal to self.d_head={self.d_head!r}, got 2 * sum(d_axes.values())={2 * sum(d_axes.values())!r}."
        for k, freqs_axis in freqs_axes.items():
            d_axis = d_axes[k]
            self.register_buffer(f"freqs_{k}", self._make_freqs(freqs_axis, d_axis))

    def _make_freqs(self, freqs_axis, d_axis):
        if freqs_axis.log_spacing:
            (min_freq, max_freq) = (math.log(freqs_axis.min_freq), math.log(freqs_axis.max_freq))
        else:
            (min_freq, max_freq) = (freqs_axis.min_freq, freqs_axis.max_freq)
        freqs = torch.linspace(min_freq, max_freq, self.n_heads * d_axis + 1)[:-1]
        if freqs_axis.log_spacing:
            freqs = freqs.exp()
        return freqs.view(d_axis, self.n_heads).mT.contiguous()

    def apply_emb(self, x, theta):
        dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
        d = theta.shape[-1]
        assert d * 2 <= x.shape[-1]
        (x1, x2, x3) = (x[..., :d], x[..., d : d * 2], x[..., d * 2 :])
        (x1, x2, theta) = (x1.to(dtype), x2.to(dtype), theta.to(dtype))
        (cos, sin) = (torch.cos(theta), torch.sin(theta))
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        (y1, y2) = (y1.to(x.dtype), y2.to(x.dtype))
        return torch.cat((y1, y2, x3), dim=-1)


class AxialRoPE2D(GenericRoPEBase):

    def __init__(self, d_head, n_heads, half_embedding=True):
        d_axis = d_head // 2 // 2 // (2 if half_embedding else 1)
        super().__init__(
            d_head,
            n_heads,
            freqs_axes={"h": RelativeRoPEAxisFreqs(), "w": RelativeRoPEAxisFreqs()},
            d_axes={"h": d_axis, "w": d_axis},
        )

    def forward(self, pos):
        assert isinstance(pos, Pos2D), f"Pos must be a Pos2D, got {type(pos)}."
        theta_h = pos.y[..., None, None] * self.freqs_h.to(pos.y.dtype)
        theta_w = pos.x[..., None, None] * self.freqs_w.to(pos.x.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class AxialRoPE2DT(GenericRoPEBase):

    def __init__(self, d_head, n_heads, partial_embedding=True):
        d_axis = d_head // 2 // (4 if partial_embedding else 3)
        super().__init__(
            d_head,
            n_heads,
            freqs_axes={
                "t": AbsoluteTemporalRoPEAxisFreqs(),
                "h": RelativeRoPEAxisFreqs(),
                "w": RelativeRoPEAxisFreqs(),
            },
            d_axes={"t": d_axis, "h": d_axis, "w": d_axis},
        )

    def forward(self, pos):
        assert isinstance(pos, Pos2DT), f"Pos must be a Pos2DT, got {type(pos)}."
        theta_t = pos.t[..., None, None] * self.freqs_t.to(pos.t.dtype)
        theta_h = pos.y[..., None, None] * self.freqs_h.to(pos.y.dtype)
        theta_w = pos.x[..., None, None] * self.freqs_w.to(pos.x.dtype)
        return torch.cat((theta_t, theta_h, theta_w), dim=-1)


DEFAULT_GLU_FF_EXPAND = 3


class GenericTransformerLayer(nn.Module):

    def __init__(
        self,
        d_model,
        pos_enc_cls=None,
        pos_enc_params={},
        self_attn_params={},
        ffn_params={},
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=DEFAULT_GLU_FF_EXPAND,
    ):
        super().__init__()
        d_ff = d_model * ff_expand
        self.self_attn = GenericAttentionBlock(
            d_model=d_model,
            pos_enc_cls=pos_enc_cls,
            pos_enc_params=pos_enc_params,
            d_head=d_head,
            d_cond_norm=d_cond_norm,
            dropout=dropout,
            **self_attn_params,
        )
        self.ff = FeedForwardBlock(
            d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm, dropout=dropout, **ffn_params
        )

    def forward(self, x, pos, cond_norm=None, **kwargs):
        x = self.self_attn(x, pos=pos, cond_norm=cond_norm, **kwargs)
        x = self.ff(x, cond_norm=cond_norm, **kwargs)
        return x


class GenericAttentionBlock(nn.Module):

    def __init__(
        self, d_model, pos_enc_cls=None, pos_enc_params={}, d_head=64, d_cond_norm=None, dropout=0.0
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.eps = 1e-06
        self.pos_emb = None
        if pos_enc_cls is not None:
            self.pos_emb = pos_enc_cls(d_head, self.n_heads, **pos_enc_params)

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(self, x, pos, cond_norm=None, **kwargs):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        (B, *DIMS, _) = x.shape
        x = rearrange(x, "b ... d -> b (...) d")
        qkv = self.qkv_proj(x)
        (q, k, v) = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        (q, k) = scale_for_cosine_sim(
            q, k, self.scale[:, None, None], torch.tensor(self.eps, device=x.device)
        )
        if self.pos_emb is not None:
            assert pos is not None
            pos = pos.to(qkv.dtype).rearrange("b ... -> b (...)")
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.view(B, *DIMS, x.size(-1)) + skip


class TokenMerge2D(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Linear(in_features * self.h * self.w, out_features, bias=False)

    def forward(self, x, pos=None, **kwargs):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        if pos is not None:
            pos = pos.reduce("... (h nh) (w nw) -> ... h w", reduction="mean", nh=self.h, nw=self.w)
        return (self.proj(x), pos)


class AddExtraTokensProj(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.extra_proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, pos, x_extra, pos_extra, **kwargs):
        return (
            torch.cat([self.proj(x), self.extra_proj(x_extra)], dim=-2),
            type(pos).cat((pos, pos_extra), dim=-1),
        )


class RemoveExtraTokensProj(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, x_extra, **kwargs):
        pos = kwargs.get("pos", None)
        num_extra = x_extra.shape[-2]
        x = x[..., :-num_extra, :]
        if pos is not None:
            pos = pos.map(lambda t: t[..., :-num_extra])
            return (self.proj(x), pos)
        return self.proj(x)


class AddExtraTokensMerge2D(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.proj = TokenMerge2D(in_features, out_features, patch_size)

    def forward(self, x, pos, x_extra=None, pos_extra=None, **kwargs):
        (x, pos) = self.proj(x, pos=pos)
        x = rearrange(x, "... h w e -> ... (h w) e")
        pos = pos.rearrange("... h w -> ... (h w)")
        if x_extra is not None:
            x = torch.cat((x, x_extra), dim=-2)
            pos = type(pos).cat((pos, pos_extra), dim=-1)
        return (x, pos)


class RemoveExtraTokensSplitLast2D(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(2, 2), zero_init=True):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = nn.Linear(in_features, out_features * self.h * self.w, bias=False)
        if zero_init:
            nn.init.zeros_(self.proj.weight)

    def forward(self, x, x_extra=None, skip=None, **kwargs):
        if x_extra is not None:
            x = x[..., : -x_extra.shape[-2], :]
        if skip is not None and x.ndim == skip.ndim - 1:
            h = skip.shape[-3] // self.h
            w = skip.shape[-2] // self.w
            x = rearrange(x, "... (h w) e -> ... h w e", h=h, w=w)
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return x


class Transformer(nn.Module):

    def __init__(self, in_features, out_features, width, depth, d_head, temporal=False, d_cond_norm=None):
        super().__init__()
        rope = AxialRoPE2DT if temporal else AxialRoPE2D
        self.mid_level = nn.ModuleList(
            [
                GenericTransformerLayer(
                    d_model=width, d_head=d_head, pos_enc_cls=rope, d_cond_norm=d_cond_norm
                )
                for _ in range(depth)
            ]
        )
        merge = AddExtraTokensProj if temporal else AddExtraTokensMerge2D
        split = RemoveExtraTokensProj if temporal else RemoveExtraTokensSplitLast2D
        self.mid_merge = merge(in_features, width)
        self.mid_split = split(width, out_features)

    def forward(self, x, pos, **kwargs):
        skip = x
        (x, pos) = self.mid_merge(x, pos=pos, **kwargs)
        for layer in self.mid_level:
            x = layer(x, pos=pos, **kwargs)
        return self.mid_split(x, skip=skip, **kwargs)


# DINOv2 feature encoder (Apache-2.0)

def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        (B, N, C) = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        (q, k, v) = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer=Mlp,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        (w0, h0) = (w // self.patch_size, h // self.patch_size)
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            (sx, sy) = (float(w0 + self.interpolate_offset) / M, float(h0 + self.interpolate_offset) / M)
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = (
            nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                **kwargs,
            )
            .permute(0, 2, 3, 1)
            .view(1, -1, dim)
        )
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(x.dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        (B, nc, w, h) = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.size(0), -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }


def vit_base(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


vit_base_reg = partial(vit_base, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0)
vit_large_reg = partial(vit_large, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0)


class MinDinoV2Reg(nn.Module):

    def __init__(self, model_version="dinov2_vitl14_reg", gradient_last_blocks=2):
        super().__init__()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*xFormers.*")
            original = torch.hub.load(
                "facebookresearch/dinov2", model_version, verbose=False, skip_validation=True
            )
        self.model = {"dinov2_vitb14_reg": vit_base_reg, "dinov2_vitl14_reg": vit_large_reg}[model_version]()
        self.model.load_state_dict(original.state_dict(), strict=True)
        self.model.requires_grad_(False).eval()
        for block in self.model.blocks[-gradient_last_blocks:]:
            block.requires_grad_(True).train()

    def forward(self, imgs):
        assert imgs.ndim == 4 and imgs.shape[1] == imgs.shape[2]
        imgs = imgs.movedim(-1, 1)
        factor = imgs.shape[-1] // 224
        if factor > 1:
            imgs = F.avg_pool2d(imgs, factor)
        imgs = F.interpolate(imgs, [224, 224], mode="bilinear", antialias=True)
        imgs = (imgs + 1.0) / 2.0
        imgs = (imgs - torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225], device=imgs.device
        )[:, None, None]
        features = self.model.forward_features(imgs)["x_norm_patchtokens"]
        return rearrange(features, "b (h w) d -> b h w d", h=16, w=16)


# Image latents and motion encoder

class TinyAutoencoderKL(nn.Module):

    def __init__(self, repo="madebyollin/taesd"):
        super().__init__()
        self.ae = diffusers.AutoencoderTiny.from_pretrained(repo)
        self.ae.eval()
        self.ae.requires_grad_(False)

    def forward(self, img):
        return self.encode(img=img)

    @torch.no_grad()
    def encode(self, img):
        img = img.movedim(-1, 1)
        latent = self.ae.encode(img, return_dict=False)[0]
        latent = latent.movedim(1, -1)
        return latent

    @torch.no_grad()
    def decode(self, latent):
        latent = latent.movedim(-1, 1)
        rec = self.ae.decode(latent, return_dict=False)[0]
        rec = rec.movedim(1, -1)
        return rec


class MotionEncoder(Transformer):

    def __init__(self, d_motion, d_frame, num_queries_per_frame=1, num_register_tokens=0, **kwargs):
        super().__init__(**kwargs)
        width = self.mid_merge.proj.out_features
        self.motion_dim = d_motion
        self.num_queries_per_frame = num_queries_per_frame
        self.num_register_tokens = num_register_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries_per_frame, width))
        self.frame_proj = nn.Linear(d_frame, width, bias=False)
        self.head = nn.Sequential(
            RMSNorm(width),
            FeedForwardBlock(width, 3 * width),
            FeedForwardBlock(width, 3 * width, d_out=d_motion),
            RMSNorm(d_motion),
        )

    def forward(self, x, **kwargs):
        (B, T, H, W, D) = x.shape
        x = self.frame_proj(x)
        x_pos = Pos2DT.make_grid(B, T, H, W, device=x.device)
        query_tokens = repeat(self.query_tokens, "1 n d -> b (t n) d", b=B, t=T - 1)
        query_pos = Pos2DT.stack([x_pos[:, 1:], x_pos[:, :-1]], dim=-3).mean(dim=(2, 3, 4))
        query_pos = query_pos.repeat("b t -> b (t n)", n=self.num_queries_per_frame)
        motion_embeddings = super().forward(
            x=query_tokens,
            pos=query_pos,
            x_extra=rearrange(x, "b t h w d -> b (t h w) d"),
            pos_extra=x_pos.rearrange("b t h w -> b (t h w)"),
        )
        motion_embeddings = self.head(motion_embeddings)
        motion_embeddings = rearrange(
            motion_embeddings, "b (t n) d -> b t n d", t=T - 1, n=self.num_queries_per_frame
        )
        return motion_embeddings


@contextmanager
def eval_mode(model):
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in modes.items():
            module.training = training


# Latent flow matching and the StaDy objective

class FM(nn.Module, ABC):

    def __init__(self, backbone, time_mapping, ae=None, cond_dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.ae = ae
        self.cond_dropout = cond_dropout
        self.time_emb = FourierFeatures(1, time_mapping.width)
        self.time_in_proj = nn.Linear(time_mapping.width, time_mapping.width, bias=False)
        self.time_mapping = MappingNetwork(
            n_layers=time_mapping.depth,
            d_model=time_mapping.width,
            d_ff=time_mapping.d_ff,
            dropout=time_mapping.dropout,
        )

    def sample_train_t(self, data_dims, device):
        return torch.sigmoid(torch.randn((data_dims[0], *[1] * (len(data_dims) - 1)), device=device) - 0.0)

    def get_sampling_t(self, data_dims, steps, device):
        (B, *sample_dims) = data_dims
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)[:, None].expand(steps + 1, B)
        return ts.view(steps + 1, B, *[1] * len(sample_dims))

    def sample_cond_mask(self, data_dims, device, force_all=None):
        (B, *sample_dims) = data_dims
        dropout_shape = (B, *[1] * len(sample_dims))
        if force_all is None and (not self.training):
            force_all = True
        if force_all is not None:
            return torch.full(dropout_shape, force_all, device=device, dtype=torch.bool)
        else:
            return torch.rand(dropout_shape, device=device) >= self.cond_dropout

    def encode(self, x):
        if self.ae is None:
            return x
        return self.ae.encode(x)

    def decode(self, x):
        if self.ae is None:
            return x
        return self.ae.decode(x)

    def get_cond(self, t, **data_kwargs):
        return {"cond_norm": self.time_mapping(self.time_in_proj(self.time_emb(t)))}

    def get_loss_weight(self, t):
        return 1.0

    def get_target(self, *, x_0, x_1, x_t, t):
        return x_1 - x_0

    @abstractmethod
    def predict_v(self, x_t, cond_dropout_mask, **cond_kwargs):
        pass

    @torch.no_grad()
    def sample(self, x_0, steps=50, **data_kwargs):
        with eval_mode(self):
            ts = self.get_sampling_t(x_0.shape, steps=steps, device=x_0.device)
            mask_keep = self.sample_cond_mask(x_0.shape, device=x_0.device, force_all=True)
            x_t = x_0
            for t_start, t_end in zip(ts[:-1], ts[1:]):
                v_pred = self.predict_v(
                    x_t=x_t, t=t_start, cond_dropout_mask=mask_keep, **self.get_cond(t_start, **data_kwargs)
                )
                x_t = x_t + (t_end - t_start) * v_pred
            return self.decode(x_t)


class FrameToFrameFM(FM):

    def __init__(self, frame_embedder, frame_embedder_dim, **kwargs):
        super().__init__(**kwargs)
        width = self.backbone.mid_merge.proj.proj.out_features
        self.frame_embedder = frame_embedder
        self.frame_proj = nn.Linear(frame_embedder_dim, width, bias=False)
        self.null_token = nn.Parameter(torch.zeros(width))

    def get_cond(self, t, x_cond, x_cond_mask=None, **data_kwargs):
        cond_dict = dict(super().get_cond(t, **data_kwargs))
        frame_embs = self.frame_proj(self.frame_embedder(x_cond))
        frame_pos = Pos2D.make_grid(*frame_embs.shape[:3], device=x_cond.device).rearrange("b h w -> b (h w)")
        frame_embs = rearrange(frame_embs, "b h w c -> b (h w) c")
        if x_cond_mask is None and self.cond_dropout > 0:
            x_cond_mask = self.sample_cond_mask(frame_embs.shape, device=frame_embs.device)
        if x_cond_mask is not None:
            frame_embs = torch.where(x_cond_mask, frame_embs, self.null_token[None, None, :])
        cond_dict["x_extra"] = frame_embs
        cond_dict["pos_extra"] = frame_pos
        return cond_dict

    def predict_v(self, x_t, cond_dropout_mask, **cond_kwargs):
        (B, H, W, C) = x_t.shape
        if "cond_norm" in cond_kwargs and cond_kwargs["cond_norm"].ndim > 2:
            cond_kwargs["cond_norm"] = rearrange(cond_kwargs["cond_norm"], "b ... d -> b (...) d")
        return self.backbone(x=x_t, pos=Pos2D.make_grid(B, H, W, device=x_t.device), **cond_kwargs)


class FrameAndMotionToFrameFM(FrameToFrameFM):

    def __init__(self, motion_encoder, **kwargs):
        super().__init__(**kwargs)
        width = self.backbone.mid_merge.proj.proj.out_features
        motion_dim = motion_encoder.motion_dim
        self.num_motion_embeddings = motion_encoder.num_queries_per_frame + motion_encoder.num_register_tokens
        self.motion_encoder = motion_encoder
        self.motion_proj = nn.Linear(motion_dim, width, bias=False)
        self.motion_null_token = nn.Parameter(torch.zeros(width))

    def forward(self, video):
        model = self
        (b, frames) = video.shape[:2]
        features = model.frame_embedder(video.flatten(0, 1))
        features = features.reshape(b, frames, *features.shape[1:])
        motion = model.motion_encoder(features)[:, 0]
        (source, target) = (video[:, 0], video[:, 1])
        clean = model.encode(target)
        noise = torch.randn_like(clean)
        t = model.sample_train_t(clean.shape, device=video.device)
        x_t = (1 - t) * noise + t * clean
        velocity = model.get_target(x_0=noise, x_1=clean, x_t=x_t, t=t)

        def mask(shape, keep):
            return model.sample_cond_mask(shape, device=video.device, force_all=keep)

        source_mask = torch.cat([mask((b, 1, 1), keep) for keep in (True, False, False)])
        motion_mask = torch.cat([mask(motion.shape, keep) for keep in (True, True, False)])
        (full, probe, uncond) = model.predict_v(
            x_t=torch.cat([x_t] * 3),
            t=torch.cat([t] * 3),
            cond_dropout_mask=None,
            **model.get_cond(
                torch.cat([t] * 3),
                x_cond=torch.cat([source] * 3),
                motion_embeddings=torch.cat([motion, motion.detach(), torch.zeros_like(motion)]),
                x_cond_mask=source_mask,
                motion_cond_mask=motion_mask,
            ),
        ).chunk(3)
        frozen = {
            name: p if name.startswith("motion_") else p.detach() for (name, p) in model.named_parameters()
        }
        frozen.update(dict(model.named_buffers()))
        with _reparametrize_module(model, frozen, tie_weights=True, strict=True):
            regularized = model.predict_v(
                x_t=x_t,
                t=t,
                cond_dropout_mask=None,
                **model.get_cond(
                    t,
                    x_cond=source,
                    motion_embeddings=motion,
                    x_cond_mask=mask((b, 1, 1), False),
                    motion_cond_mask=mask(motion.shape, True),
                ),
            )
        weight = model.get_loss_weight(t)

        def loss(pred, target):
            return (weight * (pred - target)).square().mean(tuple(range(1, clean.ndim)))

        return (
            dict(
                full=loss(full, velocity),
                motion=loss(probe, velocity),
                uncond=loss(uncond, velocity),
                reg=loss(regularized, uncond.detach()),
            ),
            {},
        )

    def get_cond(self, t, x_cond, motion_embeddings, x_cond_mask=None, motion_cond_mask=None, **data_kwargs):
        cond_dict = dict(super().get_cond(t, x_cond, x_cond_mask=x_cond_mask, **data_kwargs))
        if motion_embeddings is not None:
            motion_embeddings = self.motion_proj(motion_embeddings)
            if motion_cond_mask is not None:
                motion_embeddings = torch.where(
                    motion_cond_mask, motion_embeddings, self.motion_null_token[None, None, :]
                )
        else:
            motion_embeddings = self.motion_null_token.expand(x_cond.shape[0], self.num_motion_embeddings, -1)
        motion_pos = Pos2D.make_grid(motion_embeddings.shape[0], 1, 1, device=x_cond.device).repeat(
            "b h w -> b (n h w)", n=self.num_motion_embeddings
        )
        x_key = "x_extra"
        pos_key = "pos_extra"
        cond_dict[x_key] = (
            torch.cat([cond_dict[x_key], motion_embeddings], dim=1)
            if x_key in cond_dict
            else motion_embeddings
        )
        cond_dict[pos_key] = (
            Pos2D.cat([cond_dict[pos_key], motion_pos], dim=1) if pos_key in cond_dict else motion_pos
        )
        return cond_dict


class StaDy(FrameAndMotionToFrameFM):
    """Paper open-world architecture. Inputs are RGB tensors in [-1, 1].

    The regularizer retains the paper run's ``motion_`` gradient boundary:
    motion encoder, motion projection and motion null token remain trainable.
    """

    def __init__(
        self,
        motion_dim=96,
        decoder_width=1152,
        decoder_depth=24,
        decoder_head_dim=72,
        motion_width=768,
        motion_depth=12,
        motion_head_dim=64,
        num_motion_queries=1,
    ):
        config = dict(
            motion_dim=motion_dim,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            decoder_head_dim=decoder_head_dim,
            motion_width=motion_width,
            motion_depth=motion_depth,
            motion_head_dim=motion_head_dim,
            num_motion_queries=num_motion_queries,
        )
        time_mapping = SimpleNamespace(depth=2, width=256, d_ff=768, dropout=0.0)
        ae = TinyAutoencoderKL()
        frame_embedder = MinDinoV2Reg(model_version="dinov2_vitl14_reg", gradient_last_blocks=2)
        motion_encoder = MotionEncoder(
            d_motion=motion_dim,
            d_frame=1024,
            num_queries_per_frame=num_motion_queries,
            num_register_tokens=0,
            in_features=motion_width,
            out_features=motion_width,
            width=motion_width,
            depth=motion_depth,
            d_head=motion_head_dim,
            temporal=True,
        )
        backbone = Transformer(
            in_features=4,
            out_features=4,
            width=decoder_width,
            depth=decoder_depth,
            d_head=decoder_head_dim,
            temporal=False,
            d_cond_norm=256,
        )
        super().__init__(
            backbone=backbone,
            time_mapping=time_mapping,
            ae=ae,
            frame_embedder=frame_embedder,
            frame_embedder_dim=1024,
            motion_encoder=motion_encoder,
        )
        self.config = config

    def encode_motion(self, video):
        """Encode [B, T, H, W, 3] into [B, T-1, num_motion_queries, motion_dim]."""
        assert video.ndim == 5 and video.shape[1] >= 2 and (video.shape[-1] == 3)
        features = self.frame_embedder(video.flatten(0, 1))
        features = features.reshape(*video.shape[:2], *features.shape[1:])
        return self.motion_encoder(features)

    @torch.no_grad()
    def transfer(self, source, motion, steps=50):
        """Generate [B,H,W,3] from a source image and [B,num_motion_queries,motion_dim]."""
        assert steps > 0 and source.ndim == 4 and (motion.ndim == 3)
        with eval_mode(self):
            noise = torch.randn_like(self.encode(source))
            return self.sample(noise, steps=steps, x_cond=source, motion_embeddings=motion)

    @classmethod
    def from_pretrained(cls, checkpoint, device="cpu", **model_kwargs):
        """Load a training checkpoint or a bare paper-model state dict."""
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(payload.get("config", {}))
        config.update(model_kwargs)
        model = cls(**config)
        model.load_state_dict(payload.get("model", payload), strict=True)
        return model.to(device).eval()


LOSS_WEIGHTS = dict(full=10.0, motion=1.0, uncond=1.0, reg=2.0)


# CelebV checkpoint architecture

class CelebVMotionExtractor(Transformer):

    def __init__(self, frame_encoder, d_motion, d_content, num_queries_per_frame=1, **kwargs):
        super().__init__(**kwargs)
        width = self.mid_merge.proj.out_features
        self.frame_encoder = frame_encoder
        self.frame_proj = nn.Linear(d_content, width)
        self.head = nn.Sequential(
            RMSNorm(width),
            FeedForwardBlock(width, 3 * width),
            FeedForwardBlock(width, 3 * width, d_out=d_motion),
            RMSNorm(d_motion),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries_per_frame, width))
        self.num_queries_per_frame = num_queries_per_frame

    def forward(self, x, **kwargs):
        frame_tokens = self.frame_encoder(rearrange(x, "b t h w c -> (b t) h w c"))
        frame_tokens = rearrange(self.frame_proj(frame_tokens), "(b t) h w d -> b t h w d", b=x.shape[0])
        (b, t, h, w, d) = frame_tokens.shape
        frame_pos = Pos2DT.make_grid(b, t, h, w, device=x.device)
        query_tokens = repeat(self.query_tokens, "1 n d -> b (t n) d", b=b, t=t - 1)
        query_pos = Pos2DT.stack([frame_pos[:, 1:], frame_pos[:, :-1]], dim=-3).mean(dim=(2, 3, 4))
        query_pos = query_pos.repeat("b t -> b (t n)", n=self.num_queries_per_frame)
        motion_embeddings = super().forward(
            x=query_tokens,
            pos=query_pos,
            x_extra=rearrange(frame_tokens, "b t h w d -> b (t h w) d"),
            pos_extra=frame_pos.rearrange("b t h w -> b (t h w)"),
            **kwargs,
        )
        motion_embeddings = rearrange(self.head(motion_embeddings), "b (t n) d -> b t n d", t=t - 1)
        return motion_embeddings


class LatentFrameGenerator(FM):

    def __init__(self, num_register_tokens=0, d_motion=128, **kwargs):
        super().__init__(**kwargs)
        width = self.backbone.mid_merge.proj.proj.out_features
        self.num_register_tokens = num_register_tokens
        self.start_proj = TokenMerge2D(
            3 if self.ae is None else 4,
            width,
            patch_size=(self.backbone.mid_merge.proj.h, self.backbone.mid_merge.proj.w),
        )
        self.motion_proj = nn.Linear(d_motion, width)

    def get_cond(self, t, **data_kwargs):
        cond = dict(super().get_cond(t, **data_kwargs))
        if "start_frames" in data_kwargs:
            cond["start_frames"] = data_kwargs["start_frames"]
        if "motion_embeddings" in data_kwargs:
            cond["motion_embeddings"] = data_kwargs["motion_embeddings"]
        return cond

    def predict_v(self, x_t, t, **cond_kwargs):
        (B, H, W, C) = x_t.shape
        x_pos = Pos2D.make_grid(B, H, W, device=x_t.device)
        if "cond_norm" in cond_kwargs and cond_kwargs["cond_norm"].ndim > 2:
            cond_kwargs["cond_norm"] = rearrange(cond_kwargs["cond_norm"], "b ... d -> b (...) d")
        (x_extra, pos_extra) = ([], [])
        if "start_frames" in cond_kwargs:
            start_frames = cond_kwargs.pop("start_frames")
            if self.ae is not None:
                start_frames = self.ae(start_frames)
            (start_tokens, start_pos) = self.start_proj(start_frames, pos=x_pos)
            x_extra.append(rearrange(start_tokens, "b h w d -> b (h w) d"))
            pos_extra.append(start_pos.rearrange("b h w -> b (h w)"))
        if "motion_embeddings" in cond_kwargs:
            x_extra.append(self.motion_proj(cond_kwargs.pop("motion_embeddings")))
            pos_extra.append(x_pos.mean(dim=(1, 2)).repeat("b -> b n", n=x_extra[-1].shape[1]))
        x_extra = torch.cat(x_extra, dim=1) if len(x_extra) > 0 else None
        pos_extra = type(x_pos).cat(pos_extra, dim=1) if len(pos_extra) > 0 else None
        cond_kwargs["x"] = x_t
        cond_kwargs["pos"] = x_pos
        cond_kwargs["x_extra"] = x_extra
        cond_kwargs["pos_extra"] = pos_extra
        x_pred = self.backbone(**cond_kwargs)
        return x_pred


class CelebVStaDy(nn.Module):
    """CelebV-HQ model with 64 motion queries of width eight."""

    def __init__(self):
        super().__init__()
        self.motion_extractor = CelebVMotionExtractor(
            frame_encoder=MinDinoV2Reg(model_version="dinov2_vitb14_reg", gradient_last_blocks=2),
            d_motion=8,
            d_content=768,
            num_queries_per_frame=64,
            in_features=384,
            out_features=384,
            width=384,
            depth=6,
            d_head=64,
            temporal=True,
        )
        self.frame_generator = LatentFrameGenerator(
            num_register_tokens=0,
            d_motion=8,
            time_mapping=SimpleNamespace(depth=2, width=256, d_ff=768, dropout=0.0),
            ae=TinyAutoencoderKL(),
            backbone=Transformer(
                in_features=4, out_features=4, width=384, depth=12, d_head=64, temporal=False, d_cond_norm=256
            ),
        )
        self.unconditional_frame_generator = self.frame_generator

    def encode_motion(self, video):
        """Encode RGB [B,T,H,W,3] in [-1,1] into [B,T-1,64,8]."""
        return self.motion_extractor(video)

    @torch.no_grad()
    def transfer(self, source, motion, steps=50):
        """Transfer [B,64,8] motion onto an RGB [B,H,W,3] source."""
        assert steps > 0 and source.ndim == 4 and (motion.ndim == 3)
        with eval_mode(self):
            generator = self.frame_generator
            noise = torch.randn_like(generator.encode(source))
            return generator.sample(noise, steps=steps, start_frames=source, motion_embeddings=motion)
