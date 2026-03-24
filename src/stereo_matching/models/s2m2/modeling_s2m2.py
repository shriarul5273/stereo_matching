"""
S2M2: Scalable Stereo Matching Model (ICCV 2025).

Vendored from third-party/s2m2/src/s2m2/core/model/ in dependency order:
  1. utils.py          — _s2m2_custom_sinc, _s2m2_custom_unfold, _s2m2_coords_grid,
                         _s2m2_bilinear_sampler, _s2m2_get_pe
  2. feature_fusion.py — _S2M2_FeatureFusion
  3. attentions.py     — _S2M2_SelfAttn, _S2M2_CrossAttn, _S2M2_SelfAttnBlock1D,
                         _S2M2_CrossAttnBlock1D, _S2M2_SelfAttnBlock2D,
                         _S2M2_CrossAttnBlock2D, _S2M2_FFN, _S2M2_ConvBlock2D,
                         _S2M2_GlobalAttnBlock, _S2M2_BasicAttnBlock
  4. unet.py           — _S2M2_Unet
  5. submodules.py     — _S2M2_CostVolume, _S2M2_CNNEncoder, _S2M2_UpsampleMask4x,
                         _S2M2_UpsampleMask1x, _s2m2_logsumexp_stable, _S2M2_DispInit
  6. stacked_MRT.py    — _S2M2_MRT, _S2M2_StackedMRT
  7. refinenet.py      — _S2M2_ConvGRU, _S2M2_GlobalRefiner, _S2M2_LocalRefiner
  8. s2m2.py           — _S2M2
  9. Public wrapper    — S2M2Model

All internal class/function names are prefixed _S2M2_ / _s2m2_ to avoid
collisions with other models registered in the same process.

Original code: https://github.com/junhong-3dv/s2m2
License: see third-party/s2m2/LICENSE.md
"""

import logging
import os
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from ...modeling_utils import BaseStereoModel
from .configuration_s2m2 import S2M2Config, _S2M2_VARIANT_MAP

logger = logging.getLogger(__name__)


# ── Section 1: utils ──────────────────────────────────────────────────────── #

def _s2m2_custom_sinc(x):
    return torch.where(
        torch.abs(x) < 1e-6,
        torch.ones_like(x),
        (torch.sin(3.1415 * x) / (3.1415 * x)).to(x.dtype),
    )


def _s2m2_custom_unfold(x, kernel_size=3, padding=1):
    B, C, H, W = x.shape
    p2d = (padding, padding, padding, padding)
    x_pad = F.pad(x, p2d, "replicate")
    x_list = []
    for ind_i in range(kernel_size):
        for ind_j in range(kernel_size):
            x_list.append(x_pad[:, :, ind_i:ind_i + H, ind_j:ind_j + W])
    return torch.cat(x_list, dim=1)


def _s2m2_coords_grid(b, h, w, device):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    stacks = [x, y]
    grid = torch.stack(stacks, dim=0)       # [2, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)   # [B, 2, H, W]
    return grid.to(device)


def _s2m2_bilinear_sampler(img, coords, mode='bilinear'):
    """Wrapper for grid_sample, uses pixel coordinates."""
    W = torch.tensor(img.shape[-1], dtype=img.dtype, device=img.device)
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, mode=mode, align_corners=False)


def _s2m2_get_pe(h: int, w: int, pe_dim: int, dtype, device):
    """Relative positional encoding."""
    with torch.no_grad():
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h - 1, h).to(device).to(dtype),
            torch.linspace(0, w - 1, w).to(device).to(dtype),
            indexing='ij',
        )
        rel_x_pos = (grid_x.reshape(-1, 1) - grid_x.reshape(1, -1)).long()
        rel_y_pos = (grid_y.reshape(-1, 1) - grid_y.reshape(1, -1)).long()

        L = 2 * w + 1
        sig = 5 / pe_dim
        x_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        dim_t = torch.linspace(-1, 1, pe_dim // 2).to(device).to(dtype)
        pe_x = _s2m2_custom_sinc((dim_t[None, :] - x_pos[:, None]) / sig)
        pe_x = F.normalize(pe_x, p=2, dim=-1)
        rel_pe_x = pe_x[rel_x_pos + w - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        L = 2 * h + 1
        y_pos = torch.linspace(-3, 3, L).to(device).to(dtype).tanh()
        pe_y = _s2m2_custom_sinc((dim_t[None, :] - y_pos[:, None]) / sig)
        pe_y = F.normalize(pe_y, p=2, dim=-1)
        rel_pe_y = pe_y[rel_y_pos + h - 1].reshape(h * w, h * w, pe_dim // 2).to(dtype)

        pe = 0.5 * torch.cat([rel_pe_x, rel_pe_y], dim=2)
    return pe.clone()


# ── Section 2: feature_fusion ─────────────────────────────────────────────── #

class _S2M2_FeatureFusion(nn.Module):
    def __init__(self, dim: int, kernel_size: int, use_gate=True):
        super().__init__()
        pad = kernel_size // 2
        self.use_gate = use_gate
        if use_gate:
            self.feature_gate = nn.Sequential(
                nn.Conv2d(2 * dim, dim, kernel_size=kernel_size, padding=pad),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.Sigmoid(),
            )
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1),
        )

    def forward(self, z0: Tensor, z1: Tensor) -> Tensor:
        z = torch.cat([z0, z1], dim=1)
        if self.use_gate:
            eps = 0.01
            w = self.feature_gate(z).clamp(min=eps, max=1 - eps)
            z_out = self.feature_fusion(z) + w * z0 + (1 - w) * z1
        else:
            z_out = self.feature_fusion(z)
        return z_out


# ── Section 3: attentions ─────────────────────────────────────────────────── #

class _S2M2_SelfAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pe = use_pe
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)
        if self.use_pe:
            self.pe_proj = nn.Linear(32, self.head_dim)

    def forward(self, x: Tensor, pe: Tensor = None) -> Tensor:
        B, N, C = x.shape
        scale = self.scale
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_pe:
            score = torch.einsum('...ic, ...jc -> ...ij', scale * q, k)
            attn = score.reshape(B, self.num_heads, N, N).softmax(dim=-1)
            out = torch.einsum('...ij, ...jc -> ...ic', attn, v)
            pe_sum = torch.einsum('...nij, ijc -> ...nic', attn, pe)
            out = out + self.pe_proj(pe_sum)
        else:
            out = cp.checkpoint(F.scaled_dot_product_attention, q, k, v, use_reentrant=False)
        out = self.proj(out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        return out


class _S2M2_CrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)

    def forward(self, x: Tensor, y: Tensor):
        B, N, C = x.shape
        qx = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ky = self.k(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vy = self.v(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_out = F.scaled_dot_product_attention(qx, ky, vy)

        kx = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        qy = self.q(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vx = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_out = F.scaled_dot_product_attention(qy, kx, vx)

        x_out = self.proj(x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        y_out = self.proj(y_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))
        return x_out, y_out


class _S2M2_SelfAttnBlock1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.dim = dim
        self.attn = _S2M2_SelfAttn(dim=dim, num_heads=num_heads,
                                    dim_expansion=dim_expansion, use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        B, H, W, C = z.shape
        z = z.reshape(B * H, W, C)
        z_norm = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z
        return z.reshape(B, H, W, C)


class _S2M2_CrossAttnBlock1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = _S2M2_CrossAttn(dim, num_heads, dim_expansion=dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        z_norm = self.norm_pre(z)
        x, y = z_norm.chunk(2, dim=0)
        B, H, W, C = x.shape
        x, y = x.reshape(B * H, W, C), y.reshape(B * H, W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        return torch.cat([x, y], dim=0) + z


class _S2M2_SelfAttnBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool):
        super().__init__()
        self.dim = dim
        self.attn = _S2M2_SelfAttn(dim=dim, num_heads=num_heads,
                                    dim_expansion=dim_expansion, use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        B, H, W, C = z.shape
        z = z.reshape(B, H * W, C).contiguous()
        z_norm = self.norm_pre(z)
        z = self.attn(z_norm, pe) + z
        return z.reshape(B, H, W, C).contiguous()


class _S2M2_CrossAttnBlock2D(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int):
        super().__init__()
        self.attn = _S2M2_CrossAttn(dim, num_heads, dim_expansion=dim_expansion)
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        z_norm = self.norm_pre(z)
        x, y = z_norm.chunk(2, dim=0)
        B, H, W, C = x.shape
        x, y = x.reshape(B, H * W, C), y.reshape(B, H * W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        return torch.cat([x, y], dim=0) + z


class _S2M2_FFN(nn.Module):
    def __init__(self, dim: int, dim_expansion: int):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_expansion * dim),
            nn.GELU(),
            nn.Linear(dim_expansion * dim, dim),
        )
        self.norm_pre = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.ffn(self.norm_pre(z)) + z


class _S2M2_ConvBlock2D(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dim_expansion: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv2d(dim_expansion * dim, dim, kernel_size, padding=kernel_size // 2),
        )
        self.convs_1x = nn.Sequential(
            nn.Conv2d(dim, dim_expansion * dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim_expansion * dim, dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.convs(z) + self.convs_1x(z)


class _S2M2_GlobalAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int,
                 use_cross_attn: bool = False, use_pe: bool = False):
        super().__init__()
        self.self_attn = _S2M2_SelfAttnBlock2D(dim=dim, num_heads=num_heads,
                                                dim_expansion=dim_expansion, use_pe=use_pe)
        if use_cross_attn:
            self.cross_attn = _S2M2_CrossAttnBlock2D(dim=dim, num_heads=num_heads,
                                                      dim_expansion=dim_expansion)
            self.ffn_c = _S2M2_FFN(dim=dim, dim_expansion=dim_expansion)
        else:
            self.cross_attn = None
        self.ffn = _S2M2_FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        if self.cross_attn is not None:
            z = self.cross_attn(z)
            z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        return z.permute(0, 3, 1, 2).contiguous()


class _S2M2_BasicAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dim_expansion: int, use_pe: bool = False):
        super().__init__()
        self.cross_attn = _S2M2_CrossAttnBlock1D(dim=dim, num_heads=num_heads,
                                                  dim_expansion=dim_expansion)
        self.self_attn = _S2M2_SelfAttnBlock1D(dim=dim, num_heads=num_heads,
                                                dim_expansion=dim_expansion, use_pe=use_pe)
        self.ffn_c = _S2M2_FFN(dim=dim, dim_expansion=dim_expansion)
        self.ffn = _S2M2_FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: Tensor, pe: Tensor = None) -> Tensor:
        z = z.permute(0, 2, 3, 1)
        z = self.cross_attn(z)
        z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        return z.permute(0, 3, 1, 2)


# ── Section 4: unet ───────────────────────────────────────────────────────── #

class _S2M2_Unet(nn.Module):
    def __init__(self, dims: list, dim_expansion: int, use_pe: bool,
                 n_attn: int = 1, use_gate_fusion: bool = True):
        super().__init__()
        self.dims = dims
        self.use_pe = use_pe

        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[0], dims[1], kernel_size=1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[1], dims[2], kernel_size=1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[1], dims[0], kernel_size=1))
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[1], kernel_size=1))
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.concat_conv0 = _S2M2_FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv1 = _S2M2_FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv2 = _S2M2_FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc0 = _S2M2_ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.enc1 = _S2M2_ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.enc2 = _S2M2_ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.enc3s = nn.ModuleList([
            _S2M2_GlobalAttnBlock(dim=dims[2], num_heads=8, dim_expansion=dim_expansion,
                                  use_cross_attn=False, use_pe=use_pe)
            for _ in range(n_attn)
        ])

        self.dec0 = _S2M2_ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.dec1 = _S2M2_ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.dec2 = _S2M2_ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.dec3s = nn.ModuleList([
            _S2M2_GlobalAttnBlock(dim=dims[2], num_heads=8, dim_expansion=dim_expansion,
                                  use_cross_attn=False, use_pe=False)
            for _ in range(n_attn)
        ])

    def forward(self, z: Tensor):
        if self.use_pe:
            H, W = z.shape[-2:]
            pe = _s2m2_get_pe(H // 8, W // 8, 32, z.dtype, z.device)
        else:
            pe = None

        z0 = self.enc0(z)
        z1 = self.down_conv0(z0)
        z1 = self.enc1(z1)
        z2 = self.down_conv1(z1)
        z2 = self.enc2(z2)
        z3 = self.down_conv2(z2)
        for block in self.enc3s:
            z3 = block(z3, pe)

        for block in self.dec3s:
            z3 = block(z3, pe)
        z3_new = z3

        z2_new = self.up_conv2(z3_new)
        z2_new = self.concat_conv2(z2, z2_new)
        z2_new = self.dec2(z2_new)

        z1_new = self.up_conv1(z2_new)
        z1_new = self.concat_conv1(z1, z1_new)
        z1_new = self.dec1(z1_new)

        z0_new = self.up_conv0(z1_new)
        z0_new = self.concat_conv0(z0, z0_new)
        z0_new = self.dec0(z0_new)

        return z0_new, z1_new, z2_new, z3_new


# ── Section 5: submodules ─────────────────────────────────────────────────── #

class _S2M2_CostVolume:
    """Cost volume for iterative refinement (plain class, not nn.Module)."""

    def __init__(self, cv: Tensor, coords: Tensor, radius: int):
        self.radius = radius
        r = self.radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=cv.device, dtype=cv.dtype)
        self.dx = dx.reshape(1, 1, 2 * r + 1, 1)

        b, h, w, w2 = cv.shape
        self.cv = cv.reshape(b * h * w, 1, 1, w2)
        self.cv_2x = F.avg_pool2d(self.cv, kernel_size=[1, 2])
        self.coords = coords.reshape(b * h * w, 1, 1, 1)

    def __call__(self, disp: Tensor):
        b, _, h, w = disp.shape
        dx = self.dx

        x0 = self.coords - disp.reshape(b * h * w, 1, 1, 1) + dx
        y0 = 0 * x0
        corrs = _s2m2_bilinear_sampler(self.cv, torch.cat([x0, y0], dim=-1))
        corrs = corrs.reshape(b, h, w, 2 * self.radius + 1).permute(0, 3, 1, 2)

        x0 = self.coords / 2 - disp.reshape(b * h * w, 1, 1, 1) / 2 + dx
        y0 = 0 * x0
        corrs_2x = _s2m2_bilinear_sampler(self.cv_2x, torch.cat([x0, y0], dim=-1))
        corrs_2x = corrs_2x.reshape(b, h, w, 2 * self.radius + 1).permute(0, 3, 1, 2)

        return corrs, corrs_2x


class _S2M2_CNNEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=1),
        )
        self.conv1_down = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.norm1 = nn.GroupNorm(8, output_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_down = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: Tensor):
        x = self.conv0(x)
        x_2x = self.norm1(self.conv1_down(x))
        x_2x = self.conv2(x_2x) + x_2x
        x_4x = self.conv2_down(x_2x)
        return x_4x, x_2x


class _S2M2_UpsampleMask4x(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_x = nn.ConvTranspose2d(dim, 64, kernel_size=2, stride=2)
        self.conv_y = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 9, kernel_size=2, stride=2),
        )

    def forward(self, feat_x: Tensor, feat_y: Tensor) -> Tensor:
        feat_x = self.conv_x(feat_x)
        feat_y = self.conv_y(feat_y)
        return self.conv_concat(torch.cat([feat_x, feat_y], dim=1))


class _S2M2_UpsampleMask1x(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv_disp = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.conv_rgb = nn.Sequential(
            nn.ConvTranspose2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.conv_ctx = nn.ConvTranspose2d(dim, 16, kernel_size=2, stride=2)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(48, 9, kernel_size=1),
        )

    def forward(self, disp: Tensor, rgb: Tensor, ctx: Tensor) -> Tensor:
        feat_disp = self.conv_disp(disp)
        feat_rgb = self.conv_rgb(rgb)
        feat_ctx = self.conv_ctx(ctx)
        return self.conv_concat(torch.cat([feat_disp, feat_rgb, feat_ctx], dim=1))


def _s2m2_logsumexp_stable(x, dim, keepdim=False, eps=1e-30):
    m, _ = x.max(dim=dim, keepdim=True)
    y = (x - m).exp().sum(dim=dim, keepdim=True)
    y = m + torch.log(torch.clamp(y, min=eps))
    return y if keepdim else y.squeeze(dim)


class _S2M2_DispInit(nn.Module):
    """Initial disparity estimation using Optimal Transport."""

    def __init__(self, dim: int, ot_iter: int, use_positivity: bool):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.ot_iter = ot_iter
        self.use_positivity = use_positivity

    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor) -> Tensor:
        v = log_nu - _s2m2_logsumexp_stable(attn, dim=2)
        u = log_mu - _s2m2_logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        for _ in range(self.ot_iter - 1):
            v = log_nu - _s2m2_logsumexp_stable(attn + u.unsqueeze(3), dim=2)
            u = log_mu - _s2m2_logsumexp_stable(attn + v.unsqueeze(2), dim=3)
        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor) -> Tensor:
        bs, h, w, _ = attn.shape
        dtype = attn.dtype
        marginal = torch.cat(
            [torch.ones([w], device=attn.device),
             torch.tensor([w], device=attn.device)]
        ) / (2 * w)
        log_mu = marginal.log().reshape(1, 1, w + 1)
        log_nu = marginal.log().reshape(1, 1, w + 1)
        attn = F.pad(attn, (0, 1, 0, 1), "constant", 0)
        attn = self._sinkhorn(attn, log_mu, log_nu)
        w_tensor = torch.tensor(w, dtype=dtype, device=attn.device)
        log_const = torch.log(2 * w_tensor)
        return (attn[:, :, :-1, :-1] + log_const).exp().to(dtype)

    def forward(self, feature: Tensor):
        dtype = feature.dtype
        device = feature.device
        w = feature.shape[-1]
        x_grid = torch.linspace(0, w - 1, w, device=device, dtype=feature.dtype)
        if self.use_positivity:
            mask = torch.triu(torch.ones((w, w), dtype=torch.bool, device=device), diagonal=1)
        else:
            mask = torch.zeros((w, w), dtype=torch.bool, device=device)

        feature0, feature1 = self.layer_norm(feature.permute(0, 2, 3, 1)).chunk(2, dim=0)
        cv = torch.einsum('...hic,...hjc -> ...hij', feature0, feature1)

        cv_mask = cv.masked_fill(mask, -1e4)
        prob = self._optimal_transport(cv_mask)
        masked_prob = prob.masked_fill(mask, 0)

        prob_max_ind = masked_prob.argmax(dim=3).unsqueeze(3)
        prob_l = 2
        masked_prob_pad = F.pad(masked_prob, (prob_l, prob_l), "constant", 0)
        conf = 0
        correspondence_left = 0
        for idx in range(2 * prob_l + 1):
            weight = torch.gather(masked_prob_pad, index=prob_max_ind + idx, dim=3)
            conf += weight
            correspondence_left += weight * (prob_max_ind + idx - prob_l)
        eps = 1e-4
        correspondence_left = (correspondence_left + eps) / (conf + eps)
        disparity = (x_grid.reshape(1, 1, w) - correspondence_left.squeeze(3)).unsqueeze(1)
        conf = conf.unsqueeze(1).squeeze(-1)
        occ = masked_prob.sum(dim=3).unsqueeze(1)

        return disparity, conf, occ, cv


# ── Section 6: stacked_MRT ────────────────────────────────────────────────── #

class _S2M2_MRT(nn.Module):
    """Multi Resolution Transformer."""

    def __init__(self, dims: list, num_heads: int, dim_expansion: int, use_gate_fusion: bool):
        super().__init__()
        self.dims = dims

        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[0], dims[1], kernel_size=1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[1], dims[2], kernel_size=1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2),
                                        nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.up_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[1], dims[0], kernel_size=1))
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[1], kernel_size=1))
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.down_concat1 = _S2M2_FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat2 = _S2M2_FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat3 = _S2M2_FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.up_concat0 = _S2M2_FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat1 = _S2M2_FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat2 = _S2M2_FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc_attn0 = _S2M2_BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion, use_pe=False)
        self.enc_attn1 = _S2M2_BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion, use_pe=False)
        self.enc_attn2 = _S2M2_BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion, use_pe=False)
        self.enc_attn3s = nn.ModuleList([
            _S2M2_GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion,
                                  use_cross_attn=True, use_pe=False)
            for _ in range(2)
        ])

        self.dec_attn0 = _S2M2_BasicAttnBlock(dims[0], 1 * num_heads, dim_expansion, use_pe=False)
        self.dec_attn1 = _S2M2_BasicAttnBlock(dims[1], 2 * num_heads, dim_expansion, use_pe=False)
        self.dec_attn2 = _S2M2_BasicAttnBlock(dims[2], 4 * num_heads, dim_expansion, use_pe=False)
        self.dec_attn3s = nn.ModuleList([
            _S2M2_GlobalAttnBlock(dims[2], 8 * num_heads, dim_expansion,
                                  use_cross_attn=True, use_pe=False)
            for _ in range(2)
        ])

    def forward(self, z0: Tensor, z1: Tensor, z2: Tensor, z3: Tensor):
        z0 = self.enc_attn0(z0)
        z1 = self.down_concat1(z1, self.down_conv0(z0))
        z1 = self.enc_attn1(z1)
        z2 = self.down_concat2(z2, self.down_conv1(z1))
        z2 = self.enc_attn2(z2)
        z3 = self.down_concat3(z3, self.down_conv2(z2))
        for block in self.enc_attn3s:
            z3 = block(z3)

        for block in self.dec_attn3s:
            z3 = block(z3)

        z3_up = self.up_conv2(z3)
        z2 = self.up_concat2(z2, z3_up)
        z2 = self.dec_attn2(z2)

        z2_up = self.up_conv1(z2)
        z1 = self.up_concat1(z1, z2_up)
        z1 = self.dec_attn1(z1)

        z1_up = self.up_conv0(z1)
        z0 = self.up_concat0(z0, z1_up)
        z0 = self.dec_attn0(z0)

        return z0, z1, z2, z3


class _S2M2_StackedMRT(nn.Module):
    def __init__(self, num_transformer: int, dims: list, num_heads: int,
                 dim_expansion: int, use_gate_fusion: bool):
        super().__init__()
        self.uformer_list = nn.ModuleList([
            _S2M2_MRT(dims=dims, num_heads=num_heads, dim_expansion=dim_expansion,
                      use_gate_fusion=use_gate_fusion)
            for _ in range(num_transformer)
        ])

    def forward(self, z0: Tensor, z1: Tensor, z2: Tensor, z3: Tensor) -> Tensor:
        for uformer in self.uformer_list:
            z0, z1, z2, z3 = uformer(z0, z1, z2, z3)
        return z0.contiguous()


# ── Section 7: refinenet ──────────────────────────────────────────────────── #

class _S2M2_ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [kernel_size, 1], padding=[kernel_size // 2, 0])
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [kernel_size, 1], padding=[kernel_size // 2, 0])
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [kernel_size, 1], padding=[kernel_size // 2, 0])
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [1, kernel_size], padding=[0, kernel_size // 2])
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [1, kernel_size], padding=[0, kernel_size // 2])
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim,
                                [1, kernel_size], padding=[0, kernel_size // 2])

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h.to(x.dtype)


class _S2M2_GlobalRefiner(nn.Module):
    def __init__(self, feature_channels: int):
        super().__init__()
        self.init_feat = nn.Sequential(
            nn.Conv2d(2 + feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )
        self.refine_unet = _S2M2_Unet(
            dims=[feature_channels, feature_channels, feature_channels],
            dim_expansion=1, use_pe=False, use_gate_fusion=True, n_attn=1,
        )
        self.out_feat = nn.Sequential(nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1))

    def forward(self, ctx: Tensor, disp: Tensor, conf: Tensor) -> Tensor:
        disp_nor = disp / 1e2
        mask = 1.0 * (conf > 0.2)
        conf_logit = (mask * conf).logit(eps=1e-1)
        feat = self.init_feat(
            torch.cat([disp_nor * mask, conf_logit, ctx], dim=1).to(disp.dtype)
        )
        refine_feat = self.refine_unet(feat)[0]
        disp_update = self.out_feat(refine_feat) * 1e2
        return (mask * disp + (1 - mask) * disp_update).to(disp.dtype)


class _S2M2_LocalRefiner(nn.Module):
    def __init__(self, feature_channels: int, dim_expansion: int,
                 radius: int, use_gate_fusion: bool):
        super().__init__()
        self.disp_feat = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
        )
        self.corr_feat1 = nn.Sequential(
            nn.Conv2d((2 * radius + 1), 96, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(96, 64, kernel_size=1),
        )
        self.corr_feat2 = nn.Sequential(
            nn.Conv2d((2 * radius + 1), 96, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(96, 64, kernel_size=1),
        )
        self.conf_occ_feat = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=1),
        )
        self.disp_corr_ctx_cat = nn.Sequential(
            nn.Conv2d(256 + feature_channels, 2 * feature_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * feature_channels, feature_channels, kernel_size=3, padding=1),
        )
        self.refine_unet = _S2M2_Unet(
            dims=[feature_channels, feature_channels, 2 * feature_channels],
            dim_expansion=dim_expansion, use_pe=False,
            use_gate_fusion=use_gate_fusion, n_attn=1,
        )
        self.disp_update = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1, bias=False),
        )
        self.conf_occ_update = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, 2, kernel_size=3, padding=1, bias=False),
        )
        self.gru = _S2M2_ConvGRU(feature_channels, feature_channels, 3)

    def forward(self, hidden: Tensor, ctx: Tensor, disp: Tensor,
                conf: Tensor, occ: Tensor, cv_fn: Callable):
        conf_logit = conf.logit(eps=1e-2)
        occ_logit = occ.logit(eps=1e-2)

        corr1, corr2 = cv_fn(disp)
        corr_feat1 = self.corr_feat1(corr1 / 16)
        corr_feat2 = self.corr_feat2(corr2 / 16)
        disp_feat = self.disp_feat(disp / 1e2)
        conf_feat = self.conf_occ_feat(
            torch.cat([conf_logit, occ_logit], dim=1).to(disp.dtype)
        )
        disp_corr_ctx_feat = self.disp_corr_ctx_cat(
            torch.cat([disp_feat, corr_feat1, corr_feat2, ctx, conf_feat], dim=1).to(disp.dtype)
        )
        refine_feat = self.refine_unet(disp_corr_ctx_feat)[0]
        hidden_new = self.gru(hidden, refine_feat)
        disp_update = self.disp_update(hidden_new)
        conf_update, occ_update = self.conf_occ_update(hidden_new).chunk(2, dim=1)

        conf_new = torch.sigmoid(conf_update + conf_logit).to(disp.dtype)
        occ_new = torch.sigmoid(occ_update + occ_logit).to(disp.dtype)
        disp_new = (disp + disp_update).to(disp.dtype)

        return hidden_new.to(disp.dtype), disp_new, conf_new, occ_new


# ── Section 8: _S2M2 (vendored core, renamed from S2M2) ──────────────────── #

class _S2M2(nn.Module):
    def __init__(self, feature_channels: int, dim_expansion: int, num_transformer: int,
                 use_positivity: bool = False, output_upsample: bool = False,
                 refine_iter: int = 3):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_transformer = num_transformer
        self.use_positivity = use_positivity
        self.refine_iter = refine_iter
        self.output_upsample = output_upsample

        self.cnn_backbone = _S2M2_CNNEncoder(output_dim=feature_channels)

        self.feat_pyramid = _S2M2_Unet(
            dims=[feature_channels, feature_channels, 2 * feature_channels],
            dim_expansion=dim_expansion, use_gate_fusion=True,
            use_pe=True, n_attn=num_transformer * 2,
        )
        self.transformer = _S2M2_StackedMRT(
            num_transformer=num_transformer,
            dims=[feature_channels, feature_channels, 2 * feature_channels],
            num_heads=1, dim_expansion=dim_expansion, use_gate_fusion=True,
        )
        self.disp_init = _S2M2_DispInit(
            dim=feature_channels, ot_iter=3, use_positivity=use_positivity,
        )
        self.upsample_mask_1x = _S2M2_UpsampleMask1x(feature_channels)
        self.upsample_mask_4x_refine = _S2M2_UpsampleMask4x(feature_channels)

        self.global_refiner = _S2M2_GlobalRefiner(feature_channels=feature_channels)

        self.feat_fusion_layer = _S2M2_FeatureFusion(
            dim=feature_channels, kernel_size=3, use_gate=True,
        )
        self.refiner = _S2M2_LocalRefiner(
            feature_channels=feature_channels, dim_expansion=dim_expansion,
            radius=4, use_gate_fusion=True,
        )
        self.ctx_feat = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )

    def normalize_img(self, img0: Tensor, img1: Tensor):
        """Convert [0, 255] → [-1, 1]."""
        img0 = (img0 / 255.0 - 0.5) * 2
        img1 = (img1 / 255.0 - 0.5) * 2
        return img0, img1

    def upsample4x(self, x: Tensor, up_weights: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x_unfold = _s2m2_custom_unfold(x.reshape(b, c, h, w), 3, 1)
        x_unfold = F.interpolate(x_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)
        up_weights = up_weights.softmax(dim=1)
        return (x_unfold * up_weights).sum(1, keepdim=True)

    def upsample1x(self, disp: Tensor, filter_weights: Tensor) -> Tensor:
        disp_unfold = _s2m2_custom_unfold(disp, 3, 1)
        if self.output_upsample:
            upsample_factor = 2
            disp_unfold = F.interpolate(disp_unfold, scale_factor=upsample_factor, mode='nearest')
            filter_weights = F.interpolate(filter_weights, scale_factor=upsample_factor,
                                           mode='bilinear', align_corners=False)
            filter_weights = filter_weights.softmax(dim=1).to(disp.dtype)
        else:
            filter_weights = filter_weights.softmax(dim=1)
        return (disp_unfold * filter_weights).sum(1, keepdim=True)

    def forward(self, img0: Tensor, img1: Tensor):
        img0_nor, img1_nor = self.normalize_img(img0, img1)

        feature_4x, feature_2x = self.cnn_backbone(torch.cat([img0_nor, img1_nor], dim=0))
        feature0_2x, _ = feature_2x.chunk(2, dim=0)

        feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x = self.feat_pyramid(feature_4x)
        feature_tr_4x = self.transformer(feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x)

        disp, conf, occ, cv = self.disp_init(feature_tr_4x)

        feature0_tr_4x, _ = feature_tr_4x.chunk(2, dim=0)
        feature0_py_4x, _ = feature_py_4x.chunk(2, dim=0)

        disp = self.global_refiner(feature0_tr_4x.contiguous(), disp.detach(), conf.detach())
        if self.use_positivity:
            disp = disp.clamp(min=0)

        feature0_fusion_4x = self.feat_fusion_layer(feature0_tr_4x, feature0_py_4x)
        ctx0 = self.ctx_feat(feature0_fusion_4x)
        hidden = torch.tanh(ctx0)

        b, c, h, w = feature0_fusion_4x.shape
        coords_4x = torch.arange(w, device=feature0_fusion_4x.device,
                                 dtype=torch.float32).to(feature0_fusion_4x.dtype)
        cv_fn = _S2M2_CostVolume(cv, coords_4x.reshape(1, 1, w, 1).repeat(b, h, 1, 1), radius=4)

        for _ in range(self.refine_iter):
            hidden, disp, conf, occ = self.refiner(hidden, ctx0, disp, conf, occ, cv_fn)
            if self.use_positivity:
                disp = disp.clamp(min=0)
            occ_mask = torch.ge((coords_4x.reshape(1, 1, 1, -1) - disp), 0)
            occ = occ * occ_mask

        upsample_mask = self.upsample_mask_4x_refine(hidden, feature0_2x)
        disp_up = self.upsample4x(disp * 4, upsample_mask)
        occ_up = self.upsample4x(occ, upsample_mask)
        conf_up = self.upsample4x(conf, upsample_mask)

        filter_weights = self.upsample_mask_1x(disp_up, img0_nor, feature0_2x)
        disp_up = self.upsample1x(disp_up, filter_weights)
        occ_up = self.upsample1x(occ_up, filter_weights)
        conf_up = self.upsample1x(conf_up, filter_weights)
        if self.output_upsample:
            disp_up = 2 * disp_up

        return disp_up, occ_up, conf_up


# ── Section 9: public wrapper ─────────────────────────────────────────────── #

class S2M2Model(BaseStereoModel):
    config_class = S2M2Config

    def __init__(self, config: S2M2Config):
        super().__init__(config)
        self.net = _S2M2(
            feature_channels=config.feature_channels,
            dim_expansion=config.dim_expansion,
            num_transformer=config.num_transformer,
            use_positivity=config.use_positivity,
            output_upsample=config.output_upsample,
            refine_iter=config.refine_iter,
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        # Denormalize from ImageNet [0,1] → [0,255]
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        left_255 = (left * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        # Pad to multiples of 32
        h, w = left_255.shape[-2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            left_255 = F.pad(left_255, [0, pad_w, 0, pad_h], mode="replicate")
            right_255 = F.pad(right_255, [0, pad_w, 0, pad_h], mode="replicate")

        # Run network (internally normalizes to [-1, 1])
        disp_up, _occ_up, _conf_up = self.net(left_255, right_255)

        # Unpad and squeeze channel dim → [B, H, W]
        disp_up = disp_up[..., :h, :w].squeeze(1)

        if self.training:
            return [disp_up]  # List[Tensor(B, H, W)] — single-scale
        return disp_up  # Tensor(B, H, W)

    def _backbone_module(self):
        return self.net.cnn_backbone

    @classmethod
    def _load_pretrained_weights(cls, model_id: str, device: str = "cpu",
                                 for_training: bool = False, **kwargs):
        if os.path.isfile(model_id):
            ckpt_path = model_id
            variant = kwargs.get("variant")
            if variant is None:
                raise ValueError(
                    "When loading S2M2 from a local path, pass variant= "
                    "(e.g. variant='S', 'M', 'L', or 'XL')."
                )
            cfg = S2M2Config(variant=variant, **{
                k: v for k, v in kwargs.items()
                if k in ("feature_channels", "num_transformer", "refine_iter",
                         "use_positivity", "dim_expansion")
            })
        else:
            from huggingface_hub import hf_hub_download
            cfg = S2M2Config.from_variant(model_id)
            try:
                ckpt_path = hf_hub_download(
                    repo_id=cfg.hub_repo_id,
                    filename=cfg.checkpoint_filename,
                    repo_type="model",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download S2M2 checkpoint for '{model_id}' "
                    f"from HuggingFace Hub ({cfg.hub_repo_id}/{cfg.checkpoint_filename}).\n"
                    f"Error: {exc}"
                ) from exc

        model = cls(cfg)

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # S2M2 checkpoints store weights under "state_dict" key
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Add "net." prefix to match wrapper structure (self.net = _S2M2(...))
        state_dict = {"net." + k: v for k, v in state_dict.items()}

        # Remap checkpoint keys from older naming to current model attribute names:
        #   disp_init.layer_norm1 → disp_init.layer_norm  (renamed in UNet-based release)
        #   upsample_mask_4x.*   → upsample_mask_4x_refine.*  (renamed in UNet-based release)
        #   feature_flow_attn.*  → dropped (replaced by UNet-based refinement)
        remapped = {}
        for k, v in state_dict.items():
            if ".disp_init.layer_norm1." in k:
                k = k.replace(".disp_init.layer_norm1.", ".disp_init.layer_norm.")
            elif ".upsample_mask_4x." in k and ".upsample_mask_4x_refine." not in k:
                k = k.replace(".upsample_mask_4x.", ".upsample_mask_4x_refine.")
            remapped[k] = v
        state_dict = remapped

        # Keys known to exist in older checkpoints but removed in the UNet-based release.
        # Silently discard them instead of logging warnings.
        _KNOWN_REMOVED_PREFIXES = ("net.feature_flow_attn.",)

        # Shape-skipping load (replicates my_load_state_dict logic)
        model_state = model.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in _KNOWN_REMOVED_PREFIXES):
                continue
            if k in model_state:
                if v.shape != model_state[k].shape:
                    logger.warning(
                        "Skip loading parameter: %s, required shape: %s, loaded shape: %s",
                        k, model_state[k].shape, v.shape,
                    )
                else:
                    filtered[k] = v
            else:
                logger.warning("Unexpected key in checkpoint: %s", k)

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s ...", len(missing), missing[:5])

        logger.info("Loaded S2M2 (%s) from '%s'", cfg.variant, ckpt_path)
        model = model.to(device)
        if not for_training:
            model.eval()
        return model
