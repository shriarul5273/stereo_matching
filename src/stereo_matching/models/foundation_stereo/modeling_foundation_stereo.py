"""
FoundationStereoModel — fully self-contained implementation.

All architecture code from the original FoundationStereo repository is inlined
here in dependency order.  No sys.path manipulation or third-party directory
links are required.  The only external packages needed are:
    pip install torch torchvision timm einops huggingface_hub scipy

depth_anything (DPTHead, FeatureFusionBlock, etc.) is inlined directly into this file.

Original source: https://github.com/NVlabs/FoundationStereo
License: NVIDIA proprietary — see third-party/FoundationStereo/LICENSE
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.backends.cudnn as cudnn
from einops import rearrange
from torch import einsum

from ...modeling_utils import BaseStereoModel
from .configuration_foundation_stereo import (
    FoundationStereoConfig,
    _FS_VARIANT_MAP,
)

logger = logging.getLogger(__name__)

# ── AMP autocast shim ────────────────────────────────────────────────────── #
try:
    autocast = torch.cuda.amp.autocast
except Exception:
    class autocast:  # noqa: N801
        def __init__(self, enabled): pass
        def __enter__(self): pass
        def __exit__(self, *a): pass


# ===========================================================================
# Section 1 — Utils  (from core/utils/utils.py)
# ===========================================================================

class _FS_InputPadder:
    """Pads images so that dimensions are divisible by ``divis_by``."""
    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all(x.ndim == 4 for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def _fs_bilinear_sampler(img, coords, mode='bilinear', mask=False, low_memory=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1
    grid = torch.cat([xgrid, ygrid], dim=-1).to(img.dtype)
    with cudnn.flags(enabled=False):
        img = F.grid_sample(img.contiguous(), grid.contiguous(), align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def _fs_coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


# ===========================================================================
# Section 2 — Utils helpers  (from Utils.py)
# ===========================================================================

def _fs_freeze_model(model):
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.buffers():
        p.requires_grad = False
    return model


def _fs_get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
    assert max_H % divider == 0
    assert max_W % divider == 0

    def round_by_divider(x):
        return int(np.ceil(x / divider) * divider)

    H_resize = round_by_divider(H)
    W_resize = round_by_divider(W)
    if H_resize > max_H or W_resize > max_W:
        if H_resize > W_resize:
            W_resize = round_by_divider(W_resize * max_H / H_resize)
            H_resize = max_H
        else:
            H_resize = round_by_divider(H_resize * max_W / W_resize)
            W_resize = max_W
    return int(H_resize), int(W_resize)


# ===========================================================================
# Section 3 — Submodules  (from core/submodule.py)
# ===========================================================================

def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    return tensor.is_contiguous(memory_format=torch.contiguous_format)


class _FS_LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        s, u = torch.var_mean(x, dim=1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class _FS_BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super().__init__()
        self.relu = relu
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm3d(out_channels) if norm == 'batch' else nn.InstanceNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
                self.bn = nn.BatchNorm2d(out_channels) if norm == 'batch' else nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class _FS_Conv3dNormActReduced(nn.Module):
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
            kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size // 2, kernel_size // 2), stride=(1, stride, stride)),
            norm(hidden), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp // 2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out), nn.ReLU(),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class _FS_ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _FS_ResnetBasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
        super().__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class _FS_FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1, -1)):
        B, L, C = query.shape
        Q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(B, L, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(B, L, self.num_heads, self.head_dim)
        attn_output = F.scaled_dot_product_attention(Q, K, V)
        return self.out_proj(attn_output.reshape(B, L, -1))


class _FS_FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = _FS_FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class _FS_BasicConv_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super().__init__()
        self.relu = relu
        self.use_in = IN
        if is_3d:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs) if deconv else nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class _FS_Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super().__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        if deconv and is_3d and keep_dispc:
            self.conv1 = _FS_BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        else:
            self.conv1 = _FS_BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)
        mul = 2 if keep_concat else 1
        if self.concat:
            self.conv2 = _FS_BasicConv(out_channels * 2, out_channels * mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = _FS_BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        x = torch.cat((x, rem), 1) if self.concat else x + rem
        return self.conv2(x)


class _FS_Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super().__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        if deconv and is_3d and keep_dispc:
            self.conv1 = _FS_BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        else:
            self.conv1 = _FS_BasicConv_IN(in_channels, out_channels, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)
        mul = 2 if keep_concat else 1
        if self.concat:
            self.conv2 = _FS_ResnetBasicBlock(out_channels * 2, out_channels * mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = _FS_BasicConv_IN(out_channels, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        x = torch.cat((x, rem), 1) if self.concat else x + rem
        return self.conv2(x)


def _fs_groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    channels_per_group = C // num_groups
    fea1 = fea1.reshape(B, num_groups, channels_per_group, H, W)
    fea2 = fea2.reshape(B, num_groups, channels_per_group, H, W)
    with torch.cuda.amp.autocast(enabled=False):
        cost = (F.normalize(fea1.float(), dim=2) * F.normalize(fea2.float(), dim=2)).sum(dim=2)
    return cost


def _fs_build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, stride=1):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = _fs_groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups)
        else:
            volume[:, :, i, :, :] = _fs_groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def _fs_build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    return volume.contiguous()


def _fs_disparity_regression(x, maxdisp):
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device).reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)


class _FS_FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super().__init__()
        self.feat_att = nn.Sequential(
            _FS_BasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        return torch.sigmoid(feat_att) * cv


def _fs_context_upsample(disp_low, up_weights):
    b, c, h, w = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)
    return (disp_unfold * up_weights).sum(1)


class _FS_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()[None]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x, resize_embed=False):
        self.pe = self.pe.to(x.device).to(x.dtype)
        pe = self.pe
        if pe.shape[1] < x.shape[1]:
            if resize_embed:
                pe = F.interpolate(pe.permute(0, 2, 1), size=x.shape[1], mode='linear', align_corners=False).permute(0, 2, 1)
            else:
                raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
        return x + pe[:, :x.size(1)]


class _FS_CostVolumeDisparityAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
        super().__init__()
        self.resize_embed = resize_embed
        self.sa = nn.ModuleList([
            _FS_FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout)
            for _ in range(num_transformer)
        ])
        self.pos_embed0 = _FS_PositionalEmbedding(d_model, max_len=max_len)

    def forward(self, cv, window_size=(-1, -1)):
        B, C, D, H, W = cv.shape
        x = cv.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
        x = self.pos_embed0(x, resize_embed=self.resize_embed)
        for layer in self.sa:
            x = layer(x, window_size=window_size)
        return x.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2)


class _FS_ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 16, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class _FS_SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.samconv(torch.cat([avg_out, max_out], dim=1)))


class _FS_EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = _FS_LayerNorm2d(dim, eps=1e-6) if norm == 'layer' else nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.norm(self.dwconv(x))
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma * x
        return input + x.permute(0, 3, 1, 2)


# ===========================================================================
# Section 4 — Extractor  (from core/extractor.py)
# ===========================================================================

class _FS_ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8

        def make_norm(fn, c):
            return {'group': nn.GroupNorm(num_groups, c), 'batch': nn.BatchNorm2d(c),
                    'instance': nn.InstanceNorm2d(c), 'layer': _FS_LayerNorm2d(c),
                    'none': nn.Sequential()}[fn]

        self.norm1 = make_norm(norm_fn, planes)
        self.norm2 = make_norm(norm_fn, planes)
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.norm3 = make_norm(norm_fn, planes)
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class _FS_MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], norm_fn='batch', dropout=0.0, downsample=3):
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        norm_map = {'group': nn.GroupNorm(8, 64), 'batch': nn.BatchNorm2d(64),
                    'instance': nn.InstanceNorm2d(64), 'layer': _FS_LayerNorm2d(64), 'none': nn.Sequential()}
        self.norm1 = norm_map[norm_fn]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.outputs04 = nn.ModuleList([nn.Sequential(_FS_ResidualBlock(128, 128, norm_fn, stride=1), nn.Conv2d(128, dim[2], 3, padding=1)) for dim in output_dim])
        self.outputs08 = nn.ModuleList([nn.Sequential(_FS_ResidualBlock(128, 128, norm_fn, stride=1), nn.Conv2d(128, dim[1], 3, padding=1)) for dim in output_dim])
        self.outputs16 = nn.ModuleList([nn.Conv2d(128, dim[0], 3, padding=1) for dim in output_dim])
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer = nn.Sequential(_FS_ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride), _FS_ResidualBlock(dim, dim, self.norm_fn, stride=1))
        self.in_planes = dim
        return layer

    def forward(self, x, dual_inp=False, num_layers=3):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        if dual_inp:
            v = x
            x = x[:x.shape[0] // 2]
        outputs04 = [f(x) for f in self.outputs04]
        if num_layers == 1:
            return (outputs04, v) if dual_inp else (outputs04,)
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]
        if num_layers == 2:
            return (outputs04, outputs08, v) if dual_inp else (outputs04, outputs08)
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]
        return (outputs04, outputs08, outputs16, v) if dual_inp else (outputs04, outputs08, outputs16)


# ===========================================================================
# Section — depth_anything (inlined from depth_anything/blocks.py + dpt.py)
# ===========================================================================

def _fs_make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class _FS_ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class _FS_FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = _FS_ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = _FS_ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)
        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _fs_make_fusion_block(features, use_bn, size=None):
    return _FS_FeatureFusionBlock(
        features, nn.ReLU(False), deconv=False, bn=use_bn,
        expand=False, align_corners=True, size=size,
    )


class _FS_DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super().__init__()
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channel,
                      kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                for _ in range(len(self.projects))
            ])
        self.scratch = _fs_make_scratch(out_channels, features, groups=1, expand=False)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _fs_make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _fs_make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _fs_make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _fs_make_fusion_block(features, use_bn)
        head_features_1 = features
        head_features_2 = 32
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

    def forward(self, out_features, patch_h, patch_w, return_intermediate=False, patch_size=14):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * patch_size), int(patch_w * patch_size)),
                            mode="bilinear", align_corners=True)
        if return_intermediate:
            depth = self.scratch.output_conv2(out)
            depth = F.relu(depth)
            disp = 1 / depth
            disp[depth == 0] = 0
            disp = disp / disp.max()
            return out, path_1, path_2, path_3, path_4, disp
        else:
            return self.scratch.output_conv2(out)


class _FS_DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
                 use_bn=False, use_clstoken=False, pretrained_dino=False):
        super().__init__()
        assert encoder in ['vits', 'vitb', 'vitl']
        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder),
                                         pretrained=pretrained_dino)
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = _FS_DPTHead(1, dim, features, use_bn,
                                       out_channels=out_channels, use_clstoken=use_clstoken)


class _FS_DepthAnything(_FS_DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        patch_size = self.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        depth = self.depth_head(features, patch_h, patch_w, patch_size=patch_size)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return F.relu(depth).squeeze(1)


class _FS_DepthAnythingFeature(nn.Module):
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    }

    def __init__(self, encoder='vits'):
        super().__init__()
        self.encoder = encoder
        self.depth_anything = _FS_DepthAnything(self.model_configs[encoder])
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11], 'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23], 'vitg': [9, 19, 29, 39],
        }

    def forward(self, x):
        h, w = x.shape[-2:]
        features = self.depth_anything.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        patch_size = self.depth_anything.pretrained.patch_size
        patch_h, patch_w = h // patch_size, w // patch_size
        out, path_1, path_2, path_3, path_4, disp = self.depth_anything.depth_head.forward(features, patch_h, patch_w, return_intermediate=True)
        return {'out': out, 'path_1': path_1, 'path_2': path_2, 'path_3': path_3, 'path_4': path_4, 'features': features, 'disp': disp}


class _FS_ContextNetDino(_FS_MultiBasicEncoder):
    def __init__(self, args, output_dim=[128], norm_fn='batch', downsample=3):
        nn.Module.__init__(self)
        self.args = args
        self.patch_size = 14
        self.image_size = 518
        self.vit_feat_dim = 384
        self.out_dims = output_dim
        self.norm_fn = norm_fn

        norm_map = {'group': nn.GroupNorm(8, 64), 'batch': nn.BatchNorm2d(64),
                    'instance': nn.InstanceNorm2d(64), 'layer': _FS_LayerNorm2d(64), 'none': nn.Sequential()}
        self.norm1 = norm_map[norm_fn]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.down = nn.Sequential(nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=0), nn.BatchNorm2d(128))
        vit_dim = _FS_DepthAnythingFeature.model_configs[self.args.vit_size]['features'] // 2
        self.conv2 = _FS_BasicConv(128 + vit_dim, 128, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(256)
        self.outputs04 = nn.ModuleList([nn.Sequential(_FS_ResidualBlock(128, 128, norm_fn, stride=1), nn.Conv2d(128, dim[2], 3, padding=1)) for dim in output_dim])
        self.outputs08 = nn.ModuleList([nn.Sequential(_FS_ResidualBlock(128, 128, norm_fn, stride=1), nn.Conv2d(128, dim[1], 3, padding=1)) for dim in output_dim])
        self.outputs16 = nn.ModuleList([nn.Conv2d(128, dim[0], 3, padding=1) for dim in output_dim])

    def forward(self, x_in, vit_feat, dual_inp=False, num_layers=3):
        B, C, H, W = x_in.shape
        x = self.relu1(self.norm1(self.conv1(x_in)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.cat([x, vit_feat], dim=1)
        x = self.conv2(x)
        outputs04 = [f(x) for f in self.outputs04]
        y = self.layer4(x)
        outputs08 = [f(y) for f in self.outputs08]
        z = self.layer5(y)
        outputs16 = [f(z) for f in self.outputs16]
        return (outputs04, outputs08, outputs16)


class _FS_Feature(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        import timm
        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        self.dino = _FS_DepthAnythingFeature(encoder=self.args.vit_size)
        self.dino = _fs_freeze_model(self.dino)
        vit_feat_dim = _FS_DepthAnythingFeature.model_configs[self.args.vit_size]['features'] // 2
        self.deconv32_16 = _FS_Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = _FS_Conv2x_IN(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = _FS_Conv2x_IN(chans[1] * 2, chans[0], deconv=True, concat=True)
        self.conv4 = nn.Sequential(
            _FS_BasicConv(chans[0] * 2 + vit_feat_dim, chans[0] * 2 + vit_feat_dim, kernel_size=3, stride=1, padding=1, norm='instance'),
            _FS_ResidualBlock(chans[0] * 2 + vit_feat_dim, chans[0] * 2 + vit_feat_dim, norm_fn='instance'),
            _FS_ResidualBlock(chans[0] * 2 + vit_feat_dim, chans[0] * 2 + vit_feat_dim, norm_fn='instance'),
        )
        self.patch_size = 14
        self.d_out = [chans[0] * 2 + vit_feat_dim, chans[1] * 2, chans[2] * 2, chans[3]]

    def forward(self, x):
        B, C, H, W = x.shape
        divider = np.lcm(self.patch_size, 16)
        H_resize, W_resize = _fs_get_resize_keep_aspect_ratio(H, W, divider=divider, max_H=1344, max_W=1344)
        x_in_ = F.interpolate(x, size=(H_resize, W_resize), mode='bicubic', align_corners=False)
        self.dino = self.dino.eval()
        with torch.no_grad():
            output = self.dino(x_in_)
        vit_feat = F.interpolate(output['out'], size=(H // 4, W // 4), mode='bilinear', align_corners=True)
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)
        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(torch.cat([x4, vit_feat], dim=1))
        return [x4, x8, x16, x32], vit_feat


# ===========================================================================
# Section 5 — Geometry  (from core/geometry.py)
# ===========================================================================

class _FS_Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, dx=None):
        self.num_levels = num_levels
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.dx = dx
        init_corr = _FS_Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)
        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, 1, d).contiguous()
        init_corr = init_corr.reshape(b * h * w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for _ in range(self.num_levels - 1):
            geo_volume = F.avg_pool2d(geo_volume, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume)
        for _ in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords, low_memory=False):
        b, _, h, w = disp.shape
        self.dx = self.dx.to(disp.device)
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = self.dx + disp.reshape(b * h * w, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)
            geo_volume = _fs_bilinear_sampler(geo_volume, torch.cat([x0, y0], dim=-1), low_memory=low_memory)
            geo_volume = geo_volume.reshape(b, h, w, -1)
            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b * h * w, 1, 1, 1) / 2 ** i - disp.reshape(b * h * w, 1, 1, 1) / 2 ** i + self.dx
            init_corr = _fs_bilinear_sampler(init_corr, torch.cat([init_x0, y0], dim=-1), low_memory=low_memory)
            init_corr = init_corr.reshape(b, h, w, -1)
            out_pyramid.extend([geo_volume, init_corr])
        return torch.cat(out_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        with torch.cuda.amp.autocast(enabled=False):
            corr = torch.einsum('aijk,aijh->ajkh', F.normalize(fmap1.float(), dim=1), F.normalize(fmap2.float(), dim=1))
        return corr.reshape(B, H, W1, 1, W2)


# ===========================================================================
# Section 6 — Update  (from core/update.py)
# ===========================================================================

class _FS_DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            _FS_EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            _FS_EdgeNextConvEncoder(input_dim, expan_ratio=4, kernel_size=7, norm=None),
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class _FS_RaftConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, kernel_size=3):
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, x, hx):
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class _FS_SelectiveConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256, small_kernel_size=1, large_kernel_size=3, patch_size=None):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(input_dim + hidden_dim, input_dim + hidden_dim, kernel_size=3, padding=1), nn.ReLU())
        self.small_gru = _FS_RaftConvGRU(hidden_dim, input_dim, small_kernel_size)
        self.large_gru = _FS_RaftConvGRU(hidden_dim, input_dim, large_kernel_size)

    def forward(self, att, h, *x):
        x = self.conv0(torch.cat(x, dim=1))
        hx = self.conv1(torch.cat([x, h], dim=1))
        return self.small_gru(h, x, hx) * att + self.large_gru(h, x, hx) * (1 - att)


class _FS_BasicMotionEncoder(nn.Module):
    def __init__(self, args, ngroup=8):
        super().__init__()
        self.args = args
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) * (ngroup + 1)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 256, 128 - 1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        disp_ = F.relu(self.convd2(F.relu(self.convd1(disp))))
        return torch.cat([F.relu(self.conv(torch.cat([cor, disp_], dim=1))), disp], dim=1)


def _fs_pool2x(x): return F.avg_pool2d(x, 3, stride=2, padding=1)
def _fs_pool4x(x): return F.avg_pool2d(x, 5, stride=4, padding=1)
def _fs_interp(x, dest): return F.interpolate(x, dest.shape[2:], mode='bilinear', align_corners=True)


class _FS_BasicSelectiveMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, volume_dim=8):
        super().__init__()
        self.args = args
        self.encoder = _FS_BasicMotionEncoder(args, volume_dim)
        if args.n_gru_layers == 3:
            self.gru16 = _FS_SelectiveConvGRU(hidden_dim, hidden_dim * 2)
        if args.n_gru_layers >= 2:
            self.gru08 = _FS_SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers == 3) + hidden_dim * 2)
        self.gru04 = _FS_SelectiveConvGRU(hidden_dim, hidden_dim * (args.n_gru_layers > 1) + hidden_dim * 2)
        self.disp_head = _FS_DispHead(hidden_dim, 256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, net, inp, corr, disp, att):
        if self.args.n_gru_layers == 3:
            net[2] = self.gru16(att[2], net[2], inp[2], _fs_pool2x(net[1]))
        if self.args.n_gru_layers >= 2:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(att[1], net[1], inp[1], _fs_pool2x(net[0]), _fs_interp(net[2], net[1]))
            else:
                net[1] = self.gru08(att[1], net[1], inp[1], _fs_pool2x(net[0]))
        motion_features = torch.cat([inp[0], self.encoder(disp, corr)], dim=1)
        if self.args.n_gru_layers > 1:
            net[0] = self.gru04(att[0], net[0], motion_features, _fs_interp(net[1], net[0]))
        delta_disp = self.disp_head(net[0])
        mask = .25 * self.mask(net[0])
        return net, mask, delta_disp


# ===========================================================================
# Section 7 — FoundationStereo net  (from core/foundation_stereo.py)
# ===========================================================================

def _fs_normalize_image(img):
    tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    return tf(img / 255.0).contiguous()


class _FS_Hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(_FS_BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1), _FS_Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17))
        self.conv2 = nn.Sequential(_FS_BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1), _FS_Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17))
        self.conv3 = nn.Sequential(_FS_BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1), _FS_Conv3dNormActReduced(in_channels * 6, in_channels * 6, kernel_size=3, kernel_disp=17))
        self.conv3_up = _FS_BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv2_up = _FS_BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv1_up = _FS_BasicConv(in_channels * 2, in_channels, deconv=True, is_3d=True, bn=True, relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(_FS_Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17), _FS_Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17))
        self.agg_0 = nn.Sequential(_FS_BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1), _FS_Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17), _FS_Conv3dNormActReduced(in_channels * 4, in_channels * 4, kernel_size=3, kernel_disp=17))
        self.agg_1 = nn.Sequential(_FS_BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1), _FS_Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17), _FS_Conv3dNormActReduced(in_channels * 2, in_channels * 2, kernel_size=3, kernel_disp=17))
        self.atts = nn.ModuleDict({"4": _FS_CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp'] // 16)})
        self.conv_patch = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels), nn.BatchNorm3d(in_channels))
        self.feature_att_8 = _FS_FeatureAtt(in_channels * 2, feat_dims[1])
        self.feature_att_16 = _FS_FeatureAtt(in_channels * 4, feat_dims[2])
        self.feature_att_32 = _FS_FeatureAtt(in_channels * 6, feat_dims[3])
        self.feature_att_up_16 = _FS_FeatureAtt(in_channels * 4, feat_dims[2])
        self.feature_att_up_8 = _FS_FeatureAtt(in_channels * 2, feat_dims[1])

    def forward(self, x, features):
        conv1 = self.feature_att_8(self.conv1(x), features[1])
        conv2 = self.feature_att_16(self.conv2(conv1), features[2])
        conv3 = self.feature_att_32(self.conv3(conv2), features[3])
        conv3_up = self.conv3_up(conv3)
        conv2 = self.feature_att_up_16(self.agg_0(torch.cat((conv3_up, conv2), dim=1)), features[2])
        conv2_up = self.conv2_up(conv2)
        conv1 = self.feature_att_up_8(self.agg_1(torch.cat((conv2_up, conv1), dim=1)), features[1])
        conv = self.conv1_up(conv1)
        x = F.interpolate(self.atts["4"](self.conv_patch(x)), scale_factor=4, mode='trilinear', align_corners=False)
        return self.conv_out(conv + x)


class _FS_Net(nn.Module):
    """FoundationStereo network — inlined, no third-party path dependency."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        self.cv_group = 8
        volume_dim = 28

        self.cnet = _FS_ContextNetDino(args, output_dim=[args.hidden_dims, context_dims], downsample=args.n_downsample)
        self.update_block = _FS_BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
        self.sam = _FS_SpatialAttentionExtractor()
        self.cam = _FS_ChannelAttentionEnhancement(self.args.hidden_dims[0])
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, kernel_size=3, padding=1) for i in range(self.args.n_gru_layers)])
        self.feature = _FS_Feature(args)
        self.proj_cmb = nn.Conv2d(self.feature.d_out[0], 12, kernel_size=1, padding=0)
        self.stem_2 = nn.Sequential(_FS_BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1), nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.InstanceNorm2d(32), nn.ReLU())
        self.stem_4 = nn.Sequential(_FS_BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1), nn.Conv2d(48, 48, 3, 1, 1, bias=False), nn.InstanceNorm2d(48), nn.ReLU())
        self.spx_2_gru = _FS_Conv2x(32, 32, True, bn=False)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))
        self.corr_stem = nn.Sequential(nn.Conv3d(32, volume_dim, kernel_size=1), _FS_BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True), _FS_ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1), _FS_ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1))
        self.corr_feature_att = _FS_FeatureAtt(volume_dim, self.feature.d_out[0])
        self.cost_agg = _FS_Hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
        self.classifier = nn.Sequential(_FS_BasicConv(volume_dim, volume_dim // 2, kernel_size=3, padding=1, is_3d=True), _FS_ResnetBasicBlock3D(volume_dim // 2, volume_dim // 2, kernel_size=3, stride=1, padding=1), nn.Conv3d(volume_dim // 2, 1, kernel_size=7, padding=3))
        r = self.args.corr_radius
        self.dx = torch.linspace(-r, r, 2 * r + 1, requires_grad=False).reshape(1, 1, 2 * r + 1, 1)

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = F.softmax(self.spx_gru(xspx), 1)
            up_disp = _fs_context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp.float()

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False, low_memory=False, init_disp=None):
        B = len(image1)
        low_memory = low_memory or (self.args.get('low_memory', False))
        image1 = _fs_normalize_image(image1)
        image2 = _fs_normalize_image(image2)
        with autocast(enabled=self.args.mixed_precision):
            out, vit_feat = self.feature(torch.cat([image1, image2], dim=0))
            vit_feat = vit_feat[:B]
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = self.stem_2(image1)
            gwc_volume = _fs_build_gwc_volume(features_left[0], features_right[0], self.args.max_disp // 4, self.cv_group)
            left_tmp = self.proj_cmb(features_left[0])
            right_tmp = self.proj_cmb(features_right[0])
            concat_volume = _fs_build_concat_volume(left_tmp, right_tmp, maxdisp=self.args.max_disp // 4)
            del left_tmp, right_tmp
            comb_volume = self.corr_stem(torch.cat([gwc_volume, concat_volume], dim=1))
            comb_volume = self.cost_agg(self.corr_feature_att(comb_volume, features_left[0]), features_left)
            prob = F.softmax(self.classifier(comb_volume).squeeze(1), dim=1)
            if init_disp is None:
                init_disp = _fs_disparity_regression(prob, self.args.max_disp // 4)
            cnet_list = list(self.cnet(image1, vit_feat=vit_feat, num_layers=self.args.n_gru_layers))
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [self.cam(torch.relu(x[1])) * torch.relu(x[1]) for x in cnet_list]
            att = [self.sam(x) for x in inp_list]

        geo_fn = _FS_Combined_Geo_Encoding_Volume(features_left[0].float(), features_right[0].float(), comb_volume.float(), num_levels=self.args.corr_levels, dx=self.dx)
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp.float()
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, low_memory=low_memory)
            with autocast(enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
            disp = disp + delta_disp.float()
            if test_mode and itr < iters - 1:
                continue
            disp_up = self.upsample_disp(disp.float(), mask_feat_4.float(), stem_2x.float())
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up
        return init_disp, disp_preds


# ===========================================================================
# Section 8 — _FSArgs + config → args conversion
# ===========================================================================

class _FSArgs:
    """Namespace supporting both attribute access and dict-style .get() / [key].

    FoundationStereo accesses args as ``args.hidden_dims`` (attribute) AND
    ``self.cfg['max_disp']`` (dict key), so SimpleNamespace alone is insufficient.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __contains__(self, key):
        return key in self.__dict__


def _config_to_fs_args(config: FoundationStereoConfig) -> _FSArgs:
    return _FSArgs(
        hidden_dims     = config.hidden_dims,
        max_disp        = config.max_disp,
        n_gru_layers    = config.n_gru_layers,
        corr_levels     = config.corr_levels,
        corr_radius     = config.corr_radius,
        n_downsample    = config.n_downsample,
        mixed_precision = config.mixed_precision,
        vit_size        = config.vit_size,
        low_memory      = config.low_memory,
    )


# ===========================================================================
# Section 9 — Public wrapper
# ===========================================================================

class FoundationStereoModel(BaseStereoModel):
    """BaseStereoModel wrapper around the inlined FoundationStereo network."""

    config_class = FoundationStereoConfig

    def __init__(self, config: FoundationStereoConfig):
        super().__init__(config)
        self.net = _FS_Net(_config_to_fs_args(config))

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        """
        Args:
            left, right: ImageNet-normalized (B, 3, H, W) tensors in [0, 1].
        Returns:
            Training: List[Tensor(B, H, W)] — one per GRU iteration.
            Inference: Tensor(B, H, W) — final disparity in pixels.
        """
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self.config.std,  device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        left_255  = (left  * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        padder = _FS_InputPadder(left_255.shape, divis_by=32)
        left_255, right_255 = padder.pad(left_255, right_255)

        if self.training:
            init_disp, disp_preds = self.net(left_255, right_255, iters=12, test_mode=False)
            return [padder.unpad(p).squeeze(1) for p in disp_preds]
        else:
            disp_up = self.net(left_255, right_255, iters=12, test_mode=True)
            return padder.unpad(disp_up.float()).squeeze(1)

    @classmethod
    def _load_pretrained_weights(cls, model_id: str, device: str = "cpu", for_training: bool = False, **kwargs) -> "FoundationStereoModel":
        _GDRIVE_URL  = "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
        _CKPT_FNAME  = "model_best_bp2.pth"
        _VARIANT_DIR = {"standard": "11-33-40", "large": "23-51-11"}

        def _read_ckpt(path: str) -> dict:
            """Load .pth and return a flat state dict (handles nested {'model': ...} format)."""
            try:
                raw = torch.load(path, map_location=device, weights_only=True)
            except Exception:
                raw = torch.load(path, map_location=device, weights_only=False)
            sd = raw["model"] if (isinstance(raw, dict) and "model" in raw) else raw
            return {f"net.{k.removeprefix('module.')}": v for k, v in sd.items()}

        def _apply_state(model, new_sd: dict) -> None:
            try:
                model.load_state_dict(new_sd, strict=True)
            except RuntimeError as exc:
                logger.warning("strict=True failed: %s\nRetrying strict=False.", exc)
                incompatible = model.load_state_dict(new_sd, strict=False)
                if incompatible.missing_keys:
                    logger.warning("Missing keys: %s", incompatible.missing_keys)
                if incompatible.unexpected_keys:
                    logger.warning("Unexpected keys: %s", incompatible.unexpected_keys)

        # ── registered variant ID (e.g. "foundation-stereo") ─────────────────
        if model_id in _FS_VARIANT_MAP:
            config   = FoundationStereoConfig.from_variant(model_id)
            dirname  = _VARIANT_DIR[config.variant]
            cache_path = os.path.expanduser(
                f"~/.cache/foundation_stereo/{dirname}/{_CKPT_FNAME}"
            )

            if not os.path.isfile(cache_path):
                # Try auto-download via gdown
                try:
                    import gdown  # type: ignore
                    cache_dir = os.path.dirname(cache_path)
                    os.makedirs(cache_dir, exist_ok=True)
                    logger.info("Downloading FoundationStereo (%s) from Google Drive …", config.variant)
                    gdown.download_folder(_GDRIVE_URL, output=os.path.expanduser("~/.cache/foundation_stereo"), quiet=False)
                except Exception as gdrive_err:
                    raise RuntimeError(
                        f"\nFoundationStereo weights are NOT on HuggingFace Hub.\n"
                        f"Download the '{dirname}' folder from Google Drive:\n"
                        f"  {_GDRIVE_URL}\n"
                        f"Place the downloaded folder so that:\n"
                        f"  {cache_path}\n"
                        f"exists, then re-run.  Alternatively load a local file:\n"
                        f"  FoundationStereoModel.from_pretrained(\n"
                        f"      '/path/to/{dirname}/{_CKPT_FNAME}', variant='{config.variant}')\n"
                        f"(gdown error: {gdrive_err})"
                    ) from gdrive_err

            if not os.path.isfile(cache_path):
                raise RuntimeError(
                    f"Checkpoint not found at '{cache_path}'.\n"
                    f"Download '{dirname}' from {_GDRIVE_URL} and place it there."
                )

            model = cls(config)
            _apply_state(model, _read_ckpt(cache_path))
            logger.info("Loaded FoundationStereoModel (%s).", config.variant)
            return model.to(device).eval()

        # ── local .pth file ───────────────────────────────────────────────────
        if os.path.isfile(model_id):
            variant = kwargs.pop("variant", "standard")
            config  = FoundationStereoConfig(variant=variant, **kwargs)
            model   = cls(config)
            _apply_state(model, _read_ckpt(model_id))
            logger.info("Loaded FoundationStereoModel (%s) from '%s'.", config.variant, model_id)
            return model.to(device).eval()

        raise ValueError(
            f"Unknown model_id '{model_id}'. "
            f"Pass a registered variant ID ({list(_FS_VARIANT_MAP.keys())}) "
            f"or an absolute path to a .pth checkpoint.\n"
            f"Weights can be downloaded from: {_GDRIVE_URL}"
        )
