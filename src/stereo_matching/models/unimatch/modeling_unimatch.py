"""
UniMatch stereo model — vendored from third-party/unimatch.

This file contains the stereo-only disparity path.
"""

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ...modeling_utils import BaseStereoModel
from .configuration_unimatch import UniMatchConfig, _UNIMATCH_VARIANT_MAP

logger = logging.getLogger(__name__)


class _UniMatchInputPadder:
    def __init__(self, dims, padding_factor=32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class _UniMatchPositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x):
        b, _, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _UniMatchMultiScaleTridentConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        strides=1,
        paddings=0,
        dilations=1,
        dilation=1,
        groups=1,
        num_branch=1,
        test_branch_idx=-1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_branch = num_branch
        self.stride = _pair(stride)
        self.groups = groups
        self.with_bias = bias
        self.dilation = dilation
        if isinstance(paddings, int):
            paddings = [paddings] * self.num_branch
        if isinstance(dilations, int):
            dilations = [dilations] * self.num_branch
        if isinstance(strides, int):
            strides = [strides] * self.num_branch
        self.paddings = [_pair(padding) for padding in paddings]
        self.dilations = [_pair(dilation_item) for dilation_item in dilations]
        self.strides = [_pair(stride_item) for stride_item in strides]
        self.test_branch_idx = test_branch_idx
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        num_branch = self.num_branch if self.training or self.test_branch_idx == -1 else 1
        assert len(inputs) == num_branch

        if self.training or self.test_branch_idx == -1:
            outputs = [
                F.conv2d(input_item, self.weight, self.bias, stride, padding, self.dilation, self.groups)
                for input_item, stride, padding in zip(inputs, self.strides, self.paddings)
            ]
        else:
            outputs = [
                F.conv2d(
                    inputs[0],
                    self.weight,
                    self.bias,
                    self.strides[self.test_branch_idx] if self.test_branch_idx == -1 else self.strides[-1],
                    self.paddings[self.test_branch_idx] if self.test_branch_idx == -1 else self.paddings[-1],
                    self.dilation,
                    self.groups,
                )
            ]

        if self.norm is not None:
            outputs = [self.norm(x) for x in outputs]
        if self.activation is not None:
            outputs = [self.activation(x) for x in outputs]
        return outputs


def _unimatch_split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        return (
            feature.view(b, num_splits, h_new, num_splits, w_new, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )

    b, c, h, w = feature.size()
    assert h % num_splits == 0 and w % num_splits == 0
    b_new = b * num_splits * num_splits
    h_new = h // num_splits
    w_new = w // num_splits
    return (
        feature.view(b, c, num_splits, h_new, num_splits, w_new)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b_new, c, h_new, w_new)
    )


def _unimatch_merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        return (
            splits.view(new_b, num_splits, num_splits, h, w, c)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )

    b, c, h, w = splits.size()
    new_b = b // num_splits // num_splits
    return (
        splits.view(new_b, num_splits, num_splits, c, h, w)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(new_b, c, num_splits * h, num_splits * w)
    )


def _unimatch_split_feature_1d(feature, num_splits=2):
    b, w, c = feature.size()
    assert w % num_splits == 0
    return feature.view(b, num_splits, w // num_splits, c).view(b * num_splits, w // num_splits, c)


def _unimatch_merge_splits_1d(splits, h, num_splits=2):
    b, w, c = splits.size()
    new_b = b // num_splits // h
    return splits.view(new_b, h, num_splits, w, c).view(new_b, h, num_splits * w, c)


def _unimatch_window_partition_1d(x, window_size_w):
    b, w, c = x.shape
    return x.view(b, w // window_size_w, window_size_w, c).view(-1, window_size_w, c)


def _unimatch_generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w, shift_size_h, shift_size_w, device):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1), device=device)
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = _unimatch_split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


def _unimatch_generate_shift_window_attn_mask_1d(input_w, window_size_w, shift_size_w, device):
    img_mask = torch.zeros((1, input_w, 1), device=device)
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for w_slice in w_slices:
        img_mask[:, w_slice, :] = cnt
        cnt += 1
    mask_windows = _unimatch_window_partition_1d(img_mask, window_size_w)
    mask_windows = mask_windows.view(-1, window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


def _unimatch_feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = _UniMatchPositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        feature0_splits = _unimatch_split_feature(feature0, num_splits=attn_splits)
        feature1_splits = _unimatch_split_feature(feature1, num_splits=attn_splits)
        position = pos_enc(feature0_splits)
        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position
        feature0 = _unimatch_merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = _unimatch_merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position
    return feature0, feature1


def _unimatch_upsample_flow_with_mask(flow, up_mask, upsample_factor):
    b, flow_channel, h, w = flow.shape
    mask = up_mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)
    mask = torch.softmax(mask, dim=2)
    up_flow = F.unfold(upsample_factor * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)
    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(b, flow_channel, upsample_factor * h, upsample_factor * w)


def _unimatch_single_head_full_attention(q, k, v):
    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** 0.5)
    attn = torch.softmax(scores, dim=2)
    return torch.matmul(attn, v)


def _unimatch_single_head_full_attention_1d(q, k, v, h, w):
    b, _, c = q.size()
    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)
    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (c ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v).view(b, -1, c)


def _unimatch_single_head_split_window_attention(q, k, v, num_splits, with_shift, h, w, attn_mask):
    b, _, c = q.size()
    b_new = b * num_splits * num_splits
    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    if with_shift:
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2
        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = _unimatch_split_feature(q, num_splits=num_splits, channel_last=True)
    k = _unimatch_split_feature(k, num_splits=num_splits, channel_last=True)
    v = _unimatch_split_feature(v, num_splits=num_splits, channel_last=True)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / (c ** 0.5)
    if with_shift:
        scores = scores + attn_mask.repeat(b, 1, 1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.view(b_new, -1, c))
    out = _unimatch_merge_splits(out.view(b_new, h // num_splits, w // num_splits, c), num_splits=num_splits, channel_last=True)
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
    return out.view(b, -1, c)


def _unimatch_single_head_split_window_attention_1d(q, k, v, num_splits, with_shift, h, w, attn_mask):
    b, _, c = q.size()
    b_new = b * num_splits * h
    window_size_w = w // num_splits
    q = q.view(b * h, w, c)
    k = k.view(b * h, w, c)
    v = v.view(b * h, w, c)

    if with_shift:
        shift_size_w = window_size_w // 2
        q = torch.roll(q, shifts=-shift_size_w, dims=1)
        k = torch.roll(k, shifts=-shift_size_w, dims=1)
        v = torch.roll(v, shifts=-shift_size_w, dims=1)

    q = _unimatch_split_feature_1d(q, num_splits=num_splits)
    k = _unimatch_split_feature_1d(k, num_splits=num_splits)
    v = _unimatch_split_feature_1d(v, num_splits=num_splits)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)) / (c ** 0.5)
    if with_shift:
        scores = scores + attn_mask.repeat(b * h, 1, 1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.view(b_new, -1, c))
    out = _unimatch_merge_splits_1d(out, h, num_splits=num_splits)
    if with_shift:
        out = torch.roll(out, shifts=shift_size_w, dims=2)
    return out.view(b, -1, c)


class _UniMatchSelfAttnPropagation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow, local_window_attn=False, local_window_radius=1):
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow, local_window_radius)

        b, c, h, w = feature0.size()
        query = feature0.view(b, c, h * w).permute(0, 2, 1)
        query = self.q_proj(query)
        key = self.k_proj(query)
        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, value)
        return out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)

    def forward_local_window_attn(self, feature0, flow, local_window_radius):
        b, c, h, w = feature0.size()
        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)).reshape(b * h * w, 1, c)
        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)
        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size, padding=local_window_radius)
        feature0_window = (
            feature0_window.view(b, c, kernel_size ** 2, h, w)
            .permute(0, 3, 4, 1, 2)
            .reshape(b * h * w, c, kernel_size ** 2)
        )
        flow_window = F.unfold(flow, kernel_size=kernel_size, padding=local_window_radius)
        flow_window = (
            flow_window.view(b, value_channel, kernel_size ** 2, h, w)
            .permute(0, 3, 4, 2, 1)
            .reshape(b * h * w, kernel_size ** 2, value_channel)
        )
        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, flow_window).view(b, h, w, value_channel)
        return out.permute(0, 3, 1, 2).contiguous()


def _unimatch_coords_grid(b, h, w, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack((x, y), dim=0).float()[None].repeat(b, 1, 1, 1)
    return grid.to(device) if device is not None else grid


def _unimatch_generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device):
    y, x = torch.meshgrid(
        torch.linspace(h_min, h_max, len_h, device=device),
        torch.linspace(w_min, w_max, len_w, device=device),
        indexing="ij",
    )
    return torch.stack((x, y), -1).float()


def _unimatch_normalize_coords(coords, h, w):
    c = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], dtype=torch.float32, device=coords.device)
    return (coords - c) / c


def _unimatch_bilinear_sample(img, sample_coords, mode="bilinear", padding_mode="zeros"):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    _, _, h, w = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def _unimatch_flow_warp(feature, flow):
    b, _, h, w = feature.size()
    grid = _unimatch_coords_grid(b, h, w, device=flow.device) + flow
    return _unimatch_bilinear_sample(feature, grid, padding_mode="zeros")


def _unimatch_global_correlation_softmax_stereo(feature0, feature1):
    b, c, h, w = feature0.shape
    x_grid = torch.linspace(0, w - 1, w, device=feature0.device)
    feature0 = feature0.permute(0, 2, 3, 1)
    feature1 = feature1.permute(0, 2, 1, 3)
    correlation = torch.matmul(feature0, feature1) / (c ** 0.5)
    mask = torch.triu(torch.ones((w, w), device=feature0.device), diagonal=1).type_as(feature0)
    valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
    correlation[~valid_mask] = -1e9
    prob = F.softmax(correlation, dim=-1)
    correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)
    disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence
    return disparity.unsqueeze(1), prob


def _unimatch_local_correlation_softmax_stereo(feature0, feature1, local_radius):
    b, c, h, w = feature0.size()
    coords_init = _unimatch_coords_grid(b, h, w, device=feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1).contiguous()
    window_grid = _unimatch_generate_window_grid(0, 0, -local_radius, local_radius, 1, 2 * local_radius + 1, feature0.device)
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = _unimatch_normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode="zeros", align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).contiguous().view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr[~valid] = -1e9
    prob = F.softmax(corr, dim=-1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords).squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2).contiguous()
    flow = correspondence - coords_init
    return -flow[:, :1], prob


def _unimatch_local_correlation_with_flow(feature0, feature1, flow, local_radius, dilation=1):
    b, c, h, w = feature0.size()
    coords_init = _unimatch_coords_grid(b, h, w, device=feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    window_grid = _unimatch_generate_window_grid(
        -local_radius,
        local_radius,
        -local_radius,
        local_radius,
        2 * local_radius + 1,
        2 * local_radius + 1,
        feature0.device,
    )
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation
    sample_coords = sample_coords + flow.view(b, 2, -1).permute(0, 2, 1).unsqueeze(-2)
    sample_coords_norm = _unimatch_normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm, padding_mode="zeros", align_corners=True).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    return corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class _UniMatchResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if stride != 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)
        else:
            self.downsample = None

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class _UniMatchCNNEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=2):
        super().__init__()
        self.num_branch = num_output_scales
        feature_dims = [64, 96, 128]
        self.conv1 = nn.Conv2d(3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], stride=2, norm_layer=norm_layer)
        stride = 2 if num_output_scales == 1 else 1
        self.layer3 = self._make_layer(feature_dims[2], stride=stride, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)
        if self.num_branch > 1:
            self.trident_conv = _UniMatchMultiScaleTridentConv(
                output_dim,
                output_dim,
                kernel_size=3,
                strides=(1, 2),
                paddings=1,
                num_branch=self.num_branch,
            )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = _UniMatchResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = _UniMatchResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.num_branch > 1:
            return self.trident_conv([x] * self.num_branch)
        return [x]


class _UniMatchTransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=1, no_ffn=False, ffn_dim_expansion=4):
        super().__init__()
        self.no_ffn = no_ffn
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target, height, width, shifted_window_attn_mask, shifted_window_attn_mask_1d, attn_num_splits, with_shift):
        query, key, value = source, target, target
        is_self_attn = (query - key).abs().max() < 1e-6
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        if is_self_attn:
            if attn_num_splits > 1:
                message = _unimatch_single_head_split_window_attention(
                    query, key, value, attn_num_splits, with_shift, height, width, shifted_window_attn_mask
                )
            else:
                message = _unimatch_single_head_full_attention(query, key, value)
        else:
            if attn_num_splits > 1:
                message = _unimatch_single_head_split_window_attention_1d(
                    query, key, value, attn_num_splits, with_shift, height, width, shifted_window_attn_mask_1d
                )
            else:
                message = _unimatch_single_head_full_attention_1d(query, key, value, height, width)

        message = self.merge(message)
        message = self.norm1(message)
        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)
        return source + message


class _UniMatchTransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=1, ffn_dim_expansion=4):
        super().__init__()
        self.self_attn = _UniMatchTransformerLayer(d_model=d_model, nhead=nhead, no_ffn=True, ffn_dim_expansion=ffn_dim_expansion)
        self.cross_attn_ffn = _UniMatchTransformerLayer(d_model=d_model, nhead=nhead, ffn_dim_expansion=ffn_dim_expansion)

    def forward(self, source, target, height, width, shifted_window_attn_mask, shifted_window_attn_mask_1d, attn_num_splits, with_shift):
        source = self.self_attn(
            source,
            source,
            height,
            width,
            shifted_window_attn_mask,
            shifted_window_attn_mask_1d,
            attn_num_splits,
            with_shift,
        )
        return self.cross_attn_ffn(
            source,
            target,
            height,
            width,
            shifted_window_attn_mask,
            shifted_window_attn_mask_1d,
            attn_num_splits,
            with_shift,
        )


class _UniMatchFeatureTransformer(nn.Module):
    def __init__(self, num_layers=6, d_model=128, nhead=1, ffn_dim_expansion=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _UniMatchTransformerBlock(d_model=d_model, nhead=nhead, ffn_dim_expansion=ffn_dim_expansion)
                for _ in range(num_layers)
            ]
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1, attn_num_splits):
        b, c, h, w = feature0.shape
        feature0 = feature0.flatten(-2).permute(0, 2, 1)
        feature1 = feature1.flatten(-2).permute(0, 2, 1)

        if attn_num_splits > 1:
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits
            shifted_window_attn_mask = _unimatch_generate_shift_window_attn_mask(
                (h, w),
                window_size_h,
                window_size_w,
                window_size_h // 2,
                window_size_w // 2,
                feature0.device,
            )
            shifted_window_attn_mask_1d = _unimatch_generate_shift_window_attn_mask_1d(
                w,
                window_size_w,
                window_size_w // 2,
                feature0.device,
            )
        else:
            shifted_window_attn_mask = None
            shifted_window_attn_mask_1d = None

        concat0 = torch.cat((feature0, feature1), dim=0)
        concat1 = torch.cat((feature1, feature0), dim=0)

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                h,
                w,
                shifted_window_attn_mask,
                shifted_window_attn_mask_1d,
                attn_num_splits,
                attn_num_splits > 1 and i % 2 == 1,
            )
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return feature0, feature1


class _UniMatchFlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, out_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class _UniMatchSepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, kernel_size=5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, kernel_size), padding=(0, padding))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (kernel_size, 1), padding=(padding, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class _UniMatchBasicMotionEncoder(nn.Module):
    def __init__(self, corr_channels=81, flow_channels=1):
        super().__init__()
        self.convc1 = nn.Conv2d(corr_channels, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(flow_channels, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - flow_channels, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class _UniMatchBasicUpdateBlock(nn.Module):
    def __init__(self, corr_channels=81, hidden_dim=128, context_dim=128, downsample_factor=4, flow_dim=1):
        super().__init__()
        self.encoder = _UniMatchBasicMotionEncoder(corr_channels=corr_channels, flow_channels=flow_dim)
        self.gru = _UniMatchSepConvGRU(hidden_dim=hidden_dim, input_dim=context_dim + hidden_dim)
        self.flow_head = _UniMatchFlowHead(hidden_dim, hidden_dim=256, out_dim=flow_dim)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, downsample_factor ** 2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, self.mask(net), delta_flow


class _UniMatchStereoNet(nn.Module):
    def __init__(self, config: UniMatchConfig):
        super().__init__()
        self.feature_channels = config.feature_channels
        self.num_scales = config.num_scales
        self.upsample_factor = config.upsample_factor
        self.backbone = _UniMatchCNNEncoder(output_dim=config.feature_channels, num_output_scales=config.num_scales)
        self.transformer = _UniMatchFeatureTransformer(
            num_layers=config.num_transformer_layers,
            d_model=config.feature_channels,
            nhead=config.num_head,
            ffn_dim_expansion=config.ffn_dim_expansion,
        )
        self.feature_flow_attn = _UniMatchSelfAttnPropagation(in_channels=config.feature_channels)
        self.refine_proj = nn.Conv2d(config.feature_channels, 256, 1)
        self.refine = _UniMatchBasicUpdateBlock(
            corr_channels=(2 * 4 + 1) ** 2,
            downsample_factor=config.upsample_factor,
            flow_dim=1,
        )

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)
        features = self.backbone(concat)[::-1]
        feature0, feature1 = [], []
        for feature in features:
            left_feature, right_feature = torch.chunk(feature, 2, dim=0)
            feature0.append(left_feature)
            feature1.append(right_feature)
        return feature0, feature1

    def upsample_flow(self, flow, upsample_factor):
        return F.interpolate(flow, scale_factor=upsample_factor, mode="bilinear", align_corners=True) * upsample_factor

    def forward(self, img0, img1, attn_splits_list, corr_radius_list, prop_radius_list, num_reg_refine):
        disp_preds = []
        feature0_list, feature1_list = self.extract_feature(img0, img1)
        disparity = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0 = feature0_list[scale_idx]
            feature1 = feature1_list[scale_idx]
            feature0_ori = feature0
            feature1_ori = feature1
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if disparity is not None:
                disparity = F.interpolate(disparity, scale_factor=2, mode="bilinear", align_corners=True) * 2
                displace = torch.cat((-disparity, torch.zeros_like(disparity)), dim=1)
                feature1 = _unimatch_flow_warp(feature1, displace)

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            feature0, feature1 = _unimatch_feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            if corr_radius == -1:
                disparity_pred = _unimatch_global_correlation_softmax_stereo(feature0, feature1)[0]
            else:
                disparity_pred = _unimatch_local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]

            disparity = disparity + disparity_pred if disparity is not None else disparity_pred
            disparity = disparity.clamp(min=0)

            if self.training:
                disp_bilinear = self.upsample_flow(disparity, upsample_factor=upsample_factor)
                disp_preds.append(disp_bilinear)

            disparity = self.feature_flow_attn(
                feature0,
                disparity.detach(),
                local_window_attn=prop_radius > 0,
                local_window_radius=prop_radius,
            )

            if self.training and scale_idx < self.num_scales - 1:
                disp_up = self.upsample_flow(disparity, upsample_factor=upsample_factor)
                disp_preds.append(disp_up)

            if scale_idx == self.num_scales - 1:
                if self.training:
                    disp_up = self.upsample_flow(disparity, upsample_factor=upsample_factor)
                    disp_preds.append(disp_up)

                for refine_iter_idx in range(num_reg_refine):
                    disparity = disparity.detach()
                    displace = torch.cat((-disparity, torch.zeros_like(disparity)), dim=1)
                    correlation = _unimatch_local_correlation_with_flow(feature0_ori, feature1_ori, flow=displace, local_radius=4)
                    proj = self.refine_proj(feature0)
                    net, inp = torch.chunk(proj, chunks=2, dim=1)
                    net = torch.tanh(net)
                    inp = torch.relu(inp)
                    net, up_mask, residual_disp = self.refine(net, inp, correlation, disparity.clone())
                    disparity = (disparity + residual_disp).clamp(min=0)

                    if self.training or refine_iter_idx == num_reg_refine - 1:
                        disp_up = _unimatch_upsample_flow_with_mask(disparity, up_mask, upsample_factor=self.upsample_factor)
                        disp_preds.append(disp_up)

        return {"flow_preds": [pred.squeeze(1) for pred in disp_preds]}


def _resolve_local_variant(variant_name: str) -> str:
    if variant_name in _UNIMATCH_VARIANT_MAP:
        return _UNIMATCH_VARIANT_MAP[variant_name]
    return variant_name


def _extract_state_dict(raw):
    if isinstance(raw, dict):
        if "model" in raw and isinstance(raw["model"], dict):
            return raw["model"]
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return raw["state_dict"]
    return raw


class UniMatchModel(BaseStereoModel):
    config_class = UniMatchConfig

    def __init__(self, config: UniMatchConfig):
        super().__init__(config)
        self.net = _UniMatchStereoNet(config)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        padder = _UniMatchInputPadder(left.shape, padding_factor=self.config.padding_factor)
        left_pad, right_pad = padder.pad(left, right)
        preds = self.net(
            left_pad,
            right_pad,
            attn_splits_list=self.config.attn_splits_list,
            corr_radius_list=self.config.corr_radius_list,
            prop_radius_list=self.config.prop_radius_list,
            num_reg_refine=self.config.num_reg_refine,
        )["flow_preds"]
        if self.training:
            return [padder.unpad(pred) for pred in preds]
        return padder.unpad(preds[-1])

    def _backbone_module(self):
        return self.net.backbone

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "UniMatchModel":
        if model_id in _UNIMATCH_VARIANT_MAP:
            config = UniMatchConfig.from_variant(model_id)
            checkpoint_path = None
        elif os.path.isfile(model_id):
            variant_name = _resolve_local_variant(kwargs.pop("variant", "mixdata"))
            config = UniMatchConfig(variant=variant_name)
            checkpoint_path = model_id
        else:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Use one of {list(_UNIMATCH_VARIANT_MAP.keys())} or a local .pth file path."
            )

        if checkpoint_path is None:
            try:
                raw = torch.hub.load_state_dict_from_url(
                    config.checkpoint_url,
                    model_dir=os.path.expanduser("~/.cache/stereo_matching/unimatch"),
                    map_location=device,
                    progress=True,
                    file_name=config.checkpoint_filename,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download UniMatch checkpoint for '{model_id}'.\n"
                    f"URL: {config.checkpoint_url}\n"
                    f"Error: {exc}"
                ) from exc
        else:
            try:
                raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except Exception:
                raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = _extract_state_dict(raw)
        model = cls(config)

        new_state_dict = {}
        for key, value in state_dict.items():
            key = key[len("module."):] if key.startswith("module.") else key
            new_state_dict[f"net.{key}"] = value

        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("strict=True load failed for UniMatch: %s\nRetrying with strict=False.", exc)
            incompatible = model.load_state_dict(new_state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning("Missing keys: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys: %s", incompatible.unexpected_keys)

        source = checkpoint_path or config.checkpoint_url
        logger.info("Loaded UniMatchModel (%s) from '%s'", config.variant, source)
        return model
