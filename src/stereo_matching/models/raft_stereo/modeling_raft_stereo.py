"""
RaftStereoModel — RAFT-Stereo model implementation.

All RAFT-Stereo core components are vendored in this single file:
  - utils    (coords_grid, upflow8, InputPadder, bilinear_sampler)
  - extractor (ResidualBlock, BottleneckBlock, BasicEncoder, MultiBasicEncoder)
  - corr     (CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock)
  - update   (FlowHead, ConvGRU, SepConvGRU, BasicMotionEncoder, BasicMultiUpdateBlock)
  - _RAFTStereoNet (the original RAFTStereo class, renamed)
  - RaftStereoModel (public wrapper — BaseStereoModel subclass)

Original source: https://github.com/princeton-vl/RAFT-Stereo
"""

import logging
import types
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract

from ...modeling_utils import BaseStereoModel
from .configuration_raft_stereo import (
    RaftStereoConfig,
    _RAFT_VARIANT_MAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AMP autocast compatibility shim
# ---------------------------------------------------------------------------
try:
    from torch.amp import autocast as _amp_autocast

    def autocast(enabled):
        return _amp_autocast("cuda", enabled=enabled)
except ImportError:
    try:
        autocast = torch.cuda.amp.autocast
    except AttributeError:
        class autocast:  # noqa: N801
            def __init__(self, enabled):
                pass
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass


# ===========================================================================
# Section 1: Utilities  (from core/utils/utils.py)
# ===========================================================================

class InputPadder:
    """Pads images such that dimensions are divisible by ``divis_by``."""

    def __init__(self, dims, mode="sintel", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    """Forward-warp a flow field using scipy griddata (used for flow initialization)."""
    from scipy import interpolate as scipy_interp
    import numpy as np

    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]
    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))
    x1 = (x0 + dx).reshape(-1)
    y1 = (y0 + dy).reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    flow_x = scipy_interp.griddata(
        (x1[valid], y1[valid]), dx[valid], (x0, y0), method="nearest", fill_value=0
    )
    flow_y = scipy_interp.griddata(
        (x1[valid], y1[valid]), dy[valid], (x0, y0), method="nearest", fill_value=0
    )
    return torch.from_numpy(np.stack([flow_x, flow_y], axis=0)).float()


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample; uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask_out = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask_out.float()
    return img


def coords_grid(batch, ht, wd):
    """Create a coordinate grid of shape (batch, 2, ht, wd)."""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    """Upsample flow by 8× using interpolation."""
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(
        torch.arange(N).float() - N // 2,
        torch.arange(N).float() - N // 2,
        indexing="ij",
    )
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1, 1, N, N).to(input)
    output = F.conv2d(input.reshape(B * D, 1, H, W), weights, padding=N // 2)
    return output.view(B, D, H, W)


# ===========================================================================
# Section 2: Feature / Context Encoder  (from core/extractor.py)
# ===========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0, downsample=3):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = x.split(split_size=batch_dim, dim=0)
        return x


class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=None, norm_fn="batch", dropout=0.0, downsample=3):
        super(MultiBasicEncoder, self).__init__()
        if output_dim is None:
            output_dim = [128]
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[2], 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs08 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(128, 128, self.norm_fn, stride=1),
                nn.Conv2d(128, dim[1], 3, padding=1),
            )
            output_list.append(conv_out)
        self.outputs16 = nn.ModuleList(output_list)

        output_list = []
        for dim in output_dim:
            conv_out = nn.Conv2d(128, dim[0], 3, padding=1)
            output_list.append(conv_out)
        self.outputs32 = nn.ModuleList(output_list)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x, dual_inp=False, num_layers=3):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if dual_inp:
            v = x
            x = x[: (x.shape[0] // 2)]

        outputs08 = [f(x) for f in self.outputs08]
        if num_layers == 1:
            return (outputs08, v) if dual_inp else (outputs08,)

        y = self.layer4(x)
        outputs16 = [f(y) for f in self.outputs16]
        if num_layers == 2:
            return (outputs08, outputs16, v) if dual_inp else (outputs08, outputs16)

        z = self.layer5(y)
        outputs32 = [f(z) for f in self.outputs32]
        return (outputs08, outputs16, outputs32, v) if dual_inp else (outputs08, outputs16, outputs32)


# ===========================================================================
# Section 3: Correlation Volume  (from core/corr.py)
# ===========================================================================

try:
    import corr_sampler  # optional CUDA extension
except Exception:
    corr_sampler = None

try:
    import alt_cuda_corr  # optional CUDA extension
except Exception:
    alt_cuda_corr = None


class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        ctx.save_for_backward(volume, coords)
        ctx.radius = radius
        (corr,) = corr_sampler.forward(volume, coords, radius)
        return corr

    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        (grad_volume,) = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None


class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2 // 2 ** i))
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, [0]]
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), coords / 2 ** i, self.radius)
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        xgrid, ygrid = coords.split([1, 1], dim=-1)
        xgrid = 2 * xgrid / (W - 1) - 1
        ygrid = 2 * ygrid / (H - 1) - 1
        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0, 2, 3, 1)
        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        fmap2 = self.fmap2
        for i in range(self.num_levels):
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(
                torch.meshgrid(dy, dx, indexing="ij"), axis=-1
            ).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[..., 0] = centroid_lvl[..., 0] / 2 ** i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(self.fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        corr = CorrBlock1D.corr(fmap1, fmap2)
        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, 1, w2)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dx = dx.view(2 * r + 1, 1).to(coords.device)
            x0 = dx + coords.reshape(batch * h1 * w1, 1, 1, 1) / 2 ** i
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        raise NotImplementedError(
            "AlternateCorrBlock requires the alt_cuda_corr CUDA extension. "
            "Use corr_implementation='reg' or 'alt' instead."
        )


# ===========================================================================
# Section 4: Update Block  (from core/update.py)
# ===========================================================================

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, *x):
        x = torch.cat(x, dim=1)
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
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 64, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    return F.interpolate(x, dest.shape[2:], mode="bilinear", align_corners=True)


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 128

        self.gru08 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru16 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru32 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.flow_head = FlowHead(hidden_dims[2], hidden_dim=256, output_dim=2)
        factor = 2 ** self.args.n_downsample

        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr=None, flow=None, iter08=True, iter16=True, iter32=True, update=True):
        if iter32:
            net[2] = self.gru32(net[2], *(inp[2]), pool2x(net[1]))
        if iter16:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru16(net[1], *(inp[1]), pool2x(net[0]))
        if iter08:
            motion_features = self.encoder(flow, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru08(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_flow = self.flow_head(net[0])
        mask = 0.25 * self.mask(net[0])
        return net, mask, delta_flow


# ===========================================================================
# Section 5: Config → args namespace adapter
# ===========================================================================

def _config_to_args(config: RaftStereoConfig) -> types.SimpleNamespace:
    """Convert RaftStereoConfig to the args namespace the original code expects."""
    return types.SimpleNamespace(
        hidden_dims=config.hidden_dims,
        n_gru_layers=config.n_gru_layers,
        corr_levels=config.corr_levels,
        corr_radius=config.corr_radius,
        n_downsample=config.n_downsample,
        corr_implementation=config.corr_implementation,
        context_norm=config.context_norm,
        shared_backbone=config.shared_backbone,
        slow_fast_gru=config.slow_fast_gru,
        mixed_precision=config.mixed_precision,
    )


# ===========================================================================
# Section 6: _RAFTStereoNet  (original RAFTStereo, renamed; internal norm removed)
# ===========================================================================

class _RAFTStereoNet(nn.Module):
    """Internal RAFT-Stereo network.

    Renamed from RAFTStereo. The original per-image normalization
    (image / 255 → [-1, 1]) has been removed — normalization is now
    handled externally by RaftStereoModel.forward().

    Expects images in [0, 255] range (after denormalization by the wrapper).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(
            output_dim=[args.hidden_dims, context_dims],
            norm_fn=args.context_norm,
            downsample=args.n_downsample,
        )
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList(
            [
                nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2)
                for i in range(self.args.n_gru_layers)
            ]
        )

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, "instance", stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", downsample=args.n_downsample
            )

    def freeze_bn(self):
        """Freeze all BatchNorm2d layers (used during fine-tuning)."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """Run RAFT-Stereo forward pass.

        Args:
            image1: Left image (B, 3, H, W) in [0, 255] range.
            image2: Right image (B, 3, H, W) in [0, 255] range.
            iters: Number of recurrent refinement iterations.
            flow_init: Optional initial disparity estimate.
            test_mode: If True, only return the final prediction.

        Returns:
            test_mode=False: List[Tensor(B, 1, H, W)] — one per iteration.
            test_mode=True:  Tuple (coords1 - coords0, flow_up).
        """
        # Normalize to [-1, 1]
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=self.args.n_gru_layers,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [
                list(conv(i).split(split_size=conv.out_channels // 3, dim=1))
                for i, conv in zip(inp_list, self.context_zqr_convs)
            ]

        if self.args.corr_implementation == "reg":
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt":
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":
            corr_block = AlternateCorrBlock
        else:
            raise ValueError(f"Unknown corr_implementation: {self.args.corr_implementation!r}")

        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:
                    net_list = self.update_block(
                        net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False
                    )
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.args.n_gru_layers == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )
                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.args.n_gru_layers == 3,
                    iter16=self.args.n_gru_layers >= 2,
                )

            # In stereo mode, project flow onto epipolar line (zero y-component)
            delta_flow[:, 1] = 0.0
            coords1 = coords1 + delta_flow

            if test_mode and itr < iters - 1:
                continue

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


# ===========================================================================
# Section 7: RaftStereoModel — Public wrapper (BaseStereoModel subclass)
# ===========================================================================

class RaftStereoModel(BaseStereoModel):
    """RAFT-Stereo stereo matching model.

    Wraps ``_RAFTStereoNet`` with the library's standard interface.

    Usage::

        model = RaftStereoModel.from_pretrained("raft-stereo")
        # or from a local checkpoint:
        model = RaftStereoModel.from_pretrained("/path/to/raftstereo-sceneflow.pth")
    """

    config_class = RaftStereoConfig

    def __init__(self, config: RaftStereoConfig):
        super().__init__(config)
        args = _config_to_args(config)
        self.net = _RAFTStereoNet(args)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run RAFT-Stereo forward pass.

        Args:
            left:  Left image  (B, 3, H, W) in [0,1] ImageNet-normalized range.
            right: Right image (B, 3, H, W) in [0,1] ImageNet-normalized range.

        Returns:
            Inference mode: Tensor (B, H, W) — final disparity in H'/W'-pixels.
            Training mode:  List[Tensor(B, H, W)] — one per recurrent iteration.
        """
        # Denormalize [0,1] ImageNet-norm → [0,255] for _RAFTStereoNet
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        left_255 = (left * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        iters = self.config.num_iters

        if self.training:
            preds = self.net(left_255, right_255, iters=iters, test_mode=False)
            # Each pred is (B, 1, H, W) — squeeze channel dim → (B, H, W).
            # Negate: RAFT-Stereo outputs negative x-flow (right features appear
            # to the left), but library convention is positive disparity.
            return [-p.squeeze(1) for p in preds]
        else:
            _, flow_up = self.net(left_255, right_255, iters=iters, test_mode=True)
            return -flow_up.squeeze(1)  # (B, H, W) — negate to positive disparity

    def _backbone_module(self) -> Optional[nn.Module]:
        """Return the feature encoder as the backbone."""
        if hasattr(self.net, "fnet"):
            return self.net.fnet
        if hasattr(self.net, "conv2"):
            return self.net.conv2
        return None

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        for_training: bool = False,
        **kwargs,
    ) -> "RaftStereoModel":
        """Load pretrained RAFT-Stereo weights.

        Args:
            model_id: One of the registered variant IDs (e.g. "raft-stereo"),
                or a local path to a ``.pth`` checkpoint file.
            device: Device to map the weights to.
            for_training: Unused here; handled by from_pretrained().
            **kwargs: Optional ``variant`` override when loading from a local path.

        Returns:
            RaftStereoModel with loaded weights.
        """
        import os

        # 1. Resolve variant → config
        if model_id in _RAFT_VARIANT_MAP:
            config = RaftStereoConfig.from_variant(model_id)
            checkpoint_path = None
        elif os.path.isfile(model_id):
            # Local file — infer variant from kwargs or default to "standard"
            variant = kwargs.pop("variant", "standard")
            config = RaftStereoConfig(variant=variant)
            checkpoint_path = model_id
        else:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Use one of {list(_RAFT_VARIANT_MAP.keys())} or a local .pth file path."
            )

        # 2. Download from HuggingFace Hub if no local path given
        if checkpoint_path is None:
            try:
                from huggingface_hub import hf_hub_download
                checkpoint_path = hf_hub_download(
                    repo_id=config.hub_repo_id,
                    filename=config.checkpoint_filename,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download checkpoint for '{model_id}' from HuggingFace Hub. "
                    f"Pass a local .pth file path instead.\n"
                    f"HF repo: {config.hub_repo_id}\n"
                    f"Error: {exc}"
                ) from exc

        # 3. Build model
        model = cls(config)

        # 4. Load state dict
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            logger.warning(
                "weights_only=True failed; retrying with weights_only=False (legacy checkpoint)."
            )
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Unwrap nested dicts (e.g. {"model": state_dict})
        if isinstance(state_dict, dict) and "model" in state_dict and not any(
            k.startswith("net.") or k.startswith("fnet") or k.startswith("cnet")
            for k in state_dict
        ):
            state_dict = state_dict["model"]

        # 5. Remap keys to match our wrapper (self.net = _RAFTStereoNet)
        new_state_dict = {}
        for k, v in state_dict.items():
            # Strip DataParallel "module." prefix
            new_key = k[len("module."):] if k.startswith("module.") else k
            # Prefix with "net." since we wrap as self.net
            new_state_dict[f"net.{new_key}"] = v

        # 6. Load — try strict first, fall back to non-strict
        missing, unexpected = [], []
        try:
            incompatible = model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning(
                f"strict=True load failed: {exc}\n"
                "Retrying with strict=False."
            )
            incompatible = model.load_state_dict(new_state_dict, strict=False)
            missing = incompatible.missing_keys
            unexpected = incompatible.unexpected_keys

        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        logger.info(f"Loaded RaftStereoModel ({config.variant}) from '{checkpoint_path}'")
        return model
