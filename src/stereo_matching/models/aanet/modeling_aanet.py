"""
AANetModel — AANet stereo matching model.

All AANet core components are vendored in this single file in dependency order:
  1.  Warp utilities                                               (nets/warp.py)
  2.  FeatureBasicBlock (leaky-relu variant)                       (nets/feature.py)
  3.  ModulatedDeformConv, DeformConv, DeformConv2d               (nets/deform.py)
       ↳ Custom CUDA ops replaced with torchvision.ops.deform_conv2d
  4.  SimpleBottleneck, DeformSimpleBottleneck                     (nets/deform.py)
  5.  Bottleneck, DeformBottleneck                                 (nets/resnet.py)
  5b. FeaturePyramidNetwork                                        (nets/feature.py)
  6.  AANetFeature (ResNet-40)                                     (nets/resnet.py)
  7.  CostVolume, CostVolumePyramid                                (nets/cost.py)
  8.  AdaptiveAggregationModule, AdaptiveAggregation               (nets/aggregation.py)
  9.  DisparityEstimation                                          (nets/estimation.py)
 10.  StereoDRNetRefinement                                        (nets/refinement.py)
 11.  _AANet  (original AANet class, renamed)                      (nets/aanet.py)
 12.  AANetModel (public wrapper — BaseStereoModel subclass)

The deformable convolution CUDA extension (deform_conv/) is replaced with
torchvision.ops.deform_conv2d.  All module attribute names are preserved so
that checkpoints trained with the original code load without remapping.

Original paper: https://arxiv.org/abs/2004.09548
Original code:  https://github.com/haofeixu/aanet
"""

import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ...modeling_utils import BaseStereoModel
from .configuration_aanet import AANetConfig, _AANET_VARIANT_MAP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Warp utilities  (nets/warp.py)
# ---------------------------------------------------------------------------

def _aa_normalize_coords(grid):
    """Normalize image-scale coordinates to [-1, 1].  grid: [B, 2, H, W]"""
    assert grid.size(1) == 2
    h, w = grid.size()[2:]
    grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x
    grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y
    return grid.permute(0, 2, 3, 1)  # [B, H, W, 2]


def _aa_meshgrid(img):
    """Return pixel-coordinate grid [B, 2, H, W] matching img."""
    b, _, h, w = img.size()
    x_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)
    y_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)
    grid = torch.cat((x_range, y_range), dim=0)          # [2, H, W]
    return grid.unsqueeze(0).expand(b, 2, h, w)          # [B, 2, H, W]


def _aa_disp_warp(img, disp, padding_mode="border"):
    """Warp right image to left view using disparity.

    Args:
        img:  [B, 3, H, W]
        disp: [B, 1, H, W], non-negative
    Returns:
        (warped_img, valid_mask), both [B, 3, H, W]
    """
    assert disp.min() >= 0
    grid = _aa_meshgrid(img)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = _aa_normalize_coords(grid + offset)
    warped = F.grid_sample(img, sample_grid, mode="bilinear",
                           padding_mode=padding_mode, align_corners=False)
    mask = torch.ones_like(img)
    valid = F.grid_sample(mask, sample_grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)
    valid[valid < 0.9999] = 0
    valid[valid > 0] = 1
    return warped, valid


# ---------------------------------------------------------------------------
# 2. FeatureBasicBlock  (nets/feature.py — leaky-relu variant)
#    Used inside StereoDRNetRefinement's dilated blocks.
# ---------------------------------------------------------------------------

def _aa_feat_conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class _AA_FeatureBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = _aa_feat_conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = _aa_feat_conv3x3(planes, planes, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# 3. Deformable Convolution  (nets/deform.py + nets/deform_conv/)
#    The original implementation used a custom CUDA extension.  Here we
#    replace it with torchvision.ops.deform_conv2d, which is functionally
#    identical and widely available.  Module *attribute names* are kept
#    identical to the originals so checkpoint keys load without remapping.
# ---------------------------------------------------------------------------

class _AA_ModulatedDeformConv(nn.Module):
    """Modulated (DCNv2) deformable convolution — weight keys match originals."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        n = in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset, mask):
        from torchvision.ops import deform_conv2d as _tv_dcn
        return _tv_dcn(x, offset, self.weight, bias=self.bias,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation, mask=mask)


class _AA_DeformConv(nn.Module):
    """Non-modulated (DCNv1) deformable convolution — weight keys match originals."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=False):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride   = _pair(stride)
        self.padding  = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))

        n = in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        from torchvision.ops import deform_conv2d as _tv_dcn
        return _tv_dcn(x, offset, self.weight, bias=None,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation)


class _AA_DeformConv2d(nn.Module):
    """Wrapper that learns offset (and mask) then applies deformable conv.

    Sub-module names ``deform_conv`` and ``offset_conv`` are kept identical
    to the originals so that checkpoint state-dict keys load directly.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=2, groups=1,
                 deformable_groups=2, modulation=True,
                 double_mask=True, bias=False):
        super().__init__()
        self.modulation       = modulation
        self.deformable_groups = deformable_groups
        self.kernel_size      = kernel_size
        self.double_mask      = double_mask

        if modulation:
            self.deform_conv = _AA_ModulatedDeformConv(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=dilation, dilation=dilation,
                groups=groups, deformable_groups=deformable_groups,
                bias=bias)
        else:
            self.deform_conv = _AA_DeformConv(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=dilation, dilation=dilation,
                groups=groups, deformable_groups=deformable_groups,
                bias=bias)

        # offset_conv outputs: (k=3 → offset+mask) or (k=2 → offset only)
        k = 3 if modulation else 2
        offset_out_channels = deformable_groups * k * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, offset_out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=dilation, dilation=dilation,
            groups=deformable_groups, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias,   0.0)

    def forward(self, x):
        if self.modulation:
            offset_mask    = self.offset_conv(x)
            offset_channel = self.deformable_groups * 2 * self.kernel_size * self.kernel_size
            offset = offset_mask[:, :offset_channel]
            mask   = offset_mask[:, offset_channel:].sigmoid()
            if self.double_mask:
                mask = mask * 2          # initialize as ~1 to act like regular conv
            return self.deform_conv(x, offset, mask)
        else:
            offset = self.offset_conv(x)
            return self.deform_conv(x, offset)


# ---------------------------------------------------------------------------
# 4. Simple / Deformable bottleneck blocks  (nets/deform.py)
# ---------------------------------------------------------------------------

def _aa_res_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def _aa_res_conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _AA_SimpleBottleneck(nn.Module):
    """Standard bottleneck without channel expansion (used in early aggregation stages)."""

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _aa_res_conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = _aa_res_conv3x3(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = _aa_res_conv1x1(width, planes)
        self.bn3   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _AA_DeformSimpleBottleneck(nn.Module):
    """Deformable bottleneck without channel expansion (used in later aggregation stages)."""

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, norm_layer=None,
                 mdconv_dilation=2, deformable_groups=2,
                 modulation=True, double_mask=True):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _aa_res_conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = _AA_DeformConv2d(
            width, width, stride=stride,
            dilation=mdconv_dilation,
            deformable_groups=deformable_groups,
            modulation=modulation, double_mask=double_mask)
        self.bn2   = norm_layer(width)
        self.conv3 = _aa_res_conv1x1(width, planes)
        self.bn3   = norm_layer(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# 5. ResNet-style bottleneck blocks  (nets/resnet.py + nets/deform.py)
# ---------------------------------------------------------------------------

class _AA_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _aa_res_conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        self.conv2 = _aa_res_conv3x3(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = _aa_res_conv1x1(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _AA_DeformBottleneck(nn.Module):
    """Bottleneck with deformable conv2 (used in AANetFeature.layer3)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _aa_res_conv1x1(inplanes, width)
        self.bn1   = norm_layer(width)
        # DeformConv2d default: kernel=3, dilation=2  (not the ResNet dilation arg)
        self.conv2 = _AA_DeformConv2d(width, width, stride=stride)
        self.bn2   = norm_layer(width)
        self.conv3 = _aa_res_conv1x1(width, planes * self.expansion)
        self.bn3   = norm_layer(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# 6. AANetFeature — ResNet-40 feature extractor  (nets/resnet.py)
#    Outputs [layer1@H/3, layer2@H/6, layer3@H/12]
# ---------------------------------------------------------------------------

class _AA_AANetFeature(nn.Module):
    def __init__(self, in_channels=32, zero_init_residual=True,
                 groups=1, width_per_group=64,
                 feature_mdconv=True, norm_layer=None):
        super().__init__()
        norm_layer   = norm_layer or nn.BatchNorm2d
        self._norm_layer  = norm_layer
        self.inplanes     = in_channels
        self.dilation     = 1
        self.groups       = groups
        self.base_width   = width_per_group

        layers = [3, 4, 6]  # ResNet-40

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=3, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True))                                     # H/3

        self.layer1 = self._make_layer(_AA_Bottleneck,       in_channels,     layers[0])          # H/3
        self.layer2 = self._make_layer(_AA_Bottleneck,       in_channels * 2, layers[1], stride=2)# H/6
        block = _AA_DeformBottleneck if feature_mdconv else _AA_Bottleneck
        self.layer3 = self._make_layer(block,                in_channels * 4, layers[2], stride=2)# H/12

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, _AA_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _aa_res_conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers_list = [block(self.inplanes, planes, stride, downsample,
                             self.groups, self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes,
                                     groups=self.groups, base_width=self.base_width,
                                     dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers_list)

    def forward(self, x):
        x      = self.conv1(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return [layer1, layer2, layer3]


# ---------------------------------------------------------------------------
# 6b. FeaturePyramidNetwork  (nets/feature.py)
#     All pretrained checkpoints use feature_pyramid_network=True.
#     The FPN normalises [layer1@128ch, layer2@256ch, layer3@512ch] → 128ch.
# ---------------------------------------------------------------------------

class _AA_FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels=128, num_levels=3):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs     = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, 1))
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        # inputs: list of feature maps, resolution high → low
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        # top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        return [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]


# ---------------------------------------------------------------------------
# 7. Cost volume  (nets/cost.py)
# ---------------------------------------------------------------------------

class _AA_CostVolume(nn.Module):
    def __init__(self, max_disp, feature_similarity="correlation"):
        super().__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()
        if self.feature_similarity == "correlation":
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (
                        left_feature[:, :, :, i:] * right_feature[:, :, :, :-i]
                    ).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        else:
            raise NotImplementedError(
                f"feature_similarity='{self.feature_similarity}' is not supported")
        return cost_volume.contiguous()


class _AA_CostVolumePyramid(nn.Module):
    def __init__(self, max_disp, feature_similarity="correlation"):
        super().__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature_pyramid, right_feature_pyramid):
        num_scales = len(left_feature_pyramid)
        cost_volume_pyramid = []
        for s in range(num_scales):
            max_disp = self.max_disp // (2 ** s)
            cv = _AA_CostVolume(max_disp, self.feature_similarity)
            cost_volume_pyramid.append(
                cv(left_feature_pyramid[s], right_feature_pyramid[s]))
        return cost_volume_pyramid  # [H/3, H/6, H/12] scales


# ---------------------------------------------------------------------------
# 8. Adaptive Aggregation  (nets/aggregation.py)
# ---------------------------------------------------------------------------

class _AA_AdaptiveAggregationModule(nn.Module):
    """One stage of adaptive intra-scale + cross-scale aggregation."""

    def __init__(self, num_scales, num_output_branches, max_disp,
                 num_blocks=1, simple_bottleneck=False,
                 deformable_groups=2, mdconv_dilation=2):
        super().__init__()
        self.num_scales          = num_scales
        self.num_output_branches = num_output_branches
        self.max_disp            = max_disp
        self.num_blocks          = num_blocks

        # Intra-scale aggregation branches
        self.branches = nn.ModuleList()
        for i in range(self.num_scales):
            num_candidates = max_disp // (2 ** i)
            branch = nn.ModuleList()
            for _ in range(num_blocks):
                if simple_bottleneck:
                    branch.append(_AA_SimpleBottleneck(num_candidates, num_candidates))
                else:
                    branch.append(_AA_DeformSimpleBottleneck(
                        num_candidates, num_candidates,
                        modulation=True,
                        mdconv_dilation=mdconv_dilation,
                        deformable_groups=deformable_groups))
            self.branches.append(nn.Sequential(*branch))

        # Cross-scale fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(self.num_output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.num_scales):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # Upsample (1×1 conv then bilinear interpolation)
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                  kernel_size=1, bias=False),
                        nn.BatchNorm2d(max_disp // (2 ** i))))
                else:
                    # Downsample (stride-2 convs)
                    layers_list = nn.ModuleList()
                    for _ in range(i - j - 1):
                        layers_list.append(nn.Sequential(
                            nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** j),
                                      kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(max_disp // (2 ** j)),
                            nn.LeakyReLU(0.2, inplace=True)))
                    layers_list.append(nn.Sequential(
                        nn.Conv2d(max_disp // (2 ** j), max_disp // (2 ** i),
                                  kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(max_disp // (2 ** i))))
                    self.fuse_layers[-1].append(nn.Sequential(*layers_list))

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # Intra-scale aggregation
        for i in range(len(self.branches)):
            for j in range(self.num_blocks):
                x[i] = self.branches[i][j](x[i])

        if self.num_scales == 1:
            return x

        # Cross-scale aggregation
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    exchange = self.fuse_layers[i][j](x[j])
                    if exchange.size()[2:] != x_fused[i].size()[2:]:
                        exchange = F.interpolate(exchange, size=x_fused[i].size()[2:],
                                                 mode="bilinear", align_corners=False)
                    x_fused[i] = x_fused[i] + exchange

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])
        return x_fused


class _AA_AdaptiveAggregation(nn.Module):
    """Stack of AdaptiveAggregationModules with per-scale final projection."""

    def __init__(self, max_disp, num_scales=3, num_fusions=6,
                 num_stage_blocks=1, num_deform_blocks=2,
                 intermediate_supervision=True,
                 deformable_groups=2, mdconv_dilation=2):
        super().__init__()
        self.num_scales             = num_scales
        self.num_fusions            = num_fusions
        self.intermediate_supervision = intermediate_supervision

        fusions = nn.ModuleList()
        for i in range(num_fusions):
            num_out_branches = (
                num_scales if intermediate_supervision
                else (1 if i == num_fusions - 1 else num_scales))
            # Last num_deform_blocks stages use deformable conv
            use_simple = i < (num_fusions - num_deform_blocks)
            fusions.append(_AA_AdaptiveAggregationModule(
                num_scales=num_scales,
                num_output_branches=num_out_branches,
                max_disp=max_disp,
                num_blocks=num_stage_blocks,
                mdconv_dilation=mdconv_dilation,
                deformable_groups=deformable_groups,
                simple_bottleneck=use_simple))

        self.fusions = nn.Sequential(*fusions)

        self.final_conv = nn.ModuleList()
        for i in range(num_scales):
            in_ch = max_disp // (2 ** i)
            self.final_conv.append(nn.Conv2d(in_ch, in_ch, kernel_size=1))
            if not intermediate_supervision:
                break

    def forward(self, cost_volume):
        for i in range(self.num_fusions):
            cost_volume = self.fusions[i](cost_volume)
        return [self.final_conv[i](cost_volume[i]) for i in range(len(self.final_conv))]


# ---------------------------------------------------------------------------
# 9. Disparity Estimation  (nets/estimation.py)
# ---------------------------------------------------------------------------

class _AA_DisparityEstimation(nn.Module):
    def __init__(self, max_disp, match_similarity=True):
        super().__init__()
        self.max_disp = max_disp
        self.match_similarity = match_similarity

    def forward(self, cost_volume):
        cost_volume = cost_volume if self.match_similarity else -cost_volume
        prob_volume = F.softmax(cost_volume, dim=1)
        max_disp    = prob_volume.size(1)
        disp_cands  = torch.arange(0, max_disp).type_as(prob_volume).view(1, max_disp, 1, 1)
        return torch.sum(prob_volume * disp_cands, dim=1)  # [B, H, W]


# ---------------------------------------------------------------------------
# 10. StereoDRNet Refinement  (nets/refinement.py)
# ---------------------------------------------------------------------------

def _aa_ref_conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True))


class _AA_StereoDRNetRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _aa_ref_conv2d(6, 16)   # [left + warped_error]
        self.conv2 = _aa_ref_conv2d(1, 16)   # [disparity]

        dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.Sequential(*[
            _AA_FeatureBasicBlock(32, 32, stride=1, dilation=d)
            for d in dilation_list])

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        assert low_disp.dim() == 3
        low_disp    = low_disp.unsqueeze(1)                       # [B, 1, h, w]
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:],
                                 mode="bilinear", align_corners=False) * scale_factor

        warped_right = _aa_disp_warp(right_img, disp)[0]
        error        = warped_right - left_img
        concat1      = torch.cat((error, left_img), dim=1)        # [B, 6, H, W]

        out          = self.dilated_blocks(
                           torch.cat((self.conv1(concat1), self.conv2(disp)), dim=1))
        residual     = self.final_conv(out)                        # [B, 1, H, W]
        disp         = F.relu(disp + residual, inplace=True).squeeze(1)
        return disp                                                # [B, H, W]


# ---------------------------------------------------------------------------
# 11. _AANet — core stereo network  (nets/aanet.py)
#
#     Module attribute names are preserved exactly so that checkpoint keys
#     (feature_extractor.*, cost_volume.*, aggregation.*, ...) load directly
#     after prefixing with "net." in AANetModel._load_pretrained_weights.
# ---------------------------------------------------------------------------

class _AANet(nn.Module):
    def __init__(self, max_disp=192, num_downsample=2,
                 feature_similarity="correlation",
                 num_scales=3, num_fusions=6,
                 deformable_groups=2, mdconv_dilation=2,
                 num_stage_blocks=1, num_deform_blocks=3):
        super().__init__()
        self.num_downsample = num_downsample
        self.num_scales     = num_scales

        # Feature extractor — AANet variant with deformable conv in layer3
        # Outputs [layer1@128ch/H3, layer2@256ch/H6, layer3@512ch/H12]
        self.feature_extractor = _AA_AANetFeature(feature_mdconv=True)

        # FPN normalises all feature maps to 128 channels
        # (all pretrained checkpoints use feature_pyramid_network=True)
        self.fpn = _AA_FeaturePyramidNetwork(
            in_channels=[128, 256, 512], out_channels=128)

        self._max_disp = max_disp // 3       # H/3 scale base (= 64 for max_disp=192)

        # Cost volume pyramid (H/3, H/6, H/12 scales)
        self.cost_volume = _AA_CostVolumePyramid(
            self._max_disp, feature_similarity=feature_similarity)

        # Adaptive aggregation
        self.aggregation = _AA_AdaptiveAggregation(
            max_disp=self._max_disp,
            num_scales=num_scales,
            num_fusions=num_fusions,
            num_stage_blocks=num_stage_blocks,
            num_deform_blocks=num_deform_blocks,
            mdconv_dilation=mdconv_dilation,
            deformable_groups=deformable_groups,
            intermediate_supervision=True)

        # Disparity estimation
        self.disparity_estimation = _AA_DisparityEstimation(
            self._max_disp, match_similarity=True)

        # Hierarchical refinement (num_downsample=2 → H/2, H)
        self.refinement = nn.ModuleList(
            [_AA_StereoDRNetRefinement() for _ in range(num_downsample)])

    # ---- private helpers --------------------------------------------------

    def _feature_extraction(self, img):
        features = self.feature_extractor(img)   # [layer1@128, layer2@256, layer3@512]
        return self.fpn(features)                # [H/3@128, H/6@128, H/12@128]

    def _cost_volume_construction(self, left_feature, right_feature):
        cvp = self.cost_volume(left_feature, right_feature)
        if self.num_scales == 1:
            cvp = [cvp[0]]
        return cvp

    def _disparity_computation(self, aggregation):
        """Produce disparity pyramid in order [H/12, H/6, H/3]."""
        length = len(aggregation)
        return [self.disparity_estimation(aggregation[length - 1 - i])
                for i in range(length)]

    def _disparity_refinement(self, left_img, right_img, disparity):
        """Refine disparity through num_downsample steps → [H/2, H]."""
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1.0 / pow(2, self.num_downsample - i - 1)
            if scale_factor == 1.0:
                curr_left  = left_img
                curr_right = right_img
            else:
                curr_left  = F.interpolate(left_img,  scale_factor=scale_factor,
                                           mode="bilinear", align_corners=False)
                curr_right = F.interpolate(right_img, scale_factor=scale_factor,
                                           mode="bilinear", align_corners=False)
            disparity = self.refinement[i](disparity, curr_left, curr_right)
            disparity_pyramid.append(disparity)
        return disparity_pyramid

    # ---- forward ----------------------------------------------------------

    def forward(self, left_img, right_img):
        left_feature  = self._feature_extraction(left_img)
        right_feature = self._feature_extraction(right_img)
        cost_volume   = self._cost_volume_construction(left_feature, right_feature)
        aggregation   = self.aggregation(cost_volume)
        disp_pyramid  = self._disparity_computation(aggregation)
        disp_pyramid += self._disparity_refinement(
            left_img, right_img, disp_pyramid[-1])
        return disp_pyramid  # [D/12, D/6, D/3, H/2, H]  — 5 elements


# ---------------------------------------------------------------------------
# 12. AANetModel — public wrapper (BaseStereoModel subclass)
# ---------------------------------------------------------------------------

class AANetModel(BaseStereoModel):
    """AANet stereo matching model.

    Input:  ImageNet-normalised tensors (B, 3, H, W)  in  ~[-2, 2] range.
            This is exactly what StereoProcessor outputs — no extra
            denormalization is needed.
    Output: During inference → disparity tensor (B, H, W).
            During training  → list of 5 disparity tensors at
                               [H/12, H/6, H/3, H/2, H] for multi-scale loss.
    """

    config_class = AANetConfig

    def __init__(self, config: AANetConfig):
        super().__init__(config)
        self.net = _AANet(
            max_disp=config.max_disp,
            num_downsample=config.num_downsample,
            feature_similarity=config.feature_similarity,
            num_scales=config.num_scales,
            num_fusions=config.num_fusions,
            deformable_groups=config.deformable_groups,
            mdconv_dilation=config.mdconv_dilation,
            num_stage_blocks=config.num_stage_blocks,
            num_deform_blocks=config.num_deform_blocks)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        h, w = left.shape[-2:]
        pad_h = (12 - h % 12) % 12
        pad_w = (12 - w % 12) % 12
        if pad_h > 0 or pad_w > 0:
            left  = F.pad(left,  [0, pad_w, 0, pad_h], mode="replicate")
            right = F.pad(right, [0, pad_w, 0, pad_h], mode="replicate")

        preds = self.net(left, right)           # list of 5 disparity maps

        if self.training:
            return [p[..., :h, :w] for p in preds]
        return preds[-1][..., :h, :w]           # full-resolution (B, H, W)

    def _backbone_module(self):
        return self.net.feature_extractor

    @classmethod
    def _load_pretrained_weights(cls, model_id: str, device: str = "cpu", **kwargs):
        from huggingface_hub import hf_hub_download

        config   = AANetConfig.from_variant(model_id)
        model    = cls(config)
        filename = config.checkpoint_filename

        try:
            ckpt_path = hf_hub_download(
                repo_id=config.hub_repo_id,
                filename=filename,
                repo_type="dataset",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download checkpoint for '{model_id}' from HuggingFace Hub.\n"
                f"HF dataset: {config.hub_repo_id}\n"
                f"Error: {exc}"
            ) from exc

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Original checkpoint keys reference AANet modules directly.
        # Prefix with "net." because our wrapper stores the network as self.net.
        state_dict = {"net." + k: v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            logger.warning("Missing keys (%d): %s …", len(missing), missing[:3])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:3])

        return model
