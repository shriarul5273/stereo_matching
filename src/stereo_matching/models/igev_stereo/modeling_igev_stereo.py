"""
IGEV-Stereo model — adapted from third-party/IGEV/IGEV-Stereo.

This file vendors the stereo-only IGEV-Stereo path into a single module so the
library does not depend on runtime sys.path manipulation or extra scipy imports.
"""

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import BaseStereoModel
from .configuration_igev_stereo import IGEVStereoConfig, _IGEV_VARIANT_MAP

logger = logging.getLogger(__name__)


def _igev_autocast(enabled: bool, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


class _IGEVInputPadder:
    """Pad stereo inputs so width/height are divisible by the model stride."""

    def __init__(self, dims: Tuple[int, ...], divis_by: int = 32):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs: torch.Tensor) -> List[torch.Tensor]:
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def _igev_bilinear_sampler(
    img: torch.Tensor,
    coords: torch.Tensor,
    mode: str = "bilinear",
    mask: bool = False,
):
    """Stereo-specific wrapper around grid_sample using pixel coordinates."""

    height, width = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (width - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    sampled = F.grid_sample(img, grid, mode=mode, align_corners=True)
    if mask:
        valid = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return sampled, valid.float()
    return sampled


class _IGEVBasicConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        bn: bool = True,
        relu: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.relu = relu
        self.use_bn = bn

        if is_3d:
            conv_cls = nn.ConvTranspose3d if deconv else nn.Conv3d
            norm_cls = nn.BatchNorm3d
        else:
            conv_cls = nn.ConvTranspose2d if deconv else nn.Conv2d
            norm_cls = nn.BatchNorm2d

        self.conv = conv_cls(in_channels, out_channels, bias=False, **kwargs)
        self.bn = norm_cls(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        return x


class _IGEVConv2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        bn: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ):
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
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = _IGEVBasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = _IGEVBasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                bn=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = _IGEVBasicConv(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = _IGEVBasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                bn,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), dim=1)
        else:
            x = x + rem
        return self.conv2(x)


class _IGEVBasicConvIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        use_in: bool = True,
        relu: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.relu = relu
        self.use_in = use_in

        if is_3d:
            conv_cls = nn.ConvTranspose3d if deconv else nn.Conv3d
            norm_cls = nn.InstanceNorm3d
        else:
            conv_cls = nn.ConvTranspose2d if deconv else nn.Conv2d
            norm_cls = nn.InstanceNorm2d

        self.conv = conv_cls(in_channels, out_channels, bias=False, **kwargs)
        self.norm = norm_cls(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_in:
            x = self.norm(x)
        if self.relu:
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        return x


class _IGEVConv2xIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        use_in: bool = True,
        relu: bool = True,
        keep_dispc: bool = False,
    ):
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
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = _IGEVBasicConvIN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                use_in=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = _IGEVBasicConvIN(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                use_in=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = _IGEVBasicConvIN(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                use_in,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = _IGEVBasicConvIN(
                out_channels,
                out_channels,
                False,
                is_3d,
                use_in,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x: torch.Tensor, rem: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode="nearest")
        if self.concat:
            x = torch.cat((x, rem), dim=1)
        else:
            x = x + rem
        return self.conv2(x)


def _igev_groupwise_correlation(
    fea1: torch.Tensor,
    fea2: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    batch, channels, height, width = fea1.shape
    channels_per_group = channels // num_groups
    cost = (fea1 * fea2).view(batch, num_groups, channels_per_group, height, width).mean(dim=2)
    return cost


def _igev_build_gwc_volume(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
) -> torch.Tensor:
    batch, _, height, width = refimg_fea.shape
    volume = refimg_fea.new_zeros(batch, num_groups, maxdisp, height, width)
    for idx in range(maxdisp):
        if idx > 0:
            volume[:, :, idx, :, idx:] = _igev_groupwise_correlation(
                refimg_fea[:, :, :, idx:],
                targetimg_fea[:, :, :, :-idx],
                num_groups,
            )
        else:
            volume[:, :, idx, :, :] = _igev_groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def _igev_disparity_regression(x: torch.Tensor, maxdisp: int) -> torch.Tensor:
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device).view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, dim=1, keepdim=True)


class _IGEVFeatureAtt(nn.Module):
    def __init__(self, cv_chan: int, feat_chan: int):
        super().__init__()
        self.feat_att = nn.Sequential(
            _IGEVBasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat_att = self.feat_att(feat).unsqueeze(2)
        return torch.sigmoid(feat_att) * cv


def _igev_context_upsample(disp_low: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(batch, channels, height, width), 3, 1, 1)
    disp_unfold = disp_unfold.reshape(batch, -1, height, width)
    disp_unfold = F.interpolate(disp_unfold, (height * 4, width * 4), mode="nearest")
    disp_unfold = disp_unfold.reshape(batch, 9, height * 4, width * 4)
    return (disp_unfold * up_weights).sum(dim=1)


class _IGEVResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = max(planes // 8, 1)
        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class _IGEVMultiBasicEncoder(nn.Module):
    def __init__(
        self,
        output_dim: List[List[int]],
        norm_fn: str = "batch",
        dropout: float = 0.0,
        downsample: int = 3,
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.downsample = downsample

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)
        else:
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1 + (downsample > 2), padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=1 + (downsample > 1))
        self.layer3 = self._make_layer(128, stride=1 + (downsample > 0))
        self.layer4 = self._make_layer(128, stride=2)
        self.layer5 = self._make_layer(128, stride=2)

        self.outputs04 = nn.ModuleList(
            [
                nn.Sequential(
                    _IGEVResidualBlock(128, 128, self.norm_fn, stride=1),
                    nn.Conv2d(128, dim[2], 3, padding=1),
                )
                for dim in output_dim
            ]
        )
        self.outputs08 = nn.ModuleList(
            [
                nn.Sequential(
                    _IGEVResidualBlock(128, 128, self.norm_fn, stride=1),
                    nn.Conv2d(128, dim[1], 3, padding=1),
                )
                for dim in output_dim
            ]
        )
        self.outputs16 = nn.ModuleList([nn.Conv2d(128, dim[0], 3, padding=1) for dim in output_dim])
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1 = _IGEVResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = _IGEVResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x: torch.Tensor, dual_inp: bool = False, num_layers: int = 3):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if dual_inp:
            v = x
            x = x[: x.shape[0] // 2]

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


class _IGEVFeature(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("mobilenetv2_100", pretrained=False, features_only=True)
        layers = [1, 2, 3, 5, 6]
        channels = [16, 24, 32, 96, 160]

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = getattr(model, "act1", None)

        self.block0 = nn.Sequential(*model.blocks[0 : layers[0]])
        self.block1 = nn.Sequential(*model.blocks[layers[0] : layers[1]])
        self.block2 = nn.Sequential(*model.blocks[layers[1] : layers[2]])
        self.block3 = nn.Sequential(*model.blocks[layers[2] : layers[3]])
        self.block4 = nn.Sequential(*model.blocks[layers[3] : layers[4]])

        self.deconv32_16 = _IGEVConv2xIN(channels[4], channels[3], deconv=True, concat=True)
        self.deconv16_8 = _IGEVConv2xIN(channels[3] * 2, channels[2], deconv=True, concat=True)
        self.deconv8_4 = _IGEVConv2xIN(channels[2] * 2, channels[1], deconv=True, concat=True)
        self.conv4 = _IGEVBasicConvIN(channels[1] * 2, channels[1] * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.bn1(self.conv_stem(x))
        if self.act1 is not None:
            x = self.act1(x)
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]


class _IGEVDispHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class _IGEVConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=padding)

    def forward(
        self,
        h: torch.Tensor,
        cz: torch.Tensor,
        cr: torch.Tensor,
        cq: torch.Tensor,
        *x_list: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        return (1 - z) * h + z * q


class _IGEVBasicMotionEncoder(nn.Module):
    def __init__(self, args: IGEVStereoConfig):
        super().__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) * 9
        self.convc1 = nn.Conv2d(cor_planes, 64, 1, padding=0)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(128, 127, 3, padding=1)

    def forward(self, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_feat = F.relu(self.convd1(disp))
        disp_feat = F.relu(self.convd2(disp_feat))
        out = F.relu(self.conv(torch.cat([cor, disp_feat], dim=1)))
        return torch.cat([out, disp], dim=1)


def _igev_pool2x(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def _igev_interp(x: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    original_dtype = x.dtype
    if torch.cuda.is_available() and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        ctx = torch.amp.autocast("cuda", enabled=False)
    elif torch.cuda.is_available():
        ctx = torch.cuda.amp.autocast(enabled=False)
    else:
        ctx = nullcontext()
    with ctx:
        output = F.interpolate(x.float(), dest.shape[2:], mode="bilinear", align_corners=True)
    return output.to(original_dtype) if original_dtype != torch.float32 else output


class _IGEVBasicMultiUpdateBlock(nn.Module):
    def __init__(self, args: IGEVStereoConfig, hidden_dims: List[int]):
        super().__init__()
        self.args = args
        self.encoder = _IGEVBasicMotionEncoder(args)
        self.gru04 = _IGEVConvGRU(hidden_dims[2], 128 + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = _IGEVConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = _IGEVConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = _IGEVDispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.mask_feat_4 = nn.Sequential(nn.Conv2d(hidden_dims[2], 32, 3, padding=1), nn.ReLU(inplace=True))

    def forward(
        self,
        net: List[torch.Tensor],
        inp: List[List[torch.Tensor]],
        corr: Optional[torch.Tensor] = None,
        disp: Optional[torch.Tensor] = None,
        iter04: bool = True,
        iter08: bool = True,
        iter16: bool = True,
        update: bool = True,
    ):
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), _igev_pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), _igev_pool2x(net[0]), _igev_interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), _igev_pool2x(net[0]))
        if iter04:
            motion_features = self.encoder(disp, corr)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features, _igev_interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class _IGEVCombinedGeoEncodingVolume:
    def __init__(
        self,
        init_fmap1: torch.Tensor,
        init_fmap2: torch.Tensor,
        geo_volume: torch.Tensor,
        num_levels: int = 2,
        radius: int = 4,
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid: List[torch.Tensor] = []
        self.init_corr_pyramid: List[torch.Tensor] = []

        init_corr = self.corr(init_fmap1, init_fmap2)

        batch, height, width, _, width2 = init_corr.shape
        geo_batch, channels, disp_levels, geo_height, geo_width = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(geo_batch * geo_height * geo_width, channels, 1, disp_levels)
        init_corr = init_corr.reshape(batch * height * width, 1, 1, width2)

        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)

        for _ in range(self.num_levels - 1):
            geo_volume = F.avg_pool2d(geo_volume, [1, 2], stride=[1, 2])
            self.geo_volume_pyramid.append(geo_volume)

        for _ in range(self.num_levels - 1):
            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        radius = self.radius
        batch, _, height, width = disp.shape
        out_pyramid = []

        for level in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[level]
            dx = torch.linspace(-radius, radius, 2 * radius + 1, device=disp.device, dtype=disp.dtype)
            dx = dx.view(1, 1, 2 * radius + 1, 1)
            x0 = dx + disp.reshape(batch * height * width, 1, 1, 1) / (2**level)
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0, y0], dim=-1)
            sampled_geo = _igev_bilinear_sampler(geo_volume, disp_lvl)
            sampled_geo = sampled_geo.view(batch, height, width, -1)

            init_corr = self.init_corr_pyramid[level]
            init_x0 = (
                coords.reshape(batch * height * width, 1, 1, 1) / (2**level)
                - disp.reshape(batch * height * width, 1, 1, 1) / (2**level)
                + dx
            )
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            sampled_corr = _igev_bilinear_sampler(init_corr, init_coords_lvl)
            sampled_corr = sampled_corr.view(batch, height, width, -1)

            out_pyramid.extend([sampled_geo, sampled_corr])

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dims, height, width1 = fmap1.shape
        width2 = fmap2.shape[-1]
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        return corr.reshape(batch, height, width1, 1, width2).contiguous()


class _IGEVHourglass(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            _IGEVBasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVBasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            _IGEVBasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVBasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            _IGEVBasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVBasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = _IGEVBasicConv(
            in_channels * 6,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv2_up = _IGEVBasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv1_up = _IGEVBasicConv(
            in_channels * 2,
            8,
            deconv=True,
            is_3d=True,
            bn=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.agg_0 = nn.Sequential(
            _IGEVBasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            _IGEVBasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            _IGEVBasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            _IGEVBasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            _IGEVBasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            _IGEVBasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_att_8 = _IGEVFeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = _IGEVFeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = _IGEVFeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = _IGEVFeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = _IGEVFeatureAtt(in_channels * 2, 64)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        conv1 = self.feature_att_8(self.conv1(x), features[1])
        conv2 = self.feature_att_16(self.conv2(conv1), features[2])
        conv3 = self.feature_att_32(self.conv3(conv2), features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = self.feature_att_up_16(self.agg_0(torch.cat((conv3_up, conv2), dim=1)), features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = self.feature_att_up_8(self.agg_1(torch.cat((conv2_up, conv1), dim=1)), features[1])
        return self.conv1_up(conv1)


class _IGEVStereoNet(nn.Module):
    def __init__(self, config: IGEVStereoConfig):
        super().__init__()
        self.args = config
        context_dims = config.hidden_dims

        self.cnet = _IGEVMultiBasicEncoder(
            output_dim=[config.hidden_dims, context_dims],
            norm_fn="batch",
            downsample=config.n_downsample,
        )
        self.update_block = _IGEVBasicMultiUpdateBlock(config, hidden_dims=config.hidden_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[idx], config.hidden_dims[idx] * 3, 3, padding=1) for idx in range(config.n_gru_layers)]
        )

        self.feature = _IGEVFeature()

        self.stem_2 = nn.Sequential(
            _IGEVBasicConvIN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            _IGEVBasicConvIN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = _IGEVConv2xIN(24, 32, True)
        self.spx_4 = nn.Sequential(
            _IGEVBasicConvIN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24),
            nn.ReLU(),
        )

        self.spx_2_gru = _IGEVConv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(64, 9, kernel_size=4, stride=2, padding=1))

        self.conv = _IGEVBasicConvIN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = _IGEVBasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = _IGEVFeatureAtt(8, 96)
        self.cost_agg = _IGEVHourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        amp_dtype = getattr(torch, self.args.precision_dtype, torch.float16)
        with _igev_autocast(self.args.mixed_precision, amp_dtype):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = F.softmax(self.spx_gru(xspx), dim=1)
            up_disp = _igev_context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
        return up_disp

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        test_mode: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        amp_dtype = getattr(torch, self.args.precision_dtype, torch.float16)

        with _igev_autocast(self.args.mixed_precision, amp_dtype):
            features_left = self.feature(image1)
            features_right = self.feature(image2)
            stem_2x = self.stem_2(image1)
            stem_4x = self.stem_4(stem_2x)
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)

            features_left[0] = torch.cat((features_left[0], stem_4x), dim=1)
            features_right[0] = torch.cat((features_right[0], stem_4y), dim=1)

            match_left = self.desc(self.conv(features_left[0]))
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = _igev_build_gwc_volume(match_left, match_right, self.args.max_disp // 4, 8)
            gwc_volume = self.corr_feature_att(self.corr_stem(gwc_volume), features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = _igev_disparity_regression(prob, self.args.max_disp // 4)

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = F.softmax(self.spx(xspx), dim=1)

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(inp).split(split_size=conv.out_channels // 3, dim=1)) for inp, conv in zip(inp_list, self.context_zqr_convs)]

        geo_fn = _IGEVCombinedGeoEncodingVolume(
            match_left.float(),
            match_right.float(),
            geo_encoding_volume.float(),
            radius=self.args.corr_radius,
            num_levels=self.args.corr_levels,
        )

        batch, _, height, width = match_left.shape
        coords = torch.arange(width, device=match_left.device, dtype=match_left.dtype).reshape(1, 1, width, 1)
        coords = coords.repeat(batch, height, 1, 1)

        disp = init_disp
        disp_preds = []

        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with _igev_autocast(self.args.mixed_precision, amp_dtype):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list,
                    inp_list,
                    geo_feat,
                    disp,
                    iter16=self.args.n_gru_layers == 3,
                    iter08=self.args.n_gru_layers >= 2,
                )

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue
            disp_preds.append(self.upsample_disp(disp, mask_feat_4, stem_2x))

        if test_mode:
            return disp_preds[-1]

        init_disp = _igev_context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)
        return init_disp, disp_preds


def _extract_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    state_dict = raw
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model", "net"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state dict, got {type(state_dict)!r}")
    return state_dict


def _resolve_local_variant(variant: str) -> str:
    if variant in _IGEV_VARIANT_MAP:
        return _IGEV_VARIANT_MAP[variant]
    return variant


def _default_checkpoint_candidates(config: IGEVStereoConfig) -> List[str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    env_path = os.environ.get("IGEV_STEREO_CHECKPOINT")
    checkpoint_relpath = config.checkpoint_filename
    checkpoint_name = os.path.basename(checkpoint_relpath)
    candidates = [
        env_path,
        os.path.join(repo_root, "third-party", "IGEV", "IGEV-Stereo", "pretrained_models", checkpoint_relpath),
        os.path.join(repo_root, "pretrained_models", checkpoint_relpath),
        os.path.expanduser(os.path.join("~", ".cache", "stereo_matching", "igev_stereo", checkpoint_relpath)),
        os.path.join(repo_root, "third-party", "IGEV", "IGEV-Stereo", "pretrained_models", config.variant, checkpoint_name),
        os.path.join(repo_root, "pretrained_models", config.variant, checkpoint_name),
        os.path.expanduser(os.path.join("~", ".cache", "stereo_matching", "igev_stereo", config.variant, checkpoint_name)),
    ]
    return [os.path.abspath(os.path.expanduser(path)) for path in candidates if path]


class IGEVStereoModel(BaseStereoModel):
    config_class = IGEVStereoConfig

    def __init__(self, config: IGEVStereoConfig):
        super().__init__(config)
        self.net = _IGEVStereoNet(config)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)

        left_255 = (left * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        padder = _IGEVInputPadder(left_255.shape, divis_by=32)
        left_pad, right_pad = padder.pad(left_255, right_255)

        preds = self.net(left_pad, right_pad, iters=self.config.num_iters, test_mode=not self.training)
        if self.training:
            init_disp, disp_preds = preds
            full_sequence = [init_disp] + disp_preds
            return [padder.unpad(pred.float()).squeeze(1) for pred in full_sequence]
        return padder.unpad(preds.float()).squeeze(1)

    def _backbone_module(self) -> Optional[nn.Module]:
        return self.net.feature

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "IGEVStereoModel":
        if model_id in _IGEV_VARIANT_MAP:
            config = IGEVStereoConfig.from_variant(model_id)
            checkpoint_path = next((path for path in _default_checkpoint_candidates(config) if os.path.isfile(path)), None)
            if checkpoint_path is None:
                try:
                    from huggingface_hub import hf_hub_download

                    checkpoint_path = hf_hub_download(
                        repo_id=config.hub_repo_id,
                        filename=config.checkpoint_filename,
                    )
                except Exception as exc:
                    searched = "\n".join(f"  - {path}" for path in _default_checkpoint_candidates(config))
                    raise RuntimeError(
                        "Could not resolve a checkpoint for 'igev-stereo'. "
                        "Tried Hugging Face Hub and the usual local checkpoint locations.\n"
                        f"HF repo: {config.hub_repo_id}\n"
                        f"HF filename: {config.checkpoint_filename}\n"
                        f"Local paths:\n{searched}\n"
                        "For pipeline() with a local checkpoint, use model='/path/to/sceneflow.pth' "
                        "and variant='igev-stereo'.\n"
                        f"HF error: {exc}"
                    ) from exc
        elif os.path.isfile(model_id):
            config = IGEVStereoConfig(variant=_resolve_local_variant(kwargs.pop("variant", "sceneflow")))
            checkpoint_path = model_id
        else:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Use one of {list(_IGEV_VARIANT_MAP.keys())} or a local .pth file path."
            )

        try:
            raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            logger.warning("weights_only=True failed for IGEV-Stereo; retrying with weights_only=False.")
            raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = _extract_state_dict(raw)
        model = cls(config)

        remapped_state_dict = {}
        for key, value in state_dict.items():
            stripped = key[len("module.") :] if key.startswith("module.") else key
            remapped_key = stripped if stripped.startswith("net.") else f"net.{stripped}"
            remapped_state_dict[remapped_key] = value

        try:
            model.load_state_dict(remapped_state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning("strict=True load failed for IGEV-Stereo: %s\nRetrying with strict=False.", exc)
            incompatible = model.load_state_dict(remapped_state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning("Missing keys: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys: %s", incompatible.unexpected_keys)

        logger.info("Loaded IGEVStereoModel (%s) from '%s'", config.variant, checkpoint_path)
        return model
