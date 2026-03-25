"""
IGEV++ model — adapted from third-party/IGEV-plusplus.

This file vendors the stereo-only IGEV++ path into a single module so the
library does not depend on runtime sys.path manipulation or the original
project layout.
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
from .configuration_igev_plusplus import IGEVPlusPlusConfig, _IGEV_PP_VARIANT_MAP

logger = logging.getLogger(__name__)


def _igevpp_autocast(enabled: bool, dtype: torch.dtype):
    if dtype not in (torch.float16, torch.bfloat16):
        return nullcontext()
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


class _IGEVPPInputPadder:
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


def _igevpp_bilinear_sampler(
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


class _IGEVPPBasicConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        IN: bool = True,
        relu: bool = True,
        **kwargs: Any,
    ):
        super().__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_in:
            x = self.IN(x)
        if self.relu:
            x = F.leaky_relu(x, negative_slope=0.01, inplace=True)
        return x


class _IGEVPPConv2x(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv: bool = False,
        is_3d: bool = False,
        concat: bool = True,
        keep_concat: bool = True,
        IN: bool = True,
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
            self.conv1 = _IGEVPPBasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            self.conv1 = _IGEVPPBasicConv(
                in_channels,
                out_channels,
                deconv,
                is_3d,
                IN=True,
                relu=True,
                kernel_size=kernel,
                stride=2,
                padding=1,
            )

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = _IGEVPPBasicConv(
                out_channels * 2,
                out_channels * mul,
                False,
                is_3d,
                IN,
                relu,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv2 = _IGEVPPBasicConv(
                out_channels,
                out_channels,
                False,
                is_3d,
                IN,
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


def _igevpp_groupwise_correlation(
    fea1: torch.Tensor,
    fea2: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    batch, channels, height, width = fea1.shape
    channels_per_group = channels // num_groups
    cost = (fea1 * fea2).view(batch, num_groups, channels_per_group, height, width).mean(dim=2)
    return cost


def _igevpp_build_gwc_volume(
    refimg_fea: torch.Tensor,
    targetimg_fea: torch.Tensor,
    maxdisp: int,
    num_groups: int,
) -> torch.Tensor:
    batch, _, height, width = refimg_fea.shape
    volume = refimg_fea.new_zeros(batch, num_groups, maxdisp, height, width)
    for idx in range(maxdisp):
        if idx > 0:
            volume[:, :, idx, :, idx:] = _igevpp_groupwise_correlation(
                refimg_fea[:, :, :, idx:],
                targetimg_fea[:, :, :, :-idx],
                num_groups,
            )
        else:
            volume[:, :, idx, :, :] = _igevpp_groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    return volume.contiguous()


def _igevpp_disparity_regression(prob: torch.Tensor, maxdisp: int, interval: int) -> torch.Tensor:
    disp_values = torch.arange(0, maxdisp, interval, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp // interval, 1, 1)
    return torch.sum(prob * disp_values, dim=1, keepdim=True)


class _IGEVPPFeatureAtt(nn.Module):
    def __init__(self, cv_chan: int, feat_chan: int):
        super().__init__()
        self.feat_att = nn.Sequential(
            _IGEVPPBasicConv(feat_chan, feat_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
        )

    def forward(self, cv: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        feat_att = self.feat_att(feat).unsqueeze(2)
        return torch.sigmoid(feat_att) * cv


def _igevpp_context_upsample(disp_low: torch.Tensor, up_weights: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = disp_low.shape
    disp_unfold = F.unfold(disp_low.reshape(batch, channels, height, width), 3, 1, 1)
    disp_unfold = disp_unfold.reshape(batch, -1, height, width)
    disp_unfold = F.interpolate(disp_unfold, (height * 4, width * 4), mode="nearest")
    disp_unfold = disp_unfold.reshape(batch, 9, height * 4, width * 4)
    return (disp_unfold * up_weights).sum(dim=1, keepdim=True)


class _IGEVPPResidualBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, norm_fn: str = "group", stride: int = 1):
        super().__init__()

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
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class _IGEVPPMultiBasicEncoder(nn.Module):
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
                    _IGEVPPResidualBlock(128, 128, self.norm_fn, stride=1),
                    nn.Conv2d(128, dim[2], 3, padding=1),
                )
                for dim in output_dim
            ]
        )
        self.outputs08 = nn.ModuleList(
            [
                nn.Sequential(
                    _IGEVPPResidualBlock(128, 128, self.norm_fn, stride=1),
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
        layer1 = _IGEVPPResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = _IGEVPPResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x: torch.Tensor, dual_inp: bool = False, num_layers: int = 3):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
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


class _IGEVPPFeature(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("mobilenetv2_100", pretrained=False, features_only=True)
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = getattr(model, "act1", None)

        self.block0 = nn.Sequential(*model.blocks[0 : layers[0]])
        self.block1 = nn.Sequential(*model.blocks[layers[0] : layers[1]])
        self.block2 = nn.Sequential(*model.blocks[layers[1] : layers[2]])
        self.block3 = nn.Sequential(*model.blocks[layers[2] : layers[3]])
        self.block4 = nn.Sequential(*model.blocks[layers[3] : layers[4]])

        self.deconv32_16 = _IGEVPPConv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = _IGEVPPConv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = _IGEVPPConv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = _IGEVPPBasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)

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


class _IGEVPPDispHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x)))


class _IGEVPPConvGRU(nn.Module):
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


class _IGEVPPGeoEncoder(nn.Module):
    def __init__(self, geo_planes: int):
        super().__init__()
        self.convg1 = nn.Conv2d(geo_planes, 128, 1, padding=0)
        self.convg2 = nn.Conv2d(128, 96, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        return self.convg2(self.relu(self.convg1(geo)))


class _IGEVPPBasicDispEncoder(nn.Module):
    def __init__(self, args: IGEVPlusPlusConfig):
        super().__init__()
        geo_planes = (2 * args.corr_radius + 1) * 2 + 96
        self.convc1 = nn.Conv2d(geo_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 32, 7, padding=3)
        self.convd2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv = nn.Conv2d(96 + 32, 127, 3, padding=1)

    def forward(self, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_feat = F.relu(self.convd1(disp))
        disp_feat = F.relu(self.convd2(disp_feat))
        cor_disp = torch.cat([cor, disp_feat], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


def _igevpp_pool2x(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def _igevpp_interp(x: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
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


class _IGEVPPBasicMultiUpdateBlock(nn.Module):
    def __init__(self, args: IGEVPlusPlusConfig, hidden_dims: List[int]):
        super().__init__()
        self.args = args
        self.geo_encoder0 = _IGEVPPGeoEncoder(geo_planes=2 * (2 * args.corr_radius + 1) * 8)
        self.geo_encoder1 = _IGEVPPGeoEncoder(geo_planes=(2 * args.corr_radius + 1) * 8)
        self.geo_encoder2 = _IGEVPPGeoEncoder(geo_planes=(2 * args.corr_radius + 1) * 8)
        self.encoder = _IGEVPPBasicDispEncoder(args)
        encoder_output_dim = 128

        self.gru04 = _IGEVPPConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.n_gru_layers > 1))
        self.gru08 = _IGEVPPConvGRU(hidden_dims[1], hidden_dims[0] * (args.n_gru_layers == 3) + hidden_dims[2])
        self.gru16 = _IGEVPPConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = _IGEVPPDispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        net: List[torch.Tensor],
        inp: List[List[torch.Tensor]],
        geo_feat0: Optional[torch.Tensor] = None,
        geo_feat1: Optional[torch.Tensor] = None,
        geo_feat2: Optional[torch.Tensor] = None,
        init_corr: Optional[torch.Tensor] = None,
        selective_weights: Optional[torch.Tensor] = None,
        disp: Optional[torch.Tensor] = None,
        iter04: bool = True,
        iter08: bool = True,
        iter16: bool = True,
        update: bool = True,
    ):
        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), _igevpp_pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), _igevpp_pool2x(net[0]), _igevpp_interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), _igevpp_pool2x(net[0]))
        if iter04:
            geo_feat0 = self.geo_encoder0(geo_feat0)
            geo_feat1 = self.geo_encoder1(geo_feat1)
            geo_feat2 = self.geo_encoder2(geo_feat2)
            geo_feat = (
                selective_weights[:, 0:1] * geo_feat0
                + selective_weights[:, 1:2] * geo_feat1
                + selective_weights[:, 2:3] * geo_feat2
            )
            geo_feat = torch.cat([geo_feat, init_corr], dim=1)
            disp_features = self.encoder(disp, geo_feat)
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), disp_features, _igevpp_interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), disp_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp


class _IGEVPPCombinedGeoEncodingVolume:
    def __init__(
        self,
        geo_volume0: torch.Tensor,
        geo_volume1: torch.Tensor,
        geo_volume2: torch.Tensor,
        init_fmap1: torch.Tensor,
        init_fmap2: torch.Tensor,
        radius: int = 4,
        num_levels: int = 2,
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.init_corr_pyramid: List[torch.Tensor] = []
        self.geo_volume0_pyramid: List[torch.Tensor] = []

        init_corr = self.corr(init_fmap1, init_fmap2)

        batch, height, width1, _, width2 = init_corr.shape
        _, channels, d0, geo_height, geo_width = geo_volume0.shape
        d1 = geo_volume1.shape[2]
        d2 = geo_volume2.shape[2]
        geo_volume0 = geo_volume0.permute(0, 3, 4, 1, 2).reshape(batch * geo_height * geo_width, channels, 1, d0)
        self.geo_volume1 = geo_volume1.permute(0, 3, 4, 1, 2).reshape(batch * geo_height * geo_width, channels, 1, d1)
        self.geo_volume2 = geo_volume2.permute(0, 3, 4, 1, 2).reshape(batch * geo_height * geo_width, channels, 1, d2)

        init_corr = init_corr.reshape(batch * height * width1, 1, 1, width2)
        self.init_corr_pyramid.append(init_corr)
        self.geo_volume0_pyramid.append(geo_volume0)
        for _ in range(self.num_levels - 1):
            geo_volume0 = F.avg_pool2d(geo_volume0, [1, 2], stride=[1, 2])
            self.geo_volume0_pyramid.append(geo_volume0)

            init_corr = F.avg_pool2d(init_corr, [1, 2], stride=[1, 2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp: torch.Tensor, coords: torch.Tensor):
        radius = self.radius
        batch, _, height, width = disp.shape
        init_corr_pyramid = []
        geo_feat0_pyramid = []
        dx = torch.linspace(-radius, radius, 2 * radius + 1, device=disp.device, dtype=disp.dtype)
        dx = dx.view(1, 1, 2 * radius + 1, 1)

        x1 = dx + disp.reshape(batch * height * width, 1, 1, 1) / 2
        y0 = torch.zeros_like(x1)
        disp_lvl1 = torch.cat([x1, y0], dim=-1)
        geo_feat1 = _igevpp_bilinear_sampler(self.geo_volume1, disp_lvl1)
        geo_feat1 = geo_feat1.view(batch, height, width, -1)

        x2 = dx + disp.reshape(batch * height * width, 1, 1, 1) / 4
        y0 = torch.zeros_like(x2)
        disp_lvl2 = torch.cat([x2, y0], dim=-1)
        geo_feat2 = _igevpp_bilinear_sampler(self.geo_volume2, disp_lvl2)
        geo_feat2 = geo_feat2.view(batch, height, width, -1)

        for level in range(self.num_levels):
            geo_volume0 = self.geo_volume0_pyramid[level]
            x0 = dx + disp.reshape(batch * height * width, 1, 1, 1) / (2**level)
            y0 = torch.zeros_like(x0)
            disp_lvl0 = torch.cat([x0, y0], dim=-1)
            geo_feat0 = _igevpp_bilinear_sampler(geo_volume0, disp_lvl0)
            geo_feat0 = geo_feat0.view(batch, height, width, -1)
            geo_feat0_pyramid.append(geo_feat0)

            init_corr = self.init_corr_pyramid[level]
            init_x0 = (
                coords.reshape(batch * height * width, 1, 1, 1) / (2**level)
                - disp.reshape(batch * height * width, 1, 1, 1) / (2**level)
                + dx
            )
            init_coords_lvl = torch.cat([init_x0, y0], dim=-1)
            init_corr = _igevpp_bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(batch, height, width, -1)
            init_corr_pyramid.append(init_corr)

        init_corr = torch.cat(init_corr_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
        geo_feat0 = torch.cat(geo_feat0_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
        geo_feat1 = geo_feat1.permute(0, 3, 1, 2).contiguous().float()
        geo_feat2 = geo_feat2.permute(0, 3, 1, 2).contiguous().float()
        return geo_feat0, geo_feat1, geo_feat2, init_corr

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dims, height, width1 = fmap1.shape
        width2 = fmap2.shape[-1]
        corr = torch.einsum("aijk,aijh->ajkh", fmap1, fmap2)
        return corr.reshape(batch, height, width1, 1, width2).contiguous()


class _IGEVPPHourglass(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.conv0 = _IGEVPPBasicConv(in_channels, in_channels, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(
            _IGEVPPBasicConv(in_channels, in_channels * 2, is_3d=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVPPBasicConv(in_channels * 2, in_channels * 2, is_3d=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv2 = nn.Sequential(
            _IGEVPPBasicConv(in_channels * 2, in_channels * 4, is_3d=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVPPBasicConv(in_channels * 4, in_channels * 4, is_3d=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )
        self.conv3 = nn.Sequential(
            _IGEVPPBasicConv(in_channels * 4, in_channels * 8, is_3d=True, relu=True, kernel_size=3, padding=1, stride=2, dilation=1),
            _IGEVPPBasicConv(in_channels * 8, in_channels * 8, is_3d=True, relu=True, kernel_size=3, padding=1, stride=1, dilation=1),
        )

        self.conv3_up = _IGEVPPBasicConv(
            in_channels * 8,
            in_channels * 4,
            deconv=True,
            is_3d=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv2_up = _IGEVPPBasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv1_up = _IGEVPPBasicConv(
            in_channels * 2,
            in_channels,
            deconv=True,
            is_3d=True,
            IN=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.agg_0 = nn.Sequential(
            _IGEVPPBasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            _IGEVPPBasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            _IGEVPPBasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
        )
        self.agg_1 = nn.Sequential(
            _IGEVPPBasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            _IGEVPPBasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            _IGEVPPBasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
        )

        self.feature_att_4 = _IGEVPPFeatureAtt(in_channels, 96)
        self.feature_att_8 = _IGEVPPFeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = _IGEVPPFeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = _IGEVPPFeatureAtt(in_channels * 8, 160)
        self.feature_att_up_16 = _IGEVPPFeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = _IGEVPPFeatureAtt(in_channels * 2, 64)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        conv0 = self.conv0(x)
        conv0 = self.feature_att_4(conv0, features[0])

        conv1 = self.conv1(conv0)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        return self.conv1_up(conv1)


class _IGEVPlusPlusNet(nn.Module):
    def __init__(self, config: IGEVPlusPlusConfig):
        super().__init__()
        self.args = config

        context_dims = config.hidden_dims

        self.cnet = _IGEVPPMultiBasicEncoder(
            output_dim=[config.hidden_dims, context_dims],
            norm_fn="batch",
            downsample=config.n_downsample,
        )
        self.update_block = _IGEVPPBasicMultiUpdateBlock(self.args, hidden_dims=config.hidden_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], config.hidden_dims[i] * 3, 3, padding=1) for i in range(config.n_gru_layers)]
        )
        self.feature = _IGEVPPFeature()

        self.stem_2 = nn.Sequential(
            _IGEVPPBasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            _IGEVPPBasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48),
            nn.ReLU(),
        )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))
        self.spx_2 = _IGEVPPConv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            _IGEVPPBasicConv(96, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )

        self.spx_2_gru = _IGEVPPConv2x(64, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1))

        self.conv = _IGEVPPBasicConv(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.patch0 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False)
        self.patch1 = nn.Conv3d(8, 8, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=False)
        self.cost_agg0 = _IGEVPPHourglass(8)
        self.cost_agg1 = _IGEVPPHourglass(8)
        self.cost_agg2 = _IGEVPPHourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.disp_conv = nn.Sequential(
            _IGEVPPBasicConv(3, 64, kernel_size=1, stride=1, padding=0),
            _IGEVPPBasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.selective_conv = nn.Sequential(
            _IGEVPPBasicConv(96 + 64, 128, kernel_size=1, stride=1, padding=0),
            _IGEVPPBasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 3, 3, 1, 1, bias=False),
        )

    def freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def upsample_disp(self, disp: torch.Tensor, mask_feat_4: torch.Tensor, stem_2x: torch.Tensor) -> torch.Tensor:
        amp_dtype = getattr(torch, self.args.precision_dtype, torch.float16)
        with _igevpp_autocast(self.args.mixed_precision, amp_dtype):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, dim=1)
            up_disp = _igevpp_context_upsample(disp * 4.0, spx_pred)
        return up_disp

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = 12,
        test_mode: bool = False,
    ) -> Union[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        amp_dtype = getattr(torch, self.args.precision_dtype, torch.float16)

        with _igevpp_autocast(self.args.mixed_precision, amp_dtype):
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
            all_disp_volume = _igevpp_build_gwc_volume(match_left, match_right, self.args.max_disp // 4, 8)

            disp_volume0 = all_disp_volume[:, :, : self.args.s_disp_range]
            disp_volume1 = self.patch0(all_disp_volume[:, :, : self.args.m_disp_range])
            disp_volume2 = self.patch1(all_disp_volume)

            geo_encoding_volume0 = self.cost_agg0(disp_volume0, features_left)
            geo_encoding_volume1 = self.cost_agg1(disp_volume1, features_left)
            geo_encoding_volume2 = self.cost_agg2(disp_volume2, features_left)

            cost_volume0 = self.classifier(geo_encoding_volume0)
            prob_volume0 = F.softmax(cost_volume0.squeeze(1), dim=1)
            agg_disp0 = _igevpp_disparity_regression(prob_volume0, self.args.s_disp_range, self.args.s_disp_interval)

            cost_volume1 = self.classifier(geo_encoding_volume1)
            prob_volume1 = F.softmax(cost_volume1.squeeze(1), dim=1)
            agg_disp1 = _igevpp_disparity_regression(prob_volume1, self.args.m_disp_range, self.args.m_disp_interval)

            cost_volume2 = self.classifier(geo_encoding_volume2)
            prob_volume2 = F.softmax(cost_volume2.squeeze(1), dim=1)
            agg_disp2 = _igevpp_disparity_regression(prob_volume2, self.args.l_disp_range, self.args.l_disp_interval)

            disp_feature = self.disp_conv(torch.cat([agg_disp0, agg_disp1, agg_disp2], dim=1))
            selective_weights = torch.sigmoid(self.selective_conv(torch.cat([features_left[0], disp_feature], dim=1)))
            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [
                list(conv(inp).split(split_size=conv.out_channels // 3, dim=1))
                for inp, conv in zip(inp_list, self.context_zqr_convs)
            ]

        geo_fn = _IGEVPPCombinedGeoEncodingVolume(
            geo_encoding_volume0.float(),
            geo_encoding_volume1.float(),
            geo_encoding_volume2.float(),
            match_left.float(),
            match_right.float(),
            radius=self.args.corr_radius,
        )
        batch, _, height, width = match_left.shape
        coords = torch.arange(width, device=match_left.device, dtype=match_left.dtype).reshape(1, 1, width, 1)
        coords = coords.repeat(batch, height, 1, 1)
        disp = agg_disp0
        iter_preds = []

        for itr in range(iters):
            disp = disp.detach()
            geo_feat0, geo_feat1, geo_feat2, init_corr = geo_fn(disp, coords)
            with _igevpp_autocast(self.args.mixed_precision, amp_dtype):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list,
                    inp_list,
                    geo_feat0,
                    geo_feat1,
                    geo_feat2,
                    init_corr,
                    selective_weights,
                    disp,
                    iter16=self.args.n_gru_layers == 3,
                    iter08=self.args.n_gru_layers >= 2,
                )

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            iter_preds.append(disp_up)

        if test_mode:
            return disp_up

        with _igevpp_autocast(self.args.mixed_precision, amp_dtype):
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, dim=1)
        agg_disp0 = _igevpp_context_upsample(agg_disp0 * 4.0, spx_pred.float())
        agg_disp1 = _igevpp_context_upsample(agg_disp1 * 4.0, spx_pred.float())
        agg_disp2 = _igevpp_context_upsample(agg_disp2 * 4.0, spx_pred.float())
        return [agg_disp0, agg_disp1, agg_disp2], iter_preds


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
    if variant in _IGEV_PP_VARIANT_MAP:
        return _IGEV_PP_VARIANT_MAP[variant]
    return variant


def _default_checkpoint_candidates(config: IGEVPlusPlusConfig) -> List[str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    env_path = os.environ.get("IGEV_PLUSPLUS_CHECKPOINT")
    candidates = [
        env_path,
        os.path.join(
            repo_root,
            "third-party",
            "IGEV-plusplus",
            "pretrained_models",
            "igev_plusplus",
            config.checkpoint_filename,
        ),
        os.path.join(repo_root, "pretrained_models", "igev_plusplus", config.checkpoint_filename),
        os.path.expanduser(os.path.join("~", ".cache", "stereo_matching", "igev_plusplus", config.checkpoint_filename)),
    ]
    return [os.path.abspath(os.path.expanduser(path)) for path in candidates if path]


class IGEVPlusPlusModel(BaseStereoModel):
    config_class = IGEVPlusPlusConfig

    def __init__(self, config: IGEVPlusPlusConfig):
        super().__init__(config)
        self.net = _IGEVPlusPlusNet(config)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)

        left_255 = (left * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        padder = _IGEVPPInputPadder(left_255.shape, divis_by=32)
        left_pad, right_pad = padder.pad(left_255, right_255)

        preds = self.net(left_pad, right_pad, iters=self.config.num_iters, test_mode=not self.training)
        if self.training:
            agg_preds, iter_preds = preds
            full_sequence = [*agg_preds, *iter_preds]
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
    ) -> "IGEVPlusPlusModel":
        if model_id in _IGEV_PP_VARIANT_MAP:
            config = IGEVPlusPlusConfig.from_variant(model_id)
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
                        "Could not resolve a checkpoint for 'igev-plusplus'. "
                        "Tried Hugging Face Hub and the usual local checkpoint locations.\n"
                        f"HF repo: {config.hub_repo_id}\n"
                        f"HF filename: {config.checkpoint_filename}\n"
                        f"Local paths:\n{searched}\n"
                        "For pipeline() with a local checkpoint, use model='/path/to/sceneflow.pth' "
                        "and variant='igev-plusplus'.\n"
                        f"HF error: {exc}"
                    ) from exc
        elif os.path.isfile(model_id):
            config = IGEVPlusPlusConfig(variant=_resolve_local_variant(kwargs.pop("variant", "sceneflow")))
            checkpoint_path = model_id
        else:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Use one of {list(_IGEV_PP_VARIANT_MAP.keys())} or a local .pth file path."
            )

        try:
            raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            logger.warning("weights_only=True failed for IGEV++; retrying with weights_only=False.")
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
            logger.warning("strict=True load failed for IGEV++: %s\nRetrying with strict=False.", exc)
            incompatible = model.load_state_dict(remapped_state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning("Missing keys: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.warning("Unexpected keys: %s", incompatible.unexpected_keys)

        logger.info("Loaded IGEVPlusPlusModel (%s) from '%s'", config.variant, checkpoint_path)
        return model
