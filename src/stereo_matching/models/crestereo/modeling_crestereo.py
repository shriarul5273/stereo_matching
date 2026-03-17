"""
CREStereoModel — CREStereo stereo matching model.

All CREStereo core components are vendored in this single file in dependency order:
  1. bilinear_grid_sample, bilinear_sampler, coords_grid, manual_pad  (utils/utils.py)
  2. ResidualBlock, BasicEncoder                                       (extractor.py)
  3. LinearAttention, FullAttention                                    (attention/linear_attention.py)
  4. LoFTREncoderLayer, LocalFeatureTransformer                        (attention/transformer.py)
  5. PositionEncodingSine                                              (attention/position_encoding.py)
  6. AGCL                                                              (corr.py)
  7. FlowHead, SepConvGRU, BasicMotionEncoder, BasicUpdateBlock        (update.py)
  8. _CREStereoNet  (original CREStereo class, renamed)
  9. CREStereoModel (public wrapper — BaseStereoModel subclass)

Original PyTorch port: https://github.com/ibaiGorordo/CREStereo-Pytorch
"""

import copy
import logging
import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modeling_utils import BaseStereoModel
from .configuration_crestereo import CREStereoConfig, _CRESTEREO_VARIANT_MAP

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
# Section 1: Utilities  (from nets/utils/utils.py)
# ===========================================================================

def bilinear_grid_sample(im, grid, align_corners=False):
    """Bilinear grid sample — manual implementation (avoids F.grid_sample).

    Args:
        im   (torch.Tensor): (N, C, H, W)
        grid (torch.Tensor): (N, Hg, Wg, 2)  — normalised coords in [-1, 1]
        align_corners (bool)

    Returns:
        torch.Tensor: (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = torch.nn.functional.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0, device=im.device), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0, device=im.device), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=im.device), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0, device=im.device), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0, device=im.device), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=im.device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Wrapper for bilinear_grid_sample; uses pixel coordinates."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = bilinear_grid_sample(img, grid, align_corners=True)

    if mask:
        mask_out = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask_out.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing='ij',
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def manual_pad(x, pady, padx):
    pad = (padx, padx, pady, pady)
    return F.pad(x.clone().detach(), pad, "replicate")


# ===========================================================================
# Section 2: Feature Extractor  (from nets/extractor.py)
# ===========================================================================

class _CRE_ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes, affine=False)
            self.norm2 = nn.InstanceNorm2d(planes, affine=False)
            self.norm3 = nn.InstanceNorm2d(planes, affine=False)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
        )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        x = self.downsample(x)
        return self.relu(x + y)


class _CRE_BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super().__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64, affine=False)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = _CRE_ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = _CRE_ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, x.shape[0] // 2, dim=0)

        return x


# ===========================================================================
# Section 3: Linear Attention  (from nets/attention/linear_attention.py)
# ===========================================================================

def _elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class _LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = _elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length
        KV = torch.einsum("nshd,nshv->nhdv", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class _FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf')
            )

        softmax_temp = 1.0 / queries.size(3) ** 0.5
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)
        return queried_values.contiguous()


# ===========================================================================
# Section 4: Transformer  (from nets/attention/transformer.py)
# ===========================================================================

class _LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, attention='linear'):
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = _LinearAttention() if attention == 'linear' else _FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        bs = x.size(0)
        query, key, value = x, source, source

        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)
        key   = self.k_proj(key).view(bs, -1, self.nhead, self.dim)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)

        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class _LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, layer_names, attention):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        encoder_layer = _LoFTREncoderLayer(d_model, nhead, attention)
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        assert self.d_model == feat0.size(2)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError(f"Unknown layer name: {name!r}")

        return feat0, feat1


# ===========================================================================
# Section 5: Position Encoding  (from nets/attention/position_encoding.py)
# ===========================================================================

class _PositionEncodingSine(nn.Module):
    """Sinusoidal 2-D position encoding."""

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=False):
        super().__init__()
        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float()
                * (-math.log(10000.0) / (d_model // 2))
            )
        else:  # original (buggy) impl kept for backward compat
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float()
                * (-math.log(10000.0) / d_model // 2)
            )
        div_term = div_term[:, None, None]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # (1, C, H, W)

    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)].to(x.device)


# ===========================================================================
# Section 6: Adaptive Group Correlation Layer  (from nets/corr.py)
# ===========================================================================

class AGCL:
    """Adaptive Group Correlation Layer (AGCL)."""

    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.att = att
        self.coords = coords_grid(
            fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device
        )

    def __call__(self, flow, extra_offset, small_patch=False, iter_mode=False):
        if iter_mode:
            corr = self.corr_iter(self.fmap1, self.fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(
                self.fmap1, self.fmap2, flow, extra_offset, small_patch
            )
        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):
        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = manual_pad(right_feature, pady, padx)

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h:h + H, w:w + W]
                assert right_crop.shape == left_feature.shape
                corr = torch.mean(left_feature * right_crop, dim=1, keepdim=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)
        return corr_final

    def corr_iter(self, left_feature, right_feature, flow, small_patch):
        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list  = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list  = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        lefts  = torch.split(left_feature,  left_feature.shape[1]  // 4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1] // 4, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(lefts[i], rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        return torch.cat(corrs, dim=1)

    def corr_att_offset(self, left_feature, right_feature, flow, extra_offset, small_patch):
        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature  = left_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
            right_feature = right_feature.permute(0, 2, 3, 1).reshape(N, H * W, C)
            left_feature, right_feature = self.att(left_feature, right_feature)
            left_feature, right_feature = [
                x.reshape(N, H, W, C).permute(0, 3, 1, 2)
                for x in [left_feature, right_feature]
            ]

        lefts  = torch.split(left_feature,  left_feature.shape[1]  // 4, dim=1)
        rights = torch.split(right_feature, right_feature.shape[1] // 4, dim=1)

        C = C // 4

        if small_patch:
            psize_list  = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list  = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        search_num = 9
        extra_offset = extra_offset.reshape(N, search_num, 2, H, W).permute(0, 1, 3, 4, 2)

        corrs = []
        for i in range(len(psize_list)):
            left_f, right_f = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex
            x_grid, y_grid = torch.meshgrid(
                torch.arange(-rx, rx + 1, dilatex, device=self.fmap1.device),
                torch.arange(-ry, ry + 1, dilatey, device=self.fmap1.device),
                indexing='xy',
            )

            offsets = torch.stack((x_grid, y_grid))
            offsets = offsets.reshape(2, -1).permute(1, 0)
            for d in sorted((0, 2, 3)):
                offsets = offsets.unsqueeze(d)
            offsets = offsets.repeat_interleave(N, dim=0)
            offsets = offsets + extra_offset

            coords = self.coords + flow           # [N, 2, H, W]
            coords = coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = coords.reshape(N, -1, W, 2)  # [N, search_num*H, W, 2]

            right_f = bilinear_sampler(right_f, coords)          # [N, C, search_num*H, W]
            right_f = right_f.reshape(N, C, -1, H, W)            # [N, C, search_num, H, W]
            left_f  = left_f.unsqueeze(2).repeat_interleave(right_f.shape[2], dim=2)

            corr = torch.mean(left_f * right_f, dim=1)
            corrs.append(corr)

        return torch.cat(corrs, dim=1)


# ===========================================================================
# Section 7: Update Block  (from nets/update.py)
# ===========================================================================

class _CRE_FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class _CRE_SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

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
        h = (1 - z) * h + z * q

        return h


class _CRE_BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv   = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class _CRE_BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super().__init__()
        self.encoder   = _CRE_BasicMotionEncoder(cor_planes)
        self.gru       = _CRE_SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = _CRE_FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size ** 2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat((inp, motion_features), dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


# ===========================================================================
# Section 8: _CREStereoNet  (original CREStereo class, renamed)
# ===========================================================================

class _CREStereoNet(nn.Module):
    """Internal CREStereo network (renamed from CREStereo).

    Expects images in [0, 255] range (normalised internally by this forward pass).
    Outputs positive disparity (sign-corrected via ``-self.convex_upsample(...)``).
    """

    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super().__init__()

        self.max_flow        = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode       = test_mode

        self.hidden_dim  = 128
        self.context_dim = 128
        self.dropout     = 0

        self.fnet = _CRE_BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.update_block = _CRE_BasicUpdateBlock(
            hidden_dim=self.hidden_dim, cor_planes=4 * 9, mask_size=4
        )

        self.self_att_fn = _LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = _LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

        self.search_num     = 9
        self.conv_offset_16 = nn.Conv2d(256, self.search_num * 2, kernel_size=3, stride=1, padding=1)
        self.conv_offset_8  = nn.Conv2d(256, self.search_num * 2, kernel_size=3, stride=1, padding=1)
        self.range_16 = 1
        self.range_8  = 1

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate=4):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        return torch.cat((_x, _y), dim=1).to(fmap.device)

    def forward(self, image1, image2, flow_init=None, iters=10, upsample=True, test_mode=False):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        with autocast(enabled=self.mixed_precision):
            # 1/4 -> 1/8 features
            fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

            offset_dw8 = self.conv_offset_8(fmap1_dw8)
            offset_dw8 = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0

            net, inp = torch.split(fmap1, [hdim, hdim], dim=1)
            net = torch.tanh(net)
            inp = F.relu(inp)
            net_dw8 = F.avg_pool2d(net, 2, stride=2)
            inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16 features
            fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
            fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
            offset_dw16 = self.conv_offset_16(fmap1_dw16)
            offset_dw16 = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0

            net_dw16 = F.avg_pool2d(net, 4, stride=4)
            inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

            # Positional encoding + self-attention at 1/16 scale
            pos_encoding_fn_small = _PositionEncodingSine(
                d_model=256,
                max_shape=(image1.shape[2] // 16, image1.shape[3] // 16),
            )
            x_tmp = pos_encoding_fn_small(fmap1_dw16)
            fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(
                x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]
            )
            x_tmp = pos_encoding_fn_small(fmap2_dw16)
            fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(
                x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]
            )

            fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
            fmap1_dw16, fmap2_dw16 = [
                x.reshape(x.shape[0], image1.shape[2] // 16, -1, x.shape[2]).permute(0, 3, 1, 2)
                for x in [fmap1_dw16, fmap2_dw16]
            ]

        corr_fn           = AGCL(fmap1,      fmap2)
        corr_fn_dw8       = AGCL(fmap1_dw8,  fmap2_dw8)
        corr_fn_att_dw16  = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        predictions = []
        flow = None
        flow_up = None

        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            flow_dw16 = self.zero_init(fmap1_dw16)

            # RUM: 1/16
            for itr in range(iters // 2):
                small_patch = (itr % 2 != 0)
                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_fn_att_dw16(flow_dw16, offset_dw16, small_patch=small_patch)

                with autocast(enabled=self.mixed_precision):
                    net_dw16, up_mask, delta_flow = self.update_block(
                        net_dw16, inp_dw16, out_corrs, flow_dw16
                    )

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * F.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/8
            for itr in range(iters // 2):
                small_patch = (itr % 2 != 0)
                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_fn_dw8(flow_dw8, offset_dw8, small_patch=small_patch)

                with autocast(enabled=self.mixed_precision):
                    net_dw8, up_mask, delta_flow = self.update_block(
                        net_dw8, inp_dw8, out_corrs, flow_dw8
                    )

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        for itr in range(iters):
            small_patch = (itr % 2 != 0)
            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return predictions


# ===========================================================================
# Section 9: CREStereoModel — Public wrapper (BaseStereoModel subclass)
# ===========================================================================

class CREStereoModel(BaseStereoModel):
    """CREStereo stereo matching model.

    Wraps ``_CREStereoNet`` with the library's standard interface.

    Usage::

        model = CREStereoModel.from_pretrained("crestereo")
    """

    config_class = CREStereoConfig

    def __init__(self, config: CREStereoConfig):
        super().__init__(config)
        self.net = _CREStereoNet(max_disp=config.max_disp)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run CREStereo forward pass.

        Args:
            left:  Left image  (B, 3, H, W) in [0,1] ImageNet-normalized range.
            right: Right image (B, 3, H, W) in [0,1] ImageNet-normalized range.

        Returns:
            Inference mode: Tensor (B, H, W) — final disparity in pixels.
            Training mode:  List[Tensor(B, H, W)] — one per recurrent iteration.
        """
        # Denormalize [0,1] ImageNet-norm → [0,255] for _CREStereoNet
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        std  = torch.tensor(self.config.std,  device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
        left_255  = (left  * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        # _CREStereoNet always returns a list of predictions (self.test_mode is False
        # in __init__; the test_mode kwarg is not actually used in its body).
        preds = self.net(left_255, right_255)

        if self.training:
            # Each pred is (B, 2, H, W); disparity is the x-flow channel (index 0)
            return [p[:, 0] for p in preds]
        else:
            return preds[-1][:, 0]  # (B, H, W) — final iteration prediction

    def _backbone_module(self) -> Optional[nn.Module]:
        return self.net.fnet

    @classmethod
    def _load_pretrained_weights(
        cls,
        model_id: str,
        device: str = "cpu",
        for_training: bool = False,
        **kwargs,
    ) -> "CREStereoModel":
        """Load pretrained CREStereo weights.

        Args:
            model_id: One of the registered variant IDs (e.g. "crestereo"),
                or a local path to a ``.pth`` checkpoint file.
            device: Device to map the weights to.
            for_training: Unused here; handled by from_pretrained().
            **kwargs: Optional ``variant`` override when loading from a local path.

        Returns:
            CREStereoModel with loaded weights.
        """
        import os

        # 1. Resolve variant → config
        if model_id in _CRESTEREO_VARIANT_MAP:
            config = CREStereoConfig.from_variant(model_id)
            checkpoint_path = None
        elif os.path.isfile(model_id):
            variant = kwargs.pop("variant", "standard")
            config = CREStereoConfig(variant=variant)
            checkpoint_path = model_id
        else:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Use one of {list(_CRESTEREO_VARIANT_MAP.keys())} or a local .pth file path."
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
                    f"Could not download checkpoint for '{model_id}' from HuggingFace Hub.\n"
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

        # Unwrap nested dicts
        if isinstance(state_dict, dict) and len(state_dict) == 1:
            key = next(iter(state_dict))
            if isinstance(state_dict[key], dict):
                state_dict = state_dict[key]

        # 5. Remap keys: checkpoint keys are bare (e.g. "fnet.conv1.weight");
        #    prefix with "net." to match self.net = _CREStereoNet(...)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            new_state_dict[f"net.{new_key}"] = v

        # 6. Load
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as exc:
            logger.warning(f"strict=True load failed: {exc}\nRetrying with strict=False.")
            incompatible = model.load_state_dict(new_state_dict, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")

        logger.info(f"Loaded CREStereoModel ({config.variant}) from '{checkpoint_path}'")
        return model
