# Models

All model families, registered variant IDs, configuration options, and citations.

The top-level package currently registers 8 model families and 31 variant IDs from `src/stereo_matching/models/`.

| Family | Registered IDs | Weight source | Local `.pth` loading |
|---|---|---|---|
| RAFT-Stereo | 4 IDs (`raft-stereo*`) | Hugging Face model repo | ✓ |
| CREStereo | 1 ID (`crestereo`) | Hugging Face model repo | ✓ |
| AANet | 3 IDs (`aanet*`) | Hugging Face dataset repo | — |
| FoundationStereo | 2 IDs (`foundation-stereo*`) | Google Drive or local file | ✓ |
| IGEV-Stereo | 6 IDs (`igev-stereo*`) | Hugging Face model repo | ✓ |
| IGEV++ | 6 IDs (`igev-plusplus*`) | Hugging Face model repo | ✓ |
| S2M2 | 4 IDs (`s2m2*`) | Hugging Face model repo | ✓ |
| UniMatch | 5 IDs (`unimatch*`) | Direct checkpoint URLs or local file | ✓ |

---

## RAFT-Stereo

Recurrent All-Pairs Field Transforms for Stereo, adapted from optical flow estimation.

**Paper:** [RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/abs/2109.07547)
**Authors:** Lahav Lipson, Zachary Teed, Jia Deng (Princeton, 2021)

### Variants

| Variant ID | `variant` key | Training data | Notes |
|---|---|---|---|
| `raft-stereo` | `standard` | SceneFlow | General-purpose baseline |
| `raft-stereo-middlebury` | `middlebury` | SceneFlow + Middlebury | High-resolution fine-tuned; slow_fast_gru |
| `raft-stereo-eth3d` | `eth3d` | SceneFlow + ETH3D | Indoor/outdoor fine-tuned |
| `raft-stereo-realtime` | `realtime` | SceneFlow | Fast inference; 2 GRU layers, shared backbone |

Hub: `shriarul5273/RAFT-Stereo`

### Configuration (`RaftStereoConfig`)

```python
from stereo_matching.models.raft_stereo import RaftStereoConfig

config = RaftStereoConfig(
    variant="standard",        # "standard", "middlebury", "eth3d", or "realtime"
    hidden_dims=[128,128,128], # GRU hidden dims per level
    n_gru_layers=3,            # number of GRU levels
    corr_levels=4,             # correlation pyramid levels
    corr_radius=4,             # correlation search radius
    n_downsample=2,            # feature downsampling factor
    corr_implementation="reg", # "reg" (PyTorch) or "alt"
    context_norm="batch",      # "batch", "instance", "group", or "none"
    shared_backbone=False,     # share feature extractor across images
    slow_fast_gru=False,       # slow/fast GRU update strategy
    num_iters=32,              # recurrent iterations at inference
)
```

### Loading

```python
from stereo_matching.models.raft_stereo import RaftStereoModel

# From HuggingFace Hub (auto-download)
model = RaftStereoModel.from_pretrained("raft-stereo", device="cuda")

# From a local .pth checkpoint
model = RaftStereoModel.from_pretrained(
    "/path/to/raftstereo-sceneflow.pth",
    variant="standard",
    device="cuda",
)

# For training
model = RaftStereoModel.from_pretrained("raft-stereo", for_training=True)
model.train()
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `raft-stereo` | ✓ | ✓ | ✓ |
| `raft-stereo-middlebury` | ✓ | ✓ | ✓ |
| `raft-stereo-eth3d` | ✓ | ✓ | ✓ |
| `raft-stereo-realtime` | ✓ | ✓ | ✓ |

---

## CREStereo

Cascaded Recurrent Stereo with Adaptive Group Correlation Layer (AGCL) and LoFTR-style linear attention. Multi-scale refinement at 1/16, 1/8, and 1/4 resolution. PyTorch port of the original MegEngine implementation.

**Paper:** [Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation](https://arxiv.org/abs/2203.11483)
**Authors:** Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, Shuaicheng Liu (Megvii, 2022)

### Variants

| Variant ID | `variant` key | Training data | Notes |
|---|---|---|---|
| `crestereo` | `standard` | ETH3D fine-tuned | max_disp=256 |

Hub: `shriarul5273/CRE-Stereo` · Checkpoint: `crestereo_eth3d.pth`

### Configuration (`CREStereoConfig`)

```python
from stereo_matching.models.crestereo import CREStereoConfig

config = CREStereoConfig(
    variant="standard",
    max_disp=256,
)
```

### Loading

```python
from stereo_matching.models.crestereo import CREStereoModel

# From HuggingFace Hub (auto-download)
model = CREStereoModel.from_pretrained("crestereo", device="cuda")

# From a local .pth checkpoint
model = CREStereoModel.from_pretrained(
    "/path/to/crestereo_eth3d.pth",
    variant="standard",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `crestereo` | ✓ | ✓ | ✓ |

---

## AANet

Adaptive Aggregation Network — ResNet-40 feature extractor with multi-scale
adaptive cost aggregation using deformable convolutions, and StereoDRNet
hierarchical refinement.  Deformable convolutions are implemented via
`torchvision.ops.deform_conv2d` for portability.

**Paper:** [AANet: Adaptive Aggregation Network for Efficient Stereo Matching](https://arxiv.org/abs/2004.09548)
**Authors:** Haofei Xu, Juyong Zhang (2020)

### Variants

| Variant ID | `variant` key | Training data | KITTI D1-all |
|---|---|---|---|
| `aanet` | `kitti15` | KITTI 2015 fine-tuned | 2.55 % |
| `aanet-kitti2012` | `kitti12` | KITTI 2012 fine-tuned | 2.42 % (out-all) |
| `aanet-sceneflow` | `sceneflow` | Scene Flow | EPE 0.87 |

Checkpoints: downloaded from the Hugging Face dataset repo `shriarul5273/AANet`.

Architecture parameters (all variants share the same default):
`max_disp=192`, `num_scales=3`, `num_fusions=6`, `num_deform_blocks=3`,
`deformable_groups=2`, `mdconv_dilation=2`, `refinement=stereodrnet`.

### Configuration (`AANetConfig`)

```python
from stereo_matching.models.aanet import AANetConfig

config = AANetConfig(
    variant="kitti15",      # "kitti15", "kitti12", or "sceneflow"
    max_disp=192,
    num_scales=3,
    num_fusions=6,
    num_stage_blocks=1,
    num_deform_blocks=3,
    deformable_groups=2,
    mdconv_dilation=2,
)
```

### Loading

```python
from stereo_matching.models.aanet import AANetModel

# From the Hugging Face dataset repo (auto-download)
model = AANetModel.from_pretrained("aanet", device="cuda")
model = AANetModel.from_pretrained("aanet-kitti2012", device="cuda")
model = AANetModel.from_pretrained("aanet-sceneflow", device="cuda")
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `aanet` | ✓ | ✓ | ✓ |
| `aanet-kitti2012` | ✓ | ✓ | ✓ |
| `aanet-sceneflow` | ✓ | ✓ | ✓ |

> **Note:** AANet requires `torchvision` for its deformable convolution layers.
>
> The current `AANetModel.from_pretrained()` implementation resolves registered variant IDs and downloads the matching checkpoint from the Hugging Face dataset repo. Arbitrary local `.pth` paths are not handled by this loader.

---

## FoundationStereo

Zero-shot stereo matching using a foundation model backbone (DINOv2 / DepthAnything) combined with an EdgeNext feature extractor, 3D cost-volume aggregation, and a multi-level selective ConvGRU refinement stage.

**Paper:** [FoundationStereo: Zero-Shot Stereo Matching](https://arxiv.org/abs/2501.09898)
**Authors:** Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, Stan Birchfield (NVIDIA, 2025)

> The architecture is vendored into a single module. No runtime checkout of `third-party/FoundationStereo/` is required; only the original license file is kept there.

### Variants

| Variant ID | `variant` key | ViT backbone | Notes |
|---|---|---|---|
| `foundation-stereo` | `standard` | ViT-S | Faster, lower memory |
| `foundation-stereo-large` | `large` | ViT-L | Higher quality |

Weights: [Google Drive](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf) · Folders: `11-33-40` (standard), `23-51-11` (large) · File: `model_best_bp2.pth`

### Configuration (`FoundationStereoConfig`)

```python
from stereo_matching.models.foundation_stereo import FoundationStereoConfig

config = FoundationStereoConfig(
    variant="standard",          # "standard" or "large"
    vit_size="vits",             # "vits", "vitb", or "vitl"
    max_disp=192,                # maximum disparity
    n_gru_layers=3,              # GRU refinement levels
    corr_levels=2,               # geometry encoding pyramid levels
    corr_radius=4,               # correlation search radius
    n_downsample=2,              # feature encoder downsampling
    hidden_dims=[128, 128, 128], # GRU hidden dims per level
    mixed_precision=False,       # AMP
    low_memory=False,            # reduce peak VRAM
)
```

### Loading

```python
from stereo_matching.models.foundation_stereo import FoundationStereoModel

# Auto-download via gdown if it is installed.
# Weights are fetched from Google Drive on first use and cached at
# ~/.cache/foundation_stereo/{11-33-40,23-51-11}/model_best_bp2.pth
model = FoundationStereoModel.from_pretrained("foundation-stereo", device="cuda")
model = FoundationStereoModel.from_pretrained("foundation-stereo-large", device="cuda")

# If gdown is unavailable, download the folder manually or load a local .pth file.
# Download from: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
model = FoundationStereoModel.from_pretrained(
    "/path/to/11-33-40/model_best_bp2.pth",
    variant="standard",
    device="cuda",
)
model = FoundationStereoModel.from_pretrained(
    "/path/to/23-51-11/model_best_bp2.pth",
    variant="large",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `foundation-stereo` | ✓ | ✓ | ✓ |
| `foundation-stereo-large` | ✓ | ✓ | ✓ |

---

## IGEV-Stereo

Iterative Geometry Encoding Volume for Stereo Matching. Uses a MobileNetV2
feature backbone, geometry encoding volume construction, and recurrent ConvGRU
refinement.

**Paper:** [Iterative Geometry Encoding Volume for Stereo Matching](https://arxiv.org/abs/2303.06615)
**Authors:** Gangwei Xu, Xianqi Wang, Xiaohuan Ding, Xin Yang (CVPR 2023)

> The stereo path is vendored into a single module. No runtime checkout of the original IGEV repository is required. Registered variants auto-download from Hugging Face Hub repo `shriarul5273/IGEV-Stereo`.

### Variants

| Variant ID | `variant` key | Checkpoint | Notes |
|---|---|---|---|
| `igev-stereo` | `sceneflow` | `sceneflow/sceneflow.pth` | Default SceneFlow checkpoint |
| `igev-stereo-sceneflow` | `sceneflow` | `sceneflow/sceneflow.pth` | Alias of `igev-stereo` |
| `igev-stereo-kitti2012` | `kitti12` | `kitti/kitti12.pth` | KITTI 2012 fine-tuned |
| `igev-stereo-kitti2015` | `kitti15` | `kitti/kitti15.pth` | KITTI 2015 fine-tuned |
| `igev-stereo-middlebury` | `middlebury` | `middlebury/middlebury.pth` | Middlebury fine-tuned |
| `igev-stereo-eth3d` | `eth3d` | `eth3d/eth3d.pth` | ETH3D fine-tuned |

### Loading

```python
from stereo_matching.models.igev_stereo import IGEVStereoModel

# From Hugging Face Hub (auto-download)
model = IGEVStereoModel.from_pretrained("igev-stereo", device="cuda")
model = IGEVStereoModel.from_pretrained("igev-stereo-kitti2015", device="cuda")
model = IGEVStereoModel.from_pretrained("igev-stereo-middlebury", device="cuda")

# From a local .pth checkpoint
model = IGEVStereoModel.from_pretrained(
    "/path/to/sceneflow.pth",
    variant="sceneflow",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `igev-stereo` | ✓ | ✓ | ✓ |
| `igev-stereo-sceneflow` | ✓ | ✓ | ✓ |
| `igev-stereo-kitti2012` | ✓ | ✓ | ✓ |
| `igev-stereo-kitti2015` | ✓ | ✓ | ✓ |
| `igev-stereo-middlebury` | ✓ | ✓ | ✓ |
| `igev-stereo-eth3d` | ✓ | ✓ | ✓ |

---

## IGEV++

Iterative Multi-range Geometry Encoding Volumes for Stereo Matching. Uses a
MobileNetV2 feature backbone, three disparity-range geometry volumes
(small / medium / large), selective feature fusion, and multi-level ConvGRU
refinement.

**Paper:** [IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching](https://arxiv.org/abs/2409.00638)
**Authors:** Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, Xin Yang (TPAMI 2025)

> The stereo path is vendored into a single module. No runtime checkout of `third-party/IGEV-plusplus/` is required. Registered variants auto-download from Hugging Face Hub repo `shriarul5273/IGEV-plusplus-Stereo`.

### Variants

| Variant ID | `variant` key | Checkpoint | Notes |
|---|---|---|---|
| `igev-plusplus` | `sceneflow` | `sceneflow.pth` | Default SceneFlow checkpoint |
| `igev-plusplus-sceneflow` | `sceneflow` | `sceneflow.pth` | Alias of `igev-plusplus` |
| `igev-plusplus-kitti2012` | `kitti2012` | `kitti2012.pth` | KITTI 2012 fine-tuned checkpoint |
| `igev-plusplus-kitti2015` | `kitti2015` | `kitti2015.pth` | KITTI 2015 fine-tuned checkpoint |
| `igev-plusplus-middlebury` | `middlebury` | `middlebury.pth` | Middlebury fine-tuned checkpoint |
| `igev-plusplus-eth3d` | `eth3d` | `eth3d.pth` | ETH3D fine-tuned checkpoint |

### Configuration (`IGEVPlusPlusConfig`)

```python
from stereo_matching.models.igev_plusplus import IGEVPlusPlusConfig

config = IGEVPlusPlusConfig(
    variant="sceneflow",       # "sceneflow", "kitti2012", "kitti2015", "middlebury", or "eth3d"
    max_disp=768,
    s_disp_range=48,
    m_disp_range=96,
    l_disp_range=192,
    s_disp_interval=1,
    m_disp_interval=2,
    l_disp_interval=4,
    n_gru_layers=3,
    corr_radius=4,
    precision_dtype="float32",
)
```

### Loading

```python
from stereo_matching.models.igev_plusplus import IGEVPlusPlusModel

# From Hugging Face Hub (auto-download)
model = IGEVPlusPlusModel.from_pretrained("igev-plusplus", device="cuda")
model = IGEVPlusPlusModel.from_pretrained("igev-plusplus-kitti2012", device="cuda")
model = IGEVPlusPlusModel.from_pretrained("igev-plusplus-kitti2015", device="cuda")
model = IGEVPlusPlusModel.from_pretrained("igev-plusplus-middlebury", device="cuda")

# From a local .pth checkpoint
model = IGEVPlusPlusModel.from_pretrained(
    "/path/to/sceneflow.pth",
    variant="sceneflow",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `igev-plusplus` | ✓ | ✓ | ✓ |
| `igev-plusplus-sceneflow` | ✓ | ✓ | ✓ |
| `igev-plusplus-kitti2012` | ✓ | ✓ | ✓ |
| `igev-plusplus-kitti2015` | ✓ | ✓ | ✓ |
| `igev-plusplus-middlebury` | ✓ | ✓ | ✓ |
| `igev-plusplus-eth3d` | ✓ | ✓ | ✓ |

---

## S2M2

Scalable Stereo Matching Model with Multi-Resolution Transformer (ICCV 2025).
Uses a CNN feature pyramid, a stacked Multi-Resolution Transformer (MRT) with
symmetric cross-attention, Optimal Transport–based initial disparity estimation,
and iterative local/global ConvGRU refinement. Jointly estimates disparity,
occlusion, and confidence.

**Paper:** [S²M²: Scalable Stereo Matching Model for Reliable Depth Estimation](https://arxiv.org/abs/2507.13229)
**Authors:** Junhong Min, Youngpil Jeon, Jimin Kim, Minyong Choi (ICCV 2025)
**Hub:** `minimok/s2m2`

### Variants

| Variant ID | `variant` key | feature_channels | num_transformer | Checkpoint |
|---|---|---|---|---|
| `s2m2` | S | 128 | 1 | CH128NTR1.pth |
| `s2m2-m` | M | 192 | 2 | CH192NTR2.pth |
| `s2m2-l` | L | 256 | 3 | CH256NTR3.pth |
| `s2m2-xl` | XL | 384 | 3 | CH384NTR3.pth |

### Configuration (`S2M2Config`)

```python
from stereo_matching.models.s2m2 import S2M2Config

config = S2M2Config(
    variant="S",            # "S", "M", "L", or "XL"
    feature_channels=128,   # set automatically by from_variant()
    num_transformer=1,      # set automatically by from_variant()
    dim_expansion=1,        # fixed for all variants
    refine_iter=3,          # local refinement iterations
)
```

### Loading

```python
from stereo_matching.models.s2m2 import S2M2Model

# From HuggingFace Hub (auto-download)
model = S2M2Model.from_pretrained("s2m2",    device="cuda")
model = S2M2Model.from_pretrained("s2m2-m",  device="cuda")
model = S2M2Model.from_pretrained("s2m2-l",  device="cuda")
model = S2M2Model.from_pretrained("s2m2-xl", device="cuda")

# From a local .pth checkpoint
model = S2M2Model.from_pretrained(
    "/path/to/CH128NTR1.pth",
    variant="S",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `s2m2` | ✓ | ✓ | ✓ |
| `s2m2-m` | ✓ | ✓ | ✓ |
| `s2m2-l` | ✓ | ✓ | ✓ |
| `s2m2-xl` | ✓ | ✓ | ✓ |

> S2M2's `forward()` returns a single-element `List[Tensor]` during training,
> compatible with `SmoothL1StereoLoss` and `DisparityLoss`. Occlusion and
> confidence outputs are available in the vendored `_S2M2` class but discarded
> at the wrapper level.

---

## UniMatch

Stereo-only disparity path vendored from `third-party/unimatch`. Uses a CNN encoder, multi-scale feature transformer, self-attention propagation, and regression refinement.

### Variants

| Variant ID | `variant` key | Checkpoint file | Notes |
|---|---|---|---|
| `unimatch` | `mixdata` | `gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth` | Default alias |
| `unimatch-mixdata` | `mixdata` | `gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth` | Alias of `unimatch` |
| `unimatch-sceneflow` | `sceneflow` | `gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth` | Scene Flow checkpoint |
| `unimatch-kitti15` | `kitti15` | `gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth` | KITTI 2015 fine-tuned |
| `unimatch-middlebury` | `middlebury` | `gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth` | Middlebury fine-tuned |

Registered variants download with `torch.hub.load_state_dict_from_url()` and are cached under `~/.cache/stereo_matching/unimatch/`.

### Configuration (`UniMatchConfig`)

```python
from stereo_matching.models.unimatch import UniMatchConfig

config = UniMatchConfig(
    variant="mixdata",            # "mixdata", "sceneflow", "kitti15", or "middlebury"
    feature_channels=128,
    upsample_factor=4,
    num_scales=2,
    num_head=1,
    ffn_dim_expansion=4,
    num_transformer_layers=6,
    attn_splits_list=[2, 8],
    corr_radius_list=[-1, 4],
    prop_radius_list=[-1, 1],
    num_reg_refine=3,
    padding_factor=32,
)
```

### Loading

```python
from stereo_matching.models.unimatch import UniMatchModel

# From the registered release URLs
model = UniMatchModel.from_pretrained("unimatch", device="cuda")
model = UniMatchModel.from_pretrained("unimatch-kitti15", device="cuda")
model = UniMatchModel.from_pretrained("unimatch-middlebury", device="cuda")

# From a local .pth checkpoint
model = UniMatchModel.from_pretrained(
    "/path/to/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth",
    variant="mixdata",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `unimatch` | ✓ | ✓ | ✓ |
| `unimatch-mixdata` | ✓ | ✓ | ✓ |
| `unimatch-sceneflow` | ✓ | ✓ | ✓ |
| `unimatch-kitti15` | ✓ | ✓ | ✓ |
| `unimatch-middlebury` | ✓ | ✓ | ✓ |

---

## Registering a new model

See [adding_a_model.md](adding_a_model.md) for the step-by-step guide.

---

## Citations

```bibtex
@inproceedings{min2025s2m2,
  title     = {{S\textsuperscript{2}M\textsuperscript{2}}: Scalable Stereo Matching Model for Reliable Depth Estimation},
  author    = {Junhong Min and Youngpil Jeon and Jimin Kim and Minyong Choi},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}

@inproceedings{wen2025foundationstereo,
  title     = {FoundationStereo: Zero-Shot Stereo Matching},
  author    = {Wen, Bowen and Trepte, Matthew and Aribido, Joseph and Kautz, Jan and Gallo, Orazio and Birchfield, Stan},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}

@article{xu2025igevplusplus,
  title     = {IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching},
  author    = {Xu, Gangwei and Wang, Xianqi and Zhang, Zhaoxing and Cheng, Junda and Liao, Chunyuan and Yang, Xin},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2025}
}

@inproceedings{lipson2021raft,
  title     = {RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching},
  author    = {Lipson, Lahav and Teed, Zachary and Deng, Jia},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2021}
}

@inproceedings{li2022crestereo,
  title     = {Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation},
  author    = {Li, Jiankun and Wang, Peisen and Xiong, Pengfei and Cai, Tao and Yan, Ziwei and Yang, Lei and Liu, Jiangyu and Fan, Haoqiang and Liu, Shuaicheng},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}

@inproceedings{xu2020aanet,
  title     = {AANet: Adaptive Aggregation Network for Efficient Stereo Matching},
  author    = {Xu, Haofei and Zhang, Juyong},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```
