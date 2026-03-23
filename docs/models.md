# Models

All model families, registered variant IDs, configuration options, and citations.

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

# From HuggingFace Hub (auto-download)
model = AANetModel.from_pretrained("aanet", device="cuda")

# From a local checkpoint
model = AANetModel.from_pretrained(
    "/path/to/aanet_kitti15.pth",
    variant="kitti15",
    device="cuda",
)
```

### Inference / CLI / Training support

| Model | Inference | CLI | Trainable |
|---|---|---|---|
| `aanet` | ✓ | ✓ | ✓ |
| `aanet-kitti2012` | ✓ | ✓ | ✓ |
| `aanet-sceneflow` | ✓ | ✓ | ✓ |

> **Note:** AANet requires `torchvision` for its deformable convolution layers.

---

## Registering a new model

See [adding_a_model.md](adding_a_model.md) for the step-by-step guide.

---

## Citations

```bibtex
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
