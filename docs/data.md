# Datasets

This page covers the dataset classes bundled with `stereo_matching` and how to use them for training and evaluation.

---

## Overview

| Dataset | Type | Pairs | Resolution | Notes |
|---|---|---|---|---|
| SceneFlow (FlyingThings3D) | Synthetic | ~35k | 960×540 | Standard pre-training |
| SceneFlow (Driving) | Synthetic | ~4.4k | 960×540 | Driving synthetic |
| SceneFlow (Monkaa) | Synthetic | ~8.6k | 960×540 | Object animations |
| KITTI 2012 | Real | 194 train / 195 test | 1242×375 | Outdoor driving (LIDAR GT) |
| KITTI 2015 | Real | 200 train / 200 test | 1242×375 | Outdoor driving (LIDAR GT) |
| Middlebury | Real | varies | Up to 3000×2000 | High-res indoor (structured light GT) |
| ETH3D | Real | 27 train / 20 test | 1920×1080 | Indoor + outdoor, thin structures |

---

## Installation

Dataset classes require `h5py` for SceneFlow HDF5 files:

```bash
pip install stereo_matching[data]
```

---

## SceneFlow

```python
from stereo_matching.data import SceneFlowDataset

ds = SceneFlowDataset(
    root="/data/sceneflow",
    subset="things",       # "things", "driving", "monkaa"
    split="train",         # "train" or "val"
    transform=None,
)

left, right, disp = ds[0]
# left, right: PIL.Image (RGB)
# disp: np.ndarray (H, W) float32, pixels, negative = invalid
```

**Directory structure expected:**

```
/data/sceneflow/
  FlyingThings3D/
    frames_cleanpass/  (or frames_finalpass/)
    disparity/
  Driving/
    frames_cleanpass/
    disparity/
  Monkaa/
    frames_cleanpass/
    disparity/
```

---

## KITTI 2012

```python
from stereo_matching.data import KITTI2012Dataset

ds = KITTI2012Dataset(
    root="/data/kitti/kitti2012",
    split="train",   # "train" or "test"
)

left, right, disp = ds[0]
# disp: uint16 PNG loaded as float32 / 256.0 — pixels
```

**Directory structure:**

```
/data/kitti/kitti2012/
  training/
    image_0/   (left)
    image_1/   (right)
    disp_occ/  (ground truth, 16-bit PNG)
  testing/
    image_0/
    image_1/
```

---

## KITTI 2015

```python
from stereo_matching.data import KITTI2015Dataset

ds = KITTI2015Dataset(
    root="/data/kitti/kitti2015",
    split="train",
)

left, right, disp = ds[0]
```

**Directory structure:**

```
/data/kitti/kitti2015/
  training/
    image_2/   (left)
    image_3/   (right)
    disp_occ_0/ (ground truth)
  testing/
    image_2/
    image_3/
```

---

## Middlebury

```python
from stereo_matching.data import MiddleburyDataset

ds = MiddleburyDataset(
    root="/data/middlebury",
    resolution="H",   # "F" (full), "H" (half), "Q" (quarter)
    split="train",
)
```

---

## ETH3D

```python
from stereo_matching.data import ETH3DDataset

ds = ETH3DDataset(
    root="/data/eth3d",
    split="train",
)
```

---

## FolderDataset — custom data

Use `FolderDataset` for any folder of stereo pairs without pre-existing loaders.

```python
from stereo_matching.data import FolderDataset

ds = FolderDataset(
    left_dir="/data/custom/left",
    right_dir="/data/custom/right",
    disp_dir="/data/custom/disp",   # optional; omit for inference-only
    left_suffix=".png",
    right_suffix=".png",
    disp_suffix=".png",
    disp_scale=1.0,                 # scale factor for GT files
)
```

Files are matched by stem (filename without extension). Example:

```
left/   000001.png  000002.png  …
right/  000001.png  000002.png  …
disp/   000001.png  000002.png  …
```

---

## DataLoader

All dataset classes return `(left_PIL, right_PIL, disp_np)` tuples. Use `StereoProcessor` as a collation transform:

```python
import torch
from torch.utils.data import DataLoader
from stereo_matching.processing_utils import StereoProcessor

processor = StereoProcessor(model.config)

def collate_fn(batch):
    lefts, rights, disps = zip(*batch)
    inputs = processor(list(lefts), list(rights))
    gt = torch.stack([torch.from_numpy(d) for d in disps])
    return inputs, gt

loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, shuffle=True)
```
