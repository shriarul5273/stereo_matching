# Dataset Integration

`stereo_matching` currently focuses on inference and does not ship a
`stereo_matching.data` package or dataset-specific loaders. Dataset download,
parsing, augmentation, and batching must be supplied by the application.

The top-level `load_dataset()` compatibility hook is reserved for a future data
module and raises an import error in the current release. There is no `[data]`
installation extra yet.

## Expected sample format

A useful dataset contract is one stereo pair and an optional ground-truth
disparity map per sample:

```python
{
    "left": PIL.Image.Image,          # RGB
    "right": PIL.Image.Image,         # RGB
    "disparity": np.ndarray | None,   # (H, W), float32 pixels
}
```

Use positive horizontal disparity in pixels. Keep a separate boolean validity
mask when the source dataset uses zero, negative values, infinity, or a sentinel
to identify missing ground truth.

## Common stereo datasets

| Dataset | Typical use | Important format detail |
|---|---|---|
| Scene Flow | Synthetic pre-training | Read the original PFM disparity files and preserve pixel units |
| KITTI 2012 | Driving evaluation/fine-tuning | 16-bit PNG disparity is commonly stored at `value / 256` pixels |
| KITTI 2015 | Driving evaluation/fine-tuning | Use the benchmark validity/occlusion masks appropriate to the metric |
| Middlebury | High-resolution indoor evaluation | Resolution variants and calibration differ by release |
| ETH3D | Sparse indoor/outdoor evaluation | Evaluate only pixels marked valid by the benchmark mask |

Dataset layouts change between releases, so follow the documentation shipped
with the exact benchmark download instead of assuming a single directory tree.

## Minimal custom dataset

The following skeleton matches files by name. Adapt disparity decoding to the
format used by your dataset.

```python
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class StereoFolderDataset(Dataset):
    def __init__(self, root: str):
        root = Path(root)
        self.left_paths = sorted((root / "left").glob("*.png"))
        self.right_dir = root / "right"
        self.disparity_dir = root / "disparity"

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, index):
        left_path = self.left_paths[index]
        right_path = self.right_dir / left_path.name
        disparity_path = self.disparity_dir / left_path.name

        left = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")
        disparity = np.array(Image.open(disparity_path), dtype=np.float32) / 256.0
        return left, right, disparity
```

Validate at construction time that every left image has a matching right image
and, when required, a disparity file. Silent filename mismatches are especially
costly in stereo training.

## Preprocessing samples

`StereoProcessor` accepts one pair at a time. For a batch, preprocess each pair
and concatenate tensors with identical processed spatial dimensions:

```python
import torch

from stereo_matching import AutoProcessor

processor = AutoProcessor.from_pretrained("raft-stereo")


def collate_stereo(batch):
    processed = [processor(left, right) for left, right, _ in batch]
    left_values = torch.cat([item["left_values"] for item in processed])
    right_values = torch.cat([item["right_values"] for item in processed])
    original_sizes = [item["original_sizes"][0] for item in processed]
    disparities = [disparity for _, _, disparity in batch]
    return left_values, right_values, original_sizes, disparities
```

Because preprocessing preserves aspect ratio, images with different aspect
ratios can produce different widths and cannot be concatenated directly. Crop,
pad, or bucket samples by shape before batching. Ground-truth disparity must
receive the same spatial transform as the images, including horizontal scale
correction after resizing.

## Stereo-safe augmentation

- Apply the same color transform to both images unless deliberately simulating
  camera photometric differences.
- Apply identical crops and vertical transforms to both images and disparity.
- Update disparity when horizontally resizing an image.
- Avoid independent geometric transforms that break epipolar alignment.
- If swapping left and right during a horizontal flip, define and verify the
  resulting disparity sign convention explicitly.

See [training.md](training.md) for the currently supported manual training
workflow and [evaluation.md](evaluation.md) for metric definitions.
