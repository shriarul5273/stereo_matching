# Training

This page covers fine-tuning stereo models on custom datasets.

---

## Overview

All models in `stereo_matching` inherit from `BaseStereoModel`, which provides:

- `freeze_backbone()` / `unfreeze_backbone()` for partial fine-tuning
- `trainable_parameters()` for optimizer setup
- Standard `nn.Module` interface — works with any PyTorch training loop

---

## Quick start

### Manual training loop

```python
import torch
from stereo_matching import AutoStereoModel
from stereo_matching.processing_utils import StereoProcessor

# Works with any registered model — "raft-stereo", "crestereo", etc.
model = AutoStereoModel.from_pretrained(
    "raft-stereo",
    device="cuda",
    for_training=True,
)
model.train()

processor = StereoProcessor(model.config)
optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

for left_img, right_img, gt_disp in dataloader:
    inputs = processor(left_img, right_img)
    left_t  = inputs["left_values"].cuda()
    right_t = inputs["right_values"].cuda()
    gt_t    = gt_disp.cuda()

    # Training mode: returns List[Tensor(B, H, W)] — one per iteration
    predictions = model(left_t, right_t)

    loss = sequence_loss(predictions, gt_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Loss functions

### Sequence loss (`SequenceLoss`)

RAFT-Stereo uses an exponentially weighted sum over all recurrent predictions, weighting later predictions more heavily. This encourages intermediate predictions to be reasonable while focusing the model on the final output.

```python
from stereo_matching.losses import SequenceLoss

criterion = SequenceLoss(gamma=0.9, max_flow=700.0)

# predictions: List[Tensor(B, H, W)] from model.train() forward pass
# gt_disp:     Tensor(B, H, W) — ground-truth disparity, negative values = invalid
loss = criterion(predictions, gt_disp)
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `gamma` | `0.9` | Exponential weight factor (earlier iters weighted less) |
| `max_flow` | `700.0` | Exclude pixels where `abs(gt) > max_flow` from loss |

Invalid pixels (negative disparity in ground truth) are automatically masked.

**Weight schedule:** for `N` predictions, weight of prediction `i` = `gamma^(N-1-i)`.

---

## Backbone freezing

```python
# Freeze the feature extractor — train only the update / correlation heads
model.freeze_backbone()

# Inspect frozen vs trainable parameters
for name, p in model.named_parameters():
    print(name, "trainable:", p.requires_grad)

# Unfreeze all
model.unfreeze_backbone()
```

`freeze_backbone()` locates the backbone by inspecting `model.pretrained`, `model.encoder`, `model.backbone`, and `model.net.fnet` attributes (in order).

---

## Mixed precision

```python
from torch.cuda.amp import GradScaler

model.config.mixed_precision = True   # enables autocast in forward()
scaler = GradScaler()

with torch.autocast("cuda"):
    predictions = model(left_t, right_t)
    loss = criterion(predictions, gt_disp)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Data augmentation

Standard augmentations for stereo matching:

```python
import torchvision.transforms.functional as TF
import random

def augment_stereo_pair(left, right, gt_disp):
    # Random horizontal flip (must flip both images and negate disparity)
    if random.random() < 0.5:
        left, right = TF.hflip(right), TF.hflip(left)
        gt_disp = -gt_disp   # disparity sign reverses on flip

    # Color jitter (same params for both images to preserve stereo consistency)
    brightness = random.uniform(0.8, 1.2)
    contrast   = random.uniform(0.8, 1.2)
    left  = TF.adjust_brightness(TF.adjust_contrast(left,  contrast), brightness)
    right = TF.adjust_brightness(TF.adjust_contrast(right, contrast), brightness)

    return left, right, gt_disp
```

Note: do **not** apply independent random crops or vertical flips — these break the epipolar constraint.

---

## Datasets for training

| Dataset | Use | Notes |
|---|---|---|
| SceneFlow (FlyingThings3D) | Pre-training | Large synthetic, ~35k pairs |
| SceneFlow (Driving) | Pre-training | Driving synthetic |
| SceneFlow (Monkaa) | Pre-training | Object animations |
| KITTI 2012 | Fine-tuning | 194 training pairs with GT |
| KITTI 2015 | Fine-tuning | 200 training pairs with GT |
| Middlebury | Fine-tuning | High-resolution indoor scenes |
| ETH3D | Fine-tuning | Outdoor/indoor, thin structures |

See [data.md](data.md) for dataset classes and loading details.

---

## Tips

- **Batch size:** RAFT-Stereo was trained with batch size 6 on 2× A100. Start with batch size 2 on a single GPU.
- **Iterations during training:** Use fewer iterations (e.g. `num_iters=12`) to save memory; the sequence loss still trains all recurrent weights.
- **Learning rate:** `1e-4` with `OneCycleLR` or cosine decay works well for fine-tuning.
- **Warm-up:** Freeze the backbone for the first few epochs, then unfreeze at a lower LR.
- **Gradient clipping:** Clip to `max_norm=1.0` to stabilize training.
