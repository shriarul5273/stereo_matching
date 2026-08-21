# Training and Fine-Tuning

Model wrappers are regular PyTorch modules and can be used in a custom training
loop. The current package does not ship `StereoTrainer`,
`StereoTrainingArguments`, or built-in loss modules, despite those names being
reserved by the top-level lazy API. Accessing them raises an import error until
the corresponding modules are implemented.

Training behavior also differs by model family. Verify the output contract and
memory requirements of the selected wrapper before starting a long run.

## Load a model in training mode

```python
from stereo_matching import AutoStereoModel

model = AutoStereoModel.from_pretrained(
    "raft-stereo",
    device="cuda",
    for_training=True,
)
model.train()
```

`for_training=True` prevents `from_pretrained()` from applying the default
`.eval()` call. It does not create an optimizer, loss function, scheduler, or
dataset.

## Minimal sequence loss

Recurrent wrappers commonly return a list of disparity predictions during
training. A minimal masked sequence loss can be implemented as follows:

```python
import torch
import torch.nn.functional as F


def sequence_l1_loss(predictions, target, valid, gamma=0.9):
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]

    valid = valid.bool() & torch.isfinite(target)
    if not valid.any():
        raise ValueError("batch contains no valid disparity pixels")

    loss = target.new_zeros(())
    count = len(predictions)
    for index, prediction in enumerate(predictions):
        weight = gamma ** (count - index - 1)
        loss = loss + weight * F.l1_loss(prediction[valid], target[valid])
    return loss
```

This is an example, not a reproduction of every upstream model’s training
objective. Consult the original implementation when reproducing published
results.

## Optimizer groups and backbone freezing

`BaseStereoModel` provides:

- `freeze_backbone()` and `unfreeze_backbone()`
- `get_parameter_groups(backbone_lr_scale=0.1)`
- `unfreeze_top_k_backbone_layers(k)` for backbones exposing `.blocks`

Not every architecture exposes a backbone in the same way. Family wrappers may
override `_backbone_module()`; otherwise unsupported operations raise
`RuntimeError`.

```python
import torch

base_lr = 1e-4
groups = model.get_parameter_groups(backbone_lr_scale=0.1)
optimizer = torch.optim.AdamW(
    [
        {"params": groups[0]["params"], "lr": base_lr},
        {"params": groups[1]["params"], "lr": base_lr * 0.1},
    ]
)
```

The returned dictionaries contain an informational `lr_scale` field; PyTorch
optimizers do not apply that field automatically, so set each group’s `lr`
explicitly as shown above.

## Training step

```python
model.train()

left_values = left_values.cuda(non_blocking=True)
right_values = right_values.cuda(non_blocking=True)
target = target.cuda(non_blocking=True)
valid = valid.cuda(non_blocking=True)

optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", enabled=True):
    predictions = model(left_values, right_values)
    loss = sequence_l1_loss(predictions, target, valid)

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Resize and crop ground truth consistently with the image tensors. If width is
scaled by a factor, disparity values must be scaled by the same horizontal
factor.

## Model-family caveats

- Recurrent models can return multiple predictions in training mode.
- S2M2 currently returns a single-element prediction list.
- Some wrappers were primarily validated for inference; compare against the
  upstream repository before assuming the bundled forward path reproduces its
  complete training recipe.
- Large variants can exceed CPU memory or GPU VRAM even with small batches.
- `config.mixed_precision` controls family-specific internal autocast behavior;
  an outer `torch.autocast` context is still an application decision.

## Dataset and augmentation guidance

The repository does not bundle dataset loaders. See [data.md](data.md) for a
custom `Dataset` skeleton and batching constraints. Preserve epipolar geometry:
apply geometric transforms jointly to both images and update disparity values
when horizontally resizing.

## Validation

Run validation in evaluation mode and without gradients:

```python
model.eval()
with torch.no_grad():
    prediction = model(left_values, right_values)
```

Use the final prediction when a model returns a sequence, then restore original
resolution and disparity scale before computing metrics. See
[evaluation.md](evaluation.md).
