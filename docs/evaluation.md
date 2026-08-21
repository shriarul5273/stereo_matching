# Evaluation

The current package does not include a `stereo_matching.evaluation` module or
built-in benchmark dataset loaders. The CLI exposes an `evaluate` placeholder,
but it exits with an explanatory error until that module is implemented.

Use the inference pipeline with your own dataset reader and metric code. This
page defines the common metrics and the scale handling required for a correct
evaluation.

## Metrics

### EPE — End-Point Error

Average absolute disparity error over valid pixels:

```text
EPE = mean(abs(prediction - ground_truth))
```

Lower is better; the unit is pixels.

### D1 — Disparity outlier rate

Percentage of valid pixels whose error exceeds both 3 pixels and 5% of the
ground-truth magnitude:

```text
D1 = percentage(abs(error) > 3 and abs(error) / abs(ground_truth) > 0.05)
```

Lower is better. Check the benchmark protocol for whether to report all,
non-occluded, foreground, or background pixels.

### Bad-pixel rate

Percentage of valid pixels whose absolute error exceeds a selected threshold:

```text
bad_Npx = percentage(abs(error) > N)
```

Values for `N = 1`, `2`, and `3` are commonly reported.

## Reference NumPy implementation

```python
import numpy as np


def stereo_metrics(prediction, ground_truth, valid=None):
    prediction = np.asarray(prediction, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.float32)

    if prediction.shape != ground_truth.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} != {ground_truth.shape}")

    if valid is None:
        valid = np.isfinite(ground_truth) & (ground_truth > 0)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(ground_truth)

    if not valid.any():
        raise ValueError("sample contains no valid ground-truth pixels")

    error = np.abs(prediction[valid] - ground_truth[valid])
    target = np.abs(ground_truth[valid])
    relative = error / np.maximum(target, 1e-6)

    return {
        "epe": float(error.mean()),
        "d1": float(((error > 3.0) & (relative > 0.05)).mean() * 100.0),
        "bad_1px": float((error > 1.0).mean() * 100.0),
        "bad_2px": float((error > 2.0).mean() * 100.0),
        "bad_3px": float((error > 3.0).mean() * 100.0),
    }
```

## Evaluation loop

```python
from stereo_matching import pipeline

matcher = pipeline("stereo-matching", model="raft-stereo", device="cuda")

sample_metrics = []
for left, right, ground_truth, valid_mask in dataset:
    result = matcher(left, right, colorize=False)
    sample_metrics.append(
        stereo_metrics(result.disparity, ground_truth, valid_mask)
    )
```

Aggregate metrics according to the benchmark protocol. Averaging per-image
scores and computing one global pixel-weighted score are not equivalent.

## Scale and validity requirements

- `StereoProcessor.postprocess()` returns disparity at the original left-image
  resolution and restores horizontal pixel scale.
- Compare against ground truth in the same resolution and pixel units.
- Use the official validity and occlusion masks for the dataset.
- Do not include colorized disparity or metric depth in disparity metrics.
- When padding or cropping inputs externally, undo that transform before
  comparison.

## CLI status

`stereo-matching evaluate --help` documents the reserved command shape. Running
the command currently reports that `stereo_matching.evaluation` is missing. Do
not build automation around this placeholder yet.

Dataset integration guidance is available in [data.md](data.md).
