# Evaluation

This page describes the evaluation metrics and how to benchmark models on standard stereo datasets.

---

## Metrics

### EPE — End-Point Error

Average pixel-wise L1 error between predicted and ground-truth disparity over valid pixels.

```
EPE = mean(|pred - gt|)  over valid pixels
```

Lower is better. Unit: pixels.

### D1-all — Disparity Error Rate

Percentage of pixels where the disparity error exceeds both an absolute threshold and a relative threshold.

```
D1-all = % pixels where  |pred - gt| > 3px  AND  |pred - gt| / |gt| > 0.05
```

Lower is better. Unit: percent. This is the primary KITTI 2015 metric.

### bad_Npx — Bad Pixel Rate

Percentage of pixels where the absolute disparity error exceeds N pixels.

```
bad_Npx = % pixels where  |pred - gt| > N
```

Reported for N = 1, 2, 3. Lower is better. Common for Middlebury and ETH3D benchmarks.

---

## Programmatic evaluation

```python
from stereo_matching.evaluation import StereoEvaluator

evaluator = StereoEvaluator(metric="kitti2015")   # or "middlebury", "eth3d", "epe_only"

for pred_disp, gt_disp in zip(predictions, ground_truths):
    evaluator.update(pred_disp, gt_disp)

metrics = evaluator.compute()
print(metrics)
# {
#   "epe":     1.23,
#   "d1_all":  4.56,
#   "bad_1px": 12.3,
#   "bad_2px": 6.7,
#   "bad_3px": 4.6,
# }
```

### `evaluate()` convenience function

```python
from stereo_matching import pipeline
from stereo_matching.evaluation import evaluate

pipe = pipeline("stereo-matching", model="raft-stereo", device="cuda")

metrics = evaluate(
    pipe,
    dataset="kitti2015",
    data_root="/data/kitti/kitti2015",
    split="val",
    batch_size=1,
    iters=32,
)
```

### `compare()` — side-by-side comparison

```python
from stereo_matching.evaluation import compare

results = compare(
    models=["raft-stereo", "raft-stereo-middlebury"],
    dataset="kitti2015",
    data_root="/data/kitti/kitti2015",
)
# Returns a dict[model_id → metrics_dict]
```

---

## CLI evaluation

```bash
stereo-estimate evaluate \
    --model raft-stereo \
    --dataset kitti2015 \
    --data-root /data/kitti/kitti2015 \
    --split val
```

See [cli.md](cli.md) for full argument reference.

---

## Per-sample inspection

Pass `--output-dir` to the CLI (or `output_dir=` to `evaluate()`) to write per-sample disparity images alongside the error maps.

```bash
stereo-estimate evaluate \
    --model raft-stereo \
    --dataset kitti2015 \
    --data-root /data/kitti/kitti2015 \
    --output-dir eval_out/
```

Written files per sample:

| File | Description |
|---|---|
| `NNNN_pred.png` | Colored predicted disparity |
| `NNNN_error.png` | Error map (red = large error) |
| `NNNN_gt.png` | Colored ground-truth disparity |

---

## Metric notes

- **Valid pixels only:** All metrics exclude pixels where `gt_disp ≤ 0`.
- **Occluded regions:** By convention, KITTI reports metrics on all non-occluded pixels; Middlebury and ETH3D report separately for occluded and all pixels.
- **Scale:** Disparity predicted by the model is in pixels. Ensure preprocessing does not change the scale before computing metrics (`StereoProcessor.postprocess` applies the scale correction automatically).
