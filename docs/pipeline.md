# Pipeline API

The `pipeline()` factory provides a one-line inference interface for all stereo models.

---

## `pipeline()`

```python
from stereo_matching import pipeline

pipe = pipeline(task, model=None, device=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `str` | required | Must be `"stereo-matching"` |
| `model` | `str` | `None` | Registered variant ID (e.g. `"raft-stereo"`) or, for families that implement it, a local `.pth` path |
| `device` | `str` or `None` | `None` | `"cuda"`, `"cpu"`, `"mps"`, or auto-detect |

Returns a `StereoPipeline` instance.

> For arbitrary local checkpoints, the most reliable workflow is to instantiate the concrete model class plus `StereoProcessor` / `StereoPipeline` directly. The high-level `pipeline()` factory shares one `kwargs` namespace between model and processor resolution.

---

## `StereoPipeline.__call__()`

```python
results = pipe(
    left_images,
    right_images,
    batch_size=1,
    colorize=True,
    colormap="turbo",
    focal_length=None,
    baseline=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `left_images` | `str`, `PIL.Image`, or `list` | required | Left image(s) |
| `right_images` | `str`, `PIL.Image`, or `list` | required | Right image(s) |
| `batch_size` | `int` | `1` | GPU batch size |
| `colorize` | `bool` | `True` | Produce `colored_disparity` in output |
| `colormap` | `str` | `"turbo"` | Matplotlib colormap name |
| `focal_length` | `float` or `None` | `None` | Camera focal length in pixels |
| `baseline` | `float` or `None` | `None` | Camera baseline in metres |

Returns a single `StereoOutput` (single pair) or `list[StereoOutput]` (batch).

---

## `StereoOutput`

```python
@dataclass
class StereoOutput:
    disparity: np.ndarray                        # (H, W) float32 — pixels
    depth: Optional[np.ndarray] = None           # (H, W) float32 — metres
    colored_disparity: Optional[np.ndarray] = None  # (H, W, 3) uint8 RGB
    metadata: dict = field(default_factory=dict)
```

- `disparity` is always returned. Units are pixels.
- `depth` is returned when both `focal_length` and `baseline` are provided. Computed as `(focal_length * baseline) / disparity`.
- `colored_disparity` is a uint8 RGB visualization using the chosen colormap. Returned when `colorize=True`.

---

## `StereoProcessor`

Used internally by `StereoPipeline` and available directly for custom loops.

```python
from stereo_matching.processing_utils import StereoProcessor

processor = StereoProcessor(config)

# Preprocess a single stereo pair
inputs = processor(left_image, right_image)
# inputs = {
#   "left_values":    Tensor (1, 3, H', W') — ImageNet-normalized
#   "right_values":   Tensor (1, 3, H', W')
#   "original_sizes": [(H, W)]
# }

# Postprocess model output
result = processor.postprocess(
    disparity_tensor,          # Tensor (B, H', W') or (B, 1, H', W')
    inputs["original_sizes"],
    colorize=True,
    colormap="turbo",
    focal_length=721.5,        # optional
    baseline=0.54,             # optional
)
# result: StereoOutput (or list[StereoOutput] for batch > 1)
```

**Preprocessing details:**
- Height is scaled to `config.input_size` (default 384).
- Width is adjusted to preserve aspect ratio.
- Both dimensions are rounded down to the nearest multiple of 8.
- Both images in a pair are resized to the same spatial size.
- Pixel values are normalized with ImageNet mean/std to `[0, 1]`.

**Postprocessing details:**
- Disparity is upsampled (nearest-neighbor) back to original resolution.
- A scale correction `disp * (original_W / processed_W)` is applied to restore pixel units.
- Colorization uses the 95th percentile as the display maximum (suppresses outliers).
- Metric depth: `depth = (focal_length * baseline) / max(disparity, 1e-6)`.

---

## Examples

### Single pair

```python
from PIL import Image
from stereo_matching import pipeline

# Works with any registered model
pipe = pipeline("stereo-matching", model="raft-stereo", device="cuda")
# pipe = pipeline("stereo-matching", model="crestereo", device="cuda")

left  = Image.open("left.png")
right = Image.open("right.png")
result = pipe(left, right)

print(result.disparity.min(), result.disparity.max())
```

### Batch of pairs

```python
lefts  = ["left0.png", "left1.png", "left2.png"]
rights = ["right0.png", "right1.png", "right2.png"]
results = pipe(lefts, rights, batch_size=2)

for r in results:
    print(r.disparity.shape)
```

### With metric depth

```python
result = pipe(left, right, focal_length=721.5, baseline=0.54)
print(result.depth)   # (H, W) float32, metres
```

### Override number of recurrent iterations

```python
from stereo_matching import StereoPipeline
from stereo_matching.models.raft_stereo import RaftStereoModel
from stereo_matching.processing_utils import StereoProcessor

model = RaftStereoModel.from_pretrained("raft-stereo", device="cuda")
model.config.num_iters = 12   # faster inference, slightly lower accuracy
processor = StereoProcessor(model.config)

pipe = StereoPipeline(model=model, processor=processor, device="cuda")
```
