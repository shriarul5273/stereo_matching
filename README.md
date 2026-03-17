# stereo_matching

A Transformers-style Python library for stereo depth estimation — inference, evaluation, and fine-tuning.

```python
from stereo_matching import pipeline

pipe = pipeline("stereo-matching", model="raft-stereo")
result = pipe("left.png", "right.png")
print(result.disparity.shape)          # (H, W) float32 — pixels
print(result.colored_disparity.shape)  # (H, W, 3) uint8
```

---

## Features

- **Unified API** — one `pipeline()` call works for every model
- **Auto-class loading** — `AutoStereoModel`, `AutoProcessor` from any variant ID or local checkpoint
- **Metric depth** — pass `focal_length` + `baseline` to get depth in metres
- **Lazy torch import** — `import stereo_matching` does not import PyTorch
- **HuggingFace Hub** — registered variants download automatically
- **Modular** — add a new model in one file; the registry handles the rest

---

## Installation

```bash
pip install stereo_matching
```

**From source:**

```bash
git clone https://github.com/shriarul5273/stereo_matching
cd stereo_matching
pip install -e .
```

**Optional extras:**

```bash
pip install stereo_matching[data]   # h5py for dataset loading
pip install stereo_matching[dev]    # pytest, pytest-cov
```

---

## Quick Start

### Inference with `pipeline()`

```python
from stereo_matching import pipeline

# Load a registered variant (auto-downloads from HuggingFace Hub)
pipe = pipeline("stereo-matching", model="raft-stereo")
result = pipe("left.png", "right.png")

# With metric depth (requires calibrated camera)
result = pipe("left.png", "right.png", focal_length=721.5, baseline=0.54)
print(result.depth)   # (H, W) float32 — metres
```

### `AutoStereoModel` / `AutoProcessor`

```python
from stereo_matching import AutoStereoModel, AutoProcessor

model     = AutoStereoModel.from_pretrained("raft-stereo", device="cuda")
processor = AutoProcessor.from_pretrained("raft-stereo")

inputs = processor(left_img, right_img)
result = processor.postprocess(
    model(inputs["left_values"].cuda(), inputs["right_values"].cuda()),
    inputs["original_sizes"],
    colorize=True, focal_length=721.5, baseline=0.54,
)
```

### Load from a local checkpoint

```python
from stereo_matching.models.raft_stereo import RaftStereoModel

model = RaftStereoModel.from_pretrained(
    "/path/to/raftstereo-sceneflow.pth",
    variant="standard",
    device="cuda",
)
```

### Demo script

Run all registered models and save colored disparity maps to `examples/output/`:

```bash
python examples/demo.py
```

---

## Supported Models

| Model ID | Variant | Training data | Hub |
|---|---|---|---|
| `raft-stereo` | standard | SceneFlow | `shriarul5273/RAFT-Stereo` |
| `raft-stereo-middlebury` | middlebury | SceneFlow + Middlebury | `shriarul5273/RAFT-Stereo` |
| `raft-stereo-eth3d` | eth3d | SceneFlow + ETH3D | `shriarul5273/RAFT-Stereo` |
| `raft-stereo-realtime` | realtime | SceneFlow | `shriarul5273/RAFT-Stereo` |
| `crestereo` | standard | ETH3D fine-tuned | `shriarul5273/CRE-Stereo` |

See [docs/models.md](docs/models.md) for full details and citations.

---

## Output

`StereoOutput` fields:

| Field | Type | Description |
|---|---|---|
| `disparity` | `np.ndarray (H,W) float32` | Disparity in pixels |
| `depth` | `np.ndarray (H,W) float32` or `None` | Metric depth in metres |
| `colored_disparity` | `np.ndarray (H,W,3) uint8` or `None` | RGB visualization |
| `metadata` | `dict` | Optional extras |

---

## Documentation

| Topic | Link |
|---|---|
| Models & variants | [docs/models.md](docs/models.md) |
| Pipeline API | [docs/pipeline.md](docs/pipeline.md) |
| CLI reference | [docs/cli.md](docs/cli.md) |
| Training | [docs/training.md](docs/training.md) |
| Evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Datasets | [docs/data.md](docs/data.md) |
| Dependencies | [docs/dependencies.md](docs/dependencies.md) |
| Adding a model | [docs/adding_a_model.md](docs/adding_a_model.md) |
| Release notes | [docs/release_notes.md](docs/release_notes.md) |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- See [docs/dependencies.md](docs/dependencies.md) for full list

---

## License

MIT
