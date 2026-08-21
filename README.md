# stereo_matching

<p align="center">
    <a href="https://github.com/shriarul5273/stereo_matching/actions/workflows/test.yml"><img alt="CI" src="https://github.com/shriarul5273/stereo_matching/actions/workflows/test.yml/badge.svg"></a>
    <a href="https://github.com/shriarul5273/stereo_matching/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/shriarul5273/stereo_matching?color=blue"></a>
    <a href="https://pypi.org/project/stereo-matching/"><img alt="PyPI" src="https://img.shields.io/pypi/v/stereo-matching"></a>
    <a href="https://pypi.org/project/stereo-matching/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/stereo-matching"></a>
    <a href="https://huggingface.co/spaces/shriarul5273/StereoMatching_Compare_Demo"><img alt="Demo" src="https://img.shields.io/badge/Gradio-Compare%20Demo-blue"></a>
</p>

<h3 align="center">A unified Python library for stereo depth estimation</h3>

<h3 align="center">Inference - CLI - 3D Visualization - ONNX - Quantization</h3>

---

`stereo_matching` provides a single, consistent API across **8 model families and 31 registered variant IDs**. You can swap RAFT-Stereo, CREStereo, AANet, FoundationStereo, IGEV-Stereo, IGEV++, S2M2, and UniMatch without rewriting your preprocessing or postprocessing code.

It is built around the practical stereo workflow: run inference with one line, inspect models from the CLI, and turn calibrated disparity into depth maps and point clouds with the same library.

> **Current scope:** inference, model/config loading, preprocessing,
> postprocessing, CLI prediction, reduced-precision inference, ONNX export and
> quantization, and visualization are implemented. Dataset
> loaders, a packaged evaluator, trainer classes, and built-in losses are not
> included yet. Their documentation pages describe custom integration patterns
> and clearly mark reserved APIs.

## Installation

```bash
pip install stereo_matching
```

See [docs/dependencies.md](docs/dependencies.md) for runtime, development, and
model-specific dependencies.

---

## Quickstart

The `pipeline` API is the fastest way to run any registered stereo model:

```python
from stereo_matching import pipeline

pipe = pipeline("stereo-matching", model="raft-stereo")
result = pipe("left.png", "right.png", focal_length=721.5, baseline=0.54)

disparity = result.disparity          # np.ndarray, float32, (H, W)
depth_map = result.depth              # np.ndarray, float32, (H, W) or None
colored   = result.colored_disparity  # np.ndarray, uint8,   (H, W, 3)
```

For full control over preprocessing, forward pass, and postprocessing, use Auto Classes:

```python
from stereo_matching import AutoStereoModel, AutoProcessor
import torch

model = AutoStereoModel.from_pretrained("igev-stereo", device="cuda")
processor = AutoProcessor.from_pretrained("igev-stereo")

inputs = processor("left.png", "right.png")
with torch.no_grad():
    disparity = model(inputs["left_values"].cuda(), inputs["right_values"].cuda())
result = processor.postprocess(disparity, inputs["original_sizes"], colorize=True)
```

Or from the command line:

```bash
stereo-matching predict --left left.png --right right.png --model raft-stereo
```

---

## Why use stereo_matching?

**1. One API, every model.**
Switch from RAFT-Stereo to FoundationStereo or UniMatch by changing a single string. `pipeline()`, `AutoStereoModel`, and `AutoProcessor` keep the calling pattern consistent across families.

**2. Consistent model loading.**
Registered variants resolve through the same `pipeline()`, `AutoStereoModel`, and `AutoProcessor` entry points, so model selection stays simple even as the registry grows.

**3. Self-contained model packages.**
Each family lives under `src/stereo_matching/models/<family>/` with a config file, a single vendored modeling file, and lazy self-registration in the global registry.

**4. Calibrated outputs beyond disparity.**
Pass `focal_length` and `baseline` once and the library can return metric depth, colorized disparity, and point clouds for export or interactive viewing.

**5. Deployment workflows.**
Cast PyTorch models to FP16/BF16, dynamically quantize linear layers to INT8,
or export a two-input ONNX graph and quantize it with ONNX Runtime.

---

## Supported Models

8 model families - 31 registered IDs - see [docs/models.md](docs/models.md) for the full list and per-variant notes.

All families support `pipeline()`, Auto Classes, and CLI prediction.

| Family | Variants |
|---|---|
| RAFT-Stereo | `raft-stereo`, `raft-stereo-middlebury`, `raft-stereo-eth3d`, `raft-stereo-realtime` |
| CREStereo | `crestereo` |
| AANet | `aanet`, `aanet-kitti2012`, `aanet-sceneflow` |
| FoundationStereo | `foundation-stereo`, `foundation-stereo-large` |
| IGEV-Stereo | 6 registered IDs (`igev-stereo*`) |
| IGEV++ | 6 registered IDs (`igev-plusplus*`) |
| S2M2 | `s2m2`, `s2m2-m`, `s2m2-l`, `s2m2-xl` |
| UniMatch | 5 registered IDs (`unimatch*`) |

---

## What can you do?

<details>
<summary><b>Inference</b> - single pair, batch, or local script</summary>

```python
# Single stereo pair
result = pipe("left.png", "right.png")

# Batch
results = pipe(
    ["left0.png", "left1.png"],
    ["right0.png", "right1.png"],
    batch_size=2,
)
```

```bash
# CLI prediction
stereo-matching predict --left left.png --right right.png --model raft-stereo --output-dir results/

# Run the demo after selecting variants in its MODELS list
python examples/demo.py
```

</details>

<details>
<summary><b>Precision and ONNX</b> - FP16, BF16, INT8, export, and ONNX quantization</summary>

```python
from stereo_matching import AutoStereoModel, export_onnx, quantize_onnx

model = AutoStereoModel.from_pretrained("raft-stereo", device="cuda")
export_onnx(model, "raft_stereo.fp32.onnx", input_height=384, input_width=640)
quantize_onnx("raft_stereo.fp32.onnx", "raft_stereo.int8.onnx")

fp16_model = model.quantize("fp16")  # direct reduced-precision PyTorch inference
```

PyTorch dynamic INT8 and ONNX quantization are separate paths. See
[docs/quantization.md](docs/quantization.md) and
[docs/export.md](docs/export.md) before deploying reduced-precision models.

</details>

<details>
<summary><b>Auto Classes</b> - registry-based loading for registered variants</summary>

```python
from stereo_matching import AutoStereoModel, AutoProcessor

model = AutoStereoModel.from_pretrained("foundation-stereo", device="cuda")
processor = AutoProcessor.from_pretrained("foundation-stereo")
```

Use `stereo-matching list-models` to inspect the full registry and `stereo-matching info --model <id>` to print a model config from the terminal.

</details>

<details>
<summary><b>3D Visualization</b> - point clouds, PLY, and GLB export</summary>

```python
from stereo_matching import pipeline, viz
import numpy as np
from PIL import Image

pipe = pipeline("stereo-matching", model="raft-stereo")
result = pipe("left.png", "right.png", focal_length=721.5, baseline=0.54)
left_rgb = np.array(Image.open("left.png").convert("RGB"))

viz.point_cloud(
    result,
    image=left_rgb,
    focal_length=721.5,
    baseline=0.54,
    save_ply="scene.ply",
    save_glb="scene.glb",
)
```

`open3d` is currently a core dependency and is imported only when its viewer
backend is selected. See [docs/pipeline.md](docs/pipeline.md) for output details.

</details>

<details>
<summary><b>Model Comparison Demo</b> - hosted Hugging Face Space and local Gradio app</summary>

Hosted demo: [StereoMatching Compare Demo](https://huggingface.co/spaces/shriarul5273/StereoMatching_Compare_Demo)

```bash
pip install gradio gradio_sync3dcompare
python examples/compare_demo.py
```

The demo runs two stereo models on the same pair and shows disparity and 3D outputs side-by-side in a synchronized viewer.

</details>

---

## Documentation

- [docs/models.md](docs/models.md) - families, variants, and checkpoint sources
- [docs/pipeline.md](docs/pipeline.md) - `pipeline()`, `StereoOutput`, and processing details
- [docs/export.md](docs/export.md) - two-input ONNX export and runtime inference
- [docs/quantization.md](docs/quantization.md) - FP16, BF16, PyTorch INT8, and ONNX quantization
- [docs/cli.md](docs/cli.md) - implemented commands and the reserved evaluation interface
- [docs/dependencies.md](docs/dependencies.md) - runtime, development, and model-specific requirements
- [docs/data.md](docs/data.md) - custom dataset integration; no bundled loaders yet
- [docs/training.md](docs/training.md) - manual PyTorch training and current limitations
- [docs/evaluation.md](docs/evaluation.md) - metrics and a custom evaluation loop
- [docs/adding_a_model.md](docs/adding_a_model.md) - registry and package structure
- [docs/release_notes.md](docs/release_notes.md) - released and unreleased changes

## Development checks

```bash
pip install -e ".[dev]"
ruff check .
pytest tests -m "not slow"
python -m build
twine check dist/*
```

Mypy is currently advisory. Real pretrained-model inference runs in the weekly
slow workflow; see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Adding a New Model

1. Create `src/stereo_matching/models/your_model/`
2. Add `configuration_your_model.py`
3. Add `modeling_your_model.py`
4. Add `__init__.py` with `MODEL_REGISTRY.register(...)`
5. Import the package in `src/stereo_matching/__init__.py`

`AutoStereoModel`, `AutoProcessor`, and `pipeline()` resolve the new model automatically. See [docs/adding_a_model.md](docs/adding_a_model.md) for the full pattern.

---

## Acknowledgments

This library builds on the work of 8 stereo matching research families. See [docs/models.md#citations](docs/models.md#citations) for the citation block.

## License

MIT
