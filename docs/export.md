# ONNX Export

ONNX export creates a graph with two inputs and one disparity output:

```text
left_values  (B, 3, H, W) ─┐
                            ├─ stereo model ─ disparity (B, H, W)
right_values (B, 3, H, W) ─┘
```

## Installation

```bash
pip install "stereo_matching[export]"
```

The extra installs `onnx` and `onnxruntime`. ONNX is required to export and
inspect the graph; ONNX Runtime is used by verification and quantization.

## Python API

```python
from stereo_matching import AutoStereoModel, export_onnx

model = AutoStereoModel.from_pretrained("raft-stereo", device="cpu")
path = export_onnx(
    model,
    "raft_stereo.onnx",
    input_height=384,
    input_width=640,
    opset_version=17,
    dynamic_batch=True,
    dynamic_spatial=False,
    verify=True,
)
```

The same operation is available as `model.export_onnx(...)`.

| Argument | Default | Description |
|---|---|---|
| `output_path` | required | Destination `.onnx` path |
| `input_height` | `model.config.input_size` | Trace-time image height |
| `input_width` | `input_height` | Trace-time image width |
| `opset_version` | `17` | ONNX opset |
| `dynamic_batch` | `True` | Mark batch axes dynamic |
| `dynamic_spatial` | `False` | Mark height/width dynamic |
| `verify` | `False` | Compare ONNX Runtime output against PyTorch |
| `atol`, `rtol` | `1e-3` | Verification tolerances |

Export switches the model to evaluation mode and requires
`forward(left, right)` to return `Tensor(B, H, W)`.

## CLI

```bash
stereo-matching --device cpu export \
    --model raft-stereo \
    --output raft_stereo.onnx \
    --height 384 \
    --width 640 \
    --verify
```

Useful options include `--static-batch`, `--dynamic-spatial`, `--opset`,
`--iters`, and `--precision fp32|fp16|bf16`. Global options such as `--device`
must appear before `export`.

## Reduced-precision export

FP16 and BF16 models can be exported by casting first or using CLI
`--precision`. Operator availability depends on the target execution provider.
Automatic verification uses ONNX Runtime’s CPU provider:

- FP16 verification may fail when the CPU provider lacks an implementation.
- BF16 verification is rejected explicitly because NumPy’s standard array path
  cannot represent BF16 inputs. Export BF16 with `verify=False` and validate in
  the intended provider.
- Dynamic PyTorch INT8 cannot be exported. Use the export-first workflow in
  [quantization.md](quantization.md#onnx-quantization).

## Dynamic axes

Dynamic batch export is covered by the offline test suite. Dynamic spatial axes
only annotate the graph; they do not make an architecture shape-agnostic.
Padding logic, fixed correlation layouts, positional embeddings, and Python
shape branches can still bake trace-time dimensions into a model. Test every
deployment shape with ONNX Runtime before enabling `dynamic_spatial`.

## ONNX Runtime inference

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "raft_stereo.onnx",
    providers=["CPUExecutionProvider"],
)
inputs = {
    "left_values": np.random.randn(1, 3, 384, 640).astype(np.float32),
    "right_values": np.random.randn(1, 3, 384, 640).astype(np.float32),
}
disparity = session.run(["disparity"], inputs)[0]
```

The graph consumes already-preprocessed tensors, not image files. Reproduce the
same resize and normalization as `StereoProcessor`, then restore output size and
horizontal disparity scale in the deployment application.

## Model compatibility

The export helper and its two-input contract are tested with a small offline
stereo model. The 31 pretrained variants have not all been certified for ONNX.
Vendored architectures may contain unsupported operators or trace-time Python
control flow. Use `verify=True` where supported, inspect warnings, and test real
images before production deployment.
