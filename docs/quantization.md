# Precision and Quantization

`stereo_matching` provides three in-process PyTorch precision workflows and a
separate ONNX Runtime quantization workflow:

| Workflow | Target | Behavior |
|---|---|---|
| FP16 | PyTorch accelerator inference | Casts all floating parameters and buffers to `torch.float16` |
| BF16 | PyTorch accelerator or supported CPU inference | Casts to `torch.bfloat16` |
| INT8 | PyTorch CPU inference | Dynamically quantizes `nn.Linear` layers; convolution layers remain floating point |
| ONNX INT8/UINT8 | ONNX Runtime deployment | Quantizes an already-exported floating-point ONNX graph |

These workflows are lossy and model-dependent. Validate accuracy, latency, and
operator support for the selected model and deployment provider.

## FP16 and BF16

```python
from stereo_matching import AutoStereoModel, quantize_model

model = AutoStereoModel.from_pretrained("raft-stereo", device="cuda")
model = quantize_model(model, "fp16")
```

Aliases `float16` and `bfloat16` are accepted. Casting mutates the model and
returns the same object, so `model.quantize("bf16")` is equivalent.

`StereoPipeline` automatically casts processed image tensors to the first
floating model parameter’s dtype. When calling the model directly, cast inputs
yourself:

```python
inputs = processor("left.png", "right.png")
left = inputs["left_values"].to(device="cuda", dtype=next(model.parameters()).dtype)
right = inputs["right_values"].to(device="cuda", dtype=next(model.parameters()).dtype)
disparity = model(left, right)
```

FP16 has incomplete or slow CPU operator coverage on some PyTorch builds. BF16
requires hardware/provider support to deliver a speedup. A successful cast does
not guarantee that every operation in a particular architecture supports the
dtype.

## Dynamic INT8 in PyTorch

```python
from stereo_matching import AutoProcessor, AutoStereoModel, StereoPipeline

model = AutoStereoModel.from_pretrained("unimatch", device="cuda")
int8_model = model.quantize("int8")
processor = AutoProcessor.from_pretrained("unimatch")
pipeline = StereoPipeline(int8_model, processor, device="cpu")
result = pipeline("left.png", "right.png")
```

INT8 behavior differs from FP16/BF16:

- A deep-copied model is returned; the original model and device are unchanged.
- The result always runs on CPU.
- Only `nn.Linear` layers are dynamically quantized. Stereo architectures that
  spend most of their time in convolutions or correlation volumes may see
  little benefit.
- No calibration dataset is required.
- PyTorch’s eager dynamic quantization API is deprecated in favor of `torchao`.
  It remains the compatibility path used here across the declared PyTorch
  range, but may require migration in a future release.

Do not pass this dynamic INT8 model to `export_onnx()`: the exporter does not
support its `quantized::linear_dynamic` operators. Export the floating-point
model first and then quantize the ONNX file.

## ONNX quantization

Install the export dependencies:

```bash
pip install "stereo_matching[export]"
```

```python
from stereo_matching import AutoStereoModel, export_onnx, quantize_onnx

model = AutoStereoModel.from_pretrained("raft-stereo", device="cpu")
export_onnx(
    model,
    "raft_stereo.fp32.onnx",
    input_height=384,
    input_width=640,
    verify=True,
)
quantize_onnx(
    "raft_stereo.fp32.onnx",
    "raft_stereo.int8.onnx",
    weight_type="int8",
    verify=True,
)
```

`weight_type` accepts `int8` and `uint8`. Optional `per_channel=True` and
`reduce_range=True` are forwarded to ONNX Runtime. Verification defaults to
enabled and compares both stereo inputs against the floating-point graph with
`atol=rtol=0.05`.

The command-line equivalent is:

```bash
stereo-matching export --model raft-stereo \
    --output raft_stereo.fp32.onnx --height 384 --width 640 --verify

stereo-matching quantize-onnx \
    --input raft_stereo.fp32.onnx \
    --output raft_stereo.int8.onnx \
    --weight-type int8
```

## Accuracy and runtime caveats

- Dynamic ONNX quantization needs no calibration data, so it can materially
  degrade some checkpoints. Keep verification enabled and evaluate on real
  stereo data afterward.
- INT8 convolution graphs can require a recent ONNX Runtime build with
  `ConvInteger` support. Try `uint8` or update ONNX Runtime if the CPU provider
  reports `NOT_IMPLEMENTED`.
- Passing verification only checks one synthetic input. It is a structural
  smoke test, not an accuracy benchmark.
- Per-channel quantization can improve accuracy for some weights but is
  provider-dependent.
- Quantized file size does not predict model accuracy or inference speed.

See [export.md](export.md) for graph shape and provider details.
