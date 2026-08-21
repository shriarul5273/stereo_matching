# CLI Reference

Installing the package creates the `stereo-matching` command:

```bash
pip install stereo_matching
stereo-matching --help
```

The implemented commands are `predict`, `list-models`, `info`, `export`, and
`quantize-onnx`. An `evaluate` parser is present as a reserved interface, but
evaluation is not implemented in the current package.

## Global options

Global options must appear before the subcommand:

```bash
stereo-matching --device cuda predict --model raft-stereo \
    --left left.png --right right.png
```

| Option | Default | Description |
|---|---|---|
| `--device DEVICE` | auto | `cuda`, `cpu`, or `mps` |
| `--quiet` | false | Suppress non-essential output for commands that support it |
| `--verbose` | false | Reserved; currently does not change logging |

## `predict`

Run disparity estimation on one stereo pair:

```bash
stereo-matching predict \
    --left LEFT_IMAGE \
    --right RIGHT_IMAGE \
    (--model VARIANT_ID | --checkpoint PATH) \
    [options]
```

| Option | Default | Description |
|---|---|---|
| `--left PATH` | required | Left image path |
| `--right PATH` | required | Right image path |
| `--model ID` | one source required | Registered variant ID |
| `--checkpoint PATH` | one source required | Local checkpoint path; support depends on the family |
| `--variant NAME` | none | Family-specific hint for a local checkpoint |
| `--iters N` | model default | Override `config.num_iters` |
| `--focal-length F` | none | Focal length in pixels; requires `--baseline` |
| `--baseline B` | none | Baseline in metres; requires `--focal-length` |
| `--output-dir DIR` | `./output` | Output directory |
| `--colormap NAME` | `turbo` | Matplotlib colormap |
| `--no-save` | false | Print statistics without writing output files |

Both calibration arguments must be supplied together. Metric depth is computed
as `focal_length * baseline / disparity`.

### Output files

| File | Condition | Description |
|---|---|---|
| `disparity.png` | always unless `--no-save` | 16-bit PNG containing `round(max(disparity, 0) * 256)` |
| `disparity_color.png` | colorization available | Colorized disparity written through OpenCV |
| `side_by_side.png` | colorization available | Left RGB image next to colored disparity |
| `depth.npy` | calibrated run | Float32 depth map in metres |

### Examples

```bash
# Registered checkpoint on the auto-detected device
stereo-matching predict --model raft-stereo \
    --left left.png --right right.png

# Explicit device: global option comes first
stereo-matching --device cuda predict --model igev-stereo \
    --left left.png --right right.png --iters 16 --output-dir results/

# Calibrated depth
stereo-matching predict --model raft-stereo \
    --left left.png --right right.png \
    --focal-length 721.5 --baseline 0.54

# Local checkpoint for a family whose loader supports paths
stereo-matching predict \
    --checkpoint /path/to/raftstereo-sceneflow.pth \
    --variant standard --left left.png --right right.png
```

AANet currently resolves registered IDs only; its loader does not accept an
arbitrary local `.pth` path. See [models.md](models.md) for per-family loading
support.

## `list-models`

Print the 31 registered variant IDs:

```bash
stereo-matching list-models
stereo-matching list-models --json
```

The JSON form returns a JSON array of strings and is useful for scripts.

## `info`

Print the resolved configuration without downloading weights when a registered
model ID is used:

```bash
stereo-matching info --model raft-stereo
stereo-matching info --model raft-stereo --json
```

For a checkpoint path, `info` loads the model to discover its configuration:

```bash
stereo-matching --device cpu info \
    --checkpoint /path/to/model.pth --variant standard --json
```

The exact fields vary by model family.

## `export`

Export a registered model or supported local checkpoint to a two-input ONNX
graph:

```bash
stereo-matching --device cpu export \
    --model raft-stereo \
    --output raft_stereo.onnx \
    --height 384 --width 640 --verify
```

| Option | Default | Description |
|---|---|---|
| `--model ID` / `--checkpoint PATH` | one required | Model source |
| `--variant NAME` | none | Local-checkpoint family hint |
| `--output PATH` | required | Destination `.onnx` file |
| `--height N` | `config.input_size` | Trace input height |
| `--width N` | height | Trace input width |
| `--iters N` | model default | Recurrent iterations baked into the trace |
| `--opset N` | `17` | ONNX opset |
| `--precision` | `fp32` | `fp32`, `fp16`, or `bf16` |
| `--static-batch` | false | Fix batch size to one |
| `--dynamic-spatial` | false | Mark height and width dynamic |
| `--verify` | false | Compare ONNX Runtime CPU output with PyTorch |

BF16 cannot use the standard CPU verification path. Dynamic PyTorch INT8 is
also intentionally excluded; export FP32 first and use `quantize-onnx`. See
[export.md](export.md).

## `quantize-onnx`

Quantize an already-exported floating-point ONNX graph:

```bash
stereo-matching quantize-onnx \
    --input raft_stereo.onnx \
    --output raft_stereo.int8.onnx \
    --weight-type int8
```

| Option | Default | Description |
|---|---|---|
| `--input PATH` | required | Floating-point source graph |
| `--output PATH` | required | Quantized destination graph |
| `--weight-type` | `int8` | `int8` or `uint8` |
| `--no-verify` | false | Skip comparison with the source graph |
| `--atol`, `--rtol` | `0.05` | Verification tolerances |
| `--per-channel` | false | Quantize weights per output channel |
| `--reduce-range` | false | Use reduced integer range where supported |

Verification is enabled by default because dynamic quantization accuracy is
model-dependent. See [quantization.md](quantization.md).

## `evaluate` status

The following interface is reserved:

```bash
stereo-matching evaluate \
    (--model VARIANT_ID | --checkpoint PATH) \
    --dataset NAME \
    --data-root PATH \
    [--split val] \
    [--batch-size 1]
```

Running it currently exits because `stereo_matching.evaluation` is not included.
Use the custom evaluation loop in [evaluation.md](evaluation.md) instead.

## Demo scripts

`examples/demo.py` contains a `MODELS` list whose entries are commented out by
default. Select the variants you want, then run:

```bash
python examples/demo.py
```

Outputs are written under `examples/output/`. The comparison application has
additional dependencies:

```bash
pip install gradio gradio_sync3dcompare
python examples/compare_demo.py
```
