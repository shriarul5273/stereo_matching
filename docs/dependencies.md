# Dependencies

The authoritative dependency declarations are in `pyproject.toml`. This page
explains why they are present and calls out model-specific tools.

## Python and PyTorch support

- Python 3.10 or newer is required.
- CI runs the fast suite on Python 3.10, 3.11, and 3.12.
- The declared PyTorch floor is 2.9.0; CI checks both 2.9.0 and the latest CPU
  wheel available from the official PyTorch index.
- CUDA and MPS behavior depends on the installed PyTorch build and hardware.

## Core dependencies

These are installed by `pip install stereo_matching`:

| Package | Declared version | Purpose |
|---|---|---|
| `torch` | `>=2.9.0` | Models and tensor operations |
| `torchvision` | `>=0.15` | Vision operations, including AANet deformable convolution |
| `Pillow` | `>=9.0` | Image loading and RGB conversion |
| `numpy` | `>=1.24` | Array outputs and numerical operations |
| `matplotlib` | `>=3.6` | Disparity colormaps and plotting fallback |
| `opencv-python` | `>=4.8` | Image resizing, reading, and writing |
| `huggingface-hub` | `>=0.16` | Downloads for Hub-backed checkpoints |
| `einops` | `>=0.6` | Tensor rearrangement utility retained by the model stack |
| `timm` | `>=0.9.1` | Backbones used by FoundationStereo and IGEV families |
| `tqdm` | unpinned | Progress reporting in model utilities |
| `open3d` | unpinned | Interactive point-cloud viewer |

For headless servers, `opencv-python-headless` provides the same `cv2` API, but
the project currently declares `opencv-python`. Adjust the environment
deliberately if avoiding GUI libraries.

## Development extra

Install all contributor tools with:

```bash
pip install -e ".[dev]"
```

The `dev` extra includes `pytest`, `pytest-cov`, Ruff, mypy, `build`, Twine,
ONNX, and ONNX Runtime.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the commands used by CI.

## Export extra

Install ONNX export, verification, and quantization support without the other
development tools:

```bash
pip install "stereo_matching[export]"
```

This installs `onnx>=1.14` and `onnxruntime>=1.20`. See [export.md](export.md)
and [quantization.md](quantization.md).

## Model-specific dependencies

### FoundationStereo downloads

Registered FoundationStereo IDs use Google Drive. Install `gdown` to enable
automatic folder download:

```bash
pip install gdown
```

Without it, download the checkpoint manually and pass its local path. See
[models.md](models.md#foundationstereo).

### RAFT-Stereo interpolation helper

`scipy` is imported lazily only by RAFT-Stereo’s optional
`forward_interpolate()` warm-start helper:

```bash
pip install scipy
```

Ordinary inference does not call that helper.

### Optional RAFT correlation extensions

RAFT-Stereo probes for importable `corr_sampler` and `alt_cuda_corr` extension
modules and falls back to the pure-PyTorch implementation when they are absent.
Their source trees are not bundled in this repository; build them from the
compatible upstream RAFT-Stereo source if needed.

## Features not represented by extras

There is currently no `[data]` extra or bundled dataset module. Dataset
integration is application-owned; see [data.md](data.md).

There is also no `[viz]` extra at present: `open3d` is declared as a core
dependency. The viewer still imports it lazily, and `backend="matplotlib"` or
`backend="none"` can be used without opening an Open3D window.

## CI installation behavior

The fast matrix installs CPU-only PyTorch, ONNX, and ONNX Runtime wheels and
exercises export plus INT8 graph quantization on a small offline stereo model.
The scheduled slow workflow installs the full project plus `scipy` and `gdown`,
then attempts pretrained inference for every registered variant. It does not
claim that every pretrained family is ONNX-compatible.
