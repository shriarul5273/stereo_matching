# Dependencies

## Core dependencies

These are installed automatically with `pip install stereo_matching`.

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Deep learning framework |
| `torchvision` | ≥ 0.15 | Image transforms and AANet deformable convolution ops |
| `Pillow` | ≥ 9.0 | Image I/O (`PIL.Image`) |
| `numpy` | ≥ 1.24 | Array operations, output type |
| `matplotlib` | ≥ 3.6 | Colormaps for disparity visualization |
| `opencv-python` | ≥ 4.8 | Image read/write, colorspace conversion |
| `huggingface-hub` | ≥ 0.16 | Checkpoint download for Hugging Face-backed models |
| `einops` | ≥ 0.6 | Tensor rearrangement used by FoundationStereo |
| `timm` | ≥ 0.9.1 | Backbones used by FoundationStereo and the IGEV families |
| `tqdm` | any | Progress bars |

---

## Optional dependencies

### `[data]` — dataset loading

```bash
pip install stereo_matching[data]
```

| Package | Purpose |
|---|---|
| `h5py` | HDF5 file reading for SceneFlow datasets |
| `tqdm` | Progress bars during dataset preprocessing |

### `[dev]` — development and testing

```bash
pip install stereo_matching[dev]
```

| Package | Purpose |
|---|---|
| `pytest` | ≥ 7.0 — test runner |
| `pytest-cov` | Coverage reporting |

### Model-specific extras

```bash
pip install gdown
```

| Package | Purpose |
|---|---|
| `gdown` | Auto-download FoundationStereo weights from Google Drive |

---

## Optional CUDA extensions

RAFT-Stereo supports optional CUDA correlation kernels for faster inference. These are not required and the library falls back to a pure-PyTorch implementation automatically.

| Extension | Purpose | Install |
|---|---|---|
| `corr_sampler` | Fast CUDA 1D correlation | Build from `third-party/RAFT-Stereo/` |

To build:

```bash
cd third-party/RAFT-Stereo
python setup.py install
```

The library detects the extension at import time and uses it if available.

---

## Python version support

Python 3.9, 3.10, 3.11, 3.12.

---

## Hardware

| Device | Notes |
|---|---|
| CUDA GPU | Recommended for training and fast inference |
| CPU | Supported; inference is slow (seconds/pair) |
| MPS (Apple Silicon) | Supported via `--device mps`; mixed precision may be limited |

---

## Notes

- `scipy` is used only inside the `forward_interpolate()` utility (an optional warm-start helper for RAFT-Stereo). It is **not** a required dependency — the function imports scipy lazily.
- `opencv-python` and `opencv-python-headless` provide the same API. Use `opencv-python-headless` in server/docker environments where display is unavailable.
- FoundationStereo auto-download uses optional `gdown`; otherwise load a local `.pth` checkpoint manually.
- UniMatch checkpoints are fetched with `torch.hub.load_state_dict_from_url()` and cached under `~/.cache/stereo_matching/unimatch/`.
- The vendored model wrappers do not require runtime checkouts of the original third-party repositories.
