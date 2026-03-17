# CLI Reference

The `stereo-estimate` command-line tool provides inference, model listing, and evaluation from the terminal.

---

## Installation

The CLI is installed automatically with the package:

```bash
pip install stereo_matching
stereo-estimate --help
```

---

## Commands

### `predict`

Run disparity estimation on a stereo image pair.

```bash
stereo-estimate predict \
    --left  LEFT_IMAGE \
    --right RIGHT_IMAGE \
    (--model VARIANT_ID | --checkpoint PATH) \
    [options]
```

**Required arguments:**

| Argument | Description |
|---|---|
| `--left PATH` | Path to left image |
| `--right PATH` | Path to right image |
| `--model ID` | Registered variant ID — mutually exclusive with `--checkpoint` |
| `--checkpoint PATH` | Path to a local `.pth` file — mutually exclusive with `--model` |

**Supported model IDs:** `raft-stereo`, `raft-stereo-middlebury`, `raft-stereo-eth3d`, `raft-stereo-realtime`, `crestereo`

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--variant NAME` | `standard` | Variant name when using `--checkpoint` |
| `--iters N` | model default | Override recurrent iterations |
| `--device DEVICE` | auto | `cuda`, `cpu`, or `mps` |
| `--focal-length F` | — | Focal length in pixels (enables depth output) |
| `--baseline B` | — | Baseline in metres (enables depth output) |
| `--output-dir DIR` | `./output` | Directory to save results |
| `--colormap NAME` | `turbo` | Matplotlib colormap |
| `--no-save` | — | Print stats only, do not write files |

**Output files** (written to `--output-dir`):

| File | Description |
|---|---|
| `disparity.png` | 16-bit PNG, value = disparity × 256 (KITTI convention) |
| `disparity_color.png` | Colorized visualization (uint8 RGB) |
| `side_by_side.png` | Left image next to colored disparity |
| `depth_mm.png` | 16-bit PNG depth in millimetres (only if `--focal-length` + `--baseline` given) |

**Examples:**

```bash
# HuggingFace Hub model
stereo-estimate predict --left left.png --right right.png --model raft-stereo

# CREStereo
stereo-estimate predict --left left.png --right right.png --model crestereo

# Local checkpoint
stereo-estimate predict --left left.png --right right.png \
    --checkpoint /path/to/raftstereo-sceneflow.pth --variant standard

# With metric depth
stereo-estimate predict --left left.png --right right.png --model raft-stereo \
    --focal-length 721.5 --baseline 0.54

# Faster inference, custom output dir
stereo-estimate predict --left l.png --right r.png --model raft-stereo \
    --iters 12 --output-dir results/
```

---

### `list-models`

Print all registered model variant IDs.

```bash
stereo-estimate list-models
```

Example output:

```
Registered stereo model variants:
  raft-stereo
  raft-stereo-middlebury
  raft-stereo-eth3d
  raft-stereo-realtime
  crestereo
```

---

### `info`

Show configuration details for a model variant.

```bash
stereo-estimate info --model VARIANT_ID
```

Example:

```bash
stereo-estimate info --model raft-stereo
```

Output:

```
Model: raft-stereo
  model_type      : raft-stereo
  variant         : standard
  input_size      : 384
  num_iters       : 32
  corr_levels     : 4
  corr_radius     : 4
  n_gru_layers    : 3
  mixed_precision : False
```

---

### `evaluate`

Evaluate a model on a benchmark dataset.

```bash
stereo-estimate evaluate \
    --model VARIANT_ID \
    --dataset DATASET \
    --data-root PATH \
    [--split SPLIT] \
    [--checkpoint PATH] \
    [--iters N] \
    [--batch-size N] \
    [--device DEVICE] \
    [--output-dir DIR]
```

| Argument | Default | Description |
|---|---|---|
| `--model ID` | required | Variant ID or `--checkpoint` |
| `--checkpoint PATH` | — | Local checkpoint (alternative to `--model`) |
| `--dataset NAME` | required | `sceneflow`, `kitti2015`, `kitti2012`, `middlebury`, `eth3d` |
| `--data-root PATH` | required | Root directory of the dataset |
| `--split SPLIT` | `val` | Dataset split: `train`, `val`, `test` |
| `--iters N` | model default | Recurrent iterations |
| `--batch-size N` | `1` | Evaluation batch size |
| `--device DEVICE` | auto | Compute device |
| `--output-dir DIR` | — | Save per-sample results (optional) |

**Reported metrics:** EPE, D1-all (%), bad_1px (%), bad_2px (%), bad_3px (%)

See [evaluation.md](evaluation.md) for metric definitions.

---

## Demo script

Run all registered models on a stereo pair and save colored disparity maps:

```bash
python examples/demo.py
```

Results are saved to `examples/output/` as `<model_name>_disp.png`.
