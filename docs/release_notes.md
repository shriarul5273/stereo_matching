# Release Notes

## Unreleased

## v0.2.0 - 2026-08-21

### Project quality

- Added pull-request CI for Ruff, advisory mypy, package builds, and the fast
  test matrix on Python 3.10–3.12 with PyTorch 2.9 and latest.
- Added weekly pretrained inference checks for all 31 registered variants.
- Added CodeQL scanning, Dependabot configuration, contributor guidance, and a
  CI status badge.
- Added offline tests for package metadata, configuration serialization,
  registry resolution, CLI behavior, and processor behavior.
- Synchronized package metadata with the declared PyTorch floor: Python 3.10+
  is now required.

### Export and quantization

- Added in-place FP16 and BF16 model casting with pipeline input dtype handling.
- Added CPU dynamic INT8 quantization for linear layers while preserving the
  caller’s original model.
- Added two-input ONNX export with dynamic batch/spatial options and numerical
  verification through ONNX Runtime.
- Added export-first ONNX INT8/UINT8 dynamic quantization with verification,
  per-channel, and reduced-range options.
- Added `export` and `quantize-onnx` CLI commands, an `[export]` dependency
  extra, public convenience functions, and model methods.

### Documentation

- Audited every Markdown page against the source tree.
- Marked dataset loaders, packaged evaluation, trainer classes, and built-in
  losses as future/reserved APIs rather than currently importable features.
- Replaced stale dataset, training, and evaluation examples with accurate
  application-owned integration guidance.
- Corrected CLI option placement, demo behavior, dependency/extras guidance,
  processor normalization, batching constraints, and model support wording.

## v0.1.0

Initial release of `stereo_matching`.

### Added

**Core library:**
- `BaseStereoConfig` — base configuration class with stereo-specific fields (`input_size`, `max_disparity`, `num_iters`, `mixed_precision`, `is_metric`)
- `BaseStereoModel` — base model class with `forward(left, right)`, `predict()`, `from_pretrained()`, backbone freezing, and parameter-group helpers
- `StereoProcessor` — preprocessing (resize-to-height, ImageNet normalization) and postprocessing (nearest-neighbor upsample, scale correction, colorization, metric depth)
- `StereoOutput` — dataclass with `disparity`, `depth`, `colored_disparity`, `metadata`
- `ModelRegistry` — singleton registry for model families and variants
- `AutoStereoModel` — auto-class model loading from variant ID or local checkpoint
- `AutoProcessor` — auto-class processor loading from variant ID

**Pipeline:**
- `pipeline("stereo-matching", model=..., device=...)` — one-line inference factory
- `StereoPipeline` — batched inference with colorization and metric depth support

**RAFT-Stereo model:**
- `RaftStereoConfig` / `RaftStereoModel`
- Registered variants: `raft-stereo`, `raft-stereo-middlebury`, `raft-stereo-eth3d`, `raft-stereo-realtime`
- All checkpoints downloaded from `shriarul5273/RAFT-Stereo` on HuggingFace Hub
- Correct per-variant architecture flags: `slow_fast_gru`, `n_gru_layers`, `n_downsample`, `shared_backbone`, `context_norm`

**CREStereo model:**
- `CREStereoConfig` / `CREStereoModel`
- Registered variant: `crestereo` (ETH3D fine-tuned, `max_disp=256`)
- Checkpoint downloaded from `shriarul5273/CRE-Stereo` on HuggingFace Hub
- Full PyTorch port vendored from `CREStereo-Pytorch`: AGCL, LoFTR-style linear attention, multi-scale cascaded RUM (1/16 → 1/8 → 1/4), separable ConvGRU, convex upsampling

**Current source tree note:**
- Additional model families now live under `src/stereo_matching/models/`: `aanet`, `foundation-stereo`, `igev-stereo`, `igev-plusplus`, `s2m2`, and `unimatch`
- See [models.md](models.md) for the up-to-date registry and loading behavior

**CLI:**
- `stereo-matching predict` — single-pair inference with output file saving
- `stereo-matching list-models` — list all registered variants
- `stereo-matching info` — show model configuration
- `stereo-matching evaluate` — reserved parser that reports the missing evaluation module

**Examples:**
- `examples/demo.py` — runs the variants selected in its `MODELS` list on a stereo pair and saves colored disparity maps to `examples/output/<model>_disp.png`

### Design decisions

- Lazy torch import: `import stereo_matching` does not import PyTorch
- Normalization pipeline: processor scales pixels to `[0,1]`, standardizes with configured mean/std, and family wrappers convert to the range expected by their vendored architecture
- Disparity scale correction in postprocessing: `disp * (original_W / processed_W)`
- `input_size` = target height — stereo pairs are typically wider than tall (e.g. KITTI 1242×375)
- Nearest-neighbor upsampling preserves sharp disparity boundaries
- Colorization uses 95th percentile as display maximum to suppress outlier pixels
- Single-file vendoring: all architecture code inlined into `modeling_<name>.py` with prefixed class names to avoid collisions
