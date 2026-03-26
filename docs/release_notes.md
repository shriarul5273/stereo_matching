# Release Notes

## v0.1.0

Initial release of `stereo_matching`.

### Added

**Core library:**
- `BaseStereoConfig` — base configuration class with stereo-specific fields (`input_size`, `max_disparity`, `num_iters`, `mixed_precision`, `is_metric`)
- `BaseStereoModel` — base model class with `forward(left, right)`, `predict()`, `from_pretrained()`, `freeze_backbone()`, `unfreeze_backbone()`, `trainable_parameters()`
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
- `stereo-matching evaluate` — benchmark on standard datasets

**Examples:**
- `examples/demo.py` — runs all registered models on a stereo pair, saves colored disparity maps to `examples/output/<model>_disp.png`

### Design decisions

- Lazy torch import: `import stereo_matching` does not import PyTorch
- Normalization pipeline: processor outputs `[0,1]` ImageNet-normalized; model wrapper denorms to `[0,255]` before calling vendored architecture
- Disparity scale correction in postprocessing: `disp * (original_W / processed_W)`
- `input_size` = target height — stereo pairs are typically wider than tall (e.g. KITTI 1242×375)
- Nearest-neighbor upsampling preserves sharp disparity boundaries
- Colorization uses 95th percentile as display maximum to suppress outlier pixels
- Single-file vendoring: all architecture code inlined into `modeling_<name>.py` with prefixed class names to avoid collisions
