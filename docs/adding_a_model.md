# Adding a New Model

This guide walks through adding a new stereo model to the `stereo_matching` library. The pattern mirrors the existing packages under `src/stereo_matching/models/`.

---

## Overview

Every model consists of three files inside `src/stereo_matching/models/<model_name>/`:

| File | Purpose |
|---|---|
| `configuration_<name>.py` | Hyperparameters, variant map, and checkpoint metadata |
| `modeling_<name>.py` | Architecture (vendored) + `BaseStereoModel` wrapper |
| `__init__.py` | Self-registration + lazy export |

Plus one line added to `src/stereo_matching/__init__.py` to trigger registration on import.

---

## Step 1 — Create the config class

**File:** `src/stereo_matching/models/<name>/configuration_<name>.py`

```python
from ...configuration_utils import BaseStereoConfig

# Map from public variant ID → internal variant key
_MY_MODEL_VARIANT_MAP = {
    "my-model":       "standard",
    "my-model-large": "large",
}

_MY_MODEL_HUB_REPOS = {
    "standard": "your-hf-username/my-model-checkpoints",
    "large":    "your-hf-username/my-model-checkpoints",
}

_MY_MODEL_CHECKPOINT_FILES = {
    "standard": "my-model-standard.pth",
    "large":    "my-model-large.pth",
}


class MyModelConfig(BaseStereoConfig):
    model_type = "my-model"   # must be unique across all models

    def __init__(self, variant="standard", **kwargs):
        super().__init__(**kwargs)
        self.variant = variant

    @classmethod
    def from_variant(cls, variant_id: str) -> "MyModelConfig":
        if variant_id not in _MY_MODEL_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_MY_MODEL_VARIANT_MAP.keys())}"
            )
        return cls(variant=_MY_MODEL_VARIANT_MAP[variant_id])

    @property
    def hub_repo_id(self) -> str:
        return _MY_MODEL_HUB_REPOS.get(self.variant, "")

    @property
    def checkpoint_filename(self) -> str:
        return _MY_MODEL_CHECKPOINT_FILES.get(self.variant, "model.pth")
```

**Rules:**
- `model_type` must be unique — it is the registry key.
- `from_variant(variant_id)` is called by `AutoProcessor` and `pipeline()`.
- Expose the metadata your loader needs, for example `hub_repo_id`, `checkpoint_filename`, `checkpoint_url`, or `gdrive_url`.

Hugging Face is only one supported loading pattern. See `foundation_stereo` for Google Drive-backed weights and `unimatch` for direct checkpoint URLs.

---

## Step 2 — Create the model class

**File:** `src/stereo_matching/models/<name>/modeling_<name>.py`

Vendor all architecture code into this single file, then wrap it:

```python
import logging
import torch
import torch.nn as nn

from ...modeling_utils import BaseStereoModel
from .configuration_<name> import MyModelConfig, _MY_MODEL_VARIANT_MAP

logger = logging.getLogger(__name__)


# ── vendored architecture ──────────────────────────────────────────────────── #
# Paste all source files here in dependency order.
# Prefix internal class names (e.g. _MyModel_ResidualBlock) to avoid
# collisions with classes from other models in the same process.

class _MyModelNet(nn.Module):
    """Vendored architecture — internal only."""

    def __init__(self, config: MyModelConfig):
        super().__init__()
        # build layers ...

    def forward(self, image1, image2, test_mode=False):
        # Expects images in [0, 255] range (denormed by the wrapper)
        # Returns: list of Tensor(B,2,H,W) (training) or Tensor(B,2,H,W) (test)
        ...


# ── public wrapper ─────────────────────────────────────────────────────────── #

class MyModel(BaseStereoModel):
    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.net = _MyModelNet(config)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        # Denormalize [0,1] ImageNet-norm → [0,255] for vendored net
        mean = torch.tensor(self.config.mean, device=left.device, dtype=left.dtype).view(1,3,1,1)
        std  = torch.tensor(self.config.std,  device=left.device, dtype=left.dtype).view(1,3,1,1)
        left_255  = (left  * std + mean) * 255.0
        right_255 = (right * std + mean) * 255.0

        if self.training:
            preds = self.net(left_255, right_255)
            return [p[:, 0] for p in preds]   # List[Tensor(B,H,W)]
        else:
            out = self.net(left_255, right_255, test_mode=True)
            return out[:, 0]                  # Tensor(B,H,W)

    def _backbone_module(self):
        return self.net.fnet   # adjust to match your architecture

    @classmethod
    def _load_pretrained_weights(cls, model_id, device="cpu", for_training=False, **kwargs):
        import os

        if model_id in _MY_MODEL_VARIANT_MAP:
            config = MyModelConfig.from_variant(model_id)
            checkpoint_path = None
        elif os.path.isfile(model_id):
            config = MyModelConfig(variant=kwargs.pop("variant", "standard"))
            checkpoint_path = model_id
        else:
            raise ValueError(f"Unknown model_id '{model_id}'.")

        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=config.hub_repo_id,
                filename=config.checkpoint_filename,
            )

        model = cls(config)

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Strip DataParallel prefix, add wrapper prefix
        new_sd = {}
        for k, v in state_dict.items():
            k = k[len("module."):] if k.startswith("module.") else k
            new_sd[f"net.{k}"] = v

        try:
            model.load_state_dict(new_sd, strict=True)
        except RuntimeError as exc:
            logger.warning(f"strict=True failed: {exc}\nRetrying with strict=False.")
            incompatible = model.load_state_dict(new_sd, strict=False)
            if incompatible.missing_keys:
                logger.warning(f"Missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys: {incompatible.unexpected_keys}")

        logger.info(f"Loaded MyModel ({config.variant}) from '{checkpoint_path}'")
        return model
```

**Key rules:**
- `forward()` always receives `[0, 1]` ImageNet-normalized tensors from the processor.
- The wrapper denorms to `[0, 255]` before calling the vendored net.
- Training mode returns `List[Tensor(B,H,W)]`; inference mode returns `Tensor(B,H,W)`.
- Prefix all internal classes to avoid name collisions (e.g. `_MyModel_ResidualBlock`).

---

## Step 3 — Create `__init__.py`

**File:** `src/stereo_matching/models/<name>/__init__.py`

```python
from .configuration_<name> import MyModelConfig, _MY_MODEL_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_<name> import MyModel
    return MyModel


MODEL_REGISTRY.register(
    model_type="my-model",
    config_cls=MyModelConfig,
    model_cls=_load_model_cls,          # lazy — no torch import at registration time
    variant_ids=list(_MY_MODEL_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "MyModel":
        from .modeling_<name> import MyModel
        return MyModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MyModelConfig", "MyModel"]
```

---

## Step 4 — Register in top-level `__init__.py`

**File:** `src/stereo_matching/__init__.py`

Add one import (torch-free):

```python
from .models import raft_stereo
from .models import crestereo
from .models import foundation_stereo
# ...
from .models import my_model   # ← add this line
```

---

## Step 5 — Verify

```bash
# 1. Registry picks up the new variants (no torch required)
python -c "
from stereo_matching import MODEL_REGISTRY
print(MODEL_REGISTRY.list_variants())
# Should include 'my-model' and 'my-model-large'
"

# 2. Load from Hub
python -c "
from stereo_matching import AutoStereoModel
model = AutoStereoModel.from_pretrained('my-model', device='cpu')
print(sum(p.numel() for p in model.parameters()), 'params')
"

# 3. Forward pass shape
python -c "
import torch
from stereo_matching.models.my_model import MyModel
from stereo_matching.processing_utils import StereoProcessor
from PIL import Image

model = MyModel.from_pretrained('my-model', device='cpu')
proc  = StereoProcessor(model.config)

dummy = Image.new('RGB', (640, 480))
inp   = proc(dummy, dummy)
with torch.no_grad():
    out = model(inp['left_values'], inp['right_values'])
print(out.shape)   # expect (1, H, W)
"
```

---

## Checklist

- [ ] `model_type` is unique across all registered models
- [ ] `from_variant(variant_id)` raises `ValueError` for unknown IDs
- [ ] Loader metadata/properties return correct values for your chosen download path
- [ ] Internal classes are prefixed to avoid name collisions
- [ ] `forward()` denorms `[0,1]` → `[0,255]` before calling vendored net
- [ ] Training mode returns `List[Tensor(B,H,W)]`, inference mode returns `Tensor(B,H,W)`
- [ ] `_load_pretrained_weights` handles both local path and HF Hub download
- [ ] `__init__.py` calls `MODEL_REGISTRY.register()` without importing torch
- [ ] Added one-line import to `src/stereo_matching/__init__.py`
- [ ] All 3 verification commands pass
- [ ] Add variant rows to `docs/models.md` and `README.md`
