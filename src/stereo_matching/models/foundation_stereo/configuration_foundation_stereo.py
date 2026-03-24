"""
FoundationStereoConfig — Configuration for FoundationStereo models.

Inherits from BaseStereoConfig and maps to the args dict expected by the
original FoundationStereo architecture.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseStereoConfig


# Variant ID → internal variant name
_FS_VARIANT_MAP = {
    "foundation-stereo":       "standard",
    "foundation-stereo-large": "large",
}

# Google Drive folder (both variants live under the same root folder)
_FS_GDRIVE_URL = "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"

# Per-variant checkpoint directory name (inside the Drive folder / local cache)
_FS_CKPT_DIRS = {
    "standard": "11-33-40",
    "large":    "23-51-11",
}

# Variant-specific config overrides
_FS_VARIANT_CONFIGS = {
    "standard": {
        "vit_size":      "vits",
        "hidden_dims":   [128, 128, 128],
        "max_disp":      192,
        "n_gru_layers":  3,
        "corr_levels":   2,
        "corr_radius":   4,
        "n_downsample":  2,
    },
    "large": {
        "vit_size":      "vitl",
        "hidden_dims":   [192, 192, 192],
        "max_disp":      192,
        "n_gru_layers":  3,
        "corr_levels":   2,
        "corr_radius":   4,
        "n_downsample":  2,
    },
}


class FoundationStereoConfig(BaseStereoConfig):
    """Configuration for FoundationStereo models.

    Supports two variants:
        - ``foundation-stereo``:       Standard (ViT-S backbone, faster)
        - ``foundation-stereo-large``: Large (ViT-L backbone, higher quality)

    Maps to the ``args`` dict expected by the original FoundationStereo
    architecture (accessed both as attributes and via dict indexing).
    """

    model_type = "foundation-stereo"

    def __init__(
        self,
        variant: str = "standard",
        vit_size: str = "vits",
        max_disp: int = 192,
        n_gru_layers: int = 3,
        corr_levels: int = 2,
        corr_radius: int = 4,
        n_downsample: int = 2,
        hidden_dims: Optional[List[int]] = None,
        mixed_precision: bool = False,
        low_memory: bool = False,
        **kwargs: Any,
    ):
        # Apply variant-specific defaults
        overrides = _FS_VARIANT_CONFIGS.get(variant, {})
        vit_size      = overrides.get("vit_size",      vit_size)
        max_disp      = overrides.get("max_disp",      max_disp)
        n_gru_layers  = overrides.get("n_gru_layers",  n_gru_layers)
        corr_levels   = overrides.get("corr_levels",   corr_levels)
        corr_radius   = overrides.get("corr_radius",   corr_radius)
        n_downsample  = overrides.get("n_downsample",  n_downsample)
        hidden_dims   = overrides.get("hidden_dims",   hidden_dims)

        super().__init__(**kwargs)

        self.variant        = variant
        self.vit_size       = vit_size
        self.max_disp       = max_disp
        self.n_gru_layers   = n_gru_layers
        self.corr_levels    = corr_levels
        self.corr_radius    = corr_radius
        self.n_downsample   = n_downsample
        self.hidden_dims    = hidden_dims or [128, 128, 128]
        self.mixed_precision = mixed_precision
        self.low_memory     = low_memory

    @classmethod
    def from_variant(cls, variant_id: str) -> "FoundationStereoConfig":
        """Create a config from a variant identifier string."""
        if variant_id not in _FS_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_FS_VARIANT_MAP.keys())}"
            )
        return cls(variant=_FS_VARIANT_MAP[variant_id])

    @property
    def gdrive_url(self) -> str:
        """Google Drive folder URL for downloading FoundationStereo weights."""
        return _FS_GDRIVE_URL

    @property
    def ckpt_dir(self) -> str:
        """Per-variant checkpoint directory name (e.g. '11-33-40')."""
        return _FS_CKPT_DIRS.get(self.variant, "11-33-40")
