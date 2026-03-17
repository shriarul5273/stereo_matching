"""
RaftStereoConfig — Configuration for RAFT-Stereo models.

Inherits from BaseStereoConfig and maps to the original argparse namespace
used by the RAFT-Stereo codebase.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseStereoConfig


# Variant ID → internal variant name
_RAFT_VARIANT_MAP = {
    "raft-stereo":            "standard",
    "raft-stereo-middlebury": "middlebury",
    "raft-stereo-eth3d":      "eth3d",
    "raft-stereo-realtime":   "realtime",
}

# HuggingFace Hub repo IDs
_RAFT_HUB_REPOS = {
    "standard":   "shriarul5273/RAFT-Stereo",
    "middlebury": "shriarul5273/RAFT-Stereo",
    "eth3d":      "shriarul5273/RAFT-Stereo",
    "realtime":   "shriarul5273/RAFT-Stereo",
}

# Checkpoint filenames on HuggingFace Hub
_RAFT_CHECKPOINT_FILES = {
    "standard":   "raftstereo-sceneflow.pth",
    "middlebury": "raftstereo-middlebury.pth",
    "eth3d":      "raftstereo-eth3d.pth",
    "realtime":   "raftstereo-realtime.pth",
}

# Variant-specific config overrides
_RAFT_VARIANT_CONFIGS = {
    "standard":   {},
    "middlebury": {"slow_fast_gru": True},
    "eth3d":      {},
    "realtime":   {"slow_fast_gru": True, "n_gru_layers": 2, "n_downsample": 3, "shared_backbone": True},
}


class RaftStereoConfig(BaseStereoConfig):
    """Configuration for RAFT-Stereo models.

    Supports four variants:
        - ``raft-stereo``: Trained on SceneFlow (general purpose)
        - ``raft-stereo-middlebury``: Fine-tuned on Middlebury; slow_fast_gru
        - ``raft-stereo-eth3d``: Fine-tuned on ETH3D; indoor/outdoor scenes
        - ``raft-stereo-realtime``: Fast inference; 2 GRU layers, shared backbone

    Maps directly to the original argparse namespace used by the RAFT-Stereo
    training and inference scripts.
    """

    model_type = "raft-stereo"

    def __init__(
        self,
        variant: str = "standard",
        hidden_dims: Optional[List[int]] = None,
        n_gru_layers: int = 3,
        corr_levels: int = 4,
        corr_radius: int = 4,
        n_downsample: int = 2,
        corr_implementation: str = "reg",
        context_norm: str = "batch",
        shared_backbone: bool = False,
        slow_fast_gru: bool = False,
        **kwargs: Any,
    ):
        # Apply variant-specific defaults
        variant_overrides = _RAFT_VARIANT_CONFIGS.get(variant, {})
        n_gru_layers        = variant_overrides.get("n_gru_layers",        n_gru_layers)
        n_downsample        = variant_overrides.get("n_downsample",        n_downsample)
        shared_backbone     = variant_overrides.get("shared_backbone",     shared_backbone)
        corr_implementation = variant_overrides.get("corr_implementation", corr_implementation)
        slow_fast_gru       = variant_overrides.get("slow_fast_gru",       slow_fast_gru)
        context_norm        = variant_overrides.get("context_norm",        context_norm)

        super().__init__(**kwargs)

        self.variant = variant
        self.hidden_dims = hidden_dims or [128, 128, 128]
        self.n_gru_layers = n_gru_layers
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.n_downsample = n_downsample
        self.corr_implementation = corr_implementation
        self.context_norm = context_norm
        self.shared_backbone = shared_backbone
        self.slow_fast_gru = slow_fast_gru

    @classmethod
    def from_variant(cls, variant_id: str) -> "RaftStereoConfig":
        """Create a config from a variant identifier string.

        Args:
            variant_id: One of "raft-stereo", "raft-stereo-middlebury",
                "raft-stereo-eth3d", "raft-stereo-realtime".

        Returns:
            RaftStereoConfig configured for the given variant.
        """
        if variant_id not in _RAFT_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_RAFT_VARIANT_MAP.keys())}"
            )
        variant = _RAFT_VARIANT_MAP[variant_id]
        return cls(variant=variant)

    @property
    def hub_repo_id(self) -> str:
        """HuggingFace Hub repo ID for checkpoint download."""
        return _RAFT_HUB_REPOS.get(self.variant, "")

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on HuggingFace Hub."""
        return _RAFT_CHECKPOINT_FILES.get(self.variant, "model.pth")
