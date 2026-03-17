"""
CREStereoConfig — Configuration for CREStereo models.
"""

from typing import Any

from ...configuration_utils import BaseStereoConfig


# Variant ID → internal variant name
_CRESTEREO_VARIANT_MAP = {
    "crestereo": "standard",
}

# HuggingFace Hub repo IDs
_CRESTEREO_HUB_REPOS = {
    "standard": "shriarul5273/CRE-Stereo",
}

# Checkpoint filenames on HuggingFace Hub
_CRESTEREO_CHECKPOINT_FILES = {
    "standard": "crestereo_eth3d.pth",
}


class CREStereoConfig(BaseStereoConfig):
    """Configuration for CREStereo models.

    Supports one variant:
        - ``crestereo``: Trained on ETH3D dataset.
    """

    model_type = "crestereo"

    def __init__(
        self,
        variant: str = "standard",
        max_disp: int = 256,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.variant = variant
        self.max_disp = max_disp

    @classmethod
    def from_variant(cls, variant_id: str) -> "CREStereoConfig":
        """Create a config from a variant identifier string.

        Args:
            variant_id: One of the registered variant IDs (e.g. "crestereo").

        Returns:
            CREStereoConfig configured for the given variant.
        """
        if variant_id not in _CRESTEREO_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_CRESTEREO_VARIANT_MAP.keys())}"
            )
        variant = _CRESTEREO_VARIANT_MAP[variant_id]
        return cls(variant=variant)

    @property
    def hub_repo_id(self) -> str:
        """HuggingFace Hub repo ID for checkpoint download."""
        return _CRESTEREO_HUB_REPOS.get(self.variant, "")

    @property
    def checkpoint_filename(self) -> str:
        """Checkpoint filename on HuggingFace Hub."""
        return _CRESTEREO_CHECKPOINT_FILES.get(self.variant, "model.pth")
