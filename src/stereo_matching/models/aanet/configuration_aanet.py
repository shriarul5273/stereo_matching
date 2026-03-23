"""
AANetConfig — Configuration for AANet models.
"""

from typing import Any

from ...configuration_utils import BaseStereoConfig


# Variant ID → internal variant name
_AANET_VARIANT_MAP = {
    "aanet":           "kitti15",
    "aanet-kitti2012": "kitti12",
    "aanet-sceneflow": "sceneflow",
}

_AANET_HUB_REPO_ID = "shriarul5273/AANet"

# Local filename used when caching the downloaded checkpoint.
_AANET_CHECKPOINT_FILES = {
    "kitti15":   "aanet_kitti15.pth",
    "kitti12":   "aanet_kitti12.pth",
    "sceneflow": "aanet_sceneflow.pth",
}


class AANetConfig(BaseStereoConfig):
    """Configuration for AANet models.

    Supports three variants:
        - ``aanet``          : Trained on KITTI 2015 (recommended).
        - ``aanet-kitti2012``: Trained on KITTI 2012.
        - ``aanet-sceneflow``: Trained on Scene Flow.

    All variants use the default AANet architecture:
    ResNet-40 feature extractor with deformable convolutions,
    3-scale adaptive aggregation, and StereoDRNet refinement.

    Checkpoints are downloaded from the Hugging Face Hub dataset
    ``shriarul5273/AANet``.
    """

    model_type = "aanet"

    def __init__(
        self,
        variant: str = "kitti15",
        max_disp: int = 192,
        num_downsample: int = 2,
        feature_similarity: str = "correlation",
        num_scales: int = 3,
        num_fusions: int = 6,
        num_stage_blocks: int = 1,
        num_deform_blocks: int = 3,
        deformable_groups: int = 2,
        mdconv_dilation: int = 2,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.variant           = variant
        self.max_disp          = max_disp
        self.num_downsample    = num_downsample
        self.feature_similarity = feature_similarity
        self.num_scales        = num_scales
        self.num_fusions       = num_fusions
        self.num_stage_blocks  = num_stage_blocks
        self.num_deform_blocks = num_deform_blocks
        self.deformable_groups = deformable_groups
        self.mdconv_dilation   = mdconv_dilation

    @classmethod
    def from_variant(cls, variant_id: str) -> "AANetConfig":
        """Create a config from a variant identifier string."""
        if variant_id not in _AANET_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_AANET_VARIANT_MAP.keys())}"
            )
        variant = _AANET_VARIANT_MAP[variant_id]
        return cls(variant=variant)

    @property
    def checkpoint_filename(self) -> str:
        """Local filename used when caching the downloaded checkpoint."""
        return _AANET_CHECKPOINT_FILES.get(self.variant, "aanet.pth")

    # ------------------------------------------------------------------
    # Kept for interface compatibility with BaseStereoModel/AutoClasses.
    # AANet checkpoints live in a Hugging Face dataset repo.
    # ------------------------------------------------------------------
    @property
    def hub_repo_id(self) -> str:
        return _AANET_HUB_REPO_ID
