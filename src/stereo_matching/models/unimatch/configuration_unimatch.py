"""
UniMatch configuration for stereo/disparity variants.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseStereoConfig

_UNIMATCH_VARIANT_MAP = {
    "unimatch": "mixdata",
    "unimatch-mixdata": "mixdata",
    "unimatch-sceneflow": "sceneflow",
    "unimatch-kitti15": "kitti15",
    "unimatch-middlebury": "middlebury",
}

_UNIMATCH_CHECKPOINT_URLS = {
    "mixdata": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/"
               "gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth",
    "sceneflow": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/"
                 "gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth",
    "kitti15": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/"
               "gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth",
    "middlebury": "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/"
                  "gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
}

_UNIMATCH_CHECKPOINT_FILES = {
    "mixdata": "gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth",
    "sceneflow": "gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth",
    "kitti15": "gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth",
    "middlebury": "gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth",
}


class UniMatchConfig(BaseStereoConfig):
    model_type = "unimatch"

    def __init__(
        self,
        variant: str = "mixdata",
        feature_channels: int = 128,
        upsample_factor: int = 4,
        num_scales: int = 2,
        num_head: int = 1,
        ffn_dim_expansion: int = 4,
        num_transformer_layers: int = 6,
        attn_splits_list: Optional[List[int]] = None,
        corr_radius_list: Optional[List[int]] = None,
        prop_radius_list: Optional[List[int]] = None,
        num_reg_refine: int = 3,
        padding_factor: int = 32,
        **kwargs: Any,
    ):
        kwargs.setdefault("input_size", 384)
        kwargs.setdefault("max_disparity", 400)
        super().__init__(**kwargs)

        if variant not in _UNIMATCH_CHECKPOINT_URLS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(_UNIMATCH_CHECKPOINT_URLS.keys())}"
            )

        self.variant = variant
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.num_scales = num_scales
        self.num_head = num_head
        self.ffn_dim_expansion = ffn_dim_expansion
        self.num_transformer_layers = num_transformer_layers
        self.attn_splits_list = attn_splits_list if attn_splits_list is not None else [2, 8]
        self.corr_radius_list = corr_radius_list if corr_radius_list is not None else [-1, 4]
        self.prop_radius_list = prop_radius_list if prop_radius_list is not None else [-1, 1]
        self.num_reg_refine = num_reg_refine
        self.padding_factor = padding_factor

    @classmethod
    def from_variant(cls, variant_id: str) -> "UniMatchConfig":
        if variant_id not in _UNIMATCH_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_UNIMATCH_VARIANT_MAP.keys())}"
            )
        return cls(variant=_UNIMATCH_VARIANT_MAP[variant_id])

    @property
    def hub_repo_id(self) -> str:
        return ""

    @property
    def checkpoint_url(self) -> str:
        return _UNIMATCH_CHECKPOINT_URLS[self.variant]

    @property
    def checkpoint_filename(self) -> str:
        return _UNIMATCH_CHECKPOINT_FILES[self.variant]
