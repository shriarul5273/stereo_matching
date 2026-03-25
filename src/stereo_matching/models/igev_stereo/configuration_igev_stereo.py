"""
IGEV-Stereo configuration.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseStereoConfig


_IGEV_VARIANT_MAP = {
    "igev-stereo": "sceneflow",
    "igev-stereo-sceneflow": "sceneflow",
    "igev-stereo-kitti2012": "kitti12",
    "igev-stereo-kitti2015": "kitti15",
    "igev-stereo-middlebury": "middlebury",
    "igev-stereo-eth3d": "eth3d",
}

_IGEV_HUB_REPO_ID = "shriarul5273/IGEV-Stereo"

_IGEV_CHECKPOINT_FILES = {
    "sceneflow": "sceneflow/sceneflow.pth",
    "kitti12": "kitti/kitti12.pth",
    "kitti15": "kitti/kitti15.pth",
    "middlebury": "middlebury/middlebury.pth",
    "eth3d": "eth3d/eth3d.pth",
}


class IGEVStereoConfig(BaseStereoConfig):
    model_type = "igev-stereo"

    def __init__(
        self,
        variant: str = "sceneflow",
        hidden_dims: Optional[List[int]] = None,
        corr_levels: int = 2,
        corr_radius: int = 4,
        n_downsample: int = 2,
        n_gru_layers: int = 3,
        max_disp: int = 192,
        precision_dtype: str = "float32",
        **kwargs: Any,
    ):
        if variant not in _IGEV_CHECKPOINT_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(_IGEV_CHECKPOINT_FILES.keys())}"
            )

        kwargs.setdefault("backbone", "mobilenetv2_100")
        kwargs.setdefault("input_size", 384)
        kwargs.setdefault("max_disparity", max_disp)
        kwargs.setdefault("num_iters", 32)
        super().__init__(**kwargs)

        self.variant = variant
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 128, 128]
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.n_downsample = n_downsample
        self.n_gru_layers = n_gru_layers
        self.max_disp = max_disp
        self.precision_dtype = precision_dtype

    @classmethod
    def from_variant(cls, variant_id: str) -> "IGEVStereoConfig":
        if variant_id not in _IGEV_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_IGEV_VARIANT_MAP.keys())}"
            )
        return cls(variant=_IGEV_VARIANT_MAP[variant_id])

    @property
    def hub_repo_id(self) -> str:
        return _IGEV_HUB_REPO_ID

    @property
    def checkpoint_filename(self) -> str:
        return _IGEV_CHECKPOINT_FILES[self.variant]
