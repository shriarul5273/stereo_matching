"""
IGEV++ configuration.
"""

from typing import Any, List, Optional

from ...configuration_utils import BaseStereoConfig


_IGEV_PP_VARIANT_MAP = {
    "igev-plusplus": "sceneflow",
    "igev-plusplus-sceneflow": "sceneflow",
    "igev-plusplus-kitti2012": "kitti2012",
    "igev-plusplus-kitti2015": "kitti2015",
    "igev-plusplus-middlebury": "middlebury",
    "igev-plusplus-eth3d": "eth3d",
}

_IGEV_PP_HUB_REPO_ID = "shriarul5273/IGEV-plusplus-Stereo"

_IGEV_PP_CHECKPOINT_FILES = {
    "sceneflow": "sceneflow.pth",
    "kitti2012": "kitti2012.pth",
    "kitti2015": "kitti2015.pth",
    "middlebury": "middlebury.pth",
    "eth3d": "eth3d.pth",
}


class IGEVPlusPlusConfig(BaseStereoConfig):
    model_type = "igev-plusplus"

    def __init__(
        self,
        variant: str = "sceneflow",
        hidden_dims: Optional[List[int]] = None,
        corr_levels: int = 2,
        corr_radius: int = 4,
        n_downsample: int = 2,
        n_gru_layers: int = 3,
        max_disp: int = 768,
        s_disp_range: int = 48,
        m_disp_range: int = 96,
        l_disp_range: int = 192,
        s_disp_interval: int = 1,
        m_disp_interval: int = 2,
        l_disp_interval: int = 4,
        precision_dtype: str = "float32",
        **kwargs: Any,
    ):
        if variant not in _IGEV_PP_CHECKPOINT_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(_IGEV_PP_CHECKPOINT_FILES.keys())}"
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
        self.s_disp_range = s_disp_range
        self.m_disp_range = m_disp_range
        self.l_disp_range = l_disp_range
        self.s_disp_interval = s_disp_interval
        self.m_disp_interval = m_disp_interval
        self.l_disp_interval = l_disp_interval
        self.precision_dtype = precision_dtype

    @classmethod
    def from_variant(cls, variant_id: str) -> "IGEVPlusPlusConfig":
        if variant_id not in _IGEV_PP_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant_id}'. "
                f"Available: {list(_IGEV_PP_VARIANT_MAP.keys())}"
            )
        return cls(variant=_IGEV_PP_VARIANT_MAP[variant_id])

    @property
    def hub_repo_id(self) -> str:
        return _IGEV_PP_HUB_REPO_ID

    @property
    def checkpoint_filename(self) -> str:
        return _IGEV_PP_CHECKPOINT_FILES[self.variant]
