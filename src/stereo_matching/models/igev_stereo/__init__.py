"""
IGEV-Stereo model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_igev_stereo import IGEVStereoConfig, _IGEV_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_igev_stereo import IGEVStereoModel

    return IGEVStereoModel


MODEL_REGISTRY.register(
    model_type="igev-stereo",
    config_cls=IGEVStereoConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_IGEV_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "IGEVStereoModel":
        from .modeling_igev_stereo import IGEVStereoModel

        return IGEVStereoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IGEVStereoConfig", "IGEVStereoModel"]
