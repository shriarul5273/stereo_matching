"""
FoundationStereo model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_foundation_stereo import FoundationStereoConfig, _FS_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_foundation_stereo import FoundationStereoModel
    return FoundationStereoModel


MODEL_REGISTRY.register(
    model_type="foundation-stereo",
    config_cls=FoundationStereoConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_FS_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "FoundationStereoModel":
        from .modeling_foundation_stereo import FoundationStereoModel
        return FoundationStereoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FoundationStereoConfig", "FoundationStereoModel"]
