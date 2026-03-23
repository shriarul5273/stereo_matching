"""
AANet model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_aanet import AANetConfig, _AANET_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_aanet import AANetModel
    return AANetModel


MODEL_REGISTRY.register(
    model_type="aanet",
    config_cls=AANetConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_AANET_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "AANetModel":
        from .modeling_aanet import AANetModel
        return AANetModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AANetConfig", "AANetModel"]
