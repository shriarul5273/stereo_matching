"""
UniMatch stereo model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_unimatch import UniMatchConfig, _UNIMATCH_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_unimatch import UniMatchModel
    return UniMatchModel


MODEL_REGISTRY.register(
    model_type="unimatch",
    config_cls=UniMatchConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_UNIMATCH_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "UniMatchModel":
        from .modeling_unimatch import UniMatchModel
        return UniMatchModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["UniMatchConfig", "UniMatchModel"]
