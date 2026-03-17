"""
CREStereo model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_crestereo import CREStereoConfig, _CRESTEREO_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_crestereo import CREStereoModel
    return CREStereoModel


MODEL_REGISTRY.register(
    model_type="crestereo",
    config_cls=CREStereoConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_CRESTEREO_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "CREStereoModel":
        from .modeling_crestereo import CREStereoModel
        return CREStereoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CREStereoConfig", "CREStereoModel"]
