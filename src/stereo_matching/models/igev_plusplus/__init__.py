"""
IGEV++ model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_igev_plusplus import IGEVPlusPlusConfig, _IGEV_PP_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_igev_plusplus import IGEVPlusPlusModel

    return IGEVPlusPlusModel


MODEL_REGISTRY.register(
    model_type="igev-plusplus",
    config_cls=IGEVPlusPlusConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_IGEV_PP_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "IGEVPlusPlusModel":
        from .modeling_igev_plusplus import IGEVPlusPlusModel

        return IGEVPlusPlusModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IGEVPlusPlusConfig", "IGEVPlusPlusModel"]
