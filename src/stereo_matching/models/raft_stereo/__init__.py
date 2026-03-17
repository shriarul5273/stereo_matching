"""
RAFT-Stereo model package.

Importing this package self-registers the model family in MODEL_REGISTRY.
The modeling class is loaded lazily on first use (defers torch import).
"""

from .configuration_raft_stereo import RaftStereoConfig, _RAFT_VARIANT_MAP
from ...registry import MODEL_REGISTRY


def _load_model_cls():
    from .modeling_raft_stereo import RaftStereoModel
    return RaftStereoModel


MODEL_REGISTRY.register(
    model_type="raft-stereo",
    config_cls=RaftStereoConfig,
    model_cls=_load_model_cls,
    variant_ids=list(_RAFT_VARIANT_MAP.keys()),
)


def __getattr__(name):
    if name == "RaftStereoModel":
        from .modeling_raft_stereo import RaftStereoModel
        return RaftStereoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RaftStereoConfig", "RaftStereoModel"]
