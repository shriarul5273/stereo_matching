"""
stereo_matching — A Transformers-style Python library for stereo depth estimation.

Provides a unified, modular API for running, comparing, and integrating
stereo matching models.
"""

from .output import StereoOutput
from .configuration_utils import BaseStereoConfig
from .registry import MODEL_REGISTRY

# Auto classes
from .models.auto import AutoStereoModel, AutoProcessor

# Ensure model modules are imported so they self-register.
# These are torch-free — modeling classes are loaded lazily on first use.
from .models import raft_stereo
from .models import crestereo


def load_dataset(name, split="train", root=None, download=True, transform=None, **kwargs):
    """Load a stereo dataset by name. See :mod:`stereo_matching.data` for details."""
    from .data import load_dataset as _load
    return _load(name, split=split, root=root, download=download, transform=transform, **kwargs)


def __getattr__(name):
    """Defer torch-heavy imports until first use."""
    if name == "BaseStereoModel":
        from .modeling_utils import BaseStereoModel
        globals()["BaseStereoModel"] = BaseStereoModel
        return BaseStereoModel
    if name == "StereoProcessor":
        from .processing_utils import StereoProcessor
        globals()["StereoProcessor"] = StereoProcessor
        return StereoProcessor
    if name == "StereoPipeline":
        from .pipeline_utils import StereoPipeline
        globals()["StereoPipeline"] = StereoPipeline
        return StereoPipeline
    if name == "pipeline":
        from .pipeline_utils import pipeline
        globals()["pipeline"] = pipeline
        return pipeline
    # Training symbols (Phase 4)
    if name == "StereoTrainer":
        from .trainer import StereoTrainer
        globals()["StereoTrainer"] = StereoTrainer
        return StereoTrainer
    if name == "StereoTrainingArguments":
        from .training_args import StereoTrainingArguments
        globals()["StereoTrainingArguments"] = StereoTrainingArguments
        return StereoTrainingArguments
    if name == "SequenceLoss":
        from .losses import SequenceLoss
        globals()["SequenceLoss"] = SequenceLoss
        return SequenceLoss
    if name == "SmoothL1StereoLoss":
        from .losses import SmoothL1StereoLoss
        globals()["SmoothL1StereoLoss"] = SmoothL1StereoLoss
        return SmoothL1StereoLoss
    if name == "DisparityLoss":
        from .losses import DisparityLoss
        globals()["DisparityLoss"] = DisparityLoss
        return DisparityLoss
    # Visualization (Phase 6)
    if name == "viz":
        from . import viz
        globals()["viz"] = viz
        return viz
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StereoOutput",
    "BaseStereoConfig",
    "BaseStereoModel",
    "StereoProcessor",
    "StereoPipeline",
    "pipeline",
    "AutoStereoModel",
    "AutoProcessor",
    "MODEL_REGISTRY",
    "load_dataset",
    # Training
    "StereoTrainer",
    "StereoTrainingArguments",
    "SequenceLoss",
    "SmoothL1StereoLoss",
    "DisparityLoss",
    # Visualization
    "viz",
]

__version__ = "0.1.0"
