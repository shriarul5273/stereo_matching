"""
AutoStereoModel — Automatic model resolution via the global registry.
"""

from typing import Any, Optional

from ...registry import MODEL_REGISTRY


class AutoStereoModel:
    """Resolves a model identifier to the correct StereoModel subclass and loads weights.

    Usage::

        from stereo_matching import AutoStereoModel

        model = AutoStereoModel.from_pretrained("raft-stereo")
        # Internally instantiates RaftStereoModel
    """

    def __init__(self):
        raise RuntimeError(
            "AutoStereoModel is not meant to be instantiated directly. "
            "Use AutoStereoModel.from_pretrained(model_id) instead."
        )

    @staticmethod
    def from_pretrained(
        model_id: str,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        """Load a pretrained stereo matching model.

        Args:
            model_id: Model identifier (e.g. "raft-stereo") or local checkpoint path.
            device: Device to load to. Auto-detected if None.
            **kwargs: Additional args passed to the model's from_pretrained().

        Returns:
            Instantiated model with pretrained weights.
        """
        model_cls = MODEL_REGISTRY.get_model_cls(model_id)
        return model_cls.from_pretrained(model_id, device=device, **kwargs)
