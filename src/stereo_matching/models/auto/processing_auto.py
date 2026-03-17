"""
AutoProcessor — Automatic processor resolution via the global registry.
"""

from typing import Any

from ...registry import MODEL_REGISTRY


class AutoProcessor:
    """Resolves a model identifier to a correctly configured StereoProcessor.

    Usage::

        from stereo_matching import AutoProcessor

        processor = AutoProcessor.from_pretrained("raft-stereo")
    """

    def __init__(self):
        raise RuntimeError(
            "AutoProcessor is not meant to be instantiated directly. "
            "Use AutoProcessor.from_pretrained(model_id) instead."
        )

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any):
        """Create a StereoProcessor configured for the given model.

        Args:
            model_id: Model identifier (e.g. "raft-stereo").
            **kwargs: Additional args passed to the processor.

        Returns:
            StereoProcessor configured with the correct model config.
        """
        config_cls = MODEL_REGISTRY.get_config_cls(model_id)

        # Resolve variant → config via from_variant() if available
        if hasattr(config_cls, "from_variant"):
            try:
                config = config_cls.from_variant(model_id)
            except (ValueError, KeyError):
                config = config_cls()
        else:
            config = config_cls()

        from ...processing_utils import StereoProcessor
        return StereoProcessor.from_config(config)
