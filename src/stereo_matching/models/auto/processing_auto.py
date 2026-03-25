"""
AutoProcessor — Automatic processor resolution via the global registry.
"""

import os
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
        config = kwargs.pop("config", None)

        if config is None:
            resolved_model_id = model_id
            if os.path.isfile(model_id) or os.path.isdir(model_id):
                resolved_model_id = kwargs.pop("variant", None) or kwargs.pop("model_type", None)
                if resolved_model_id is None:
                    raise ValueError(
                        f"Cannot infer processor config from local checkpoint path '{model_id}'. "
                        "Pass a registered variant or model_type, for example variant='igev-stereo'."
                    )

            config_cls = MODEL_REGISTRY.get_config_cls(resolved_model_id)

            # Resolve variant → config via from_variant() if available
            if hasattr(config_cls, "from_variant"):
                try:
                    config = config_cls.from_variant(resolved_model_id)
                except (ValueError, KeyError):
                    config = config_cls()
            else:
                config = config_cls()

        from ...processing_utils import StereoProcessor
        return StereoProcessor.from_config(config)
