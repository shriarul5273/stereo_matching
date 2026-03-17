"""
BaseStereoConfig — Base configuration class for all stereo matching models.

Stores model-level metadata needed to instantiate a model and its processor.
Subclasses override default values; no new logic needed for simple configs.
"""

import copy
import json
from typing import Any, Dict, List, Optional


class BaseStereoConfig:
    """Base configuration for stereo matching models.

    Every field has a sensible default so users can override only what they need.
    Subclasses typically only change default values (Transformers modular pattern).

    Attributes:
        model_type: Unique identifier for the model family.
        backbone: Encoder backbone name (e.g. "default", "resnet50").
        input_size: Target height for preprocessing resize.
        max_disparity: Maximum expected disparity range in pixels.
        num_iters: Number of recurrent refinement iterations (for RAFT-style models).
        mixed_precision: Whether to use AMP during inference.
        mean: Per-channel normalization mean (default: ImageNet).
        std: Per-channel normalization std (default: ImageNet).
        is_metric: Always False for stereo (output is relative disparity in pixels).
    """

    model_type: str = "base"

    def __init__(
        self,
        backbone: str = "default",
        input_size: int = 384,
        max_disparity: int = 192,
        num_iters: int = 32,
        mixed_precision: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        is_metric: bool = False,
        **kwargs: Any,
    ):
        self.backbone = backbone
        self.input_size = input_size
        self.max_disparity = max_disparity
        self.num_iters = num_iters
        self.mixed_precision = mixed_precision
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        self.is_metric = is_metric

        # Store any extra kwargs for forward compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-round-trippable dictionary."""
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.model_type
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseStereoConfig":
        """Instantiate a config from a dictionary."""
        config_dict = copy.deepcopy(config_dict)
        config_dict.pop("model_type", None)
        return cls(**config_dict)

    def save_pretrained(self, save_directory: str) -> None:
        """Save config to a JSON file."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "BaseStereoConfig":
        """Load config from a directory containing config.json."""
        import os
        config_path = os.path.join(pretrained_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({json.dumps(self.to_dict(), indent=2)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseStereoConfig):
            return False
        return self.to_dict() == other.to_dict()
