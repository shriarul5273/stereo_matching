"""
StereoOutput — Standard output dataclass for all stereo matching models.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StereoOutput:
    """Standard output returned by all stereo matching inference paths.

    Attributes:
        disparity: Disparity map, shape (H, W), float32, in pixels.
        depth: Optional metric depth map, shape (H, W), float32, in metres.
            Only populated when focal_length and baseline are provided.
        colored_disparity: Colormapped RGB visualization, shape (H, W, 3), uint8.
            None if colorization was disabled.
        metadata: Dictionary containing model name, device, input resolution,
            inference latency, max disparity, and any model-specific fields.
    """

    disparity: np.ndarray
    depth: Optional[np.ndarray] = None
    colored_disparity: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
