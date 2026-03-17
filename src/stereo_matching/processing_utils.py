"""
StereoProcessor — Shared processor for all stereo matching models.

Handles image-pair-to-tensor (preprocess) and tensor-to-output (postprocess)
transformations. NOT duplicated per model — reads parameters from the config.
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import matplotlib
from PIL import Image

from .configuration_utils import BaseStereoConfig
from .output import StereoOutput

logger = logging.getLogger(__name__)


class StereoProcessor:
    """Shared image processor for stereo matching models.

    Preprocessing pipeline:
        1. Load both images (path, PIL Image, or NumPy array).
        2. Convert to RGB.
        3. Resize: height → input_size, width preserving aspect ratio,
           both dimensions divisible by 8.
        4. Normalize pixel values (ImageNet mean/std by default).
        5. Convert to torch.Tensor and add batch dimension.

    Postprocessing pipeline:
        1. Squeeze channel dimension if present.
        2. Resize disparity back to original resolution (nearest-neighbor).
        3. Scale disparity by (W / W') to restore pixel units.
        4. Optionally apply colormap (default: turbo).
        5. Optionally compute metric depth from focal_length + baseline.
        6. Return StereoOutput.
    """

    def __init__(self, config: Optional[BaseStereoConfig] = None):
        if config is None:
            config = BaseStereoConfig()

        self.input_size = config.input_size
        self.mean = config.mean
        self.std = config.std

    @classmethod
    def from_config(cls, config: BaseStereoConfig) -> "StereoProcessor":
        """Create a processor from a model config."""
        return cls(config=config)

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        left_image: Union[str, "Image.Image", np.ndarray],
        right_image: Union[str, "Image.Image", np.ndarray],
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Preprocess a stereo image pair for model inference.

        Args:
            left_image: Left image. Can be a file path, PIL Image, or NumPy (H,W,3).
            right_image: Right image. Same formats as left_image.
            return_tensors: Return format, currently only "pt" (PyTorch).

        Returns:
            Dictionary with:
                - "left_values":  Tensor of shape (1, 3, H', W')
                - "right_values": Tensor of shape (1, 3, H', W')
                - "original_sizes": List of one (H, W) tuple
        """
        left_np = self._load_image(left_image)
        right_np = self._load_image(right_image)

        original_h, original_w = left_np.shape[:2]

        left_tensor, (new_h, new_w) = self._transform(left_np)
        # Resize right to the same target dimensions as left
        right_resized = cv2.resize(right_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        right_np_f = right_resized.astype(np.float32) / 255.0
        right_np_f = (right_np_f - np.array(self.mean)) / np.array(self.std)
        right_tensor = torch.from_numpy(right_np_f.transpose(2, 0, 1)).unsqueeze(0).float()

        return {
            "left_values": left_tensor,
            "right_values": right_tensor,
            "original_sizes": [(original_h, original_w)],
        }

    # ------------------------------------------------------------------ #
    #  Postprocessing
    # ------------------------------------------------------------------ #

    def postprocess(
        self,
        disparity: torch.Tensor,
        original_sizes: List[Tuple[int, int]],
        colorize: bool = True,
        colormap: str = "turbo",
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Union["StereoOutput", List["StereoOutput"]]:
        """Convert raw model disparity output to StereoOutput(s).

        Args:
            disparity: Raw disparity from model, shape (B, H', W') or (B, 1, H', W').
                Values are in units of H'/W'-space pixels.
            original_sizes: Original (H, W) for each image in the batch.
            colorize: Whether to produce a colored disparity visualization.
            colormap: Matplotlib colormap name (default: "turbo").
            focal_length: Camera focal length in pixels (enables metric depth).
            baseline: Camera baseline in metres (enables metric depth).

        Returns:
            A single StereoOutput if batch size is 1, otherwise a list.
        """
        # Squeeze channel dim: (B,1,H,W) → (B,H,W)
        if disparity.dim() == 4:
            disparity = disparity.squeeze(1)

        outputs = []
        batch_size = disparity.shape[0]
        processed_w = disparity.shape[-1]  # W' — model output width

        for i in range(batch_size):
            disp = disparity[i]
            original_h, original_w = original_sizes[i]

            # Scale factor: restore disparity to original pixel units
            scale_x = original_w / processed_w

            # Resize to original resolution using nearest-neighbor
            disp_resized = torch.nn.functional.interpolate(
                disp.unsqueeze(0).unsqueeze(0),
                size=(original_h, original_w),
                mode="nearest",
            )[0, 0]

            disp_np = disp_resized.cpu().numpy().astype(np.float32)
            disp_np = disp_np * scale_x  # restore pixel units

            # Colorize
            colored = None
            if colorize:
                colored = self._colorize(disp_np, colormap)

            # Metric depth
            depth_np = None
            if focal_length is not None and baseline is not None:
                depth_np = (focal_length * baseline) / np.maximum(disp_np, 1e-6)
                depth_np = depth_np.astype(np.float32)

            outputs.append(
                StereoOutput(
                    disparity=disp_np,
                    depth=depth_np,
                    colored_disparity=colored,
                    metadata={},
                )
            )

        return outputs[0] if len(outputs) == 1 else outputs

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _load_image(self, image: Union[str, "Image.Image", np.ndarray]) -> np.ndarray:
        """Load an image from various sources into a NumPy RGB array."""
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                return image
            raise ValueError(f"Expected (H, W, 3) array, got shape {image.shape}")

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return self._load_from_url(image)
            return self._load_from_path(image)

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _load_from_path(path: str) -> np.ndarray:
        """Load image from a local file path."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from '{path}'")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_from_url(url: str) -> np.ndarray:
        """Load image from a URL."""
        import urllib.request
        with urllib.request.urlopen(url) as response:
            data = response.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    def _transform(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Apply stereo-specific resize + normalize + convert to tensor.

        Resize strategy for stereo:
            - Scale height → input_size, preserving aspect ratio.
            - Round both dimensions down to nearest multiple of 8.
            - Normalize with ImageNet mean/std.

        Returns:
            Tuple of (tensor (1, 3, H', W'), (new_h, new_w)).
        """
        h, w = image_rgb.shape[:2]

        # Scale height to input_size
        scale = self.input_size / h
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Ensure both dimensions are multiples of 8
        new_h = max(new_h - (new_h % 8), 8)
        new_w = max(new_w - (new_w % 8), 8)

        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        img = resized.astype(np.float32) / 255.0
        img = (img - np.array(self.mean)) / np.array(self.std)

        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, (new_h, new_w)

    @staticmethod
    def _colorize(disparity_np: np.ndarray, colormap: str = "turbo") -> np.ndarray:
        """Apply a matplotlib colormap to a disparity map.

        Uses 95th-percentile normalization to avoid outlier pixels washing
        out the colormap. Returns an (H, W, 3) uint8 RGB array.
        """
        d_max = max(float(np.percentile(disparity_np, 95)), 1.0)
        normalized = np.clip(disparity_np / d_max, 0.0, 1.0)
        cmap = matplotlib.colormaps.get_cmap(colormap)
        colored = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
        return colored
