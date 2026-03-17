"""
StereoPipeline — Highest-level abstraction for stereo matching inference.

Chains model + processor into a single callable.
``pipeline()`` factory function resolves model IDs via the Auto classes.
"""

import logging
import time
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .output import StereoOutput
from .modeling_utils import BaseStereoModel, _auto_detect_device
from .processing_utils import StereoProcessor

logger = logging.getLogger(__name__)


class StereoPipeline:
    """End-to-end stereo matching pipeline.

    Usage::

        pipe = StereoPipeline(model=model, processor=processor)
        result = pipe("left.jpg", "right.jpg")
        disparity_map = result.disparity

    Accepts single image pairs or lists of pairs (auto-batching).
    """

    def __init__(
        self,
        model: BaseStereoModel,
        processor: StereoProcessor,
        device: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device or _auto_detect_device()
        self.model = self.model.to(self.device).eval()

    def __call__(
        self,
        left_images: Union[str, "Image.Image", np.ndarray, List],
        right_images: Union[str, "Image.Image", np.ndarray, List],
        batch_size: int = 1,
        colorize: bool = True,
        colormap: str = "turbo",
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Union[StereoOutput, List[StereoOutput]]:
        """Run end-to-end stereo matching.

        Args:
            left_images: Single left image or list. Accepts paths, PIL, ndarray.
            right_images: Single right image or list. Same formats as left_images.
            batch_size: Number of pairs to process per forward pass.
            colorize: Whether to produce a colored disparity visualization.
            colormap: Matplotlib colormap name for visualization.
            focal_length: Camera focal length in pixels (enables metric depth).
            baseline: Camera baseline in metres (enables metric depth).

        Returns:
            StereoOutput for single pair, or list of StereoOutput for batch.
        """
        # Normalize to lists
        if not isinstance(left_images, list):
            left_images = [left_images]
            right_images = [right_images]
            single = True
        else:
            single = False

        all_results = []
        for i in range(0, len(left_images), batch_size):
            left_batch = left_images[i : i + batch_size]
            right_batch = right_images[i : i + batch_size]
            results = self._process_batch(
                left_batch,
                right_batch,
                colorize=colorize,
                colormap=colormap,
                focal_length=focal_length,
                baseline=baseline,
            )
            if isinstance(results, list):
                all_results.extend(results)
            else:
                all_results.append(results)

        if single and len(all_results) == 1:
            return all_results[0]
        return all_results

    def _process_batch(
        self,
        left_batch: List,
        right_batch: List,
        colorize: bool = True,
        colormap: str = "turbo",
        focal_length: Optional[float] = None,
        baseline: Optional[float] = None,
    ) -> Union[StereoOutput, List[StereoOutput]]:
        """Process a batch of stereo pairs."""
        start = time.perf_counter()

        # Preprocess each pair and stack into a batch
        lefts, rights, original_sizes = [], [], []
        for limg, rimg in zip(left_batch, right_batch):
            inp = self.processor(limg, rimg)
            lefts.append(inp["left_values"])
            rights.append(inp["right_values"])
            original_sizes.extend(inp["original_sizes"])

        left_tensor = torch.cat(lefts, dim=0).to(self.device)
        right_tensor = torch.cat(rights, dim=0).to(self.device)

        # Forward pass
        with torch.no_grad():
            disparity = self.model(left_tensor, right_tensor)

        elapsed = time.perf_counter() - start

        # Postprocess
        outputs = self.processor.postprocess(
            disparity,
            original_sizes,
            colorize=colorize,
            colormap=colormap,
            focal_length=focal_length,
            baseline=baseline,
        )

        # Inject metadata
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            out.metadata.update(
                {
                    "model_type": getattr(self.model.config, "model_type", "unknown"),
                    "backbone": getattr(self.model.config, "backbone", "unknown"),
                    "device": str(self.device),
                    "latency_seconds": round(elapsed / len(outputs), 4),
                    "input_resolution": (
                        left_tensor.shape[-2],
                        left_tensor.shape[-1],
                    ),
                    "max_disparity": float(
                        disparity.max().item() if isinstance(disparity, torch.Tensor)
                        else max(d.max().item() for d in disparity)
                    ),
                    "focal_length": focal_length,
                    "baseline": baseline,
                }
            )

        return outputs[0] if len(outputs) == 1 else outputs


def pipeline(
    task: str = "stereo-matching",
    model: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> StereoPipeline:
    """Factory function to create a stereo matching pipeline.

    Usage::

        from stereo_matching import pipeline

        pipe = pipeline("stereo-matching", model="raft-stereo")
        result = pipe("left.jpg", "right.jpg")

    Args:
        task: Task name (must be "stereo-matching").
        model: Model identifier (e.g. "raft-stereo") or local checkpoint path.
        device: Device string. Auto-detected if None.
        **kwargs: Additional arguments passed to model.from_pretrained().

    Returns:
        Configured StereoPipeline.
    """
    if task != "stereo-matching":
        raise ValueError(f"Unsupported task '{task}'. Only 'stereo-matching' is supported.")

    if model is None:
        raise ValueError("You must specify a model identifier (e.g. model='raft-stereo').")

    from .models.auto import AutoStereoModel, AutoProcessor

    loaded_model = AutoStereoModel.from_pretrained(model, device=device, **kwargs)
    processor = AutoProcessor.from_pretrained(model)

    return StereoPipeline(model=loaded_model, processor=processor, device=device)
