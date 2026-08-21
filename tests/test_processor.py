import numpy as np
import pytest
import torch

from stereo_matching.configuration_utils import BaseStereoConfig
from stereo_matching.output import StereoOutput
from stereo_matching.processing_utils import StereoProcessor


def test_preprocess_pair_shapes_match():
    processor = StereoProcessor(BaseStereoConfig(input_size=16))
    left = np.zeros((10, 20, 3), dtype=np.uint8)
    right = np.zeros((12, 24, 3), dtype=np.uint8)

    inputs = processor(left, right)

    assert inputs["left_values"].shape == (1, 3, 16, 32)
    assert inputs["right_values"].shape == inputs["left_values"].shape
    assert inputs["original_sizes"] == [(10, 20)]


def test_postprocess_restores_pixel_scale_and_depth():
    processor = StereoProcessor(BaseStereoConfig())
    disparity = torch.ones((1, 8, 16))

    result = processor.postprocess(
        disparity,
        original_sizes=[(16, 32)],
        colorize=False,
        focal_length=10.0,
        baseline=0.2,
    )

    assert isinstance(result, StereoOutput)
    assert result.disparity.shape == (16, 32)
    assert result.colored_disparity is None
    assert np.all(result.disparity == 2.0)
    assert np.allclose(result.depth, 1.0)


def test_invalid_array_shape_is_rejected():
    processor = StereoProcessor()

    with pytest.raises(ValueError, match="Expected"):
        processor._load_image(np.zeros((10, 20), dtype=np.uint8))
