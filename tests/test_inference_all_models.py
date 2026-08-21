from pathlib import Path

import numpy as np
import pytest

from stereo_matching import MODEL_REGISTRY
from stereo_matching.output import StereoOutput


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "example1"


@pytest.mark.slow
@pytest.mark.parametrize("variant_id", MODEL_REGISTRY.list_variants())
def test_pretrained_variant_inference(variant_id):
    from stereo_matching import pipeline

    matcher = pipeline("stereo-matching", model=variant_id, device="cpu")
    result = matcher(
        str(ASSET_ROOT / "left.png"),
        str(ASSET_ROOT / "right.png"),
        colorize=False,
    )

    assert isinstance(result, StereoOutput)
    assert result.disparity.ndim == 2
    assert np.isfinite(result.disparity).all()
