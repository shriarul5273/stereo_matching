import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from stereo_matching.configuration_utils import BaseStereoConfig
from stereo_matching.modeling_utils import BaseStereoModel
from stereo_matching.pipeline_utils import StereoPipeline
from stereo_matching.processing_utils import StereoProcessor
from stereo_matching.quantization import quantize_model, quantize_onnx


class TinyStereoModel(BaseStereoModel):
    def __init__(self):
        super().__init__(BaseStereoConfig(input_size=8))
        self.features = nn.Conv2d(6, 4, kernel_size=1)
        self.head = nn.Linear(4, 1)

    def forward(self, left, right):
        features = self.features(torch.cat([left, right], dim=1))
        features = features.permute(0, 2, 3, 1)
        return self.head(features).squeeze(-1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyStereoModel().eval()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_reduced_precision_casts_in_place(model, name, expected):
    result = quantize_model(model, dtype=name)

    assert result is model
    assert next(model.parameters()).dtype == expected


def test_model_quantize_convenience_method(model):
    assert model.quantize("fp16") is model
    assert next(model.parameters()).dtype == torch.float16


def test_int8_returns_new_cpu_model_and_preserves_original(model):
    original = next(model.parameters()).detach().clone()

    result = quantize_model(model, dtype="int8")

    assert result is not model
    assert next(result.parameters()).device.type == "cpu"
    assert torch.equal(next(model.parameters()), original)
    assert "quantized" in type(result.head).__module__


def test_int8_forward(model):
    result = quantize_model(model, dtype="int8")
    left = torch.randn(1, 3, 8, 12)
    right = torch.randn(1, 3, 8, 12)

    with torch.no_grad():
        disparity = result(left, right)

    assert disparity.shape == (1, 8, 12)
    assert torch.isfinite(disparity).all()


def test_unknown_precision_raises(model):
    with pytest.raises(ValueError, match="Unknown dtype"):
        quantize_model(model, dtype="int4")


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_pipeline_casts_inputs_for_reduced_precision(model, precision):
    quantize_model(model, precision)
    pipeline = StereoPipeline(
        model=model,
        processor=StereoProcessor(model.config),
        device="cpu",
    )
    left = np.zeros((8, 12, 3), dtype=np.uint8)
    right = np.ones((8, 12, 3), dtype=np.uint8)

    try:
        result = pipeline(left, right, colorize=False)
    except RuntimeError as exc:
        pytest.skip(f"This torch CPU build lacks {precision} kernels: {exc}")

    assert result.disparity.dtype == np.float32
    assert result.disparity.shape == (8, 12)


def test_onnx_quantization_verifies_by_default():
    assert inspect.signature(quantize_onnx).parameters["verify"].default is True


def test_onnx_quantization_rejects_unknown_type(tmp_path):
    source = tmp_path / "missing.onnx"
    with pytest.raises(ValueError, match="Unknown weight_type"):
        quantize_onnx(source, tmp_path / "output.onnx", weight_type="int4")
