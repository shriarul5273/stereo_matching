import numpy as np
import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from stereo_matching.configuration_utils import BaseStereoConfig  # noqa: E402
from stereo_matching.export import export_onnx  # noqa: E402
from stereo_matching.modeling_utils import BaseStereoModel  # noqa: E402
from stereo_matching.quantization import quantize_model, quantize_onnx  # noqa: E402


class TinyExportableStereoModel(BaseStereoModel):
    def __init__(self):
        super().__init__(BaseStereoConfig(input_size=8))
        self.features = nn.Conv2d(6, 4, kernel_size=1)
        self.head = nn.Linear(4, 1)

    def forward(self, left, right):
        features = self.features(torch.cat([left, right], dim=1))
        return self.head(features.permute(0, 2, 3, 1)).squeeze(-1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyExportableStereoModel().eval()


def test_export_has_two_named_inputs_and_verifies(model, tmp_path):
    output = tmp_path / "stereo.onnx"

    result = export_onnx(
        model,
        output,
        input_height=8,
        input_width=12,
        verify=True,
    )

    assert result == output
    graph = onnx.load(str(output)).graph
    assert [value.name for value in graph.input] == ["left_values", "right_values"]
    assert [value.name for value in graph.output] == ["disparity"]


def test_dynamic_batch_runs_with_another_batch_size(model, tmp_path):
    output = export_onnx(
        model,
        tmp_path / "dynamic.onnx",
        input_height=8,
        input_width=12,
        dynamic_batch=True,
    )
    session = onnxruntime.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    )
    inputs = {
        "left_values": np.random.randn(3, 3, 8, 12).astype(np.float32),
        "right_values": np.random.randn(3, 3, 8, 12).astype(np.float32),
    }

    disparity = session.run(["disparity"], inputs)[0]

    assert disparity.shape == (3, 8, 12)


def test_int8_onnx_quantization(model, tmp_path):
    source = export_onnx(
        model,
        tmp_path / "source.onnx",
        input_height=8,
        input_width=12,
    )
    output = tmp_path / "quantized.onnx"

    try:
        result = quantize_onnx(source, output, weight_type="int8", verify=True)
    except Exception as exc:
        if "ConvInteger" in str(exc) or "NOT_IMPLEMENTED" in str(exc):
            pytest.skip(f"ONNX Runtime lacks required INT8 Conv support: {exc}")
        raise

    assert result == output
    assert output.is_file()


def test_uint8_onnx_quantization_without_verification(model, tmp_path):
    source = export_onnx(
        model,
        tmp_path / "source.onnx",
        input_height=8,
        input_width=12,
    )
    output = tmp_path / "quantized.onnx"

    assert quantize_onnx(source, output, weight_type="uint8", verify=False) == output
    assert output.is_file()


def test_dynamic_int8_cannot_be_exported(model, tmp_path):
    quantized = quantize_model(model, "int8")

    with pytest.raises(ValueError, match="cannot be exported"):
        export_onnx(
            quantized,
            tmp_path / "invalid.onnx",
            input_height=8,
            input_width=12,
        )


def test_model_export_convenience_method(model, tmp_path):
    output = model.export_onnx(
        str(tmp_path / "method.onnx"),
        input_height=8,
        input_width=12,
    )

    assert output.is_file()
