"""Precision reduction for PyTorch and exported ONNX stereo models.

PyTorch dynamic INT8 and ONNX quantization are separate workflows. Dynamic
INT8 replaces PyTorch ``nn.Linear`` modules and is intended for direct CPU
inference; export the original floating-point model before calling
``quantize_onnx`` when a quantized ONNX file is required.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CAST_DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
_ONNX_WEIGHT_TYPES = {"int8", "uint8"}


def quantize_model(model: nn.Module, dtype: str = "fp16") -> nn.Module:
    """Cast or dynamically quantize a PyTorch stereo model.

    ``fp16``/``float16`` and ``bf16``/``bfloat16`` cast the supplied model
    in-place and return it. ``int8`` deep-copies the model, moves the copy to
    CPU, and dynamically quantizes ``nn.Linear`` layers; the original model is
    unchanged. Dynamic INT8 does not quantize convolution layers and is not
    exportable through ``torch.onnx.export``.
    """
    normalized = dtype.lower()
    if normalized in _CAST_DTYPES:
        result = model.to(dtype=_CAST_DTYPES[normalized])
        setattr(result, "_stereo_precision", normalized)
        return result

    if normalized == "int8":
        cpu_model = copy.deepcopy(model).to("cpu")
        try:
            from torch.ao.quantization import quantize_dynamic
        except ImportError:
            quantize_dynamic = torch.quantization.quantize_dynamic

        result = quantize_dynamic(cpu_model, {nn.Linear}, dtype=torch.qint8)
        setattr(result, "_stereo_precision", "int8_dynamic")
        return result

    raise ValueError(
        f"Unknown dtype {dtype!r}. Available: fp16, bf16, int8 "
        "(float16 and bfloat16 aliases are also accepted)."
    )


def quantize_onnx(
    onnx_path: str | Path,
    output_path: str | Path,
    weight_type: str = "int8",
    verify: bool = True,
    atol: float = 5e-2,
    rtol: float = 5e-2,
    per_channel: bool = False,
    reduce_range: bool = False,
) -> Path:
    """Dynamically quantize an existing ONNX model with ONNX Runtime.

    The source must be a floating-point ONNX model, normally created with
    :func:`stereo_matching.export.export_onnx`. No calibration dataset is
    required. Verification compares both-input inference against the original
    graph and defaults to enabled because quantization accuracy is
    model-dependent.
    """
    normalized = weight_type.lower()
    if normalized not in _ONNX_WEIGHT_TYPES:
        raise ValueError(
            f"Unknown weight_type {weight_type!r}. Available: "
            f"{sorted(_ONNX_WEIGHT_TYPES)}"
        )

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "quantize_onnx() requires ONNX Runtime: pip install onnxruntime"
        ) from exc

    source = Path(onnx_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("output_path must differ from onnx_path")

    destination.parent.mkdir(parents=True, exist_ok=True)
    type_map = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    quantize_dynamic(
        str(source),
        str(destination),
        weight_type=type_map[normalized],
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    logger.info("Quantized ONNX model (%s) written to %s", normalized, destination)

    if verify:
        _verify_onnx_quantization(source, destination, atol=atol, rtol=rtol)

    return destination


def _ort_random_inputs(session) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("ONNX verification requires numpy") from exc

    inputs: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    for metadata in session.get_inputs():
        shape = [dimension if isinstance(dimension, int) else 1 for dimension in metadata.shape]
        if metadata.type == "tensor(float16)":
            dtype = np.float16
        elif metadata.type == "tensor(float)":
            dtype = np.float32
        else:
            raise TypeError(
                f"Unsupported ONNX input type {metadata.type!r} for verification. "
                "BF16 graphs generally require provider-specific input handling; "
                "set verify=False and validate them in the deployment runtime."
            )
        inputs[metadata.name] = rng.standard_normal(shape).astype(dtype)
    return inputs


def _verify_onnx_quantization(
    original_path: Path,
    quantized_path: Path,
    atol: float,
    rtol: float,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "verify=True requires ONNX Runtime: pip install onnxruntime"
        ) from exc

    original = ort.InferenceSession(
        str(original_path), providers=["CPUExecutionProvider"]
    )
    inputs = _ort_random_inputs(original)
    expected = original.run(None, inputs)

    quantized = ort.InferenceSession(
        str(quantized_path), providers=["CPUExecutionProvider"]
    )
    actual = quantized.run(None, inputs)

    if len(expected) != len(actual):
        raise AssertionError(
            f"ONNX output count changed after quantization: {len(expected)} != {len(actual)}"
        )
    for expected_output, actual_output in zip(expected, actual):
        np.testing.assert_allclose(
            expected_output,
            actual_output,
            atol=atol,
            rtol=rtol,
            err_msg=(
                "Quantized ONNX output diverges from the floating-point model. "
                "Quantization is model-dependent; try per-channel settings, "
                "uint8, or a calibration-based workflow."
            ),
        )

    logger.info(
        "ONNX quantization verified within atol=%s and rtol=%s", atol, rtol
    )
