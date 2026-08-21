"""ONNX export for two-input stereo matching models."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ONNX_EXPORT_SUPPORTS_DYNAMO = "dynamo" in inspect.signature(
    torch.onnx.export
).parameters


def _model_device_and_dtype(model: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.device, parameter.dtype
    for buffer in model.buffers():
        if buffer.is_floating_point():
            return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_height: int | None = None,
    input_width: int | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_spatial: bool = False,
    verify: bool = False,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Path:
    """Export ``forward(left, right) -> disparity`` to ONNX.

    The graph has two image inputs named ``left_values`` and ``right_values``
    and one output named ``disparity``. Input dimensions default to the model
    configuration's ``input_size`` and a square trace shape.
    """
    if getattr(model, "_stereo_precision", None) == "int8_dynamic":
        raise ValueError(
            "PyTorch dynamic INT8 models cannot be exported to ONNX. Export "
            "the floating-point model first, then call quantize_onnx()."
        )

    default_size = int(getattr(getattr(model, "config", None), "input_size", 384))
    height = default_size if input_height is None else input_height
    width = height if input_width is None else input_width
    if height <= 0 or width <= 0:
        raise ValueError("input_height and input_width must be positive")

    try:
        import onnx
    except ImportError as exc:
        raise ImportError("ONNX export requires: pip install onnx") from exc

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device, dtype = _model_device_and_dtype(model)
    left = torch.randn(1, 3, height, width, device=device, dtype=dtype)
    right = torch.randn(1, 3, height, width, device=device, dtype=dtype)

    with torch.no_grad():
        output = model(left, right)
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise TypeError(
            "ONNX export requires eval-mode forward(left, right) to return a "
            "Tensor with shape (batch, height, width)."
        )

    dynamic_axes = None
    if dynamic_batch or dynamic_spatial:
        input_axes = {}
        output_axes = {}
        if dynamic_batch:
            input_axes[0] = "batch"
            output_axes[0] = "batch"
        if dynamic_spatial:
            input_axes[2] = "height"
            input_axes[3] = "width"
            output_axes[1] = "height"
            output_axes[2] = "width"
        dynamic_axes = {
            "left_values": dict(input_axes),
            "right_values": dict(input_axes),
            "disparity": output_axes,
        }

    export_kwargs: dict[str, Any] = {}
    if _ONNX_EXPORT_SUPPORTS_DYNAMO:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            (left, right),
            str(destination),
            input_names=["left_values", "right_values"],
            output_names=["disparity"],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            **export_kwargs,
        )

    graph = onnx.load(str(destination)).graph
    if len(graph.input) != 2:
        raise RuntimeError(
            f"Exported stereo graph must have two inputs; found {len(graph.input)}"
        )

    logger.info("Exported ONNX model to %s", destination)
    if verify:
        _verify_onnx_export(
            model,
            destination,
            left=left,
            right=right,
            atol=atol,
            rtol=rtol,
        )
    return destination


def _verify_onnx_export(
    model: torch.nn.Module,
    onnx_path: str | Path,
    left: torch.Tensor,
    right: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "verify=True requires ONNX Runtime: pip install onnxruntime"
        ) from exc

    if left.dtype == torch.bfloat16 or right.dtype == torch.bfloat16:
        raise TypeError(
            "Automatic BF16 verification is not supported by NumPy/ONNX "
            "Runtime's standard CPU input path. Export with verify=False and "
            "validate using the target execution provider."
        )

    previous_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            expected = model(left, right).detach().cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32 = previous_tf32

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    inputs = {
        "left_values": left.detach().cpu().numpy(),
        "right_values": right.detach().cpu().numpy(),
    }
    actual = session.run(["disparity"], inputs)[0]
    np.testing.assert_allclose(
        expected,
        actual,
        atol=atol,
        rtol=rtol,
        err_msg="ONNX disparity does not match the PyTorch model output.",
    )
    logger.info("ONNX export verified within atol=%s and rtol=%s", atol, rtol)
