"""PyTorch → ONNX export helpers.

Adapted from Ultralytics export utilities.
See: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@contextmanager
def arange_patch(*, dynamic: bool = False, half: bool = False, fmt: str = ""):
    """Workaround for ONNX torch.arange incompatibility with FP16."""
    if dynamic and half and fmt == "onnx":
        func = torch.arange

        def arange(*args, dtype=None, **kwargs):
            return func(*args, **kwargs).to(dtype)

        torch.arange = arange
        yield
        torch.arange = func
    else:
        yield


def torch2onnx(
    model: torch.nn.Module,
    im: torch.Tensor | tuple[torch.Tensor, ...],
    output_file: Path | str,
    opset: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic: dict | None = None,
) -> str:
    """Export a PyTorch model to ONNX format."""
    if input_names is None:
        input_names = ["images"]
    if output_names is None:
        output_names = ["output0"]

    torch.onnx.export(
        model,
        im,
        str(output_file),
        verbose=False,
        opset_version=opset,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        dynamo=False,
    )
    return str(output_file)


def export_onnx(
    model: torch.nn.Module,
    output_file: Path | str,
    im: torch.Tensor,
    *,
    dynamic: bool = False,
    simplify: bool = True,
    convert_onnx_fp16: bool = False,
) -> str:
    """Export model to ONNX, optionally simplify and convert ONNX weights to FP16."""
    import onnx

    output_file = Path(output_file)
    logger.info("Exporting ONNX with onnx %s", onnx.__version__)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}

    with arange_patch(dynamic=dynamic, half=convert_onnx_fp16, fmt="onnx"):
        torch2onnx(
            model,
            im,
            output_file,
            opset=20,
            input_names=["images"],
            output_names=["output0"],
            dynamic=dynamic_axes,
        )

    model_onnx = onnx.load(output_file)

    if simplify:
        try:
            import onnxslim

            logger.info("Slimming ONNX with onnxslim %s", onnxslim.__version__)
            model_onnx = onnxslim.slim(model_onnx)
        except Exception as exc:
            logger.warning("ONNX simplifier failed: %s", exc)

    if getattr(model_onnx, "ir_version", 0) > 10:
        logger.info("Limiting ONNX IR version %s to 10", model_onnx.ir_version)
        model_onnx.ir_version = 10

    if convert_onnx_fp16:
        try:
            from onnxruntime.transformers import float16

            logger.info("Converting ONNX graph to FP16")
            model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
        except Exception as exc:
            logger.warning("ONNX FP16 conversion failed: %s", exc)

    onnx.save(model_onnx, output_file)
    return str(output_file.resolve())
