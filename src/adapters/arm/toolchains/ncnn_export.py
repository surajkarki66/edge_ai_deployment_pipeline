"""Export PyTorch model to NCNN via PNNX.

Adapted from Ultralytics export utilities.
See: https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

from pathlib import Path

import torch


def torch2ncnn(
    model: torch.nn.Module,
    im: torch.Tensor,
    output_dir: Path | str,
    half: bool = False,
    device: str | torch.device | None = None,
) -> str:
    """Export a PyTorch model to NCNN format using PNNX.

    Logic adapted from Ultralytics (torch2ncnn / PNNX export flow).

    Args:
        model: PyTorch model in eval mode.
        im: Example input tensor for tracing, e.g. torch.randn(1, 3, 224, 224).
        output_dir: Directory for model.ncnn.param and model.ncnn.bin.
        half: Use FP16 export.
        device: "cpu", "cuda", etc. Defaults to im's device or "cpu".

    Returns:
        Path to output_dir as a string.
    """
    import pnnx

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_type = im.device.type
    elif isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)

    ncnn_args = dict(
        ncnnparam=(output_dir / "model.ncnn.param").as_posix(),
        ncnnbin=(output_dir / "model.ncnn.bin").as_posix(),
        ncnnpy=(output_dir / "model_ncnn.py").as_posix(),
    )
    pnnx_args = dict(
        ptpath=(output_dir / "model.pt").as_posix(),
        pnnxparam=(output_dir / "model.pnnx.param").as_posix(),
        pnnxbin=(output_dir / "model.pnnx.bin").as_posix(),
        pnnxpy=(output_dir / "model_pnnx.py").as_posix(),
        pnnxonnx=(output_dir / "model.pnnx.onnx").as_posix(),
    )

    model.eval()
    pnnx.export(model, inputs=im, **ncnn_args, **pnnx_args, fp16=half, device=device_type)

    for name in ("debug.bin", "debug.param", "debug2.bin", "debug2.param"):
        (output_dir / name).unlink(missing_ok=True)
    for path in pnnx_args.values():
        Path(path).unlink(missing_ok=True)

    return str(output_dir)
