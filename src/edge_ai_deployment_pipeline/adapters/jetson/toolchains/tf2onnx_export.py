from __future__ import annotations

import argparse
import logging

from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OPSET = 13
DEFAULT_INPUT_NAME = "input"


def keras_input_signature(
    channels: int,
    height: int,
    width: int,
    *,
    batch: int = 1,
    name: str = DEFAULT_INPUT_NAME,
    channels_last: bool = True,
) -> list[Any]:
    """Build ``tf.TensorSpec`` list for tf2onnx (fixed batch=1 for TensorRT)."""
    import tensorflow as tf

    if channels_last:
        shape = [batch, height, width, channels]
    else:
        shape = [batch, channels, height, width]
    return [tf.TensorSpec(shape, tf.float32, name=name)]


def export_keras_to_onnx(
    model: Any,
    output_path: Path | str,
    *,
    channels: int,
    height: int,
    width: int,
    input_name: str = DEFAULT_INPUT_NAME,
    opset: int = DEFAULT_OPSET,
    channels_last: bool = True,
) -> Path:
    """Export a ``tf.keras.Model`` to ONNX via ``tf2onnx.convert.from_keras``."""
    import tf2onnx

    _require_image_input_shape(model, channels=channels, height=height, width=width)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_signature = keras_input_signature(
        channels,
        height,
        width,
        name=input_name,
        channels_last=channels_last,
    )
    logger.info(
        "Converting Keras model to ONNX (opset=%d, shape=%s)",
        opset,
        input_signature[0].shape,
    )
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=opset,
    )
    output_path.write_bytes(onnx_model.SerializeToString())
    logger.info("Wrote ONNX model to %s", output_path.resolve())
    return output_path.resolve()


def _require_image_input_shape(model: Any, *, channels: int, height: int, width: int) -> None:
    """Reject non-vision Keras models (e.g. tf2onnx tutorial Dense on 3x3)."""
    inp_shape = getattr(model, "input_shape", None)
    if inp_shape is None:
        return
    if isinstance(inp_shape, list):
        inp_shape = inp_shape[0]
    if not inp_shape:
        return
    static = [d for d in inp_shape if d is not None]
    if len(static) < 3:
        raise ValueError(
            f"Model input_shape {inp_shape} is not a 4D image tensor (N,H,W,C or N,C,H,W). "
            "This pipeline only supports vision .keras/.h5 models."
        )
    if len(static) == 3 and static[-1] not in (channels, 1, 3) and static[1] not in (height, width):
        raise ValueError(
            f"Model input_shape {inp_shape} does not look like an image input for "
            f"--input-shape {channels},{height},{width}."
        )


def export_to_onnx(
    model: Any,
    source_path: Path | str,
    output_path: Path | str,
    *,
    channels: int,
    height: int,
    width: int,
    original_format: str = "",
    opset: int = DEFAULT_OPSET,
    input_name: str = DEFAULT_INPUT_NAME,
    channels_last: bool = True,
) -> Path:
    """Export a Keras in-memory model or .keras/.h5 file to ONNX."""
    import tensorflow as tf

    source_path = Path(source_path)
    fmt = original_format.lower()

    if fmt in ("keras", "h5") or source_path.suffix.lower() in (".keras", ".h5"):
        if isinstance(model, tf.keras.Model):
            keras_model = model
        else:
            keras_model = tf.keras.models.load_model(str(source_path), compile=False)
        return export_keras_to_onnx(
            keras_model,
            output_path,
            channels=channels,
            height=height,
            width=width,
            input_name=input_name,
            opset=opset,
            channels_last=channels_last,
        )

    raise ValueError(
        f"Unsupported TensorFlow/Keras source for ONNX export: {source_path} ({fmt}). "
        "Use a .keras or .h5 file."
    )


def resolve_trt_shape_from_onnx(
    onnx_path: Path | str,
    *,
    fallback: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Infer TensorRT profile shape from the first 4D ONNX input (same layout as ONNX)."""
    import onnx

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        if fallback:
            return fallback
        raise ValueError(f"No inputs in ONNX graph: {onnx_path}")

    dims = model.graph.input[0].type.tensor_type.shape.dim
    values: list[int] = []
    for dim in dims:
        if dim.dim_value > 0:
            values.append(int(dim.dim_value))
        else:
            values.append(1)

    if len(values) != 4:
        if fallback:
            return fallback
        raise ValueError(f"Expected 4D image ONNX input, got {values}")

    return values[0], values[1], values[2], values[3]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Keras (.keras/.h5) to ONNX using tf2onnx.",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to .keras or .h5 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="Output ONNX file (default: model.onnx)",
    )
    parser.add_argument(
        "--input-shape",
        required=True,
        metavar="C,H,W",
        help="Input shape for vision models (channels,height,width)",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME)
    parser.add_argument(
        "--channels-first",
        action="store_true",
        help="Use NCHW TensorSpec instead of NHWC (default: NHWC)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = args.model.resolve()

    if not model_path.is_file():
        logger.error("Model file not found: %s", model_path)
        return 1

    parts = [int(x.strip()) for x in args.input_shape.split(",")]
    if len(parts) != 3:
        logger.error("input-shape must be C,H,W")
        return 1
    channels, height, width = parts

    try:
        import tensorflow as tf

        keras_model = tf.keras.models.load_model(str(model_path), compile=False)
        export_keras_to_onnx(
            keras_model,
            args.output,
            channels=channels,
            height=height,
            width=width,
            input_name=args.input_name,
            opset=args.opset,
            channels_last=not args.channels_first,
        )
        logger.info("Conversion succeeded: %s", args.output.resolve())
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
