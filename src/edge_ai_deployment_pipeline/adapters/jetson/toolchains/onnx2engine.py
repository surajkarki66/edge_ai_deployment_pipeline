# ONNX → TensorRT engine builder (adapted from Ultralytics).
# Copyright (C) Ultralytics Inc.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

import json
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def _trt_major_version(trt: object) -> int:
    return int(getattr(trt, "__version__", "0").split(".", 1)[0])


def _set_builder_flag_if_available(config: object, trt: object, flag_name: str) -> bool:
    flag = getattr(getattr(trt, "BuilderFlag", None), flag_name, None)
    if flag is None:
        return False
    config.set_flag(flag)  # type: ignore[attr-defined]
    return True


def _apply_precision_config(
    *,
    trt: object,
    config: object,
    half: bool,
    int8: bool,
    dataset: object | None,
    onnx_file: str,
    dla: int | None,
    prefix: str,
) -> tuple[bool, bool]:
    """Configure FP16/INT8 flags when supported; fall back to FP32 on TensorRT 11+."""
    trt_major = _trt_major_version(trt)
    trt_version = getattr(trt, "__version__", "unknown")
    supports_legacy_precision = trt_major < 11

    if int8 and not supports_legacy_precision:
        logger.warning(
            "%sTensorRT %s removed BuilderFlag.INT8 and implicit calibration. "
            "Building an FP32 engine from the ONNX model instead. "
            "For INT8 on TensorRT 11+, quantize the ONNX model with NVIDIA ModelOpt "
            "before engine build, or use TensorRT 10.x.",
            prefix,
            trt_version,
        )
        int8 = False

    if half and not supports_legacy_precision:
        logger.warning(
            "%sTensorRT %s removed BuilderFlag.FP16. "
            "Building an FP32 engine from the ONNX model instead. "
            "For FP16 on TensorRT 11+, run NVIDIA ModelOpt AutoCast on the ONNX model first.",
            prefix,
            trt_version,
        )
        half = False

    if int8:
        if dataset is None:
            raise ValueError(
                "INT8 TensorRT export requires calibration_data (image directory or .txt list)"
            )
        if not _set_builder_flag_if_available(config, trt, "INT8"):
            raise RuntimeError(
                f"TensorRT {trt_version} does not expose BuilderFlag.INT8; cannot build INT8 engine"
            )
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # type: ignore[attr-defined]

        class EngineCalibrator(trt.IInt8Calibrator):  # type: ignore[attr-defined]
            """Custom INT8 calibrator for TensorRT engine optimization."""

            def __init__(
                self,
                dataset: object,
                cache: str = "",
            ) -> None:
                trt.IInt8Calibrator.__init__(self)  # type: ignore[attr-defined]
                self.dataset = dataset
                self.data_iter = iter(dataset)
                self.batch = getattr(dataset, "batch_size", 1)
                self.algo = (
                    trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2  # type: ignore[attr-defined]
                    if dla is not None
                    else trt.CalibrationAlgoType.MINMAX_CALIBRATION  # type: ignore[attr-defined]
                )
                self.cache = Path(cache)

            def get_algorithm(self) -> object:
                return self.algo

            def get_batch_size(self) -> int:
                return self.batch or 1

            def get_batch(self, names: object) -> list[int] | None:
                try:
                    batch = next(self.data_iter)
                    if isinstance(batch, dict):
                        batch = batch["img"] / 255.0
                    if batch.device.type == "cpu":
                        batch = batch.to("cuda")
                    return [int(batch.contiguous().data_ptr())]
                except StopIteration:
                    return None

            def read_calibration_cache(self) -> bytes | None:
                if self.cache.exists() and self.cache.suffix == ".cache":
                    return self.cache.read_bytes()
                return None

            def write_calibration_cache(self, cache: bytes) -> None:
                _ = self.cache.write_bytes(cache)

        if hasattr(config, "int8_calibrator"):
            config.int8_calibrator = EngineCalibrator(  # type: ignore[attr-defined]
                dataset=dataset,
                cache=str(Path(onnx_file).with_suffix(".cache")),
            )
        else:
            raise RuntimeError(
                f"TensorRT {trt_version} does not support int8_calibrator on IBuilderConfig"
            )

    elif half:
        if not _set_builder_flag_if_available(config, trt, "FP16"):
            raise RuntimeError(
                f"TensorRT {trt_version} does not expose BuilderFlag.FP16; cannot build FP16 engine"
            )

    return half, int8


def onnx2engine(
    onnx_file: str,
    output_file: Path | str | None = None,
    workspace: int | None = None,
    half: bool = False,
    int8: bool = False,
    dynamic: bool = False,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dla: int | None = None,
    dataset=None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "",
) -> str:
    """Export a model to TensorRT engine format.

    Args:
        onnx_file (str): Path to the ONNX file to be converted.
        output_file (Path | str | None): Path to save the generated TensorRT engine file.
        workspace (int | None): Workspace size in GB for TensorRT.
        half (bool, optional): Enable FP16 precision.
        int8 (bool, optional): Enable INT8 precision.
        dynamic (bool, optional): Enable dynamic input shapes.
        shape (tuple[int, int, int, int], optional): 4D image input shape (matches ONNX layout).
        dla (int | None): DLA core to use (Jetson devices only).
        dataset (optional): Iterable yielding CUDA float32 input batches for INT8 calibration.
        metadata (dict | None): Metadata to include in the engine file.
        verbose (bool, optional): Enable verbose logging.
        prefix (str, optional): Prefix for log messages.

    Returns:
        (str): Path to the exported engine file.

    Raises:
        ValueError: If DLA is enabled on non-Jetson devices or required precision is not set.
        RuntimeError: If the ONNX file cannot be parsed.

    Notes:
        TensorRT version compatibility is handled for workspace size and engine building.
        INT8 calibration requires a dataset and generates a calibration cache.
        Metadata is serialized and written to the engine file if provided.
    """
    import tensorrt as trt

    output_file = output_file or Path(onnx_file).with_suffix(".engine")

    log = trt.Logger(trt.Logger.INFO)
    if verbose:
        log.min_severity = trt.Logger.Severity.VERBOSE

    # Engine builder
    builder = trt.Builder(log)
    config = builder.create_builder_config()
    workspace_bytes = int((workspace or 0) * (1 << 30))
    trt_major = _trt_major_version(trt)
    is_trt10_plus = trt_major >= 10
    if is_trt10_plus and workspace_bytes > 0:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif workspace_bytes > 0:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace_bytes
    if is_trt10_plus:
        # TensorRT 10+ always uses explicit batch; EXPLICIT_BATCH flag was removed.
        network = builder.create_network()
    else:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    half = getattr(builder, "platform_has_fast_fp16", True) and half
    int8 = getattr(builder, "platform_has_fast_int8", True) and int8
    half, int8 = _apply_precision_config(
        trt=trt,
        config=config,
        half=half,
        int8=int8,
        dataset=dataset,
        onnx_file=onnx_file,
        dla=dla,
        prefix=prefix,
    )

    # Optionally switch to DLA if enabled
    if dla is not None:
        logger.info(f"{prefix} enabling DLA on core {dla}...")
        if not half and not int8:
            raise ValueError(
                "DLA requires either 'half=True' (FP16) or 'int8=True' (INT8) to be enabled. Please enable one of them and try again."
            )
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    # Read ONNX file
    parser = trt.OnnxParser(network, log)
    if not parser.parse_from_file(onnx_file):
        raise RuntimeError(f"failed to load ONNX file: {onnx_file}")

    # Network inputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        logger.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        logger.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    def _dim_is_dynamic(d: int) -> bool:
        return d == -1

    has_dynamic_input = any(
        _dim_is_dynamic(inp.shape[i]) for inp in inputs for i in range(len(inp.shape))
    )
    if dynamic or has_dynamic_input:
        profile = builder.create_optimization_profile()
        n, d1, d2, d3 = shape
        # NHWC when channel dim is last and small
        if d3 <= 4 and d1 > d3:
            min_shape = (n, 32, 32, d3)
            max_shape = (n, d2 * 2, d2 * 2, d3)
        else:
            min_shape = (n, d1, 32, 32)
            max_shape = (n, d1, d2 * 2, d3 * 2)
        for inp in inputs:
            profile.set_shape(inp.name, min=min_shape, opt=shape, max=max_shape)
        config.add_optimization_profile(profile)
        if int8 and trt_major < 10:  # deprecated in TensorRT 10+, causes internal errors
            config.set_calibration_profile(profile)

    logger.info(
        f"{prefix} building {'INT8' if int8 else 'FP' + ('16' if half else '32')} engine as {output_file}"
    )

    # Write file
    if is_trt10_plus:
        # TensorRT 10+ returns bytes directly, not a context manager
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed, check logs for errors")
        with open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine)
    else:
        with builder.build_engine(network, config) as engine, open(output_file, "wb") as t:
            if metadata is not None:
                meta = json.dumps(metadata)
                t.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                t.write(meta.encode())
            t.write(engine.serialize())
    return str(output_file)
