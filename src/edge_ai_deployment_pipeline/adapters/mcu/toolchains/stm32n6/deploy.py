from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys


from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOOLCHAIN_ROOT = Path(__file__).resolve().parent
QUANTIZE_SCRIPT = TOOLCHAIN_ROOT / "quantize_model.py"
COMPILE_SCRIPT = TOOLCHAIN_ROOT / "compile_model.py"
TFLITE_NAME = "quantized_model.tflite"


def _abs(path: Path) -> Path:
    """Resolve paths for subprocesses that run with cwd=TOOLCHAIN_ROOT."""
    return path.expanduser().resolve()


def _run_script(script: Path, args: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"STM32N6 script missing: {script}")
    cmd = [sys.executable, str(script), *args]
    logger.info("Running: %s", " ".join(shlex.quote(a) for a in cmd))
    subprocess.run(cmd, cwd=TOOLCHAIN_ROOT, check=True)


def _color_mode(channels: int) -> str:
    return "grayscale" if channels == 1 else "rgb"


def _image_size_arg(height: int, width: int) -> str:
    return f"{height},{width}"


def run_quantize(
    *,
    model_path: Path,
    calib_dir: Path,
    workspace: Path,
    channels: int,
    height: int,
    width: int,
    config: dict[str, Any],
) -> Path:
    """Keras → TFLite PTQ. Returns path to quantized .tflite."""
    model_path = _abs(model_path)
    calib_dir = _abs(calib_dir)
    workspace = _abs(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    export_dir = str(config.get("export_dir", "quantized_models"))

    args = [
        "--model",
        str(model_path),
        "--calib-dir",
        str(calib_dir),
        "--output-dir",
        str(workspace),
        "--export-dir",
        export_dir,
        "--image-size",
        _image_size_arg(height, width),
        "--color-mode",
        str(config.get("color_mode", _color_mode(channels))),
        "--granularity",
        str(config.get("granularity", "per_channel")),
        "--input-type",
        str(config.get("input_type", "uint8")),
        "--output-type",
        str(config.get("output_type", "float")),
    ]

    for flag, key in (
        ("--interpolation", "interpolation"),
        ("--aspect-ratio", "aspect_ratio"),
    ):
        if config.get(key):
            args.extend([flag, str(config[key])])

    if config.get("optimize"):
        args.append("--optimize")

    _run_script(QUANTIZE_SCRIPT, args)

    tflite = workspace / export_dir / TFLITE_NAME
    if not tflite.is_file():
        flat = workspace / TFLITE_NAME
        if flat.is_file():
            return flat
        raise FileNotFoundError(f"TFLite output not found under {workspace}")
    return tflite


def run_compile(
    *,
    tflite_path: Path,
    workspace: Path,
    c_project: Path,
    class_names: list[str],
    config: dict[str, Any],
) -> Path:
    """TFLite → STEdgeAI generated C code."""
    tflite_path = _abs(tflite_path)
    workspace = _abs(workspace)
    c_project = _abs(c_project)
    compile_workspace = workspace / "compiled"
    compile_workspace.mkdir(parents=True, exist_ok=True)

    board = str(config.get("board", "STM32N6570-DK"))
    args = [
        "--model",
        str(tflite_path),
        "--output-dir",
        str(compile_workspace.resolve()),
        "--c-project",
        str(c_project),
        "--board",
        board,
        "--class-names",
        ",".join(class_names),
        "--aspect-ratio",
        str(config.get("aspect_ratio", "crop")),
        "--color-mode",
        str(config.get("compile_color_mode", config.get("color_mode", "rgb"))),
        "--optimization",
        str(config.get("optimization", "balanced")),
    ]

    stedgeai = config.get("stedgeai") or os.environ.get("STM32_AI_EXE", "")
    if stedgeai:
        args.extend(["--stedgeai", str(stedgeai)])

    cube_ide = config.get("cube_ide") or os.environ.get("STM32_CUBE_IDE_EXE", "")
    if cube_ide:
        args.extend(["--cube-ide", str(cube_ide)])

    if config.get("build_firmware"):
        args.append("--build")
        serial = config.get("stlink_serial")
        if serial:
            args.extend(["--stlink-serial", str(serial)])

    _run_script(COMPILE_SCRIPT, args)

    generated = compile_workspace / "generated"
    if not generated.is_dir():
        raise FileNotFoundError(f"STEdgeAI output missing: {generated}")
    return generated


def export_keras_to_stm32n6(
    *,
    model_path: Path,
    calib_dir: Path,
    c_project: Path | None,
    class_names: list[str],
    channels: int,
    height: int,
    width: int,
    config: dict[str, Any],
) -> tuple[Path, Path | None]:
    """Full flow: quantize → optional compile. Returns (tflite_path, generated_dir)."""
    workspace = _abs(
        Path(str(config.get("stm32_workspace", config.get("output_dir", "."))))
    )
    platform_id = str(config.get("platform_id", config.get("board_id", "mcu")))
    board_workspace = _abs(workspace / platform_id / "stm32n6_workspace")
    board_workspace.mkdir(parents=True, exist_ok=True)

    tflite = run_quantize(
        model_path=model_path,
        calib_dir=calib_dir,
        workspace=board_workspace,
        channels=channels,
        height=height,
        width=width,
        config=config,
    )

    if config.get("quantize_only"):
        return tflite, None
    if c_project is None:
        raise ValueError("c_project_path is required unless quantize_only is set")

    generated = run_compile(
        tflite_path=tflite,
        workspace=board_workspace,
        c_project=c_project,
        class_names=class_names,
        config=config,
    )
    return tflite, generated
