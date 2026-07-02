from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Any

import yaml

from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.image_classification.compile_model import (
    resolve_stedgeai_path,
)
from edge_ai_deployment_pipeline.core.ultralytics_export import yolo_export_config

logger = logging.getLogger(__name__)

STM32_OD_APP_DIR = (
    Path(__file__).resolve().parent
    / "stm32ai-n6-yolov8-objectdetection"
    / "object_detection"
)
CONFIG_EXAMPLES_DIR = STM32_OD_APP_DIR / "config_file_examples"
DEFAULT_DEPLOYMENT_CONFIG = "deployment_n6_yolo11n_config"
RUNTIME_CONFIG_STEM = "runtime_deployment_config"


def stm32_od_app_dir() -> Path:
    if not STM32_OD_APP_DIR.is_dir():
        raise FileNotFoundError(f"STM32 object-detection app not found: {STM32_OD_APP_DIR}")
    return STM32_OD_APP_DIR


def resolve_deployment_config_name(config: dict[str, Any]) -> str:
    """Hydra config stem from platforms.json ``deployment_config_name``."""
    raw = str(config.get("deployment_config_name", DEFAULT_DEPLOYMENT_CONFIG)).strip()
    return raw.removesuffix(".yaml").removesuffix(".yml")


def resolve_od_model_type(config: dict[str, Any], source_path: Path) -> str:
    export_cfg = yolo_export_config(config)
    model_type = export_cfg.get("model_type")
    if model_type:
        return str(model_type)

    stem = source_path.stem.lower()
    if stem.startswith("yolov"):
        return stem
    if stem.startswith("yolo"):
        return f"yolov{stem[4:]}"

    raise ValueError(
        "Cannot infer STM32 OD model_type from weights name. "
        "Set yolo_export.model_type in platform config (e.g. yolov11n)."
    )


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_runtime_deploy_config(
    *,
    deploy_workspace: Path,
    base_config_name: str,
    tflite_path: Path,
    c_project_path: Path,
    class_names: list[str],
    model_type: str,
    config: dict[str, Any],
) -> Path:
    """Materialize a deploy config under *deploy_workspace* for stm32ai_main."""
    base_path = CONFIG_EXAMPLES_DIR / f"{base_config_name}.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Deployment config not found: {base_path}. "
            f"Check deployment_config_name in platforms.json ({base_config_name})."
        )

    tflite_path = tflite_path.resolve()
    deploy_workspace = deploy_workspace.resolve()
    c_project_path = c_project_path.resolve()

    if not tflite_path.is_file():
        raise FileNotFoundError(f"Quantized TFLite not found: {tflite_path}")
    if not c_project_path.is_dir():
        raise FileNotFoundError(f"C project not found: {c_project_path}")

    stedgeai = resolve_stedgeai_path(str(config.get("stedgeai") or os.environ.get("STM32_AI_EXE", "")))
    if stedgeai is None:
        raise FileNotFoundError(
            "STEdgeAI executable not found. Install STEdgeAI and set STM32_AI_EXE "
            "or pass stedgeai in the board config."
        )

    with base_path.open(encoding="utf-8") as handle:
        runtime_cfg = yaml.safe_load(handle)
    if not isinstance(runtime_cfg, dict):
        raise ValueError(f"Invalid deployment config YAML: {base_path}")

    board = str(config.get("board", "STM32N6570-DK"))
    optimization = str(config.get("optimization", "balanced"))
    aspect_ratio = str(config.get("aspect_ratio", "crop"))
    color_mode = str(config.get("color_mode", config.get("compile_color_mode", "rgb")))

    mlflow_db = (deploy_workspace / "mlflow.db").resolve()

    updates: dict[str, Any] = {
        "general": {
            "project_name": str(config.get("mlflow_project_name", "edge_ai_deploy")),
            "saved_models_dir": "saved_models",
        },
        "model": {
            "model_path": str(tflite_path),
            "model_type": model_type,
        },
        "dataset": {"class_names": class_names},
        "preprocessing": {
            "resizing": {"aspect_ratio": aspect_ratio},
            "color_mode": color_mode,
        },
        "tools": {
            "stedgeai": {
                "path_to_stedgeai": str(stedgeai),
                "optimization": optimization,
                "on_cloud": False,
            },
        },
        "deployment": {
            "c_project_path": str(c_project_path),
            "hardware_setup": {
                "serie": "STM32N6",
                "board": board,
            },
        },
        "hydra": {"run": {"dir": str(deploy_workspace)}},
        "mlflow": {"uri": f"sqlite:///{mlflow_db}"},
    }

    cube_ide = config.get("cube_ide") or os.environ.get("STM32_CUBE_IDE_EXE", "")
    if cube_ide:
        updates["tools"]["path_to_cubeIDE"] = str(Path(str(cube_ide)).resolve())

    postprocessing = config.get("od_postprocessing")
    if isinstance(postprocessing, dict):
        updates["postprocessing"] = {
            key: postprocessing[key]
            for key in (
                "confidence_thresh",
                "NMS_thresh",
                "IoU_eval_thresh",
                "max_detection_boxes",
            )
            if key in postprocessing
        }

    runtime_cfg = _deep_update(runtime_cfg, updates)
    deploy_workspace.mkdir(parents=True, exist_ok=True)
    runtime_path = deploy_workspace / f"{RUNTIME_CONFIG_STEM}.yaml"
    with runtime_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(runtime_cfg, handle, sort_keys=False)
    logger.info("Wrote STM32 OD runtime config: %s", runtime_path)
    return runtime_path


def run_stm32_od_deploy(
    *,
    tflite_path: Path,
    board_dir: Path,
    c_project_path: Path,
    class_names: list[str],
    source_path: Path,
    config: dict[str, Any],
) -> Path:
    """Invoke STM32 OD deploy; returns the Hydra output workspace under *board_dir*."""
    app_dir = stm32_od_app_dir()
    config_name = resolve_deployment_config_name(config)
    deploy_workspace = (board_dir / "stm32_od_deploy").resolve()
    model_type = resolve_od_model_type(config, source_path)

    write_runtime_deploy_config(
        deploy_workspace=deploy_workspace,
        base_config_name=config_name,
        tflite_path=tflite_path,
        c_project_path=c_project_path,
        class_names=class_names,
        model_type=model_type,
        config=config,
    )

    mlflow_db = deploy_workspace / "mlflow.db"
    if mlflow_db.is_file():
        mlflow_db.unlink()

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_db.resolve()}"
    stedgeai = resolve_stedgeai_path(str(config.get("stedgeai") or os.environ.get("STM32_AI_EXE", "")))
    if stedgeai is not None:
        env.setdefault("STM32_AI_EXE", str(stedgeai))

    # Hydra resolves --config-path from subprocess cwd (STM32 app dir); must be absolute.
    cmd = [
        sys.executable,
        "stm32ai_main.py",
        "--config-path",
        str(deploy_workspace),
        "--config-name",
        RUNTIME_CONFIG_STEM,
    ]
    logger.info("Running STM32 OD deploy: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=app_dir, check=True, env=env)

    if not deploy_workspace.is_dir():
        raise FileNotFoundError(f"STM32 OD deploy did not create workspace: {deploy_workspace}")
    return deploy_workspace


def collect_deploy_artifacts(deploy_workspace: Path, board_dir: Path) -> tuple[Path, Path | None]:
    """Copy deploy outputs into *board_dir*; return (artifact_path, generated_dir)."""
    generated_dir: Path | None = None
    for candidate in (deploy_workspace / "generated", deploy_workspace):
        if (candidate / "stai_network.c").is_file():
            generated_dir = candidate
            break

    artifact_path = deploy_workspace
    if generated_dir is not None:
        for name in ("stai_network.c", "stai_network.h", "network_atonbuf.xSPI2.raw"):
            src = generated_dir / name
            if not src.is_file():
                src = deploy_workspace / name
            if src.is_file():
                dest = board_dir / name
                shutil.copy2(src, dest)
                if name == "stai_network.c":
                    artifact_path = dest

    return artifact_path.resolve(), (
        generated_dir.resolve() if generated_dir is not None else None
    )


def _is_saved_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "saved_model.pb").is_file()


def find_saved_model_dir(board_dir: Path) -> Path | None:
    """Locate a TensorFlow SavedModel directory for STM32 PTQ under *board_dir*."""
    for candidate in sorted(board_dir.glob("**/saved_model")):
        if _is_saved_model_dir(candidate):
            return candidate

    # Ultralytics YOLO TFLite export: yolov8n_saved_model/saved_model.pb (not .../saved_model/).
    for candidate in sorted(board_dir.glob("*_saved_model")):
        if _is_saved_model_dir(candidate):
            return candidate

    return None
