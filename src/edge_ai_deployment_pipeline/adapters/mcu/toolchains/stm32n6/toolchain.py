from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.adapters.mcu.toolchains.base import McuToolchain
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.c_projects import resolve_application
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.image_classification.deploy import (
    export_keras_to_stm32n6,
)
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.object_detection.deploy_od import (
    collect_deploy_artifacts,
    find_saved_model_dir,
    run_stm32_od_deploy,
)
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.object_detection.yolov8_quantize import (
    quantize,
)
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.yolo_export import export_yolo_to_tflite
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.config.task import resolve_task
from edge_ai_deployment_pipeline.core.types import (
    CvTask,
    DeploymentArtifact,
    FrameworkTag,
    PlatformConfig,
)
from edge_ai_deployment_pipeline.core.ultralytics_export import (
    yolo_calibration_yaml,
    yolo_export_config,
    yolo_export_int8,
)
from edge_ai_deployment_pipeline.core.yolo_loader import yolo_class_names

_YOLO_CALIBRATION_HINT = (
    "Set yolo_export.calibration_data to an Ultralytics dataset .yaml (with train/val paths). "
    "The board-level calibration_data image folder is only required for Keras image classification."
)
_KERAS_CALIBRATION_HINT = (
    "Set calibration_data to an image directory (or class subfolders) for PTQ. "
    "YOLO object detection uses yolo_export.calibration_data instead."
)


class Stm32N6Toolchain(McuToolchain):
    name = "stm32n6"

    def _is_ultralytics_object_detection(self, config: PlatformConfig) -> bool:
        return (
            resolve_task(config) == CvTask.OBJECT_DETECTION
            and config.get("framework") == FrameworkTag.ULTRALYTICS.value
        )

    def _require_yolo_calibration(self, config: PlatformConfig) -> Path:
        raw = yolo_calibration_yaml(config)
        if not raw:
            raise ValueError(
                "STM32N6 YOLO object detection requires yolo_export.calibration_data. "
                + _YOLO_CALIBRATION_HINT
            )
        yaml_path = Path(str(raw)).resolve()
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"yolo_export.calibration_data not found: {yaml_path}. {_YOLO_CALIBRATION_HINT}"
            )
        images_dir = yaml_path.parent / "valid" / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"YOLO calibration images not found: {images_dir}. "
                "Ensure the dataset referenced by your Ultralytics data.yaml contains valid/images."
            )
        return yaml_path

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        supported = config.get("supported_operations")
        if supported is not None and not supported:
            raise ValueError("No supported operations defined for this STM32N6 board config")

        if self._is_ultralytics_object_detection(config):
            if not yolo_export_int8(config, default_precision="int8"):
                raise ValueError("STM32N6 YOLO export only supports INT8 TFLite")
            self._require_yolo_calibration(config)
            if config.get("framework") != FrameworkTag.ULTRALYTICS.value:
                raise ValueError(
                    "STM32N6 object detection requires an Ultralytics YOLO model (.pt)"
                )
            if not config.get("quantize_only"):
                try:
                    application = resolve_application(config)
                except ValueError as exc:
                    raise ValueError(f"STM32N6 YOLO compile setup invalid: {exc}") from exc
                if application is None and not config.get("c_project_path"):
                    raise ValueError(
                        "STM32N6 YOLO deploy requires --task object_detection "
                        "(to resolve the ObjectDetection Cube project) or an explicit c_project_path"
                    )
            return True

        if config.get("framework") != FrameworkTag.KERAS.value:
            raise ValueError(
                "STM32N6 image classification requires a Keras model (.h5 / .keras)"
            )
        if not self.is_keras_model(model):
            raise ValueError("STM32N6 image classification requires a loaded Keras model")
        fmt = str(config.get("original_format", ""))
        if fmt not in ("h5", "keras"):
            raise ValueError(f"STM32N6 expects .h5 or .keras models, got '{fmt}'")
        if not config.get("calibration_data"):
            raise ValueError(
                "STM32N6 Keras image classification requires calibration_data. "
                + _KERAS_CALIBRATION_HINT
            )
        if config.get("quantize_only"):
            return True
        if config.get("c_project_path"):
            return True
        try:
            application = resolve_application(config)
        except ValueError as exc:
            raise ValueError(f"STM32N6 compile setup invalid: {exc}") from exc
        if application is None:
            raise ValueError(
                "STM32N6 compile requires 'application' "
                "(image_classification or object_detection) or an explicit 'c_project_path'"
            )
        return True

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        channels, height, width = self._require_input_shape(config)
        source_path = config.get("source_path")
        application = resolve_application(config)

        if not source_path:
            raise ValueError("source_path is required for STM32N6 export")

        if self._is_ultralytics_object_detection(config):
            yolo_calibration = self._require_yolo_calibration(config)

            converted: dict[str, Any] = {
                "source_path": str(source_path),
                "input_shape": (channels, height, width),
                "precision": "int8",
                "toolchain": self.name,
                "export_backend": "yolo_tflite",
                "class_names": yolo_class_names(model),
                "original_format": str(config.get("original_format")),
                "yolo_calibration_yaml": str(yolo_calibration),
                "target_mcu": str(config.get("target_mcu")),
                "board": str(config.get("board")),
                "quantization": config.get("quantization"),
                "quantize_only": bool(config.get("quantize_only")),
            }

            if application:
                converted["application"] = application

            if not converted["quantize_only"]:
                converted["c_project_path"] = str(self._resolve_c_project(config))

            return converted

        calibration_data = self._require_calibration_data(config)
        self._require_keras(config)
        if not self.is_keras_model(model):
            raise TypeError("STM32N6 export requires a Keras model for Image Classification task")

        class_names = config.get("class_names")
        if class_names:
            names = [s.strip() for s in str(class_names).split(",") if s.strip()]
        else:
            names = self.infer_class_names(calibration_data)

        converted = {
            "source_path": str(source_path),
            "original_format": str(config.get("original_format", "")),
            "input_shape": (channels, height, width),
            "calibration_data": str(calibration_data),
            "class_names": names,
            "target_mcu": str(config.get("target_mcu", "STM32N6570")),
            "board": str(config.get("board", "STM32N6570-DK")),
            "quantization": config.get("quantization", "int8"),
            "toolchain": self.name,
            "quantize_only": bool(config.get("quantize_only", False)),
        }
        if application:
            converted["application"] = application

        if not converted["quantize_only"]:
            converted["c_project_path"] = str(self._resolve_c_project(config))

        return converted

    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact:
        platform_id = str(config.get("platform_id", config.get("board_id", "mcu")))
        board_dir = Path(config.get("output_dir", DEFAULT_ARTIFACTS_DIR)) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)
        channels, height, width = converted["input_shape"]

        c_project = (
            Path(str(converted["c_project_path"]))
            if not converted.get("quantize_only")
            else None
        )

        if converted.get("export_backend") == "yolo_tflite":
            source_path = Path(str(converted["source_path"]))
            exported_tflite = export_yolo_to_tflite(
                source_path=source_path,
                output_dir=board_dir,
                height=height,
                width=width,
                config=config,
                precision=str(converted["precision"]),
            )

            calibration_yaml = converted.get("yolo_calibration_yaml") or yolo_calibration_yaml(config)
            if not calibration_yaml:
                raise ValueError(
                    "yolo_export.calibration_data is required for YOLO MCU deploy. "
                    + _YOLO_CALIBRATION_HINT
                )
            data_path = Path(calibration_yaml).parent / "valid" / "images"
            if not data_path.is_dir():
                raise FileNotFoundError(f"YOLO calibration images not found: {data_path}")

            export_cfg = yolo_export_config(config)
            saved_model_dir = find_saved_model_dir(board_dir)
            if saved_model_dir is not None:
                quant_config = {
                    "model": {
                        "name": export_cfg.get("model_name", "yolo"),
                        "model_path": str(saved_model_dir),
                        "input_shape": [height, width, channels],
                    },
                    "quantization": {
                        "calib_dataset_path": data_path,
                        "export_path": board_dir,
                        "quantization_type": export_cfg.get("quantization_type", "per_tensor"),
                    },
                }
                out_path = Path(quantize(quant_config))
            else:
                out_path = exported_tflite.resolve()

            final_tflite = board_dir / out_path.name
            if out_path.resolve() != final_tflite.resolve():
                shutil.copy2(out_path, final_tflite)

            yolo_metadata: dict[str, Any] = {
                "toolchain": self.name,
                "input_shape": list(converted["input_shape"]),
                "compile_flags": config.get("compile_flags"),
                "export_backend": converted.get("export_backend"),
                "tflite_path": str(final_tflite.resolve()),
                "class_names": converted.get("class_names"),
                "yolo_calibration_yaml": str(Path(calibration_yaml).resolve()),
                "task": CvTask.OBJECT_DETECTION.value,
            }

            if converted.get("quantize_only"):
                return DeploymentArtifact(
                    platform_id=platform_id,
                    artifact_path=str(final_tflite.resolve()),
                    toolchain_used=self.name,
                    metadata=yolo_metadata,
                )

            if c_project is None:
                raise ValueError("c_project_path is required for STM32N6 YOLO deploy")

            class_names = converted.get("class_names") or []
            if not class_names:
                raise ValueError("class_names are required for STM32N6 YOLO deploy")

            deploy_workspace = run_stm32_od_deploy(
                tflite_path=final_tflite,
                board_dir=board_dir,
                c_project_path=c_project,
                class_names=list(class_names),
                source_path=source_path,
                config=config,
            )
            artifact_path, generated_dir = collect_deploy_artifacts(deploy_workspace, board_dir)

            yolo_metadata["deploy_workspace"] = str(deploy_workspace.resolve())
            yolo_metadata["c_project_path"] = str(c_project.resolve())
            if generated_dir is not None:
                yolo_metadata["generated_dir"] = str(generated_dir)
                stai_c = board_dir / "stai_network.c"
                if stai_c.is_file():
                    yolo_metadata["stai_network_c"] = str(stai_c.resolve())

            return DeploymentArtifact(
                platform_id=platform_id,
                artifact_path=str(artifact_path.resolve()),
                toolchain_used=self.name,
                metadata=yolo_metadata,
            )

        model_path = self._stage_model(board_dir, converted, config)
        calib_dir = Path(converted["calibration_data"])

        tflite_path, generated_dir = export_keras_to_stm32n6(
            model_path=model_path,
            calib_dir=calib_dir,
            c_project=c_project,
            class_names=converted["class_names"],
            channels=channels,
            height=height,
            width=width,
            config={**config, "platform_id": platform_id},
        )

        final_tflite = board_dir / tflite_path.name
        shutil.copy2(tflite_path, final_tflite)

        metadata: dict[str, Any] = {
            "toolchain": self.name,
            "target_mcu": converted["target_mcu"],
            "board": converted["board"],
            "quantization": converted["quantization"],
            "class_names": converted["class_names"],
            "tflite_path": str(final_tflite.resolve()),
        }
        if converted.get("application"):
            metadata["application"] = converted["application"]
        if converted.get("c_project_path"):
            metadata["c_project_path"] = converted["c_project_path"]

        artifact_path = final_tflite
        if generated_dir is not None:
            metadata["generated_dir"] = str(generated_dir.resolve())
            stai_c = generated_dir / "stai_network.c"
            if stai_c.is_file():
                dest = board_dir / "stai_network.c"
                shutil.copy2(stai_c, dest)
                artifact_path = dest
                metadata["stai_network_c"] = str(dest.resolve())

        return DeploymentArtifact(
            platform_id=platform_id,
            artifact_path=str(artifact_path.resolve()),
            toolchain_used=self.name,
            metadata=metadata,
        )

    def _stage_model(
        self,
        board_dir: Path,
        converted: dict[str, Any],
        config: PlatformConfig,
    ) -> Path:
        ext = Path(str(converted.get("source_path", ""))).suffix or ".h5"
        staged = board_dir / f"model{ext}"
        source = Path(str(converted.get("source_path", config.get("source_path", ""))))
        if not source.is_file():
            raise FileNotFoundError(f"Model file not found: {source}")
        shutil.copy2(source, staged)
        return staged
