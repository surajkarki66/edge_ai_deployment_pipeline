from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
import subprocess
from typing import Any

from edge_ai_deployment_pipeline.adapters.mcu.toolchains.base import McuToolchain
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.object_detection.yolov8_quantize import quantize
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.yolo_export import export_yolo_to_tflite
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.c_projects import resolve_application
from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.image_classification.deploy import export_keras_to_stm32n6
from edge_ai_deployment_pipeline.core.types import DeploymentArtifact, FrameworkTag, PlatformConfig
from edge_ai_deployment_pipeline.config.task import resolve_task
from edge_ai_deployment_pipeline.core.types import (
    CvTask,
    DeploymentArtifact,
    FrameworkTag,
    PlatformConfig,
)
from edge_ai_deployment_pipeline.core.ultralytics_export import yolo_calibration_yaml, yolo_export_config, yolo_export_int8
from edge_ai_deployment_pipeline.core.yolo_loader import yolo_class_names

logger = logging.getLogger(__name__)


class Stm32N6Toolchain(McuToolchain):
    name = "stm32n6"
    
    def _is_ultralytics_object_detection(self, config: PlatformConfig) -> bool:
        return (
            resolve_task(config) == CvTask.OBJECT_DETECTION
            and config.get("framework") == FrameworkTag.ULTRALYTICS.value
        )

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if not config.get("supported_operations"):
            return False
        
        if self._is_ultralytics_object_detection(config):
            if yolo_export_int8(config, default_precision="int8"):
                if not yolo_calibration_yaml(config):
                    logger.warning(
                        "YOLO TensorRT INT8 requires yolo_export.calibration_data "
                        "(Ultralytics dataset yaml; not the image-folder calibration_data)"
                    )
                    return False
                
                return config.get("framework") == FrameworkTag.ULTRALYTICS.value
            else:
                logger.error("Only support int8 tflite export")
                return False
        
        if config.get("framework") != FrameworkTag.KERAS.value:
            return False
        if not self.is_keras_model(model):
            return False
        fmt = str(config.get("original_format", ""))
        if fmt not in ("h5", "keras"):
            logger.warning("STM32N6 expects .h5 or .keras models")
            return False
        if not config.get("calibration_data"):
            logger.warning("STM32N6 requires calibration_data")
            return False
        if config.get("quantize_only"):
            return True
        if config.get("c_project_path"):
            return True
        try:
            return resolve_application(config) is not None
        except ValueError:
            return False

    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]:
        channels, height, width = self._require_input_shape(config)
        source_path = config.get("source_path")
        application = resolve_application(config)
        calibration_data = self._require_calibration_data(config)

        if not source_path:
            raise ValueError("source_path is required for STM32N6 export")
        
        if self._is_ultralytics_object_detection(config):
            
            converted = {
                "source_path": str(source_path),
                "input_shape": (channels, height, width),
                "precision": "int8",
                "toolchain": self.name,
                "export_backend": "yolo_tflite",
                "class_names": yolo_class_names(model),
                "original_format": str(config.get("original_format")),
                "calibration_data": str(calibration_data),
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
        
        else:
            self._require_keras(config)
            if not self.is_keras_model(model):
                raise TypeError("STM32N6 export requires a Keras model for Image Classification task")
        

            class_names = config.get("class_names")
            if class_names:
                names = [s.strip() for s in str(class_names).split(",") if s.strip()]
            else:
                names = self.infer_class_names(calibration_data)

            converted: dict[str, Any] = {
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
            
            # Export to totflite
            tflite_path = export_yolo_to_tflite(
                source_path=Path(str(converted["source_path"])),
                output_dir=board_dir,
                height=height,
                width=width,
                config=config,
                precision=str(converted["precision"]),
            )
            tflite_path = str(Path(tflite_path).parent)

            calibration_yaml = yolo_calibration_yaml(config)
            data_path = Path(calibration_yaml)
            data_path = data_path.parent / "valid" / "images"

            export_cfg = yolo_export_config(config)
            print(tflite_path)

            # Quantization
            quant_config =     {
                "model": {
                    "name": export_cfg.get("model_name"),
                    "model_path": tflite_path,
                    "input_shape": [height, width, channels],
                },
                "quantization": {
                    "calib_dataset_path": data_path,
                    "export_path": board_dir,
                    "quantization_type": export_cfg.get("quantization_type"),
                },
            }
            out_path = quantize(quant_config)
            
            # call deploy function
            PROJECT_ROOT = Path(__file__).resolve().parents[6]

            COMPILATION_CODE_DIR = PROJECT_ROOT / "data" / "artifacts" / "_stm32n6_repos" / "stm32ai-n6-yolov8-objectdetection" / "object_detection"

            env = os.environ.copy()
            env["MLFLOW_ALLOW_FILE_STORE"] = "true"

            subprocess.run(
                [
                    "python",
                    "stm32ai_main.py",
                    "--config-path",
                    "./config_file_examples/",
                    "--config-name",
                    "deployment_n6_yolov8_config.yaml",
                ],
                cwd=COMPILATION_CODE_DIR,
                check=True,
                env=env,
            )
            
            yolo_metadata = {
                "toolchain": self.name,
                "input_shape": list(converted["input_shape"]),
                "compile_flags": config.get("compile_flags"),
                "export_backend": converted.get("export_backend"),
                "tflite_path": str(out_path),
                "class_names": converted.get("class_names"),
                "task": CvTask.OBJECT_DETECTION.value,
            }

            return DeploymentArtifact(
                platform_id=platform_id,
                artifact_path=str(PROJECT_ROOT / "data" / "artifacts" ),
                toolchain_used=self.name,
                metadata=yolo_metadata,
            )


        else:
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
