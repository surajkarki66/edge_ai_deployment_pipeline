from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from adapters.mcu.toolchains.base import McuToolchain
from adapters.mcu.toolchains.stm32n6.c_projects import resolve_application
from adapters.mcu.toolchains.stm32n6.deploy import export_keras_to_stm32n6
from core.types import DeploymentArtifact, FrameworkTag, PlatformConfig

logger = logging.getLogger(__name__)


class Stm32N6Toolchain(McuToolchain):
    name = "stm32n6"

    def validate(self, model: Any, config: PlatformConfig) -> bool:
        if not config.get("supported_operations"):
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
        self._require_keras(config)
        if not self.is_keras_model(model):
            raise TypeError("STM32N6 export requires a Keras model")

        channels, height, width = self._require_input_shape(config)
        calibration_data = self._require_calibration_data(config)
        source_path = config.get("source_path")
        if not source_path:
            raise ValueError("source_path is required for STM32N6 export")

        class_names = config.get("class_names")
        if class_names:
            names = [s.strip() for s in str(class_names).split(",") if s.strip()]
        else:
            names = self.infer_class_names(calibration_data)

        application = resolve_application(config)

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
        board_dir = Path(config.get("output_dir", "artifacts")) / platform_id
        board_dir.mkdir(parents=True, exist_ok=True)

        channels, height, width = converted["input_shape"]
        model_path = self._stage_model(board_dir, converted, config)
        calib_dir = Path(converted["calibration_data"])

        c_project = (
            Path(str(converted["c_project_path"]))
            if not converted.get("quantize_only")
            else None
        )

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
