from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.config.paths import EXAMPLE_MODEL_CLASS
from edge_ai_deployment_pipeline.core.pytorch_loader import load_pytorch_from_state_dict
from edge_ai_deployment_pipeline.core.types import CvTask, FrameworkTag, LoadedModel
from edge_ai_deployment_pipeline.core.yolo_loader import load_yolo

logger = logging.getLogger(__name__)


class IntakeStage:
    """Accept model exports and emit a LoadedModel."""

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".keras", ".h5"}

    def run(
        self,
        model_path: str | Path,
        *,
        model_class: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        task: CvTask,
        patch_ultralytics_dpu: bool = False,
    ) -> LoadedModel:
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        if task == CvTask.OBJECT_DETECTION:
            if path.suffix.lower() not in {".pt", ".pth"}:
                raise ValueError(
                    "Object detection requires an Ultralytics model (.pt or .pth), "
                    f"got: {path}"
                )
            return self._load_ultralytics(path, patch_ultralytics_dpu=patch_ultralytics_dpu)

        if path.suffix.lower() in {".keras", ".h5"}:
            return self._load_keras(path)
        if path.suffix.lower() in {".pt", ".pth"}:
            return self._load_pytorch(
                path,
                model_class=model_class,
                model_kwargs=model_kwargs,
            )

        raise ValueError(f"Unsupported model format: {path}")

    def _load_ultralytics(self, path: Path, *, patch_ultralytics_dpu: bool = False) -> LoadedModel:
        logger.debug("Loading Ultralytics model from %s", path)
        model = load_yolo(path, patch_dpu_layers=patch_ultralytics_dpu)
        return LoadedModel(
            framework=FrameworkTag.ULTRALYTICS,
            source_path=str(path.resolve()),
            model=model,
            original_format="ultralytics",
        )

    def _load_pytorch(
        self,
        path: Path,
        *,
        model_class: str | None,
        model_kwargs: dict[str, Any] | None,
    ) -> LoadedModel:
        if not model_class:
            raise ValueError(
                "PyTorch weights require --model-class (network definition), e.g.\n"
                f"  --model-class {EXAMPLE_MODEL_CLASS}\n"
                "For object detection, pass --task object_detection with an "
                "Ultralytics checkpoint (.pt).\n"
                "Save weights with: torch.save(model.state_dict(), 'weights.pt')"
            )

        model = load_pytorch_from_state_dict(
            path,
            model_class,
            model_kwargs=model_kwargs,
        )
        return LoadedModel(
            framework=FrameworkTag.PYTORCH,
            source_path=str(path.resolve()),
            model=model,
            original_format="pytorch_state_dict",
            architecture_spec=model_class,
            model_kwargs=model_kwargs or None,
        )

    def _load_keras(self, path: Path) -> LoadedModel:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow required for Keras models") from exc

        logger.debug("Loading Keras model from %s", path)
        model = tf.keras.models.load_model(str(path))
        return LoadedModel(
            framework=FrameworkTag.KERAS,
            source_path=str(path.resolve()),
            model=model,
            original_format=path.suffix.lower().lstrip("."),
        )
