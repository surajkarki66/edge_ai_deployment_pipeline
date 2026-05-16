"""Stage 1a — load trained computer-vision models (.pt, .keras, SavedModel)."""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import FrameworkTag, LoadedModel

logger = logging.getLogger(__name__)


class IntakeStage:
    """Accept vision model exports and emit a LoadedModel."""

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".keras", ".h5"}
    SAVEDMODEL_MARKER = "saved_model.pb"

    def run(self, model_path: str | Path) -> LoadedModel:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        if path.is_dir():
            return self._load_savedmodel(path)
        if path.suffix.lower() in {".keras", ".h5"}:
            return self._load_keras(path)
        if path.suffix.lower() in {".pt", ".pth"}:
            return self._load_pytorch(path)

        raise ValueError(f"Unsupported model format: {path}")

    def _load_pytorch(self, path: Path) -> LoadedModel:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required for .pt models: pip install torch") from exc

        logger.debug("Loading PyTorch model from %s", path)
        try:
            graph_handle = torch.jit.load(str(path), map_location="cpu")
            original_format = "torchscript"
        except Exception:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                graph_handle = checkpoint
                original_format = "state_dict"
            else:
                graph_handle = checkpoint
                original_format = "pytorch_module"

        return LoadedModel(
            framework=FrameworkTag.PYTORCH,
            source_path=str(path.resolve()),
            graph_handle=graph_handle,
            original_format=original_format,
        )

    def _load_keras(self, path: Path) -> LoadedModel:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow required for Keras models") from exc

        logger.debug("Loading Keras model from %s", path)
        graph_handle = tf.keras.models.load_model(str(path))
        return LoadedModel(
            framework=FrameworkTag.KERAS,
            source_path=str(path.resolve()),
            graph_handle=graph_handle,
            original_format=path.suffix.lower().lstrip("."),
        )

    def _load_savedmodel(self, path: Path) -> LoadedModel:
        marker = path / self.SAVEDMODEL_MARKER
        if not marker.exists() and not any(path.glob("**/saved_model.pb")):
            raise ValueError(f"Directory does not look like SavedModel: {path}")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("TensorFlow required for SavedModel") from exc

        logger.debug("Loading SavedModel from %s", path)
        graph_handle = tf.saved_model.load(str(path))
        return LoadedModel(
            framework=FrameworkTag.TENSORFLOW,
            source_path=str(path.resolve()),
            graph_handle=graph_handle,
            original_format="savedmodel",
        )
