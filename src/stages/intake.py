from __future__ import annotations

import logging
from pathlib import Path

from core.types import FrameworkTag, LoadedModel

logger = logging.getLogger(__name__)


class IntakeStage:
    """Accept model exports and emit a LoadedModel."""

    SUPPORTED_EXTENSIONS = {".pt", ".pth", ".keras", ".h5"}

    def run(self, model_path: str | Path) -> LoadedModel:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")

        if path.suffix.lower() in {".keras", ".h5"}:
            return self._load_keras(path)
        if path.suffix.lower() in {".pt", ".pth"}:
            return self._load_pytorch(path)

        raise ValueError(f"Unsupported model format: {path}")

    def _load_pytorch(self, path: Path) -> LoadedModel:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch required for .pt models") from exc

        import torch.nn as nn

        logger.debug("Loading PyTorch model from %s", path)
        try:
            model = torch.jit.load(str(path), map_location="cpu")
            original_format = "torchscript"
        except Exception:
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                raise ValueError(
                    f"Checkpoint files are not supported: {path}. "
                    "Provide a full model (nn.Module pickle) or TorchScript export."
                ) from None
            if not isinstance(obj, nn.Module) and not hasattr(obj, "forward"):
                raise TypeError(
                    f"Expected a full PyTorch model at {path}, got {type(obj).__name__}"
                ) from None
            model = obj
            original_format = "pytorch_module"

        return LoadedModel(
            framework=FrameworkTag.PYTORCH,
            source_path=str(path.resolve()),
            model=model,
            original_format=original_format,
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
