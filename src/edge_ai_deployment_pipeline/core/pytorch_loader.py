from __future__ import annotations

import importlib
import importlib.util
import json
import sys

from pathlib import Path
from typing import Any

import torch.nn as nn


def parse_model_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-kwargs must be valid JSON, got: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a JSON object")
    return parsed


def split_model_class_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            f"Invalid model class {spec!r}. Use module.path:ClassName or file.py:ClassName"
        )
    module_part, class_name = spec.rsplit(":", 1)
    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name in model class spec: {class_name!r}")
    return module_part, class_name


def resolve_model_definition_file(spec: str) -> Path:
    """Return the ``.py`` file that defines the model class."""
    module_part, _ = split_model_class_spec(spec)
    if module_part.endswith(".py") or "/" in module_part or "\\" in module_part:
        path = Path(module_part).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model definition file not found: {path}")
        return path

    module = importlib.import_module(module_part)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ValueError(
            f"Cannot locate a .py file for model class {spec!r}; "
            "use a file path such as models/net.py:Net"
        )
    return Path(module_file).resolve()


def resolve_model_class(spec: str) -> type[nn.Module]:
    """Resolve ``module.path:ClassName`` or ``path/to/file.py:ClassName``."""
    module_part, class_name = split_model_class_spec(spec)

    if module_part.endswith(".py") or "/" in module_part or "\\" in module_part:
        path = Path(module_part).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Model definition file not found: {path}")
        module_name = f"_edge_ai_model_{path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Cannot import model class from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)

    try:
        model_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class {class_name!r} not found in {module_part!r}"
        ) from exc
    if not isinstance(model_cls, type) or not issubclass(model_cls, nn.Module):
        raise TypeError(f"{spec!r} must refer to a torch.nn.Module subclass")
    return model_cls


def extract_state_dict(obj: Any) -> dict[str, Any]:
    import torch

    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        raise ValueError(
            "Checkpoint dict must be a state_dict or contain 'state_dict' / 'model_state_dict'"
        )
    raise ValueError(
        f"Expected a PyTorch state_dict checkpoint, got {type(obj).__name__}"
    )


def _strip_module_prefix(state_dict: dict[str, Any], prefix: str = "module.") -> dict[str, Any]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_pytorch_from_state_dict(
    weights_path: Path,
    model_class: str,
    *,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Instantiate ``model_class`` and load weights from a ``.pt`` / ``.pth`` file."""
    import torch

    model_cls = resolve_model_class(model_class)
    kwargs = dict(model_kwargs or {})
    model = model_cls(**kwargs)

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    state_dict = _strip_module_prefix(extract_state_dict(checkpoint))
    model.load_state_dict(state_dict)
    return model.eval()

def load_pytorch_model(
    weights_path: str,
) -> nn.Module:
    import torch

    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
    model = checkpoint["model"]
    model.float()
    model.eval()
    return model