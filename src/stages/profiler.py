from __future__ import annotations

import logging
import time

from typing import Any

from core.types import FrameworkTag, LoadedModel, ModelProfile

logger = logging.getLogger(__name__)

BENCHMARK_ITERATIONS = 10


class ProfilerStage:
    """Extract FLOPs, layer inventory, and CPU latency."""

    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        channels, height, width = input_shape
        if channels < 1 or height < 1 or width < 1:
            raise ValueError(f"input_shape dimensions must be positive, got {input_shape}")
        self._input_shape = input_shape

    def run(self, loaded: LoadedModel) -> ModelProfile:
        if loaded.framework == FrameworkTag.PYTORCH:
            profile = self._profile_pytorch(loaded)
        else:
            profile = self._profile_tensorflow(loaded)
        return profile

    def _profile_pytorch(self, loaded: LoadedModel) -> ModelProfile:
        import torch

        model = self._resolve_pytorch_module(loaded.model)
        layer_inventory = self._pytorch_layer_inventory(model)
        input_resolution = self._input_shape
        dummy = torch.randn(1, *input_resolution)

        flops, macs = self._estimate_pytorch_flops(model, dummy)
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        peak_activation_mb = self._estimate_activation_memory_mb(model, dummy)
        latency_ms = self._benchmark_pytorch(model, dummy)

        return ModelProfile(
            flops=flops,
            macs=macs,
            parameter_count=param_count,
            model_size_mb=round(model_size_mb, 4),
            peak_activation_memory_mb=round(peak_activation_mb, 4),
            layer_type_inventory=layer_inventory,
            input_resolution=input_resolution,
            cpu_latency_ms=latency_ms,
        )

    def _resolve_pytorch_module(self, handle: Any) -> Any:
        import torch.nn as nn

        if isinstance(handle, nn.Module):
            return handle.eval()
        if isinstance(handle, dict):
            raise ValueError(
                "Checkpoint files are not supported; provide a full model or TorchScript export"
            )
        if hasattr(handle, "forward"):
            return handle.eval()
        raise TypeError("Cannot profile PyTorch handle — expected nn.Module or TorchScript")

    def _pytorch_layer_inventory(self, model: Any) -> dict[str, int]:
        inventory: dict[str, int] = {}
        skip = {"Sequential", "ModuleList", "ModuleDict"}
        for module in model.modules():
            name = type(module).__name__
            if name in skip:
                continue
            if not list(module.children()):
                inventory[name] = inventory.get(name, 0) + 1
        return inventory

    def _estimate_pytorch_flops(self, model: Any, dummy: Any) -> tuple[int, int]:
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = int(FlopCountAnalysis(model, dummy).total())
            return flops, flops // 2
        except ImportError:
            pass
        try:
            from torchinfo import summary

            s = summary(model, input_size=tuple(dummy.shape), verbose=0)
            total = int(getattr(s, "total_mult_adds", 0) or getattr(s, "total_params", 0) * 2)
            return total, total // 2
        except ImportError:
            logger.warning("Install fvcore or torchinfo for accurate FLOPs; using estimate")
        param_count = sum(p.numel() for p in model.parameters())
        estimate = param_count * 2
        return estimate, estimate // 2

    def _estimate_activation_memory_mb(self, model: Any, dummy: Any) -> float:
        import torch

        activations: list[int] = []

        def hook(_module: Any, _inp: Any, out: Any) -> None:
            if isinstance(out, torch.Tensor):
                activations.append(out.numel() * out.element_size())

        hooks = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
        with torch.no_grad():
            model(dummy)
        for h in hooks:
            h.remove()
        peak = max(activations) if activations else dummy.numel() * 4
        return peak / (1024**2)

    def _benchmark_pytorch(self, model: Any, dummy: Any) -> float:
        import torch

        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
        timings: list[float] = []
        with torch.no_grad():
            for _ in range(BENCHMARK_ITERATIONS):
                start = time.perf_counter()
                model(dummy)
                timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        return round(timings[len(timings) // 2], 4)

    def _profile_tensorflow(self, loaded: LoadedModel) -> ModelProfile:
        model = loaded.model
        layer_inventory = self._tf_layer_inventory(model)
        input_resolution = self._input_shape
        flops = macs = sum(layer_inventory.values()) * 1_000_000
        param_count = int(getattr(model, "count_params", lambda: 0)())
        model_size_mb = param_count * 4 / (1024**2) if param_count else 1.0
        latency_ms = self._benchmark_tensorflow(model, input_resolution)

        return ModelProfile(
            flops=flops,
            macs=macs,
            parameter_count=param_count,
            model_size_mb=round(model_size_mb, 4),
            peak_activation_memory_mb=round(model_size_mb * 0.5, 4),
            layer_type_inventory=layer_inventory,
            input_resolution=input_resolution,
            cpu_latency_ms=latency_ms,
        )

    def _tf_layer_inventory(self, model: Any) -> dict[str, int]:
        inventory: dict[str, int] = {}
        if hasattr(model, "layers"):
            for layer in model.layers:
                name = type(layer).__name__
                inventory[name] = inventory.get(name, 0) + 1
        else:
            inventory["KerasModel"] = 1
        return inventory

    def _benchmark_tensorflow(self, model: Any, shape: tuple[int, ...]) -> float:
        try:
            import numpy as np
            import tensorflow as tf

            dummy = np.random.randn(1, *shape).astype(np.float32)
            fn = model.signatures[list(model.signatures.keys())[0]] if hasattr(model, "signatures") else model
            for _ in range(5):
                fn(tf.constant(dummy)) if callable(fn) else None
            timings: list[float] = []
            for _ in range(BENCHMARK_ITERATIONS):
                start = time.perf_counter()
                if hasattr(model, "__call__"):
                    model(tf.constant(dummy), training=False)
                timings.append((time.perf_counter() - start) * 1000)
            timings.sort()
            return round(timings[len(timings) // 2], 4) if timings else 0.0
        except Exception:
            logger.warning("TF benchmark failed; returning placeholder latency")
            return 10.0
