from __future__ import annotations

import gc
import logging
import resource
import sys
import time

from typing import Any

from edge_ai_deployment_pipeline.core.types import FrameworkTag, LoadedModel, ModelProfile

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
            profile = self._pytorch_model_profiler(loaded)
        else:
            profile = self._tensorflow_model_profiler(loaded)
        return profile

    def _pytorch_model_profiler(self, loaded: LoadedModel) -> ModelProfile:
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
        inference_ram_mb = self._measure_pytorch_inference_ram_mb(model, dummy)

        return ModelProfile(
            flops=flops,
            macs=macs,
            parameter_count=param_count,
            model_size_mb=round(model_size_mb, 4),
            peak_activation_memory_mb=round(peak_activation_mb, 4),
            inference_ram_mb=round(inference_ram_mb, 4),
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

    def _measure_pytorch_inference_ram_mb(self, model: Any, dummy: Any) -> float:
        import torch

        if torch.cuda.is_available():
            cuda = torch.device("cuda")
            orig_device = next(model.parameters()).device
            on_cuda = orig_device.type == "cuda"
            if not on_cuda:
                model.to(cuda)
            run_input = dummy if on_cuda else dummy.to(cuda)
            torch.cuda.reset_peak_memory_stats(cuda)
            with torch.no_grad():
                model(run_input)
            peak_bytes = torch.cuda.max_memory_allocated(cuda)
            if not on_cuda:
                model.to(orig_device)
            return peak_bytes / (1024**2)

        gc.collect()
        rss_before = self._process_rss_mb()
        with torch.no_grad():
            model(dummy)
        gc.collect()
        return max(0.0, self._process_rss_mb() - rss_before)

    @staticmethod
    def _process_rss_mb() -> float:
        """Current resident set size in MB."""
        try:
            with open("/proc/self/status", encoding="ascii") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except OSError:
            pass
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024**2)
        return usage.ru_maxrss / 1024

    def _benchmark_pytorch(self, model: Any, dummy: Any) -> float:
        import torch

        cpu = torch.device("cpu")
        model.to(cpu)
        dummy = dummy.to(cpu)
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

    def _tensorflow_model_profiler(self, loaded: LoadedModel) -> ModelProfile:
        model = loaded.model
        layer_inventory = self._tf_layer_inventory(model)
        input_resolution = self._input_shape
        batch_shape = self._keras_batch_input_shape(model, input_resolution)
        flops, macs = self._estimate_tensorflow_flops(model, batch_shape)
        param_count, model_size_mb = self._keras_weight_stats(model)
        inference_ram_mb = self._measure_keras_inference_ram_mb(model, batch_shape)
        peak_activation_mb = self._estimate_keras_activation_memory_mb(model, batch_shape)
        latency_ms = self._benchmark_latency_tensorflow(model, batch_shape)

        return ModelProfile(
            flops=flops,
            macs=macs,
            parameter_count=param_count,
            model_size_mb=round(model_size_mb, 4),
            peak_activation_memory_mb=round(peak_activation_mb, 4),
            inference_ram_mb=round(inference_ram_mb, 4),
            layer_type_inventory=layer_inventory,
            input_resolution=input_resolution,
            cpu_latency_ms=latency_ms,
        )

    @staticmethod
    def _keras_batch_input_shape(
        model: Any, input_shape: tuple[int, int, int]
    ) -> tuple[int, int, int, int]:
        """Map pipeline ``(C,H,W)`` to a batch-1 tensor shape matching the Keras model layout."""
        channels, height, width = input_shape
        inp_shape = getattr(model, "input_shape", None)
        if isinstance(inp_shape, list):
            inp_shape = inp_shape[0]
        if inp_shape and len(inp_shape) == 4:
            d1, d2, d3 = (inp_shape[i] for i in (1, 2, 3))
            if d3 is not None and d3 <= 4 and (d1 is None or d1 > d3):
                return (1, height, width, channels)
            if d1 is not None and d1 <= 4 and (d2 is None or d2 > d1):
                return (1, channels, height, width)
        return (1, height, width, channels)

    @staticmethod
    def _keras_weight_stats(model: Any) -> tuple[int, float]:
        """Count parameters and on-disk weight bytes."""
        weights = getattr(model, "weights", None) or []
        if weights:
            param_count = 0
            size_bytes = 0
            for weight in weights:
                array = weight.numpy() if hasattr(weight, "numpy") else weight
                param_count += int(array.size)
                size_bytes += int(array.nbytes)
            return param_count, size_bytes / (1024**2)

        param_count = int(getattr(model, "count_params", lambda: 0)())
        return param_count, (param_count * 4 / (1024**2)) if param_count else 0.0

    def _estimate_keras_activation_memory_mb(
        self, model: Any, batch_shape: tuple[int, int, int, int]
    ) -> float:
        """Estimate peak activation bytes as the largest leaf-layer output (like PyTorch hooks)."""
        import numpy as np
        import tensorflow as tf

        activation_bytes: list[int] = []
        hooked: list[tuple[Any, Any]] = []

        def _tensor_bytes(tensor: tf.Tensor) -> int | None:
            shape = tensor.shape
            if shape.rank is None:
                return None
            if hasattr(shape, "num_elements"):
                try:
                    count = int(shape.num_elements())
                except (ValueError, TypeError):
                    return None
            else:
                count = 1
                for dim in shape:
                    if dim is None:
                        return None
                    count *= int(dim)
            return count * tensor.dtype.size

        def _record_output(outputs: Any) -> None:
            for tensor in tf.nest.flatten(outputs):
                if isinstance(tensor, tf.Tensor):
                    nbytes = _tensor_bytes(tensor)
                    if nbytes is not None:
                        activation_bytes.append(nbytes)

        def _wrap_leaf_layer(layer: Any) -> None:
            original_call = layer.call

            def wrapped_call(*args: Any, **kwargs: Any):
                outputs = original_call(*args, **kwargs)
                _record_output(outputs)
                return outputs

            hooked.append((layer, original_call))
            layer.call = wrapped_call

        def _install_hooks(container: Any) -> None:
            for layer in getattr(container, "layers", []):
                if getattr(layer, "layers", None):
                    _install_hooks(layer)
                else:
                    _wrap_leaf_layer(layer)

        fallback_bytes = int(np.prod(batch_shape)) * 4
        try:
            _install_hooks(model)
            dummy = np.random.randn(*batch_shape).astype(np.float32)
            model(tf.constant(dummy), training=False)
        except Exception as exc:
            logger.warning("Keras activation memory estimate failed (%s); using input size", exc)
            return fallback_bytes / (1024**2)
        finally:
            for layer, original_call in hooked:
                layer.call = original_call

        peak = max(activation_bytes) if activation_bytes else fallback_bytes
        return peak / (1024**2)

    def _measure_keras_inference_ram_mb(
        self, model: Any, batch_shape: tuple[int, int, int, int]
    ) -> float:
        """Measured runtime memory: GPU peak if available, else CPU RSS delta."""
        import numpy as np
        import tensorflow as tf

        dummy = tf.constant(np.random.randn(*batch_shape).astype(np.float32))
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                tf.config.experimental.reset_memory_stats("GPU:0")
                model(dummy, training=False)
                info = tf.config.experimental.get_memory_info("GPU:0")
                return info["peak"] / (1024**2)
            except Exception as exc:
                logger.warning("GPU inference RAM measurement failed (%s); using CPU RSS", exc)

        gc.collect()
        rss_before = self._process_rss_mb()
        try:
            model(dummy, training=False)
        except Exception as exc:
            logger.warning("CPU inference RAM measurement failed (%s)", exc)
            return 0.0
        gc.collect()
        return max(0.0, self._process_rss_mb() - rss_before)

    def _estimate_tensorflow_flops(
        self, model: Any, batch_shape: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        try:
            import tensorflow as tf
            from tensorflow.python.profiler import model_analyzer
            from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

            input_signature = [tf.TensorSpec(shape=batch_shape, dtype=tf.float32)]
            forward_fn = tf.function(
                lambda x: model(x, training=False),
                input_signature=input_signature,
            )
            graph_info = model_analyzer.profile(
                forward_fn.get_concrete_function().graph,
                options=ProfileOptionBuilder.float_operation(),
            )
            flops = int(graph_info.total_float_ops)
            macs = flops // 2
            logger.debug(
                "TF profiler FLOPs=%d MACs=%d (input shape=%s)",
                flops,
                macs,
                batch_shape,
            )
            return flops, macs
        except Exception as exc:
            logger.warning("TF FLOPs profiler failed (%s); using parameter estimate", exc)
            param_count, _ = self._keras_weight_stats(model)
            estimate = param_count * 2
            return estimate, estimate // 2

    def _tf_layer_inventory(self, model: Any) -> dict[str, int]:
        inventory: dict[str, int] = {}
        if hasattr(model, "layers"):
            for layer in model.layers:
                name = type(layer).__name__
                inventory[name] = inventory.get(name, 0) + 1
        return inventory

    def _benchmark_latency_tensorflow(
        self, model: Any, batch_shape: tuple[int, int, int, int]
    ) -> float:
        try:
            import numpy as np
            import tensorflow as tf

            dummy = np.random.randn(*batch_shape).astype(np.float32)
            inp = tf.constant(dummy)
            with tf.device("/CPU:0"):
                for _ in range(5):
                    model(inp, training=False)
                timings: list[float] = []
                for _ in range(BENCHMARK_ITERATIONS):
                    start = time.perf_counter()
                    model(inp, training=False)
                    timings.append((time.perf_counter() - start) * 1000)
            timings.sort()
            return round(timings[len(timings) // 2], 4) if timings else 0.0
        except Exception:
            logger.warning("TF benchmark failed; returning placeholder latency")
            return 10.0
