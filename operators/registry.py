"""Supported operations — primary source is platforms.json per toolchain."""

from __future__ import annotations

from core.registry import operator_registry
from core.types import PlatformConfig

# Optional programmatic overrides via @operator_registry.register("jetson:tensorrt")
_FALLBACK: dict[str, list[str]] = {}


def _register_fallback(key: str, ops: list[str]) -> None:
    if key not in operator_registry:

        @operator_registry.register(key)
        def _table() -> list[str]:
            return ops

    _FALLBACK[key] = ops


def get_supported_operations(
    spec: PlatformConfig,
    platform_family: str | None = None,
    toolchain: str | None = None,
) -> list[str]:
    """
    Return layer/operation names supported for this board+toolchain.

    Reads `supported_operations` from the merged catalogue spec (platforms.json).
    Falls back to the operator registry only if the field is absent.
    """
    from_config = spec.get("supported_operations")
    if isinstance(from_config, list) and from_config:
        return [str(op) for op in from_config]

    family = platform_family or str(spec.get("platform_family", ""))
    tc = toolchain or spec.get("toolchain")
    if tc:
        keyed = f"{family}:{tc}"
        if keyed in _FALLBACK:
            return list(_FALLBACK[keyed])
        if keyed in operator_registry:
            entry = operator_registry.get(keyed)
            if callable(entry):
                result = entry()
                if isinstance(result, list):
                    return [str(x) for x in result]

    if family in _FALLBACK:
        return list(_FALLBACK[family])
    return []
