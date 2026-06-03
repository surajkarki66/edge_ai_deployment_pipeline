from __future__ import annotations

from core.types import PlatformConfig


def get_supported_operations(spec: PlatformConfig) -> list[str]:
    """Return layer/operation names from `supported_operations` in the merged catalogue spec."""
    from_config = spec.get("supported_operations")
    if isinstance(from_config, list) and from_config:
        return [str(op) for op in from_config]
    return []
