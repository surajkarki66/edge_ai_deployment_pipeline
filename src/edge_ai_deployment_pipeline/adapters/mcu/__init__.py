from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_ai_deployment_pipeline.adapters.mcu.adapter import MCUAdapter

__all__ = ["MCUAdapter"]


def __getattr__(name: str) -> object:
    if name == "MCUAdapter":
        from edge_ai_deployment_pipeline.adapters.mcu.adapter import MCUAdapter

        return MCUAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
