from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.toolchain import Stm32N6Toolchain

__all__ = ["Stm32N6Toolchain"]


def __getattr__(name: str) -> object:
    if name == "Stm32N6Toolchain":
        from edge_ai_deployment_pipeline.adapters.mcu.toolchains.stm32n6.toolchain import (
            Stm32N6Toolchain,
        )

        return Stm32N6Toolchain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
