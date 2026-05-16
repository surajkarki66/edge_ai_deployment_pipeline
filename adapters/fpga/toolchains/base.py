"""Base contract for an FPGA deployment toolchain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import DeploymentArtifact, PlatformConfig


class FpgaToolchain(ABC):
    name: str = ""

    @abstractmethod
    def validate(self, model: Any, config: PlatformConfig) -> bool: ...

    @abstractmethod
    def convert(self, model: Any, config: PlatformConfig) -> dict[str, Any]: ...

    @abstractmethod
    def build_artifact(
        self, converted: dict[str, Any], config: PlatformConfig
    ) -> DeploymentArtifact: ...
