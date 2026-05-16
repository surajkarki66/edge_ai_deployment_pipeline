"""Abstract base for platform-specific CV model deployment adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.types import DeploymentArtifact, PlatformConfig


class PlatformAdapter(ABC):
    """Self-contained adapter: validate → convert → build artifact."""

    platform_id: str = ""

    @abstractmethod
    def validate(self, model: Any, config: PlatformConfig) -> bool:
        """Return True if the model can be processed with this adapter."""

    @abstractmethod
    def convert(self, model: Any, config: PlatformConfig) -> Any:
        """Convert the model to the platform-specific intermediate representation."""

    @abstractmethod
    def build_artifact(self, converted_model: Any, config: PlatformConfig) -> DeploymentArtifact:
        """Write the converted model as a deployable artifact."""
