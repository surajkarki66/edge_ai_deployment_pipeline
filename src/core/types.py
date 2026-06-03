from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

PlatformConfig = dict[str, Any]


class FrameworkTag(str, Enum):
    PYTORCH = "pytorch"
    KERAS = "keras"


class CompatibilityStatus(str, Enum):
    FULLY_COMPATIBLE = "fully_compatible"
    PARTIALLY_COMPATIBLE = "partial"
    INCOMPATIBLE = "incompatible"


@dataclass
class LoadedModel:
    framework: FrameworkTag
    source_path: str
    model: Any
    original_format: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "source_path": self.source_path,
            "original_format": self.original_format,
            "model": "<opaque>",
        }


@dataclass
class ModelProfile:
    """Profile of a trained computer-vision model (image input, spatial graph)."""

    flops: int
    macs: int
    parameter_count: int
    model_size_mb: float
    peak_activation_memory_mb: float
    inference_ram_mb: float
    layer_type_inventory: dict[str, int]
    input_resolution: tuple[int, ...]  # (channels, height, width)
    cpu_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_resolution"] = list(self.input_resolution)
        return d


@dataclass
class PlatformCompatResult:
    """Compatibility verdict for a single board target."""

    platform_id: str  # board_id — deployable hardware target
    platform_family: str  # jetson | fpga | mcu | arm
    status: CompatibilityStatus
    unsupported_layers: list[str]
    fallback_penalty_estimate: float | None
    notes: str
    toolchain: str | None = None  # deploy toolchain (tensorrt, vitis_ai, ncnn, …)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "board_id": self.platform_id,
            "platform_family": self.platform_family,
            "toolchain": self.toolchain,
            "status": self.status.value,
            "unsupported_layers": self.unsupported_layers,
            "fallback_penalty_estimate": self.fallback_penalty_estimate,
            "notes": self.notes,
        }


@dataclass
class CompatibilityReport:
    results: list[PlatformCompatResult]
    viable_platform_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "viable_platform_ids": self.viable_platform_ids,
        }


@dataclass
class UserRequirements:
    max_latency_ms: float | None = None
    max_power_watts: float | None = None
    max_cost_usd: float | None = None
    toolchain: str | None = None  # force a toolchain; None = auto-select best per board

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    platform_id: str
    total_score: float
    latency_score: float | None
    power_score: float | None
    cost_score: float | None
    fallback_penalty_applied: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedPlatform:
    """Best board target after constraint scoring."""

    platform_id: str  # board_id
    platform_family: str
    toolchain: str | None  # selected deploy toolchain for this board
    score_breakdown: ScoreBreakdown
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "board_id": self.platform_id,
            "platform_family": self.platform_family,
            "toolchain": self.toolchain,
            "score_breakdown": self.score_breakdown.to_dict(),
            "reasoning": self.reasoning,
        }


@dataclass
class DeploymentArtifact:
    platform_id: str
    artifact_path: str
    toolchain_used: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    loaded_model: LoadedModel | None = None
    profile: ModelProfile | None = None
    compatibility: CompatibilityReport | None = None
    selected_platform: SelectedPlatform | None = None
    artifact: DeploymentArtifact | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded_model": self.loaded_model.to_dict() if self.loaded_model else None,
            "profile": self.profile.to_dict() if self.profile else None,
            "compatibility": self.compatibility.to_dict() if self.compatibility else None,
            "selected_platform": (
                self.selected_platform.to_dict() if self.selected_platform else None
            ),
            "artifact": self.artifact.to_dict() if self.artifact else None,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
