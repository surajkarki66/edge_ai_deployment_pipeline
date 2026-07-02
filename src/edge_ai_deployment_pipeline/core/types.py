from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

PlatformConfig = dict[str, Any]


class FrameworkTag(str, Enum):
    PYTORCH = "pytorch"
    KERAS = "keras"
    ULTRALYTICS = "ultralytics"


class CompatibilityStatus(str, Enum):
    FULLY_COMPATIBLE = "fully_compatible"
    PARTIALLY_COMPATIBLE = "partial"
    INCOMPATIBLE = "incompatible"


class LatencyType(str, Enum):
    """Target latency class for board selection."""

    ULTRA_LOW = "ultra_low"
    LOW = "low"
    MEDIUM = "medium"
    MODERATE = "moderate"


class CvTask(str, Enum):
    """Computer-vision deployment task — drives platform-specific adaptations."""

    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"


@dataclass
class LoadedModel:
    framework: FrameworkTag
    source_path: str
    model: Any
    original_format: str
    architecture_spec: str | None = None
    model_kwargs: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "framework": self.framework.value,
            "source_path": self.source_path,
            "original_format": self.original_format,
            "model": "<opaque>",
        }
        if self.architecture_spec:
            payload["architecture_spec"] = self.architecture_spec
        if self.model_kwargs:
            payload["model_kwargs"] = self.model_kwargs
        return payload


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
    unsupported_operations: list[str]
    notes: str
    toolchain: str | None = None  # deploy toolchain (tensorrt, vitis_ai, ncnn, …)
    unsupported_layer_ratio: float = 0.0  # share of layers (by count) that are unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_id": self.platform_id,
            "board_id": self.platform_id,
            "platform_family": self.platform_family,
            "toolchain": self.toolchain,
            "status": self.status.value,
            "unsupported_operations": self.unsupported_operations,
            "unsupported_layer_ratio": round(self.unsupported_layer_ratio, 4),
            "notes": self.notes,
        }


@dataclass
class CompatibilityReport:
    results: list[PlatformCompatResult]
    viable_platform_ids: list[str]  # all evaluated boards (included in rankings)
    deployable_platform_ids: list[str] = field(default_factory=list)  # not incompatible

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "viable_platform_ids": self.viable_platform_ids,
            "deployable_platform_ids": self.deployable_platform_ids,
        }


@dataclass
class UserRequirements:
    latency_type: LatencyType
    task: CvTask
    max_power_watts: float = 25.0
    max_cost_usd: float = 500.0
    toolchain: str | None = None  # force a toolchain; None = auto-select best per board

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["latency_type"] = self.latency_type.value
        d["task"] = self.task.value
        return d


@dataclass
class ScoreBreakdown:
    platform_id: str
    total_score: float
    power_score: float | None
    cost_score: float | None
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
class PlatformRanking:
    """One board × toolchain candidate with constraint score (sorted best-first)."""

    rank: int
    platform_id: str
    platform_family: str
    toolchain: str | None
    score_breakdown: ScoreBreakdown
    reasoning: str
    selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "platform_id": self.platform_id,
            "board_id": self.platform_id,
            "platform_family": self.platform_family,
            "toolchain": self.toolchain,
            "selected": self.selected,
            "score_breakdown": self.score_breakdown.to_dict(),
            "reasoning": self.reasoning,
        }


@dataclass
class ConstraintCheckResult:
    selected: SelectedPlatform
    rankings: list[PlatformRanking]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "rankings": [r.to_dict() for r in self.rankings],
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
    platform_rankings: list[PlatformRanking] | None = None
    artifact: DeploymentArtifact | None = None
    run_id: str | None = None
    artifacts_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "loaded_model": self.loaded_model.to_dict() if self.loaded_model else None,
            "profile": self.profile.to_dict() if self.profile else None,
            "compatibility": self.compatibility.to_dict() if self.compatibility else None,
            "selected_platform": self._selected_platform_dict(),
            "artifact": self.artifact.to_dict() if self.artifact else None,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.artifacts_dir is not None:
            payload["artifacts_dir"] = self.artifacts_dir
        return payload

    def _selected_platform_dict(self) -> dict[str, Any] | None:
        if self.selected_platform is None:
            return None
        payload = self.selected_platform.to_dict()
        if self.platform_rankings:
            payload["rankings"] = [r.to_dict() for r in self.platform_rankings]
        return payload

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
