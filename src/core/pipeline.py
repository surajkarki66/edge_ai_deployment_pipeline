"""Pipeline orchestrator — runs stages sequentially, resumable from any stage."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from core.types import (
    CompatibilityReport,
    DeploymentArtifact,
    LoadedModel,
    ModelProfile,
    PipelineResult,
    SelectedPlatform,
    UserRequirements,
)
import adapters.arm  # noqa: F401 — register platform adapters
import adapters.fpga.adapter  # noqa: F401
import adapters.jetson  # noqa: F401
import adapters.mcu  # noqa: F401
from stages.compatibility import CompatibilityStage
from stages.constraint_check import ConstraintCheckStage
from stages.deployment import DeploymentStage
from stages.intake import IntakeStage
from stages.profiler import ProfilerStage

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    INTAKE = "intake"
    PROFILER = "profiler"
    COMPATIBILITY = "compatibility"
    CONSTRAINT = "constraint"
    DEPLOYMENT = "deployment"


class Pipeline:
    """Four-stage edge deployment pipeline for computer-vision models (optional resume)."""

    def __init__(
        self,
        model_path: str | Path,
        requirements: UserRequirements,
        output_dir: str | Path | None = None,
        platform_config_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.requirements = requirements
        self.output_dir = Path(output_dir or "artifacts")
        self.platform_config_path = platform_config_path

        self._intake = IntakeStage()
        self._profiler = ProfilerStage()
        self._compatibility = CompatibilityStage(platform_config_path)
        self._constraint = ConstraintCheckStage(platform_config_path)
        self._deployment = DeploymentStage(self.output_dir)

    def run(
        self,
        *,
        start_from: PipelineStage | None = None,
        loaded_model: LoadedModel | None = None,
        profile: ModelProfile | None = None,
        compatibility: CompatibilityReport | None = None,
        selected_platform: SelectedPlatform | None = None,
        artifact: DeploymentArtifact | None = None,
    ) -> PipelineResult:
        """Execute stages from `start_from` (or from the beginning) through deployment."""
        result = PipelineResult(
            loaded_model=loaded_model,
            profile=profile,
            compatibility=compatibility,
            selected_platform=selected_platform,
            artifact=artifact,
        )

        stages = [
            (PipelineStage.INTAKE, self._run_intake),
            (PipelineStage.PROFILER, self._run_profiler),
            (PipelineStage.COMPATIBILITY, self._run_compatibility),
            (PipelineStage.CONSTRAINT, self._run_constraint),
            (PipelineStage.DEPLOYMENT, self._run_deployment),
        ]

        start_idx = 0
        if start_from is not None:
            start_idx = next(i for i, (s, _) in enumerate(stages) if s == start_from)

        for _, handler in stages[start_idx:]:
            handler(result)

        return result

    def _run_intake(self, result: PipelineResult) -> None:
        if result.loaded_model is not None:
            logger.info("Skipping intake — LoadedModel already provided")
            return
        logger.info("Stage 1a: Model intake — %s", self.model_path)
        result.loaded_model = self._intake.run(self.model_path)
        logger.info("Loaded model: %s", result.loaded_model.to_dict())

    def _run_profiler(self, result: PipelineResult) -> None:
        if result.profile is not None:
            logger.info("Skipping profiler — ModelProfile already provided")
            return
        if result.loaded_model is None:
            raise ValueError("Profiler requires LoadedModel; run intake first or pass loaded_model")
        logger.info("Stage 1b: Model profiling")
        result.profile = self._profiler.run(result.loaded_model)
        logger.info("Model profile: %s", result.profile.to_dict())

    def _run_compatibility(self, result: PipelineResult) -> None:
        if result.compatibility is not None:
            logger.info("Skipping compatibility — report already provided")
            return
        if result.profile is None:
            raise ValueError("Compatibility requires ModelProfile")
        logger.info("Stage 2: Compatibility analysis")
        result.compatibility = self._compatibility.run(result.profile)
        logger.info("Compatibility: %s", result.compatibility.to_dict())

    def _run_constraint(self, result: PipelineResult) -> None:
        if result.selected_platform is not None:
            logger.info("Skipping constraint check — SelectedPlatform already provided")
            return
        if result.profile is None or result.compatibility is None:
            raise ValueError("Constraint check requires ModelProfile and CompatibilityReport")
        logger.info("Stage 3: Constraint checking")
        result.selected_platform = self._constraint.run(
            result.profile,
            result.compatibility,
            self.requirements,
        )
        logger.info("Selected platform: %s", result.selected_platform.to_dict())

    def _run_deployment(self, result: PipelineResult) -> None:
        if result.artifact is not None:
            logger.info("Skipping deployment — artifact already provided")
            return
        if result.loaded_model is None or result.selected_platform is None:
            raise ValueError("Deployment requires LoadedModel and SelectedPlatform")
        logger.info("Stage 4: Deployment artifact generation")
        result.artifact = self._deployment.run(
            result.loaded_model,
            result.selected_platform,
            self.platform_config_path,
        )
        logger.info("Deployment artifact: %s", result.artifact.to_dict())
