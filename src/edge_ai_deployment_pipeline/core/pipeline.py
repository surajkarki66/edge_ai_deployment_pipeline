from __future__ import annotations

import logging

from enum import Enum
from pathlib import Path

from edge_ai_deployment_pipeline.core.types import (
    CompatibilityReport,
    LoadedModel,
    ModelProfile,
    PipelineResult,
    SelectedPlatform,
    UserRequirements,
)
from edge_ai_deployment_pipeline.stages.compatibility import CompatibilityStage
from edge_ai_deployment_pipeline.stages.constraint_check import ConstraintCheckStage
from edge_ai_deployment_pipeline.stages.intake import IntakeStage
from edge_ai_deployment_pipeline.stages.profiler import ProfilerStage

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    INTAKE = "intake"
    PROFILER = "profiler"
    COMPATIBILITY = "compatibility"
    CONSTRAINT = "constraint"


class Pipeline:
    """Edge deployment pipeline: intake, profile, compatibility, constraint scoring."""

    def __init__(
        self,
        model_path: str | Path,
        input_shape: tuple[int, int, int],
        requirements: UserRequirements,
        platform_config_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.requirements = requirements
        self.platform_config_path = platform_config_path

        self._intake = IntakeStage()
        self._profiler = ProfilerStage(input_shape)
        self._compatibility = CompatibilityStage(platform_config_path)
        self._constraint = ConstraintCheckStage(platform_config_path)

    def run(
        self,
        *,
        loaded_model: LoadedModel | None = None,
        profile: ModelProfile | None = None,
        compatibility: CompatibilityReport | None = None,
        selected_platform: SelectedPlatform | None = None,
    ) -> PipelineResult:
        result = PipelineResult(
            loaded_model=loaded_model,
            profile=profile,
            compatibility=compatibility,
            selected_platform=selected_platform,
        )

        self._run_intake(result)
        self._run_profiler(result)
        self._run_compatibility(result)
        self._run_constraint(result)
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
        logger.info("Stage 1b: Model profiling (input_shape=%s)", self.input_shape)
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
