from __future__ import annotations

import logging

from enum import Enum
from pathlib import Path

from edge_ai_deployment_pipeline.core.board_selection import apply_selection, resolve_board_choice
from edge_ai_deployment_pipeline.core.types import (
    CompatibilityReport,
    DeploymentArtifact,
    LoadedModel,
    ModelProfile,
    PipelineResult,
    SelectedPlatform,
    UserRequirements,
)
from edge_ai_deployment_pipeline.stages.compatibility import CompatibilityStage
from edge_ai_deployment_pipeline.stages.constraint_check import ConstraintCheckStage
from edge_ai_deployment_pipeline.stages.deployment import DeploymentStage
from edge_ai_deployment_pipeline.stages.intake import IntakeStage
from edge_ai_deployment_pipeline.stages.profiler import ProfilerStage

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    INTAKE = "intake"
    PROFILER = "profiler"
    COMPATIBILITY = "compatibility"
    CONSTRAINT = "constraint"
    DEPLOYMENT = "deployment"


class Pipeline:
    """Edge deployment pipeline: intake, profile, compatibility, constraint, deploy."""

    def __init__(
        self,
        model_path: str | Path,
        input_shape: tuple[int, int, int],
        requirements: UserRequirements,
        output_dir: str | Path | None = None,
        platform_config_path: str | Path | None = None,
        board_id: str | None = None,
        auto_select_board: bool = False,
        interactive: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.requirements = requirements
        self.output_dir = Path(output_dir or "artifacts")
        self.platform_config_path = platform_config_path
        self.board_id = board_id
        self.auto_select_board = auto_select_board
        self.interactive = interactive

        self._intake = IntakeStage()
        self._profiler = ProfilerStage(input_shape)
        self._compatibility = CompatibilityStage(platform_config_path)
        self._constraint = ConstraintCheckStage(platform_config_path)
        self._deployment = DeploymentStage(self.output_dir)

    def run(
        self,
        *,
        loaded_model: LoadedModel | None = None,
        profile: ModelProfile | None = None,
        compatibility: CompatibilityReport | None = None,
        selected_platform: SelectedPlatform | None = None,
        artifact: DeploymentArtifact | None = None,
    ) -> PipelineResult:
        result = PipelineResult(
            loaded_model=loaded_model,
            profile=profile,
            compatibility=compatibility,
            selected_platform=selected_platform,
            artifact=artifact,
        )

        self._run_intake(result)
        self._run_profiler(result)
        self._run_compatibility(result)
        self._run_constraint(result)
        self._run_board_selection(result)
        self._run_deployment(result)
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
        selection = self._constraint.run(
            result.profile,
            result.compatibility,
            self.requirements,
        )
        result.selected_platform = selection.selected
        result.platform_rankings = selection.rankings
        logger.info("Platform rankings: %d candidates", len(result.platform_rankings))

    def _run_board_selection(self, result: PipelineResult) -> None:
        if result.selected_platform is not None and result.platform_rankings is None:
            logger.info("Using pre-selected platform: %s", result.selected_platform.platform_id)
            return
        if not result.platform_rankings:
            raise ValueError("Board selection requires platform rankings from constraint check")

        chosen = resolve_board_choice(
            result.platform_rankings,
            board_id=self.board_id,
            auto_select=self.auto_select_board,
            interactive=self.interactive,
        )
        result.selected_platform, result.platform_rankings = apply_selection(
            result.platform_rankings,
            chosen,
        )
        logger.info(
            "User selected platform: %s (rank %d)",
            result.selected_platform.platform_id,
            next(r.rank for r in result.platform_rankings if r.selected),
        )

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
            input_shape=self.input_shape,
        )
        logger.info("Deployment artifact: %s", result.artifact.to_dict())
