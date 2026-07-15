from __future__ import annotations

import logging

from enum import Enum
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.config.catalogue import targets_vitis_ai_flow
from edge_ai_deployment_pipeline.config.paths import DEFAULT_ARTIFACTS_DIR
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
from edge_ai_deployment_pipeline.steps.compatibility import CompatibilityStep
from edge_ai_deployment_pipeline.steps.constraint_check import ConstraintCheckStep
from edge_ai_deployment_pipeline.steps.deployment import DeploymentStep
from edge_ai_deployment_pipeline.steps.intake import IntakeStep
from edge_ai_deployment_pipeline.steps.profiler import ProfilerStep

logger = logging.getLogger(__name__)


class PipelineStep(str, Enum):
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
        model_class: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_shape = input_shape
        self.requirements = requirements
        self.output_dir = Path(output_dir or DEFAULT_ARTIFACTS_DIR)
        self.platform_config_path = platform_config_path
        self.board_id = board_id
        self.auto_select_board = auto_select_board
        self.interactive = interactive
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.run_id = run_id

        self._intake = IntakeStep()
        self._profiler = ProfilerStep(input_shape)
        self._compatibility = CompatibilityStep(platform_config_path)
        self._constraint = ConstraintCheckStep(platform_config_path)
        self._deployment = DeploymentStep(self.output_dir)

    def run(
        self,
        *,
        loaded_model: LoadedModel | None = None,
        profile: ModelProfile | None = None,
        compatibility: CompatibilityReport | None = None,
        selected_platform: SelectedPlatform | None = None,
        artifact: DeploymentArtifact | None = None,
        stop_after: PipelineStep | None = None,
    ) -> PipelineResult:
        result = PipelineResult(
            loaded_model=loaded_model,
            profile=profile,
            compatibility=compatibility,
            selected_platform=selected_platform,
            artifact=artifact,
            run_id=self.run_id,
            artifacts_dir=str(self.output_dir),
        )

        self._run_intake(result)
        if stop_after == PipelineStep.INTAKE:
            return result
        self._run_profiler(result)
        if stop_after == PipelineStep.PROFILER:
            return result
        self._run_compatibility(result)
        if stop_after == PipelineStep.COMPATIBILITY:
            return result
        self._run_constraint(result)
        if stop_after == PipelineStep.CONSTRAINT:
            return result
        self._run_board_selection(result)
        self._run_deployment(result)
        return result

    def _run_intake(self, result: PipelineResult) -> None:
        if result.loaded_model is not None:
            logger.info("Skipping intake — LoadedModel already provided")
            return
        logger.info("Step 1a: Model intake — %s", self.model_path)
        patch_ultralytics_dpu = targets_vitis_ai_flow(
            board_id=self.board_id,
            toolchain=self.requirements.toolchain,
            latency_type=self.requirements.latency_type.value,
            platform_config_path=self.platform_config_path,
        )
        if patch_ultralytics_dpu:
            logger.info("Vitis-AI flow detected — patching Ultralytics DPU layers before load")
        result.loaded_model = self._intake.run(
            self.model_path,
            model_class=self.model_class,
            model_kwargs=self.model_kwargs,
            task=self.requirements.task,
            patch_ultralytics_dpu=patch_ultralytics_dpu,
        )
        logger.info("Loaded model: %s", result.loaded_model.to_dict())

    def _run_profiler(self, result: PipelineResult) -> None:
        if result.profile is not None:
            logger.info("Skipping profiler — ModelProfile already provided")
            return
        if result.loaded_model is None:
            raise ValueError("Profiler requires LoadedModel; run intake first or pass loaded_model")
        logger.info("Step 1b: Model profiling (input_shape=%s)", self.input_shape)
        result.profile = self._profiler.run(result.loaded_model)
        logger.info("Model profile: %s", result.profile.to_dict())

    def _run_compatibility(self, result: PipelineResult) -> None:
        if result.compatibility is not None:
            logger.info("Skipping compatibility — report already provided")
            return
        if result.profile is None:
            raise ValueError("Compatibility requires ModelProfile")
        if result.loaded_model is None:
            raise ValueError("Compatibility requires LoadedModel")
        logger.info("Step 2: Compatibility analysis")
        result.compatibility = self._compatibility.run(
            result.profile,
            loaded_model=result.loaded_model,
            requirements=self.requirements,
        )
        logger.info("Compatibility: %s", result.compatibility.to_dict())

    def _run_constraint(self, result: PipelineResult) -> None:
        if result.selected_platform is not None:
            logger.info("Skipping constraint check — SelectedPlatform already provided")
            return
        if result.profile is None or result.compatibility is None:
            raise ValueError("Constraint check requires ModelProfile and CompatibilityReport")
        logger.info("Step 3: Constraint checking")
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
        logger.info("Step 4: Deployment artifact generation")
        result.artifact = self._deployment.run(
            result.loaded_model,
            result.selected_platform,
            self.platform_config_path,
            input_shape=self.input_shape,
            requirements=self.requirements,
        )
        logger.info("Deployment artifact: %s", result.artifact.to_dict())
