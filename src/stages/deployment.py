"""Stage 4 — adapter dispatch and artifact generation."""

from __future__ import annotations

import logging
from pathlib import Path

from config.catalogue import DEFAULT_CATALOGUE_PATH, PlatformCatalogue
from core.registry import platform_registry
from core.types import (
    DeploymentArtifact,
    LoadedModel,
    PlatformConfig,
    SelectedPlatform,
)

logger = logging.getLogger(__name__)


class DeploymentStage:
    """Dispatch to the platform-family adapter for the selected board."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        loaded: LoadedModel,
        selected: SelectedPlatform,
        platform_config_path: str | Path | None = None,
    ) -> DeploymentArtifact:
        board_id = selected.platform_id
        config = self._load_board_config(
            board_id, platform_config_path, toolchain=selected.toolchain
        )
        adapter_id = str(config.get("adapter_id", selected.platform_family))

        adapter = platform_registry.get_adapter(adapter_id)
        tc_note = f", toolchain={selected.toolchain}" if selected.toolchain else ""
        logger.info(
            "Deploying to %s (%s%s) via %s",
            board_id,
            selected.platform_family,
            tc_note,
            type(adapter).__name__,
        )

        if not adapter.validate(loaded.graph_handle, config):
            raise ValueError(f"Adapter validation failed for board '{board_id}'")

        converted = adapter.convert(loaded.graph_handle, config)
        artifact = adapter.build_artifact(converted, config)
        artifact.metadata["board_id"] = board_id
        artifact.metadata["platform_family"] = selected.platform_family
        if selected.toolchain:
            artifact.metadata["toolchain"] = selected.toolchain
        return artifact

    def _load_board_config(
        self,
        board_id: str,
        config_path: str | Path | None,
        *,
        toolchain: str | None = None,
    ) -> PlatformConfig:
        catalogue = PlatformCatalogue.load(config_path)
        resolved_tc = toolchain or catalogue.list_toolchains(board_id)[0]
        spec = catalogue.resolve_board_config(board_id, resolved_tc)
        spec["output_dir"] = str(self.output_dir)
        spec["platform_id"] = board_id
        spec["board_id"] = board_id
        return spec
