from __future__ import annotations

import logging

from pathlib import Path

from edge_ai_deployment_pipeline.config.catalogue import DEFAULT_CATALOGUE_PATH, PlatformCatalogue
from edge_ai_deployment_pipeline.core.types import (
    CompatibilityReport,
    CompatibilityStatus,
    ModelProfile,
    PlatformCompatResult,
    PlatformConfig,
)
logger = logging.getLogger(__name__)


def _get_supported_operations(spec: PlatformConfig) -> list[str]:
    from_config = spec.get("supported_operations")
    if isinstance(from_config, list) and from_config:
        return [str(op) for op in from_config]
    return []


def _get_configured_unsupported_operations(spec: PlatformConfig) -> list[str]:
    from_config = spec.get("unsupported_operations")
    if isinstance(from_config, list):
        return [str(op) for op in from_config]
    return []


def _get_layer_check_mode(spec: PlatformConfig) -> str:
    mode = spec.get("layer_check_mode")
    if mode in ("allowlist", "denylist"):
        return str(mode)
    if _get_supported_operations(spec):
        return "allowlist"
    if _get_configured_unsupported_operations(spec):
        return "denylist"
    return "denylist"


class CompatibilityStage:
    """Run supported-operation and unsupported-operation checks for every board × toolchain."""

    def __init__(self, platform_config_path: str | Path | None = None) -> None:
        self._config_path = Path(platform_config_path or DEFAULT_CATALOGUE_PATH)

    def run(self, profile: ModelProfile) -> CompatibilityReport:
        catalogue = PlatformCatalogue.load(self._config_path)
        board_ids = catalogue.board_ids()

        results: list[PlatformCompatResult] = []
        for board_id in board_ids:
            family = catalogue.platform_family(board_id)
            for toolchain in catalogue.list_toolchains(board_id):
                spec = catalogue.resolve_board_config(board_id, toolchain)
                results.append(
                    self._analyze_board(board_id, toolchain, spec, family, profile)
                )

        viable_boards: set[str] = set()
        for r in results:
            if r.status != CompatibilityStatus.INCOMPATIBLE:
                viable_boards.add(r.platform_id)
        return CompatibilityReport(
            results=results,
            viable_platform_ids=sorted(viable_boards),
        )

    def _analyze_board(
        self,
        board_id: str,
        toolchain: str,
        board_spec: PlatformConfig,
        platform_family: str,
        profile: ModelProfile,
    ) -> PlatformCompatResult:
        unsupported = self._collect_unsupported_operations(board_spec, profile)
        status = self._compatibility_status(unsupported, profile)

        notes_parts = [f"toolchain={toolchain}"]
        if unsupported:
            notes_parts.append(f"unsupported operations: {', '.join(unsupported)}")

        if status == CompatibilityStatus.INCOMPATIBLE:
            return PlatformCompatResult(
                platform_id=board_id,
                platform_family=platform_family,
                toolchain=toolchain,
                status=CompatibilityStatus.INCOMPATIBLE,
                unsupported_operations=unsupported,
                notes="; ".join(notes_parts),
            )

        return PlatformCompatResult(
            platform_id=board_id,
            platform_family=platform_family,
            toolchain=toolchain,
            status=status,
            unsupported_operations=unsupported,
            notes="; ".join(notes_parts) if unsupported else "Compatible",
        )

    def _collect_unsupported_operations(
        self, spec: PlatformConfig, profile: ModelProfile
    ) -> list[str]:
        """Collect unsupported operations using allowlist, denylist, or both."""
        mode = _get_layer_check_mode(spec)
        hits: list[str] = []
        if mode == "denylist":
            hits.extend(self._unsupported_from_denylist(spec, profile))
        else:
            hits.extend(self._unsupported_from_allowlist(spec, profile))
            if _get_configured_unsupported_operations(spec):
                hits.extend(self._unsupported_from_denylist(spec, profile))

        merged: list[str] = []
        seen: set[str] = set()
        for layer in hits:
            if layer not in seen:
                seen.add(layer)
                merged.append(layer)
        return merged

    def _unsupported_from_allowlist(
        self, spec: PlatformConfig, profile: ModelProfile
    ) -> list[str]:
        supported_ops = set(_get_supported_operations(spec))
        if not supported_ops:
            return []

        return [
            layer
            for layer in profile.layer_type_inventory
            if not self._layer_supported(layer, supported_ops)
        ]

    def _unsupported_from_denylist(
        self, spec: PlatformConfig, profile: ModelProfile
    ) -> list[str]:
        denylist = _get_configured_unsupported_operations(spec)
        return [
            layer
            for layer in profile.layer_type_inventory
            if self._layer_in_denylist(layer, denylist)
        ]

    def _layer_in_denylist(self, layer: str, denylist: list[str]) -> bool:
        deny = set(denylist)
        return layer in deny or self._fuzzy_match(layer, deny)

    def _compatibility_status(
        self, unsupported: list[str], profile: ModelProfile
    ) -> CompatibilityStatus:
        if not unsupported:
            return CompatibilityStatus.FULLY_COMPATIBLE

        ratio = len(unsupported) / max(len(profile.layer_type_inventory), 1)
        if ratio > 0.5:
            return CompatibilityStatus.INCOMPATIBLE

        return CompatibilityStatus.PARTIALLY_COMPATIBLE

    def _layer_supported(self, layer: str, supported: set[str]) -> bool:
        return layer in supported or self._fuzzy_match(layer, supported)

    def _fuzzy_match(self, layer: str, supported: set[str]) -> bool:
        layer_lower = layer.lower()
        for name in supported:
            name_lower = name.lower()
            if layer_lower == name_lower:
                return True
            if len(name_lower) < 4:
                continue
            if name_lower in layer_lower or layer_lower in name_lower:
                return True
        return False
