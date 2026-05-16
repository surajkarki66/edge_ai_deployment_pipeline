"""Stage 2 — per-board compatibility for computer-vision models (parallel checks)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.catalogue import DEFAULT_CATALOGUE_PATH, PlatformCatalogue
from core.types import (
    CompatibilityReport,
    CompatibilityStatus,
    ModelProfile,
    PlatformCompatResult,
    PlatformConfig,
)
from operators.registry import get_supported_operations

logger = logging.getLogger(__name__)

PROBLEM_PATTERNS: dict[str, list[str]] = {
    "jetson": ["Einsum"],
    "fpga": ["MultiheadAttention", "Attention", "Transpose", "Einsum"],
    "mcu": ["LayerNorm", "GroupNorm", "InstanceNorm2d"],
    "arm": [],
}


class CompatibilityStage:
    """Run supported-operation and problem-pattern checks for every board × toolchain."""

    def __init__(self, platform_config_path: str | Path | None = None) -> None:
        self._config_path = Path(platform_config_path or DEFAULT_CATALOGUE_PATH)

    def run(self, profile: ModelProfile) -> CompatibilityReport:
        catalogue = PlatformCatalogue.load(self._config_path)
        board_ids = catalogue.board_ids()

        tasks: list[tuple[str, str, PlatformConfig, str]] = []
        for board_id in board_ids:
            family = catalogue.platform_family(board_id)
            for toolchain in catalogue.list_toolchains(board_id):
                spec = catalogue.resolve_board_config(board_id, toolchain)
                tasks.append((board_id, toolchain, spec, family))

        results: list[PlatformCompatResult] = []
        with ThreadPoolExecutor(max_workers=len(tasks) or 1) as pool:
            futures = {
                pool.submit(self._analyze_board, bid, tc, spec, fam, profile): (bid, tc)
                for bid, tc, spec, fam in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: (r.platform_id, r.toolchain or ""))
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
        op_status, unsupported, penalty = self._check_supported_operations(
            board_spec, profile
        )
        problem_notes = self._check_problem_patterns(platform_family, profile)

        notes_parts = [f"toolchain={toolchain}", problem_notes]
        if unsupported:
            notes_parts.append(f"unsupported ops: {', '.join(unsupported)}")

        if op_status == CompatibilityStatus.INCOMPATIBLE:
            return PlatformCompatResult(
                platform_id=board_id,
                platform_family=platform_family,
                toolchain=toolchain,
                status=CompatibilityStatus.INCOMPATIBLE,
                unsupported_layers=unsupported,
                fallback_penalty_estimate=None,
                notes="; ".join(filter(None, notes_parts)) or "Unsupported operations",
            )

        return PlatformCompatResult(
            platform_id=board_id,
            platform_family=platform_family,
            toolchain=toolchain,
            status=op_status,
            unsupported_layers=unsupported,
            fallback_penalty_estimate=penalty,
            notes="; ".join(filter(None, notes_parts)) or "Compatible",
        )

    def _check_supported_operations(
        self,
        spec: PlatformConfig,
        profile: ModelProfile,
    ) -> tuple[CompatibilityStatus, list[str], float | None]:
        supported_ops = set(get_supported_operations(spec))
        if not supported_ops:
            return CompatibilityStatus.FULLY_COMPATIBLE, [], None

        unsupported = [
            layer
            for layer in profile.layer_type_inventory
            if not self._layer_supported(layer, supported_ops)
        ]

        if not unsupported:
            return CompatibilityStatus.FULLY_COMPATIBLE, [], None

        ratio = len(unsupported) / max(len(profile.layer_type_inventory), 1)
        if ratio > 0.5:
            return CompatibilityStatus.INCOMPATIBLE, unsupported, None

        penalty = round(ratio * profile.cpu_latency_ms * 0.3, 2)
        return CompatibilityStatus.PARTIALLY_COMPATIBLE, unsupported, penalty

    def _layer_supported(self, layer: str, supported: set[str]) -> bool:
        return layer in supported or self._fuzzy_match(layer, supported)

    def _fuzzy_match(self, layer: str, supported: set[str]) -> bool:
        layer_lower = layer.lower()
        return any(s.lower() in layer_lower or layer_lower in s.lower() for s in supported)

    def _check_problem_patterns(self, platform_family: str, profile: ModelProfile) -> str:
        patterns = PROBLEM_PATTERNS.get(platform_family, [])
        found = [p for p in patterns if p in profile.layer_type_inventory]
        if found:
            return f"Known problem patterns: {', '.join(found)}"
        return ""
