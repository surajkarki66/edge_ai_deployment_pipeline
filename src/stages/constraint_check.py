"""Stage 3 — CV model requirement validation and board ranking."""

from __future__ import annotations

import logging
from pathlib import Path

from config.catalogue import DEFAULT_CATALOGUE_PATH, PlatformCatalogue
from core.types import (
    CompatibilityReport,
    CompatibilityStatus,
    ModelProfile,
    PlatformCompatResult,
    PlatformConfig,
    ScoreBreakdown,
    SelectedPlatform,
    UserRequirements,
)

logger = logging.getLogger(__name__)


class ConstraintCheckStage:
    """Score viable boards; pick the best toolchain per board, then rank boards."""

    def __init__(self, platform_config_path: str | Path | None = None) -> None:
        self._config_path = Path(platform_config_path or DEFAULT_CATALOGUE_PATH)

    def run(
        self,
        profile: ModelProfile,
        compatibility: CompatibilityReport,
        requirements: UserRequirements,
    ) -> SelectedPlatform:
        catalogue = PlatformCatalogue.load(self._config_path)
        candidates: list[tuple[str, str, str | None, ScoreBreakdown, str]] = []

        compat_for_board = self._group_compat(compatibility)

        for board_id in compatibility.viable_platform_ids:
            family = catalogue.platform_family(board_id)
            board_candidates = self._candidates_for_board(
                board_id,
                family,
                catalogue,
                compat_for_board.get(board_id, []),
                profile,
                requirements,
            )
            if board_candidates:
                candidates.append(max(board_candidates, key=lambda c: c[3].total_score))

        if not candidates:
            raise ValueError("No viable boards remain after compatibility filtering")

        candidates.sort(key=lambda c: c[3].total_score, reverse=True)
        best_id, best_family, best_toolchain, best_breakdown, best_reasoning = candidates[0]

        return SelectedPlatform(
            platform_id=best_id,
            platform_family=best_family,
            toolchain=best_toolchain,
            score_breakdown=best_breakdown,
            reasoning=best_reasoning,
        )

    def _group_compat(
        self, compatibility: CompatibilityReport
    ) -> dict[str, list[PlatformCompatResult]]:
        grouped: dict[str, list[PlatformCompatResult]] = {}
        for r in compatibility.results:
            grouped.setdefault(r.platform_id, []).append(r)
        return grouped

    def _candidates_for_board(
        self,
        board_id: str,
        family: str,
        catalogue: PlatformCatalogue,
        compat_results: list[PlatformCompatResult],
        profile: ModelProfile,
        requirements: UserRequirements,
    ) -> list[tuple[str, str, str | None, ScoreBreakdown, str]]:
        viable = [r for r in compat_results if r.status != CompatibilityStatus.INCOMPATIBLE]
        if not viable:
            return []

        toolchains = catalogue.list_toolchains(board_id)
        if requirements.toolchain:
            toolchains = [t for t in toolchains if t == requirements.toolchain]
            if not toolchains:
                raise ValueError(
                    f"Requested toolchain '{requirements.toolchain}' "
                    f"is not available on board '{board_id}'"
                )

        compat_by_tc = {r.toolchain: r for r in viable if r.toolchain}
        scored: list[tuple[str, str, str | None, ScoreBreakdown, str]] = []
        for toolchain in toolchains:
            compat = compat_by_tc.get(toolchain)
            if compat is None:
                continue
            spec = catalogue.resolve_board_config(board_id, toolchain)
            breakdown, reasoning = self._score_board(
                board_id, toolchain, profile, compat, spec, requirements
            )
            scored.append((board_id, family, toolchain, breakdown, reasoning))
        return scored

    def _score_board(
        self,
        board_id: str,
        toolchain: str | None,
        profile: ModelProfile,
        compat: PlatformCompatResult,
        spec: PlatformConfig,
        requirements: UserRequirements,
    ) -> tuple[ScoreBreakdown, str]:
        base_latency = float(spec.get("baseline_latency_ms", profile.cpu_latency_ms))
        estimated_latency = base_latency + (compat.fallback_penalty_estimate or 0.0)
        if compat.status == CompatibilityStatus.PARTIALLY_COMPATIBLE:
            estimated_latency *= 1.1

        power_w = float(spec.get("tdp_watts", 10.0))
        cost_usd = float(spec.get("cost_usd", 100.0))
        memory_mb = float(spec.get("memory_mb", 4096))

        latency_score = self._latency_score(estimated_latency, requirements.max_latency_ms)
        power_score = self._threshold_score(power_w, requirements.max_power_watts, lower_is_better=True)
        cost_score = self._threshold_score(cost_usd, requirements.max_cost_usd, lower_is_better=True)

        runtime_mb = max(profile.peak_activation_memory_mb, profile.inference_ram_mb)
        memory_ok = profile.model_size_mb + runtime_mb <= memory_mb
        memory_penalty = 0.0 if memory_ok else 50.0

        fallback_penalty = compat.fallback_penalty_estimate or 0.0
        weights = {"latency": 0.5, "power": 0.25, "cost": 0.25}
        scores = [s for s in (latency_score, power_score, cost_score) if s is not None]

        if not scores:
            total = 100.0 - memory_penalty - fallback_penalty
        else:
            total = 0.0
            if latency_score is not None:
                total += latency_score * weights["latency"]
            if power_score is not None:
                total += power_score * weights["power"]
            if cost_score is not None:
                total += cost_score * weights["cost"]
            total -= memory_penalty + fallback_penalty

        display_name = spec.get("display_name", board_id)
        details: dict[str, object] = {
            "board_id": board_id,
            "platform_family": compat.platform_family,
            "display_name": display_name,
            "estimated_latency_ms": estimated_latency,
            "power_watts": power_w,
            "cost_usd": cost_usd,
            "memory_mb": memory_mb,
            "memory_ok": memory_ok,
        }
        if toolchain:
            details["toolchain"] = toolchain

        reasoning_parts = [
            f"{display_name} ({board_id}): score={total:.1f}",
            f"est. latency {estimated_latency:.1f}ms",
        ]
        if toolchain:
            reasoning_parts[0] = f"{display_name} [{toolchain}]: score={total:.1f}"
        if compat.status == CompatibilityStatus.PARTIALLY_COMPATIBLE:
            reasoning_parts.append(f"partial compat (+{fallback_penalty:.1f}ms penalty)")
        if not memory_ok:
            reasoning_parts.append("memory constraint exceeded")

        return (
            ScoreBreakdown(
                platform_id=board_id,
                total_score=round(max(total, 0.0), 2),
                latency_score=latency_score,
                power_score=power_score,
                cost_score=cost_score,
                fallback_penalty_applied=fallback_penalty,
                details=details,
            ),
            "; ".join(reasoning_parts),
        )

    def _latency_score(self, estimated_ms: float, max_ms: float | None) -> float | None:
        if max_ms is None:
            return None
        if estimated_ms <= max_ms:
            return 100.0 * (1.0 - estimated_ms / max_ms * 0.5)
        overrun = (estimated_ms - max_ms) / max_ms
        return max(0.0, 100.0 - overrun * 100)

    def _threshold_score(
        self,
        value: float,
        limit: float | None,
        *,
        lower_is_better: bool,
    ) -> float | None:
        if limit is None:
            return None
        if lower_is_better:
            if value <= limit:
                return 100.0 * (1.0 - value / limit * 0.3)
            return max(0.0, 100.0 - (value - limit) / limit * 100)
        if value >= limit:
            return 100.0
        return 100.0 * value / limit
