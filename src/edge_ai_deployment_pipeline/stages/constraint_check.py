from __future__ import annotations

import logging
from pathlib import Path

from edge_ai_deployment_pipeline.config.catalogue import DEFAULT_CATALOGUE_PATH, PlatformCatalogue
from edge_ai_deployment_pipeline.core.types import (
    CompatibilityReport,
    CompatibilityStatus,
    ConstraintCheckResult,
    ModelProfile,
    PlatformCompatResult,
    PlatformConfig,
    PlatformRanking,
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
    ) -> ConstraintCheckResult:
        catalogue = PlatformCatalogue.load(self._config_path)
        candidates: list[tuple[str, str, str | None, ScoreBreakdown, str]] = []
        rejections: list[tuple[str, str, list[str]]] = []

        compat_for_board = self._group_compat(compatibility)

        for board_id in compatibility.viable_platform_ids:
            board_spec = catalogue.get_board(board_id)
            display_name = str(board_spec.get("display_name", board_id))
            reasons = self._rejection_reasons(board_spec, requirements)
            if reasons:
                rejections.append((board_id, display_name, reasons))
                continue

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
            else:
                rejections.append(
                    (
                        board_id,
                        display_name,
                        ["no compatible toolchain available for this board"],
                    )
                )

        if not candidates:
            raise ValueError(self._format_no_selection_error(requirements, rejections))

        candidates.sort(key=lambda c: c[3].total_score, reverse=True)
        rankings = [
            PlatformRanking(
                rank=index,
                platform_id=board_id,
                platform_family=family,
                toolchain=toolchain,
                score_breakdown=breakdown,
                reasoning=reasoning,
                selected=False,
            )
            for index, (board_id, family, toolchain, breakdown, reasoning) in enumerate(
                candidates, start=1
            )
        ]
        best_id, best_family, best_toolchain, best_breakdown, best_reasoning = candidates[0]

        return ConstraintCheckResult(
            selected=SelectedPlatform(
                platform_id=best_id,
                platform_family=best_family,
                toolchain=best_toolchain,
                score_breakdown=best_breakdown,
                reasoning=best_reasoning,
            ),
            rankings=rankings,
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

    @staticmethod
    def _rejection_reasons(
        board_spec: PlatformConfig,
        requirements: UserRequirements,
    ) -> list[str]:
        reasons: list[str] = []
        configured = board_spec.get("latency_type")
        if configured is None:
            board_id = board_spec.get("board_id", "<unknown>")
            raise ValueError(f"Board '{board_id}' is missing latency_type in platform config")
        if str(configured) != requirements.latency_type.value:
            reasons.append(
                f"latency_type {configured!r} does not match requested "
                f"{requirements.latency_type.value!r}"
            )

        power_w = float(board_spec.get("tdp_watts", 0))
        cost_usd = float(board_spec.get("cost_usd", 0))
        if requirements.max_power_watts is not None and power_w > requirements.max_power_watts:
            reasons.append(
                f"power {power_w:.1f}W exceeds max budget {requirements.max_power_watts:.1f}W"
            )
        if requirements.max_cost_usd is not None and cost_usd > requirements.max_cost_usd:
            reasons.append(
                f"cost ${cost_usd:.0f} exceeds max budget ${requirements.max_cost_usd:.0f}"
            )
        return reasons

    @staticmethod
    def _format_no_selection_error(
        requirements: UserRequirements,
        rejections: list[tuple[str, str, list[str]]],
    ) -> str:
        lines = [
            "No board selected for the given constraints:",
            f"  latency_type={requirements.latency_type.value!r}",
            f"  max_power_w={requirements.max_power_watts:.1f}W",
            f"  max_cost_usd=${requirements.max_cost_usd:.0f}",
        ]
        if rejections:
            lines.append("")
            lines.append("Rejected boards:")
            for board_id, display_name, reasons in rejections:
                reason_text = "; ".join(reasons)
                lines.append(f"  - {display_name} ({board_id}): {reason_text}")
        else:
            lines.append("")
            lines.append("No compatible boards were available to evaluate.")
        return "\n".join(lines)

    def _score_board(
        self,
        board_id: str,
        toolchain: str | None,
        profile: ModelProfile,
        compat: PlatformCompatResult,
        spec: PlatformConfig,
        requirements: UserRequirements,
    ) -> tuple[ScoreBreakdown, str]:
        power_w = float(spec.get("tdp_watts"))
        cost_usd = float(spec.get("cost_usd"))
        memory_mb = float(spec.get("memory_mb"))

        power_score = self._threshold_score(
            power_w, requirements.max_power_watts, lower_is_better=True
        )
        cost_score = self._threshold_score(
            cost_usd, requirements.max_cost_usd, lower_is_better=True
        )

        runtime_mb = max(profile.peak_activation_memory_mb, profile.inference_ram_mb)
        memory_ok = profile.model_size_mb + runtime_mb <= memory_mb
        memory_penalty = 0.0 if memory_ok else 50.0

        weights = {"power": 0.5, "cost": 0.5}
        total = 0.0
        if power_score is not None:
            total += power_score * weights["power"]
        if cost_score is not None:
            total += cost_score * weights["cost"]
        total -= memory_penalty

        display_name = spec.get("display_name", board_id)
        details: dict[str, object] = {
            "board_id": board_id,
            "platform_family": compat.platform_family,
            "display_name": display_name,
            "latency_type": spec.get("latency_type"),
            "power_watts": power_w,
            "cost_usd": cost_usd,
            "memory_mb": memory_mb,
            "memory_ok": memory_ok,
        }
        if toolchain:
            details["toolchain"] = toolchain

        reasoning_parts = [
            f"{display_name} ({board_id}): score={total:.1f}",
            f"power {power_w:.1f}W",
            f"cost ${cost_usd:.0f}",
        ]
        if toolchain:
            reasoning_parts[0] = f"{display_name} [{toolchain}]: score={total:.1f}"
        if compat.status == CompatibilityStatus.PARTIALLY_COMPATIBLE:
            reasoning_parts.append("partial compatibility")
        if not memory_ok:
            reasoning_parts.append("memory constraint exceeded")

        return (
            ScoreBreakdown(
                platform_id=board_id,
                total_score=round(max(total, 0.0), 2),
                power_score=power_score,
                cost_score=cost_score,
                details=details,
            ),
            "; ".join(reasoning_parts),
        )

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
            return 0.0
        if value >= limit:
            return 100.0
        return 100.0 * value / limit
