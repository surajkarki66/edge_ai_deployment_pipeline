from __future__ import annotations

import sys

from edge_ai_deployment_pipeline.core.types import PlatformRanking, SelectedPlatform


def ranking_to_selected(ranking: PlatformRanking) -> SelectedPlatform:
    return SelectedPlatform(
        platform_id=ranking.platform_id,
        platform_family=ranking.platform_family,
        toolchain=ranking.toolchain,
        score_breakdown=ranking.score_breakdown,
        reasoning=ranking.reasoning,
    )


def find_ranking(rankings: list[PlatformRanking], board_id: str) -> PlatformRanking:
    matches = [r for r in rankings if r.platform_id == board_id]
    if not matches:
        available = ", ".join(r.platform_id for r in rankings)
        raise ValueError(
            f"Board {board_id!r} is not among ranked candidates. Available: {available}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Board {board_id!r} matched multiple ranked entries; use rank number interactively"
        )
    return matches[0]


def apply_selection(
    rankings: list[PlatformRanking],
    chosen: PlatformRanking,
) -> tuple[SelectedPlatform, list[PlatformRanking]]:
    updated = [
        PlatformRanking(
            rank=ranking.rank,
            platform_id=ranking.platform_id,
            platform_family=ranking.platform_family,
            toolchain=ranking.toolchain,
            score_breakdown=ranking.score_breakdown,
            reasoning=ranking.reasoning,
            selected=ranking.rank == chosen.rank,
        )
        for ranking in rankings
    ]
    return ranking_to_selected(chosen), updated


def format_ranking_line(ranking: PlatformRanking) -> str:
    toolchain = ranking.toolchain or "default"
    score = ranking.score_breakdown.total_score
    return (
        f"  {ranking.rank}. {ranking.platform_id} [{toolchain}] "
        f"score={score:.1f} — {ranking.reasoning}"
    )


def prompt_board_selection(rankings: list[PlatformRanking]) -> PlatformRanking:
    if not rankings:
        raise ValueError("No ranked boards available to select")

    lines = ["", "Ranked boards (best first):"]
    lines.extend(format_ranking_line(ranking) for ranking in rankings)
    lines.append("")
    lines.append(
        "Select a board by rank number or board id "
        f"(press Enter for rank 1: {rankings[0].platform_id}): "
    )
    sys.stderr.write("\n".join(lines))

    while True:
        sys.stderr.write("> ")
        sys.stderr.flush()
        try:
            choice = input().strip()
        except EOFError as exc:
            raise ValueError(
                "Board selection required but no input available. "
                "Use --board-id or --auto-select."
            ) from exc

        if not choice:
            return rankings[0]

        if choice.isdigit():
            rank = int(choice)
            for ranking in rankings:
                if ranking.rank == rank:
                    return ranking
            sys.stderr.write(f"Invalid rank {rank}. Choose 1–{len(rankings)}.\n")
            continue

        try:
            return find_ranking(rankings, choice)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")


def resolve_board_choice(
    rankings: list[PlatformRanking],
    *,
    board_id: str | None = None,
    auto_select: bool = False,
    interactive: bool = True,
) -> PlatformRanking:
    if not rankings:
        raise ValueError("No ranked boards available to select")

    if len(rankings) == 1:
        return rankings[0]

    if board_id is not None:
        return find_ranking(rankings, board_id)

    if auto_select:
        return rankings[0]

    if interactive and sys.stdin.isatty():
        return prompt_board_selection(rankings)

    board_list = ", ".join(r.platform_id for r in rankings)
    raise ValueError(
        f"{len(rankings)} boards qualify ({board_list}). "
        "Pass --board-id to choose one, or --auto-select to deploy to the top-ranked board."
    )
