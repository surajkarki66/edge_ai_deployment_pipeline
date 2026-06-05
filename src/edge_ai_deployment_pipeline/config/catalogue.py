from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from edge_ai_deployment_pipeline.core.types import PlatformConfig

DEFAULT_CATALOGUE_PATH = Path(__file__).resolve().parent / "platforms.json"

_PLATFORM_META_KEYS = frozenset({"boards", "adapter_id", "toolchains", "default_toolchain"})


class PlatformCatalogue:
    """Flattened view: each board is a deployable target with merged platform + board specs."""

    def __init__(self, boards: dict[str, PlatformConfig]) -> None:
        self._boards = boards

    @classmethod
    def load(cls, path: str | Path | None = None) -> PlatformCatalogue:
        catalogue_path = Path(path or DEFAULT_CATALOGUE_PATH)
        if not catalogue_path.exists():
            return cls({})

        with open(catalogue_path, encoding="utf-8") as f:
            data: Any = json.load(f)

        platforms = data.get("platforms", {})
        if not isinstance(platforms, dict):
            return cls({})

        boards: dict[str, PlatformConfig] = {}
        for platform_family, platform_spec in platforms.items():
            if not isinstance(platform_spec, dict):
                continue
            board_entries = platform_spec.get("boards", {})
            if not isinstance(board_entries, dict):
                continue

            platform_defaults = {
                k: v for k, v in platform_spec.items() if k not in _PLATFORM_META_KEYS
            }
            adapter_id = platform_spec.get("adapter_id", platform_family)
            platform_toolchains = platform_spec.get("toolchains", {})

            for board_id, board_spec in board_entries.items():
                if not isinstance(board_spec, dict):
                    continue
                merged: PlatformConfig = {
                    **platform_defaults,
                    **board_spec,
                    "board_id": board_id,
                    "platform_family": platform_family,
                    "adapter_id": adapter_id,
                }
                if isinstance(platform_toolchains, dict):
                    merged["toolchains"] = platform_toolchains
                if "default_toolchain" in platform_spec:
                    merged["default_toolchain"] = platform_spec["default_toolchain"]
                boards[board_id] = merged

        return cls(boards)

    def board_ids(self) -> list[str]:
        return list(self._boards.keys())

    def get_board(self, board_id: str) -> PlatformConfig:
        if board_id not in self._boards:
            raise KeyError(f"Unknown board: {board_id}")
        return dict(self._boards[board_id])

    def all_boards(self) -> dict[str, PlatformConfig]:
        return {k: dict(v) for k, v in self._boards.items()}

    def platform_family(self, board_id: str) -> str:
        return str(self.get_board(board_id).get("platform_family", board_id))

    def adapter_id(self, board_id: str) -> str:
        return str(self.get_board(board_id).get("adapter_id", self.platform_family(board_id)))

    def list_toolchains(self, board_id: str) -> list[str]:
        """Return deployable toolchains declared on the board."""
        board = self.get_board(board_id)
        available = board.get("available_toolchains")
        if not isinstance(available, list) or not available:
            raise ValueError(
                f"Board '{board_id}' must define a non-empty available_toolchains list"
            )
        return [str(t) for t in available]

    def resolve_board_config(self, board_id: str, toolchain: str) -> PlatformConfig:
        """Merge platform, board, toolchain, and per-board toolchain overrides."""
        base = self.get_board(board_id)
        toolchains_cfg = base.get("toolchains", {})
        if not isinstance(toolchains_cfg, dict):
            toolchains_cfg = {}

        tc_spec = dict(toolchains_cfg.get(toolchain, {}))
        overrides = base.get("toolchain_overrides", {})
        if isinstance(overrides, dict) and toolchain in overrides:
            board_tc = overrides[toolchain]
            if isinstance(board_tc, dict):
                tc_spec = {**tc_spec, **board_tc}

        merged: PlatformConfig = {**base, **tc_spec, "toolchain": toolchain}
        if "supported_operations" in tc_spec:
            merged["supported_operations"] = tc_spec["supported_operations"]
        if "layer_check_mode" in tc_spec:
            merged["layer_check_mode"] = tc_spec["layer_check_mode"]
        denylist = tc_spec.get("unsupported_operations")
        if isinstance(denylist, list):
            merged["unsupported_operations"] = denylist
        else:
            merged.pop("unsupported_operations", None)
        return merged
