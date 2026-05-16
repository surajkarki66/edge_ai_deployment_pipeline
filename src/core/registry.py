"""Decorator-based registries for platform adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from adapters.base import PlatformAdapter

T = TypeVar("T")


class Registry:
    """Generic name → object registry with decorator registration."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, object] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._entries:
                raise ValueError(f"{self._name}: duplicate key '{key}'")
            self._entries[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> object:
        if key not in self._entries:
            raise KeyError(f"{self._name}: unknown key '{key}'")
        return self._entries[key]

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def items(self) -> list[tuple[str, object]]:
        return list(self._entries.items())

    def __contains__(self, key: str) -> bool:
        return key in self._entries


class PlatformAdapterRegistry(Registry):
    def get_adapter(self, platform_id: str) -> PlatformAdapter:
        from adapters.base import PlatformAdapter

        entry = self.get(platform_id)
        if isinstance(entry, PlatformAdapter):
            return entry
        if isinstance(entry, type):
            instance = entry()
            if isinstance(instance, PlatformAdapter):
                return instance
        raise TypeError(f"Registered object for '{platform_id}' is not a PlatformAdapter")


platform_registry = PlatformAdapterRegistry("platform_adapter")
