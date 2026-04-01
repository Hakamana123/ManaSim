"""
Memory backend factory.

Select the backend at startup via the MEMORY_BACKEND environment variable
(default: "supabase").  Any module placed under services/memory/ that
exports a class named MemoryBackendImpl and inherits from MemoryBackend
will be automatically available as a named backend.
"""

from __future__ import annotations

import importlib

from .base import (
    EdgeData,
    EntityNode,
    EpisodeStatus,
    FilteredEntities,
    MemoryBackend,
    NodeData,
    SearchResults,
)

_backend_singleton: MemoryBackend | None = None


def get_memory_backend() -> MemoryBackend:
    """
    Return a singleton instance of the configured memory backend.

    The backend is selected via Config.MEMORY_BACKEND (default: "supabase").
    The first call instantiates the backend; subsequent calls return the
    same instance.
    """
    global _backend_singleton
    if _backend_singleton is None:
        # Deferred import avoids circular dependency at module load time.
        from ...config import Config

        backend_name = Config.MEMORY_BACKEND.lower()
        try:
            module = importlib.import_module(f".{backend_name}", package=__name__)
            impl_class = getattr(module, "MemoryBackendImpl")
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                f"Unknown MEMORY_BACKEND '{backend_name}'. "
                f"Create services/memory/{backend_name}.py "
                f"that exports MemoryBackendImpl(MemoryBackend)."
            ) from exc

        _backend_singleton = impl_class()

    return _backend_singleton


__all__ = [
    "MemoryBackend",
    "NodeData",
    "EdgeData",
    "EpisodeStatus",
    "SearchResults",
    "EntityNode",
    "FilteredEntities",
    "get_memory_backend",
]
