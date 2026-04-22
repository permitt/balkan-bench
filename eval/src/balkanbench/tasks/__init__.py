"""Task registry.

``@register_task(task_type_1, task_type_2, ...)`` decorates a ``Task`` subclass
with one or more ``task_type`` strings from ``schemas/task_spec.json`` enum.
``get_task_class(task_type)`` looks up a registered class.

Community contributions can register new task types without touching core code
by importing their module; the decorator auto-registers on import.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from balkanbench.tasks.base import Task

_T = TypeVar("_T", bound=type[Task])


class TaskNotFoundError(KeyError):
    """Raised when a task_type is not in the registry."""


_REGISTRY: dict[str, type[Task]] = {}


def register_task(*task_types: str) -> Callable[[_T], _T]:
    """Register ``cls`` under every ``task_type`` in ``task_types``.

    Raises ``ValueError`` if any ``task_type`` is already registered so
    contributors cannot silently shadow core implementations.
    """
    if not task_types:
        raise ValueError("register_task() requires at least one task_type")

    def decorator(cls: _T) -> _T:
        for t in task_types:
            if t in _REGISTRY:
                raise ValueError(f"task_type {t!r} is already registered")
            _REGISTRY[t] = cls
        return cls

    return decorator


def get_task_class(task_type: str) -> type[Task]:
    """Return the registered ``Task`` subclass for ``task_type``."""
    try:
        return _REGISTRY[task_type]
    except KeyError as exc:
        raise TaskNotFoundError(
            f"task_type {task_type!r} is not registered; known: {sorted(_REGISTRY)}"
        ) from exc


def list_task_types() -> list[str]:
    """Return sorted list of registered task_type strings."""
    return sorted(_REGISTRY)


__all__ = [
    "Task",
    "TaskNotFoundError",
    "register_task",
    "get_task_class",
    "list_task_types",
]


def _autoregister_builtin_tasks() -> None:
    """Import every builtin task module so their decorators populate the registry."""
    # Imports have side effects (the @register_task decorator runs at import time).
    # Keep this list sorted and in sync with new builtin task modules.
    from balkanbench.tasks import classification as _classification  # noqa: F401
    from balkanbench.tasks import diagnostic as _diagnostic  # noqa: F401
    from balkanbench.tasks import multiple_choice as _multiple_choice  # noqa: F401
    from balkanbench.tasks import multirc as _multirc  # noqa: F401
    from balkanbench.tasks import wsc as _wsc  # noqa: F401


_autoregister_builtin_tasks()
