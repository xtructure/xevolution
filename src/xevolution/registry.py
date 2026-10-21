"""Explicit factory names and imports; no plugin discovery or object loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

from .config import RunConfig
from .errors import ConfigurationError
from .experiment import Experiment
from .randomness import RunRandom

type ExperimentFactory = Callable[[RunConfig, RunRandom], Experiment[Any]]


class ExperimentRegistry:
    """Resolve registered aliases or trusted ``package.module:factory`` code.

    Configuration factory references execute Python on construction. Load only
    configurations from trusted sources. Registration is explicit and local to
    this registry; importing xevolution performs no discovery.
    """

    def __init__(
        self, factories: Mapping[str, ExperimentFactory] | None = None
    ) -> None:
        self._factories: dict[str, ExperimentFactory] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    def register(self, name: str, factory: ExperimentFactory) -> None:
        """Register a unique short name; accidental replacement is an error."""
        if not isinstance(name, str) or not name or ":" in name:
            raise ConfigurationError(
                "factory aliases must be nonempty names without ':'"
            )
        if name in self._factories:
            raise ConfigurationError(f"factory alias already registered: {name}")
        if not callable(factory):
            raise ConfigurationError("experiment factory must be callable")
        self._factories[name] = factory

    def resolve(self, reference: str) -> ExperimentFactory:
        """Resolve one exact factory reference without invoking it."""
        if not isinstance(reference, str) or not reference:
            raise ConfigurationError("an experiment factory reference is required")
        if reference in self._factories:
            return self._factories[reference]
        module_name, separator, attribute = reference.partition(":")
        if not separator or not module_name or not attribute or ":" in attribute:
            raise ConfigurationError(
                f"unknown factory {reference!r}; use a registered alias or package.module:factory"
            )
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise ConfigurationError(
                f"cannot resolve factory {reference!r}: {exc}"
            ) from exc
        if not callable(factory):
            raise ConfigurationError(f"factory {reference!r} is not callable")
        return cast(ExperimentFactory, factory)
