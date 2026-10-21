"""Shared records; genotypes remain application-defined and unconstrained."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, overload

from .errors import ConfigurationError, RunStateError

if TYPE_CHECKING:
    from .config import EvolutionSettings

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
type JSONDict = dict[str, JSONValue]


def _nonnegative(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True, eq=False)
class Individual[G]:
    """A candidate identified by ID, never by equality of its genotype.

    The wrapper is immutable. The application owns genotype cloning; freezing
    this record does not freeze the genotype's internal data.
    """

    id: int
    genotype: G = field(repr=False)
    birth_generation: int = 0
    parent_ids: tuple[int, ...] = ()
    fitness: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    evaluation_count: int = 0
    metadata: Mapping[str, JSONValue] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _nonnegative(self.id, "individual ID")
        _nonnegative(self.birth_generation, "birth generation")
        _nonnegative(self.evaluation_count, "evaluation count")
        for parent in self.parent_ids:
            _nonnegative(parent, "parent ID")
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata)))
        )

    def age(self, generation: int) -> int:
        """Return age at a completed generation, with newborn age zero."""
        _nonnegative(generation, "generation")
        if generation < self.birth_generation:
            raise ValueError("generation precedes this individual's birth")
        return generation - self.birth_generation


@dataclass(frozen=True, slots=True, eq=False)
class Population[G]:
    """An ordered population with independent, unique ID lookup."""

    individuals: tuple[Individual[G], ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict, repr=False)
    by_id: Mapping[int, Individual[G]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        individuals = tuple(self.individuals)
        lookup = {individual.id: individual for individual in individuals}
        if len(lookup) != len(individuals):
            raise ConfigurationError("population contains duplicate individual IDs")
        object.__setattr__(self, "individuals", individuals)
        object.__setattr__(self, "by_id", MappingProxyType(lookup))
        object.__setattr__(
            self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata)))
        )

    def __iter__(self) -> Iterator[Individual[G]]:
        return iter(self.individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    @overload
    def __getitem__(self, index: int) -> Individual[G]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Individual[G], ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Individual[G] | tuple[Individual[G], ...]:
        return self.individuals[index]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One accepted scalar fitness observation and optional numeric metrics."""

    fitness: float
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class VariationResult[G]:
    """Application-produced child data; the framework assigns its identity."""

    genotype: G = field(repr=False)
    details: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """An immutable wrapper for one birth and its operation provenance."""

    individual_id: int
    birth_generation: int
    parent_ids: tuple[int, ...]
    operator: str
    details: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(
            self, "details", MappingProxyType(copy.deepcopy(dict(self.details)))
        )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Statistics with an explicit number of contributing observations."""

    count: int
    minimum: float | None
    mean: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class StopDecision:
    """The saved outcome of one completed generation's stopping policy."""

    stop: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.stop) is not bool:
            raise ConfigurationError("stop decision must contain a boolean")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ConfigurationError("stop reason must be a string or None")


@dataclass(frozen=True, slots=True, eq=False)
class RunState[G]:
    """Framework state at a generation boundary.

    ``generation_settings`` describes the completed population;
    ``effective_settings`` applies to its successor after adaptive updates.
    """

    run_id: str
    generation: int
    population: Population[G]
    generation_settings: EvolutionSettings
    effective_settings: EvolutionSettings
    next_individual_id: int
    evaluations: int = 0
    best: Individual[G] | None = None
    last_improvement_generation: int = 0
    statistics: Mapping[str, MetricSummary] = field(default_factory=dict)
    stop: StopDecision = field(default_factory=StopDecision)
    phase: Literal["ready", "terminated"] = "ready"
    next_event_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", MappingProxyType(dict(self.statistics)))


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """A deeply immutable scientific observation represented as canonical JSON.

    ``to_dict`` returns a fresh value so a consumer cannot alter retained
    history through a nested dictionary or list reference.
    """

    generation: int
    sequence: int
    data_json: str

    def to_dict(self) -> JSONDict:
        value: object = json.loads(self.data_json)
        if not isinstance(value, dict):
            raise RunStateError("generation record must contain a JSON object")
        return cast(JSONDict, value)
