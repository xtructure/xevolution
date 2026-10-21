"""Structural extension contracts for application code."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from random import Random
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import (
    EvaluationResult,
    Individual,
    JSONValue,
    Population,
    RunState,
    StopDecision,
    VariationResult,
)

if TYPE_CHECKING:
    from .config import EvolutionSettings
    from .randomness import EvaluationContext


class Initializer[G](Protocol):
    def __call__(self, rng: Random) -> G: ...


class Clone[G](Protocol):
    def __call__(self, genotype: G) -> G: ...


class Decoder[G, P](Protocol):
    def __call__(self, genotype: G) -> P: ...


class IndividualEvaluator[G](Protocol):
    def __call__(
        self, individual: Individual[G], context: EvaluationContext
    ) -> EvaluationResult: ...


class PopulationEvaluator[G](Protocol):
    def __call__(
        self, population: Population[G], context: EvaluationContext
    ) -> Mapping[int, EvaluationResult] | Iterable[tuple[int, EvaluationResult]]: ...


class ParentSelector[G](Protocol):
    def __call__(self, population: Population[G], rng: Random) -> int: ...


class SurvivorSelector[G](Protocol):
    def __call__(
        self, population: Population[G], target_size: int, rng: Random
    ) -> Sequence[int]: ...


class Mutation[G](Protocol):
    def __call__(self, genotype: G, rng: Random) -> VariationResult[G]: ...


class Crossover[G](Protocol):
    def __call__(self, first: G, second: G, rng: Random) -> VariationResult[G]: ...


class StopPredicate[G](Protocol):
    def __call__(self, state: RunState[G]) -> StopDecision | bool: ...


class AdaptiveHook[G](Protocol):
    def __call__(self, state: RunState[G]) -> EvolutionSettings | None: ...


@runtime_checkable
class Stateful(Protocol):
    """Export every field that affects future scientific behavior."""

    def state_dict(self) -> JSONValue: ...

    def load_state_dict(self, state: JSONValue) -> None: ...


class AttachmentWriter(Protocol):
    def put(
        self, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> JSONValue: ...


class AttachmentReader(Protocol):
    def get(self, reference: JSONValue) -> bytes: ...


class PayloadCodec[G](Protocol):
    @property
    def codec_id(self) -> str: ...

    @property
    def version(self) -> int: ...

    def encode(self, value: G, writer: AttachmentWriter | None = None) -> JSONValue: ...

    def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> G: ...
