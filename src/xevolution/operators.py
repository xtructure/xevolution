"""Application variation operators and per-attempt weighted operator pools."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random

from .config import EvolutionSettings
from .errors import ConfigurationError
from .models import JSONDict, JSONValue, VariationResult
from .protocols import Clone, Crossover, Mutation
from .selection import weighted_index


def _weight(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError("Operator weight must be a nonnegative finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigurationError(
            "Operator weight is outside the finite range"
        ) from error
    if not isfinite(result) or result < 0:
        raise ConfigurationError("Operator weight must be a nonnegative finite number")
    return result


class OperatorSpec[O]:
    """A named operation with checkpointed mutable weight and explicit fallback.

    Name, operation, and fallback designation are construction-time identity.
    Only ``weight`` can be adapted; operation-owned state is registered separately.
    Fallback operations are attempted in registration order after weighted ones.
    """

    __slots__ = ("_name", "_operation", "_weight", "_fallback")

    def __init__(
        self, name: str, operation: O, weight: float = 1.0, fallback: bool = False
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("An operator requires a nonempty name")
        if not callable(operation):
            raise ConfigurationError(f"Operator {name!r} must be callable")
        if not isinstance(fallback, bool):
            raise ConfigurationError("Operator fallback designation must be boolean")
        self._name = name
        self._operation = operation
        self._weight = _weight(weight)
        self._fallback = fallback

    @property
    def name(self) -> str:
        return self._name

    @property
    def operation(self) -> O:
        return self._operation

    @property
    def fallback(self) -> bool:
        return self._fallback

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = _weight(value)

    def state_dict(self) -> JSONDict:
        return {"weight": self.weight}

    def load_state_dict(self, state: JSONValue) -> None:
        if not isinstance(state, dict) or set(state) != {"weight"}:
            raise ConfigurationError("Operator state must contain only 'weight'")
        self._weight = _weight(state["weight"])


def validate_pool[O](pool: Sequence[OperatorSpec[O]]) -> bool:
    """Validate stable identities and return whether any operation can be tried."""
    names: set[str] = set()
    available = False
    for spec in pool:
        if not isinstance(spec, OperatorSpec):
            raise ConfigurationError("Operator pools must contain OperatorSpec entries")
        if spec.name in names:
            raise ConfigurationError(
                f"Duplicate operator name {spec.name!r} in one pool"
            )
        names.add(spec.name)
        _weight(spec.weight)
        available = available or spec.fallback or spec.weight > 0
    return available


def validate_routes[G](
    settings: EvolutionSettings,
    mutations: Sequence[OperatorSpec[Mutation[G]]],
    crossovers: Sequence[OperatorSpec[Crossover[G]]],
) -> None:
    """Require an available operation for each enabled variation route.

    Recheck after adaptation: operator weights may change even when no new
    settings record is returned. Explicit fallbacks make a route available
    independently of its weighted pool.
    """
    if not isinstance(settings, EvolutionSettings):
        raise ConfigurationError("Variation settings must be EvolutionSettings")
    have_mutation = validate_pool(mutations)
    have_crossover = validate_pool(crossovers)
    if settings.mutation_probability > 0 and not have_mutation:
        raise ConfigurationError(
            "The enabled mutation route has no operator or fallback"
        )
    if settings.mutation_probability < 1 and not have_crossover:
        raise ConfigurationError(
            "The enabled crossover route has no operator or fallback"
        )


def weighted_order[O](
    pool: Sequence[OperatorSpec[O]], rng: Random
) -> Iterator[OperatorSpec[O]]:
    """Lazily try weighted operations without replacement, then fixed fallbacks.

    Each invocation has its own remaining pool. Unused operations consume no
    draws after a caller obtains a successful result and stops iteration.
    """
    validate_pool(pool)
    remaining = [spec for spec in pool if not spec.fallback and spec.weight > 0]
    while remaining:
        index = weighted_index([spec.weight for spec in remaining], rng)
        yield remaining.pop(index)
    yield from (spec for spec in pool if spec.fallback)


@dataclass(frozen=True, slots=True)
class CloneMutation[G]:
    """Create an independent copy using the application's clone contract."""

    clone: Clone[G]

    def __call__(self, genotype: G, rng: Random) -> VariationResult[G]:
        return VariationResult(self.clone(genotype), {"kind": "clone"})


@dataclass(frozen=True, slots=True)
class CloneCrossover[G]:
    """Copy one parent's genotype; this is a fallback, not domain recombination."""

    clone: Clone[G]
    copy_first: float = 0.5

    def __post_init__(self) -> None:
        probability = _weight(self.copy_first)
        if probability > 1:
            raise ConfigurationError("copy_first must be a probability in [0, 1]")

    def __call__(self, first: G, second: G, rng: Random) -> VariationResult[G]:
        index = 0 if rng.random() < self.copy_first else 1
        genotype = first if index == 0 else second
        return VariationResult(self.clone(genotype), {"copied_parent_index": index})
