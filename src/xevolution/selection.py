"""Parent and survivor selection with explicit, stable numerical policies."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from typing import Any

from .errors import ConfigurationError, SelectionError
from .models import JSONValue, Population
from .protocols import ParentSelector, SurvivorSelector


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{label} must be a finite number, not {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise SelectionError(f"{label} is outside the finite float range") from error
    if not isfinite(number):
        raise SelectionError(f"{label} must be finite")
    return number


def _scores[G](population: Population[G]) -> list[float]:
    if not population:
        raise SelectionError("Cannot select from an empty parent population")
    return [
        _finite_number(individual.fitness, f"Fitness of individual {individual.id}")
        for individual in population
    ]


def weighted_index(weights: Sequence[float], rng: Random) -> int:
    """Select an index from nonnegative finite weights; all-zero means uniform.

    Scaling by the largest weight avoids overflowing their sum. Strict interval
    boundaries ensure a zero-weight entry cannot win when the draw is zero.
    The last positive entry handles a final floating-point rounding boundary.
    """
    if not weights:
        raise SelectionError("Cannot sample an empty set of weights")
    checked = [_finite_number(value, "Selection weight") for value in weights]
    if any(weight < 0.0 for weight in checked):
        raise SelectionError(
            "Roulette selection requires nonnegative weights; choose tournament "
            "selection or supply an explicit score-to-weight transformation"
        )
    largest = max(checked)
    if largest == 0.0:
        return rng.randrange(len(checked))
    scaled = [weight / largest for weight in checked]
    total = fsum(scaled)
    target = rng.random() * total
    cumulative = 0.0
    last_positive = 0
    for index, weight in enumerate(scaled):
        if weight <= 0.0:
            continue
        last_positive = index
        cumulative += weight
        if target < cumulative:
            return index
    return last_positive


def roulette[G](population: Population[G], rng: Random) -> int:
    """Sample a parent using direct fitness weights, with replacement."""
    return population[weighted_index(_scores(population), rng)].id


def uniform[G](population: Population[G], rng: Random) -> int:
    """Sample an evaluated individual uniformly, with replacement."""
    _scores(population)
    return population[rng.randrange(len(population))].id


@dataclass(frozen=True, slots=True)
class Tournament:
    """Choose the fittest of ``size`` draws, resolving ties by smallest ID."""

    size: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int):
            raise ConfigurationError("Tournament size must be a positive integer")
        if self.size < 1:
            raise ConfigurationError("Tournament size must be a positive integer")

    def __call__[G](self, population: Population[G], rng: Random) -> int:
        scores = _scores(population)
        choices = [rng.randrange(len(population)) for _ in range(self.size)]
        winner = max(choices, key=lambda index: (scores[index], -population[index].id))
        return population[winner].id


def _validate_target(target_size: int) -> None:
    if isinstance(target_size, bool) or not isinstance(target_size, int):
        raise SelectionError("Target population size must be a positive integer")
    if target_size < 1:
        raise SelectionError("Target population size must be a positive integer")


def replace_all[G](
    population: Population[G], target_size: int, rng: Random
) -> tuple[int, ...]:
    """Retain no individuals; nonsurvivors remain eligible parents."""
    _validate_target(target_size)
    _scores(population)
    return ()


@dataclass(frozen=True, slots=True)
class Elitism:
    """Retain up to ``count`` individuals, capped by available and target size."""

    count: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ConfigurationError("Elite count must be a nonnegative integer")
        if self.count < 0:
            raise ConfigurationError("Elite count must be a nonnegative integer")

    def __call__[G](
        self, population: Population[G], target_size: int, rng: Random
    ) -> tuple[int, ...]:
        _validate_target(target_size)
        scores = _scores(population)
        order = sorted(
            range(len(population)),
            key=lambda index: (-scores[index], population[index].id),
        )
        return tuple(
            population[index].id for index in order[: min(self.count, target_size)]
        )


def _selector_config(config: Mapping[str, JSONValue], allowed: set[str]) -> str:
    unknown = set(config) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown selector fields: {sorted(unknown)}")
    name = config.get("name")
    if not isinstance(name, str):
        raise ConfigurationError("A selector requires a string 'name'")
    return name


def parent_selector(config: Mapping[str, JSONValue]) -> ParentSelector[Any]:
    """Resolve a built-in selector at the JSON-to-application type boundary."""
    name = _selector_config(config, {"name", "size"})
    if name in {"roulette", "uniform"}:
        if "size" in config:
            raise ConfigurationError(f"{name} selection does not accept 'size'")
        return roulette if name == "roulette" else uniform
    if name == "tournament":
        size = config.get("size", 3)
        if isinstance(size, bool) or not isinstance(size, int):
            raise ConfigurationError("Tournament size must be a positive integer")
        return Tournament(size)
    raise ConfigurationError(f"Unknown parent selector {name!r}")


def survivor_selector(config: Mapping[str, JSONValue]) -> SurvivorSelector[Any]:
    """Resolve a built-in survival policy from validated JSON settings."""
    name = _selector_config(config, {"name", "count"})
    if name == "replace_all":
        if "count" in config:
            raise ConfigurationError("Complete replacement does not accept 'count'")
        return replace_all
    if name in {"elitism", "elitist"}:
        count = config.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ConfigurationError("Elite count must be a nonnegative integer")
        return Elitism(count)
    raise ConfigurationError(f"Unknown survivor selector {name!r}")
