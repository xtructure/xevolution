"""Finite offspring production and framework-owned identity and lineage."""

from collections.abc import Sequence
from math import isfinite
from random import Random

from .config import EvolutionSettings, validate_json
from .errors import ConfigurationError, OperatorInapplicable, ReproductionError
from .models import (
    Individual,
    JSONDict,
    LineageRecord,
    Population,
    VariationResult,
)
from .operators import OperatorSpec, validate_routes, weighted_order
from .protocols import Crossover, Mutation, ParentSelector, SurvivorSelector


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReproductionError(f"{label} must be an integer >= {minimum}")
    return value


def _parent[G](
    population: Population[G], selector: ParentSelector[G], rng: Random
) -> Individual[G]:
    selected = selector(population, rng)
    identifier = _integer(selected, "Selected parent ID")
    if identifier not in population.by_id:
        raise ReproductionError(
            f"Selected parent {identifier} is not in the parent population"
        )
    return population.by_id[identifier]


def _details[G](result: VariationResult[G], name: str) -> JSONDict:
    try:
        details = validate_json(dict(result.details))
    except (ConfigurationError, TypeError, ValueError) as error:
        raise ReproductionError(
            f"Operator {name!r} returned invalid provenance"
        ) from error
    if not isinstance(details, dict):
        raise ReproductionError(f"Operator {name!r} provenance must be an object")
    return details


def reproduce[G](
    population: Population[G],
    settings: EvolutionSettings,
    *,
    generation: int,
    next_id: int,
    rng: Random,
    select_parent: ParentSelector[G],
    select_survivors: SurvivorSelector[G],
    mutations: Sequence[OperatorSpec[Mutation[G]]],
    crossovers: Sequence[OperatorSpec[Crossover[G]]],
) -> tuple[Population[G], tuple[LineageRecord, ...], int]:
    """Assemble one unevaluated generation without modifying its parents.

    The returned population contains retained survivor wrappers plus offspring.
    The engine must reevaluate all of them. Failure leaves input bookkeeping
    untouched but does not roll back random draws or application-owned objects.
    """
    _integer(generation, "Offspring generation", minimum=1)
    _integer(next_id, "Next individual ID")
    if not population:
        raise ReproductionError("Cannot reproduce an empty population")
    if next_id <= max(population.by_id):
        raise ReproductionError(
            "Next individual ID must exceed all current individual IDs"
        )
    for individual in population:
        fitness = individual.fitness
        if isinstance(fitness, bool) or not isinstance(fitness, (int, float)):
            raise ReproductionError(
                f"Parent {individual.id} does not have finite fitness"
            )
        try:
            valid_fitness = isfinite(fitness)
        except OverflowError as error:
            raise ReproductionError(
                f"Parent {individual.id} fitness exceeds the finite range"
            ) from error
        if not valid_fitness:
            raise ReproductionError(
                f"Parent {individual.id} does not have finite fitness"
            )
        if individual.birth_generation >= generation:
            raise ReproductionError(
                "Offspring generation must follow every parent's birth"
            )
    validate_routes(settings, mutations, crossovers)

    survivor_ids = select_survivors(population, settings.population_size, rng)
    if not isinstance(survivor_ids, Sequence):
        raise ReproductionError("Survivor selection must return a sequence of IDs")
    survivors: list[Individual[G]] = []
    seen: set[int] = set()
    for selected in survivor_ids:
        identifier = _integer(selected, "Selected survivor ID")
        if identifier not in population.by_id:
            raise ReproductionError(
                f"Selected survivor {identifier} is not in the population"
            )
        if identifier in seen:
            raise ReproductionError(f"Duplicate survivor ID {identifier}")
        seen.add(identifier)
        survivors.append(population.by_id[identifier])
    if len(survivors) > settings.population_size:
        raise ReproductionError("Survivor selection exceeds the target population size")

    children: list[Individual[G]] = []
    births: list[LineageRecord] = []
    required = settings.population_size - len(survivors)
    for _ in range(required):
        result: VariationResult[G] | None = None
        parent_ids: tuple[int, ...] = ()
        operator_name = ""
        last_failure: str | None = None
        for _attempt in range(settings.max_attempts_per_child):
            do_mutation = rng.random() < settings.mutation_probability
            first = _parent(population, select_parent, rng)
            if do_mutation:
                parent_ids = (first.id,)
                for mutation in weighted_order(mutations, rng):
                    try:
                        result = mutation.operation(first.genotype, rng)
                    except OperatorInapplicable as error:
                        last_failure = str(error)
                        continue
                    operator_name = f"mutation:{mutation.name}"
                    break
            else:
                second = _parent(population, select_parent, rng)
                parent_ids = (first.id, second.id)
                for crossover in weighted_order(crossovers, rng):
                    try:
                        result = crossover.operation(
                            first.genotype, second.genotype, rng
                        )
                    except OperatorInapplicable as error:
                        last_failure = str(error)
                        continue
                    operator_name = f"crossover:{crossover.name}"
                    break
            if operator_name:
                if not isinstance(result, VariationResult):
                    raise ReproductionError(
                        f"Operator {operator_name!r} must return VariationResult"
                    )
                break
        if result is None:
            suffix = f"; last inapplicability: {last_failure}" if last_failure else ""
            raise ReproductionError(
                f"Could not produce individual {next_id} after "
                f"{settings.max_attempts_per_child} attempts{suffix}"
            )
        details = _details(result, operator_name)
        child = Individual(
            id=next_id,
            genotype=result.genotype,
            birth_generation=generation,
            parent_ids=parent_ids,
        )
        children.append(child)
        births.append(
            LineageRecord(next_id, generation, parent_ids, operator_name, details)
        )
        next_id += 1
    return (
        Population(tuple(survivors + children), metadata=population.metadata),
        tuple(births),
        next_id,
    )
