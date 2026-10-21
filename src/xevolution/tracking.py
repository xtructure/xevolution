"""Deterministic summaries, independent historical bests, and genealogy projections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from .config import as_object, strict_dumps, validate_json
from .errors import RunStateError
from .models import (
    GenerationRecord,
    Individual,
    JSONDict,
    JSONValue,
    LineageRecord,
    MetricSummary,
    RunState,
)
from .protocols import Clone
from .serialization import encode_metrics


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunStateError(f"{label} must be finite numeric observations")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RunStateError(f"{label} must be finite numeric observations") from exc
    if not math.isfinite(number):
        raise RunStateError(f"{label} must be finite numeric observations")
    return number


def summarize(values: Sequence[float]) -> MetricSummary:
    """Summarize observations without treating missing values as zero.

    Scaling avoids overflowing an otherwise finite mean when several very
    large finite observations have the same sign.
    """
    if not values:
        return MetricSummary(0, None, None, None)
    numbers = [_number(value, "Statistics") for value in values]
    minimum, maximum = min(numbers), max(numbers)
    scale = max(abs(minimum), abs(maximum))
    mean = (
        0.0
        if scale == 0.0
        else (math.fsum(value / scale for value in numbers) / len(numbers)) * scale
    )
    # At a bound, roundoff must not make a valid summary contradict its extrema.
    mean = min(maximum, max(minimum, mean))
    return MetricSummary(len(values), float(minimum), mean, float(maximum))


def _fitness[G](individual: Individual[G]) -> float:
    return _number(individual.fitness, f"Fitness of individual {individual.id}")


def update_statistics[G](state: RunState[G], clone: Clone[G]) -> RunState[G]:
    """Return summaries and a genotype-independent best snapshot for this generation."""
    population = state.population
    ordered = tuple(sorted(population, key=lambda item: item.id))
    statistics = {"fitness": summarize([_fitness(item) for item in ordered])}
    metric_names = sorted({name for item in ordered for name in item.metrics})
    if "fitness" in metric_names:
        raise RunStateError("Supplementary metrics must not shadow primary fitness")
    for name in metric_names:
        statistics[name] = summarize(
            [item.metrics[name] for item in ordered if name in item.metrics]
        )
    if not ordered:
        return replace(state, statistics=statistics)
    current_best = min(ordered, key=lambda item: (-_fitness(item), item.id))
    best = state.best
    last_improved = state.last_improvement_generation
    if best is None or _fitness(current_best) > _fitness(best):
        best = replace(
            current_best,
            genotype=clone(current_best.genotype),
            metrics=dict(current_best.metrics),
            metadata=as_object(validate_json(dict(current_best.metadata)), "metadata"),
        )
        last_improved = state.generation
    return replace(
        state,
        statistics=statistics,
        best=best,
        last_improvement_generation=last_improved,
    )


def make_record[G](
    state: RunState[G], births: Sequence[LineageRecord]
) -> GenerationRecord:
    """Freeze a finalized generation as canonical JSON, excluding large genotypes."""
    if (
        type(state.next_event_sequence) is not int
        or state.next_event_sequence != state.generation + 1
    ):
        raise RunStateError("Allocate a generation event sequence before recording it")
    birth_ids = [birth.individual_id for birth in births]
    if len(set(birth_ids)) != len(birth_ids):
        raise RunStateError("Duplicate birth identities in a generation")
    if any(
        birth.birth_generation != state.generation
        or birth.individual_id not in state.population.by_id
        for birth in births
    ):
        raise RunStateError("Birth records must describe this generation's population")
    if set(birth_ids) != {
        item.id
        for item in state.population
        if item.birth_generation == state.generation
    }:
        raise RunStateError("Every newborn needs exactly one birth record")
    if any(
        birth.parent_ids != state.population.by_id[birth.individual_id].parent_ids
        for birth in births
    ):
        raise RunStateError("Birth parent identities disagree with individual records")
    best: JSONValue = (
        None
        if state.best is None
        else {"id": state.best.id, "fitness": _fitness(state.best)}
    )
    record: JSONDict = {
        "schema_version": 1,
        "type": "generation",
        "run_id": state.run_id,
        "generation": state.generation,
        "sequence": state.next_event_sequence - 1,
        "population_ids": [item.id for item in state.population],
        "population_metadata": validate_json(dict(state.population.metadata)),
        "statistics": encode_metrics(state.statistics),
        "evaluations": state.evaluations,
        "observations": [
            {
                "id": item.id,
                "fitness": _fitness(item),
                "metrics": dict(item.metrics),
                "evaluation_count": item.evaluation_count,
                "age": item.age(state.generation),
                "metadata": validate_json(dict(item.metadata)),
            }
            for item in state.population
        ],
        "births": [
            {
                "individual_id": birth.individual_id,
                "birth_generation": birth.birth_generation,
                "parent_ids": list(birth.parent_ids),
                "operator": birth.operator,
                "details": validate_json(dict(birth.details)),
            }
            for birth in sorted(births, key=lambda item: item.individual_id)
        ],
        "generation_settings": state.generation_settings.to_dict(),
        "effective_settings": state.effective_settings.to_dict(),
        "stop": {"stop": state.stop.stop, "reason": state.stop.reason},
        "phase": state.phase,
        "best": best,
        "last_improvement_generation": state.last_improvement_generation,
    }
    return GenerationRecord(
        generation=state.generation,
        sequence=state.next_event_sequence - 1,
        data_json=strict_dumps(record),
    )


def analyze(records: Sequence[JSONDict]) -> JSONDict:
    """Build idempotent summaries and genealogy from one committed journal prefix.

    Birth relationships are authoritative. Each individual's changing fitness
    belongs to its observation history, not to its immutable birth record.
    """
    run_id: str | None = None
    generations: list[JSONValue] = []
    individuals: dict[int, JSONDict] = {}
    edges: set[tuple[int, int]] = set()
    for index, source in enumerate(records):
        record = as_object(validate_json(source), "generation record")
        if (
            type(record.get("sequence")) is not int
            or record["sequence"] != index
            or type(record.get("generation")) is not int
            or record["generation"] != index
            or record.get("type") != "generation"
            or type(record.get("schema_version")) is not int
            or record.get("schema_version") != 1
        ):
            raise RunStateError(
                "Analysis requires ordered contiguous generation records"
            )
        record_run = record.get("run_id")
        if not isinstance(record_run, str) or not record_run:
            raise RunStateError("Generation record has no run identity")
        if run_id is None:
            run_id = record_run
        elif run_id != record_run:
            raise RunStateError("Cannot combine different runs in one analysis")
        raw_births = record.get("births")
        raw_observations = record.get("observations")
        if not isinstance(raw_births, list) or not isinstance(raw_observations, list):
            raise RunStateError("Generation records need births and observations")
        for raw_birth in raw_births:
            birth = as_object(raw_birth, "birth record")
            individual_id = birth.get("individual_id")
            parents = birth.get("parent_ids")
            if type(individual_id) is not int or individual_id < 0:
                raise RunStateError("Invalid birth identity")
            if individual_id in individuals:
                raise RunStateError(f"Duplicate birth for individual {individual_id}")
            if (
                type(birth.get("birth_generation")) is not int
                or birth.get("birth_generation") != index
                or not isinstance(parents, list)
            ):
                raise RunStateError("Invalid birth generation or parent list")
            for parent in parents:
                if (
                    type(parent) is not int
                    or parent not in individuals
                    or individuals[parent]["birth_generation"] == index
                ):
                    raise RunStateError("Genealogy refers to an unknown parent")
                edges.add((parent, individual_id))
            individuals[individual_id] = {**birth, "observations": []}
        observed: set[int] = set()
        for raw_observation in raw_observations:
            observation = as_object(raw_observation, "observation")
            individual_id = observation.get("id")
            if (
                type(individual_id) is not int
                or individual_id not in individuals
                or individual_id in observed
            ):
                raise RunStateError(
                    "Observation has unknown or duplicate individual identity"
                )
            observed.add(individual_id)
            observations = individuals[individual_id]["observations"]
            if not isinstance(observations, list):
                raise RunStateError("Invalid internal observation collection")
            observations.append({**observation, "generation": index})
        ids = record.get("population_ids")
        if (
            not isinstance(ids, list)
            or any(type(value) is not int or value < 0 for value in ids)
            or len(ids) != len(observed)
            or set(ids) != observed
        ):
            raise RunStateError("Population membership and observations disagree")
        generations.append(
            {
                key: value
                for key, value in record.items()
                if key not in ("births", "observations")
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generation_count": len(generations),
        "latest_generation": len(generations) - 1 if generations else None,
        "generations": generations,
        "individuals": [individuals[key] for key in sorted(individuals)],
        "edges": [
            {"parent_id": parent, "child_id": child} for parent, child in sorted(edges)
        ],
    }
