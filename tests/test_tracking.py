"""Scientific summaries and retained records obey ownership and identity contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from xevolution.config import EvolutionSettings
from xevolution.errors import RunStateError
from xevolution.models import (
    Individual,
    JSONValue,
    LineageRecord,
    MetricSummary,
    Population,
    RunState,
)
from xevolution.serialization import JsonCodec
from xevolution.tracking import analyze, make_record, summarize, update_statistics


def state_for[G](population: Population[G], generation: int = 0) -> RunState[G]:
    settings = EvolutionSettings(population_size=len(population))
    return RunState(
        run_id="tracking-test",
        generation=generation,
        population=population,
        generation_settings=settings,
        effective_settings=settings,
        next_individual_id=max(item.id for item in population) + 1,
        next_event_sequence=generation + 1,
        evaluations=sum(item.evaluation_count for item in population),
    )


def clone_json(value: JSONValue) -> JSONValue:
    return JsonCodec().decode(JsonCodec().encode(value))


def test_metrics_use_only_contributing_observations() -> None:
    population = Population[JSONValue](
        (
            Individual(
                0, None, fitness=-2.0, metrics={"length": 8.0}, evaluation_count=1
            ),
            Individual(1, None, fitness=0.0, evaluation_count=1),
        )
    )
    state = update_statistics(state_for(population), clone_json)
    assert state.statistics["fitness"] == MetricSummary(2, -2.0, -1.0, 0.0)
    assert state.statistics["length"] == MetricSummary(1, 8.0, 8.0, 8.0)
    assert summarize([]) == MetricSummary(0, None, None, None)
    assert state.best is not None
    assert state.best.id == 1


def test_summary_remains_finite_for_extreme_finite_inputs() -> None:
    assert summarize([1e308, 1e308]).mean == 1e308
    assert summarize([-1e308, 1e308]).mean == 0.0
    assert summarize([-1e308, -1e308]).mean == -1e308
    for invalid in (float("nan"), float("inf"), True, 10**1000):
        with pytest.raises(RunStateError, match="finite"):
            summarize([invalid])


class OpaquePayload:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        raise AssertionError("Framework must not compare arbitrary genotypes")

    def __hash__(self) -> int:
        raise AssertionError("Framework must not hash arbitrary genotypes")

    def __bool__(self) -> bool:
        raise AssertionError("Framework must not test arbitrary genotype truth")

    def __len__(self) -> int:
        raise AssertionError("Framework must not inspect arbitrary genotype length")


def test_historical_best_has_independent_payload_and_nested_metadata() -> None:
    current = Individual(
        id=0,
        genotype=OpaquePayload([2]),
        fitness=2.0,
        evaluation_count=1,
        metadata={"notes": ["original"]},
    )
    state = update_statistics(
        state_for(Population((current,))),
        lambda value: OpaquePayload(list(value.values)),
    )
    assert state.best is not None
    current.genotype.values.append(99)
    notes = current.metadata["notes"]
    assert isinstance(notes, list)
    notes.append("changed")
    assert state.best.genotype.values == [2]
    assert state.best.metadata["notes"] == ["original"]
    assert state.best is not current


def test_equal_fitness_different_identity_is_not_an_improvement() -> None:
    initial = Population[JSONValue](
        (Individual(0, [1], fitness=3.0, evaluation_count=1),)
    )
    first = update_statistics(state_for(initial), clone_json)
    descendant = Individual[JSONValue](
        1, [2], birth_generation=1, parent_ids=(0,), fitness=3.0, evaluation_count=1
    )
    second = update_statistics(
        replace(
            first,
            generation=1,
            population=Population((descendant,)),
            next_individual_id=2,
            next_event_sequence=2,
            evaluations=2,
        ),
        clone_json,
    )
    assert second.best is first.best
    assert second.last_improvement_generation == 0


def test_generation_record_and_analysis_are_detached_and_idempotent() -> None:
    first_population = Population[JSONValue](
        (
            Individual(0, [1], fitness=2.0, evaluation_count=1),
            Individual(1, [2], fitness=1.0, evaluation_count=1),
        )
    )
    first = update_statistics(state_for(first_population), clone_json)
    first_record = make_record(
        first,
        (LineageRecord(0, 0, (), "initialize"), LineageRecord(1, 0, (), "initialize")),
    )
    survivor = replace(first_population[1], fitness=4.0, evaluation_count=2)
    newborn = Individual[JSONValue](
        2, [3], birth_generation=1, parent_ids=(0, 1), fitness=3.0, evaluation_count=1
    )
    second = update_statistics(
        replace(
            first,
            generation=1,
            population=Population((survivor, newborn)),
            next_individual_id=3,
            evaluations=4,
            next_event_sequence=2,
        ),
        clone_json,
    )
    second_record = make_record(second, (LineageRecord(2, 1, (0, 1), "cross"),))
    data = [first_record.to_dict(), second_record.to_dict()]
    report = analyze(data)
    assert report == analyze(data)
    assert report["edges"] == [
        {"parent_id": 0, "child_id": 2},
        {"parent_id": 1, "child_id": 2},
    ]
    individuals = report["individuals"]
    assert isinstance(individuals, list)
    assert len(individuals) == 3  # Extinct ancestors remain in genealogy.
    survivor_history = individuals[1]
    assert isinstance(survivor_history, dict)
    observations = survivor_history["observations"]
    assert isinstance(observations, list)
    assert [value["fitness"] for value in observations if isinstance(value, dict)] == [
        1.0,
        4.0,
    ]
    assert second_record.to_dict()["population_ids"] == [1, 2]
    data[0]["births"] = []
    assert first_record.to_dict()["births"] != []
    assert report == analyze([first_record.to_dict(), second_record.to_dict()])


def test_analysis_rejects_invalid_identities_without_hashing_payloads() -> None:
    population = Population[JSONValue](
        (Individual(0, None, fitness=0.0, evaluation_count=1),)
    )
    state = update_statistics(state_for(population), clone_json)
    source = make_record(state, (LineageRecord(0, 0, (), "initialize"),)).to_dict()
    source["population_ids"] = [{"malformed": "identity"}]
    with pytest.raises(RunStateError, match="membership"):
        analyze([source])


def test_genealogy_rejects_missing_ancestors_and_duplicate_births() -> None:
    population = Population[JSONValue](
        (Individual(1, None, parent_ids=(0,), fitness=0.0, evaluation_count=1),)
    )
    state = update_statistics(state_for(population), clone_json)
    record = make_record(state, (LineageRecord(1, 0, (0,), "mutate"),)).to_dict()
    with pytest.raises(RunStateError, match="unknown parent"):
        analyze([record])
    with pytest.raises(RunStateError, match="Duplicate birth"):
        make_record(state, (LineageRecord(1, 0, (), "initialize"),) * 2)
