"""Observable evaluation contracts, including invalid and genuinely joint batches."""

from collections.abc import Iterator

import pytest

from xevolution.errors import EvaluationError
from xevolution.evaluation import evaluate_population
from xevolution.models import EvaluationResult, Individual, Population
from xevolution.randomness import EvaluationContext, RunRandom


class Opaque:
    """A valid application payload that rejects generic framework operations."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        raise AssertionError("Framework compared genotype payloads")

    def __hash__(self) -> int:
        raise AssertionError("Framework hashed a genotype")

    def __bool__(self) -> bool:
        raise AssertionError("Framework truth-tested a genotype")


def population() -> Population[Opaque]:
    return Population(
        (
            Individual(7, Opaque(3), evaluation_count=2),
            Individual(2, Opaque(8), evaluation_count=5),
        )
    )


def context() -> EvaluationContext:
    return RunRandom(19).context(4, "evaluation-test")


def test_individual_adapter_owns_counts_and_preserves_identity_order() -> None:
    original = population()
    seen: list[tuple[int, int, int | None]] = []

    def evaluate(
        individual: Individual[Opaque], context: EvaluationContext
    ) -> EvaluationResult:
        seen.append((individual.id, context.generation, context.individual_id))
        return EvaluationResult(individual.genotype.value, {"complexity": 2.0})

    result = evaluate_population(original, context(), evaluate=evaluate)
    assert seen == [(7, 4, 7), (2, 4, 2)]
    assert [individual.id for individual in result] == [7, 2]
    assert [individual.fitness for individual in result] == [3.0, 8.0]
    assert [individual.evaluation_count for individual in result] == [3, 6]
    assert [individual.fitness for individual in original] == [None, None]
    assert [individual.evaluation_count for individual in original] == [2, 5]
    assert result[0].genotype is original[0].genotype


def test_genuine_joint_evaluation_takes_precedence_over_individual_hook() -> None:
    def forbidden(
        individual: Individual[Opaque], context: EvaluationContext
    ) -> EvaluationResult:
        raise AssertionError("The individual hook must not run")

    def joint(
        population: Population[Opaque], context: EvaluationContext
    ) -> dict[int, EvaluationResult]:
        total = sum(individual.genotype.value for individual in population)
        # Reverse result order to verify deterministic association by ID.
        return {
            individual.id: EvaluationResult(individual.genotype.value - total)
            for individual in reversed(population.individuals)
        }

    result = evaluate_population(
        population(), context(), evaluate=forbidden, evaluate_population=joint
    )
    assert [individual.id for individual in result] == [7, 2]
    assert [individual.fitness for individual in result] == [-8.0, -3.0]


@pytest.mark.parametrize(
    ("results", "message"),
    [
        ({7: EvaluationResult(1.0)}, "Missing"),
        (
            {7: EvaluationResult(1), 2: EvaluationResult(2), 99: EvaluationResult(3)},
            "Unexpected",
        ),
        ([(7, EvaluationResult(1)), (7, EvaluationResult(2))], "Duplicate"),
        ([(True, EvaluationResult(1)), (2, EvaluationResult(2))], "integer"),
        ([("7", EvaluationResult(1)), (2, EvaluationResult(2))], "integer"),
        ([(7, EvaluationResult(1)), (2, None)], "EvaluationResult"),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(float("nan")))], "finite"),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(True))], "numeric"),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(10**1000))], "finite"),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(float("inf")))], "finite"),
        (
            [(7, EvaluationResult(1)), (2, EvaluationResult(1, {"x": float("nan")}))],
            "finite",
        ),
        (
            [(7, EvaluationResult(1)), (2, EvaluationResult(1, {"fitness": 2.0}))],
            "reserved",
        ),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(1, {"x": True}))], "numeric"),
        ([(7, EvaluationResult(1)), (2, EvaluationResult(1), "extra")], "pair"),
        (None, "ID-associated"),
    ],
)
def test_invalid_batches_leave_original_population_untouched(results, message) -> None:
    original = population()
    with pytest.raises(EvaluationError, match=message):
        evaluate_population(
            original, context(), evaluate_population=lambda population, context: results
        )
    assert [individual.fitness for individual in original] == [None, None]
    assert [individual.evaluation_count for individual in original] == [2, 5]


def test_late_generator_failure_propagates_without_partial_application() -> None:
    original = population()

    def broken(
        population: Population[Opaque], context: EvaluationContext
    ) -> Iterator[tuple[int, EvaluationResult]]:
        yield 7, EvaluationResult(1)
        raise RuntimeError("application failure")

    with pytest.raises(RuntimeError, match="application failure"):
        evaluate_population(original, context(), evaluate_population=broken)
    assert original[0].fitness is None
    assert original[0].evaluation_count == 2


def test_accepted_metrics_do_not_alias_the_evaluators_dictionary() -> None:
    metrics = {"complexity": 2.0}
    result = evaluate_population(
        population(),
        context(),
        evaluate=lambda individual, context: EvaluationResult(0.0, metrics),
    )
    metrics["complexity"] = 99.0
    assert result[0].metrics["complexity"] == 2.0
    assert result[1].fitness == 0.0


def test_job_randomness_is_stable_across_run_names_and_population_order() -> None:
    def stochastic(
        individual: Individual[Opaque], context: EvaluationContext
    ) -> EvaluationResult:
        return EvaluationResult(context.rng.random())

    first = population()
    second = Population(tuple(reversed(first.individuals)))
    a = evaluate_population(
        first, RunRandom(123).context(5, "run-a"), evaluate=stochastic
    )
    b = evaluate_population(
        second, RunRandom(123).context(5, "run-b"), evaluate=stochastic
    )
    assert {item.id: item.fitness for item in a} == {
        item.id: item.fitness for item in b
    }


def test_missing_evaluator_is_reported() -> None:
    with pytest.raises(EvaluationError, match="required"):
        evaluate_population(population(), context())
