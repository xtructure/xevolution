"""Atomic validation and application of individual or whole-population results."""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from math import isfinite

from .errors import EvaluationError
from .models import EvaluationResult, Population
from .protocols import IndividualEvaluator, PopulationEvaluator
from .randomness import EvaluationContext


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label} must be a finite numeric value")
    try:
        number = float(value)
    except OverflowError as error:
        raise EvaluationError(f"{label} is outside the finite float range") from error
    if not isfinite(number):
        raise EvaluationError(f"{label} must be finite")
    return number


def _validate_result(result: object, individual_id: int) -> EvaluationResult:
    if not isinstance(result, EvaluationResult):
        raise EvaluationError(
            f"Individual {individual_id} requires an EvaluationResult, "
            f"received {type(result).__name__}"
        )
    fitness = _number(result.fitness, f"Fitness for individual {individual_id}")
    if not isinstance(result.metrics, Mapping):
        raise EvaluationError(
            f"Metrics for individual {individual_id} must be a mapping"
        )
    metrics: dict[str, float] = {}
    for key, value in result.metrics.items():
        if not isinstance(key, str) or not key:
            raise EvaluationError("Metric names must be nonempty strings")
        if key == "fitness":
            raise EvaluationError("'fitness' is reserved for the primary objective")
        metrics[key] = _number(value, f"Metric {key!r} for individual {individual_id}")
    return EvaluationResult(fitness, metrics)


def _collect_results(
    raw: Mapping[int, EvaluationResult] | Iterable[tuple[int, EvaluationResult]],
    expected: set[int],
) -> dict[int, EvaluationResult]:
    entries = raw.items() if isinstance(raw, Mapping) else raw
    if not isinstance(entries, Iterable):
        raise EvaluationError(
            "A population evaluator must return ID-associated results"
        )
    results: dict[int, EvaluationResult] = {}
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise EvaluationError(
                "Each evaluation result must be an (individual_id, result) pair"
            )
        individual_id, result = entry
        if isinstance(individual_id, bool) or not isinstance(individual_id, int):
            raise EvaluationError(
                "Evaluation result IDs must be integer individual IDs"
            )
        if individual_id not in expected:
            raise EvaluationError(
                f"Unexpected evaluation result for individual {individual_id}"
            )
        if individual_id in results:
            raise EvaluationError(
                f"Duplicate evaluation result for individual {individual_id}"
            )
        results[individual_id] = _validate_result(result, individual_id)
    missing = expected - results.keys()
    if missing:
        raise EvaluationError(
            f"Missing evaluation results for individuals {sorted(missing)}"
        )
    return results


def evaluate_population[G](
    population: Population[G],
    context: EvaluationContext,
    *,
    evaluate: IndividualEvaluator[G] | None = None,
    evaluate_population: PopulationEvaluator[G] | None = None,
) -> Population[G]:
    """Evaluate once and return new wrappers only after every result is valid.

    A configured population evaluator takes precedence. The individual adapter
    supplies a fresh deterministic context for each logical individual job.
    Application failures propagate; framework result errors use EvaluationError.
    No genotype is compared, hashed, cloned, or interpreted by this coordinator.
    """
    if evaluate_population is not None:
        raw = evaluate_population(population, context)
    elif evaluate is not None:
        raw = [
            (individual.id, evaluate(individual, context.for_individual(individual.id)))
            for individual in population
        ]
    else:
        raise EvaluationError("An individual or population evaluator is required")
    results = _collect_results(raw, set(population.by_id))
    updated = tuple(
        replace(
            individual,
            fitness=results[individual.id].fitness,
            metrics=results[individual.id].metrics,
            evaluation_count=individual.evaluation_count + 1,
        )
        for individual in population
    )
    return Population(updated, metadata=population.metadata)
