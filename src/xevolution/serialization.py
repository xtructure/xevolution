"""Explicit, versioned conversion of framework state and application payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .config import EvolutionSettings, as_object, validate_json
from .errors import CheckpointCompatibilityError, CheckpointError, ConfigurationError
from .models import (
    Individual,
    JSONDict,
    JSONValue,
    MetricSummary,
    Population,
    RunState,
    StopDecision,
)
from .protocols import AttachmentReader, AttachmentWriter, PayloadCodec

STATE_SCHEMA_VERSION = 1


class JsonCodec:
    """Copy JSON-native payloads without silently coercing custom Python types."""

    codec_id = "xevolution.json"
    version = 1

    def encode(
        self, value: JSONValue, writer: AttachmentWriter | None = None
    ) -> JSONValue:
        return validate_json(value)

    def decode(
        self, value: JSONValue, reader: AttachmentReader | None = None
    ) -> JSONValue:
        return validate_json(value)


def _object(value: JSONValue, label: str) -> JSONDict:
    try:
        return as_object(value, label)
    except ConfigurationError as exc:
        raise CheckpointError(str(exc)) from exc


def _keys(data: JSONDict, expected: set[str], label: str) -> None:
    missing = expected - data.keys()
    extra = data.keys() - expected
    if missing or extra:
        raise CheckpointError(
            f"Invalid {label} fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _integer(value: JSONValue, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CheckpointError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: JSONValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"{label} must be a nonempty string")
    return value


def _number(value: JSONValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise CheckpointError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise CheckpointError(f"{label} must be a finite number")
    return result


def _metrics(value: JSONValue) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, number in _object(value, "metrics").items():
        if not name or name == "fitness":
            raise CheckpointError(
                "Metric names must be nonempty and cannot be 'fitness'"
            )
        metrics[name] = _number(number, f"metric {name!r}")
    return metrics


def _encode_individual[G](
    individual: Individual[G],
    codec: PayloadCodec[G],
    writer: AttachmentWriter | None,
) -> JSONDict:
    return {
        "id": individual.id,
        "genotype": validate_json(codec.encode(individual.genotype, writer)),
        "birth_generation": individual.birth_generation,
        "parent_ids": list(individual.parent_ids),
        "fitness": individual.fitness,
        "metrics": dict(individual.metrics),
        "evaluation_count": individual.evaluation_count,
        "metadata": validate_json(dict(individual.metadata)),
    }


def _decode_individual[G](
    value: JSONValue,
    codec: PayloadCodec[G],
    reader: AttachmentReader | None,
    *,
    generation: int,
    next_id: int,
) -> Individual[G]:
    data = _object(value, "individual")
    _keys(
        data,
        {
            "id",
            "genotype",
            "birth_generation",
            "parent_ids",
            "fitness",
            "metrics",
            "evaluation_count",
            "metadata",
        },
        "individual",
    )
    individual_id = _integer(data["id"], "individual id")
    birth = _integer(data["birth_generation"], "birth generation")
    if individual_id >= next_id or birth > generation:
        raise CheckpointError(
            "Individual identity or birth lies beyond saved run state"
        )
    raw_parents = data["parent_ids"]
    if not isinstance(raw_parents, list):
        raise CheckpointError("parent_ids must be a JSON array")
    parents = tuple(_integer(item, "parent id") for item in raw_parents)
    if any(parent >= individual_id for parent in parents):
        raise CheckpointError("Parent IDs must precede their child's allocated ID")
    fitness = _number(data["fitness"], "fitness")
    count = _integer(data["evaluation_count"], "evaluation count", minimum=1)
    metadata = _object(data["metadata"], "individual metadata")
    return Individual(
        id=individual_id,
        genotype=codec.decode(data["genotype"], reader),
        birth_generation=birth,
        parent_ids=parents,
        fitness=fitness,
        metrics=_metrics(data["metrics"]),
        evaluation_count=count,
        metadata=metadata,
    )


def _encode_summary(summary: MetricSummary) -> JSONDict:
    return {
        "count": summary.count,
        "minimum": summary.minimum,
        "mean": summary.mean,
        "maximum": summary.maximum,
    }


def _decode_summary(value: JSONValue, population_size: int) -> MetricSummary:
    data = _object(value, "metric summary")
    _keys(data, {"count", "minimum", "mean", "maximum"}, "metric summary")
    count = _integer(data["count"], "summary count")
    if count > population_size:
        raise CheckpointError("Summary count exceeds population size")
    if count == 0:
        if any(data[key] is not None for key in ("minimum", "mean", "maximum")):
            raise CheckpointError("An empty summary must have null statistics")
        return MetricSummary(0, None, None, None)
    minimum = _number(data["minimum"], "summary minimum")
    mean = _number(data["mean"], "summary mean")
    maximum = _number(data["maximum"], "summary maximum")
    if not minimum <= mean <= maximum:
        raise CheckpointError("Summary minimum, mean and maximum are inconsistent")
    return MetricSummary(count, minimum, mean, maximum)


def encode_state[G](
    state: RunState[G],
    codec: PayloadCodec[G],
    writer: AttachmentWriter | None = None,
) -> JSONDict:
    """Encode a finalized state without copying arbitrary objects implicitly.

    Codecs may use the supplied attachment writer; they must not mutate the
    genotype or consume scientific randomness. Only completed states may be
    persisted, including the evaluated initial generation.
    """
    try:
        _validate_completed(state)
        _text(codec.codec_id, "codec id")
        _integer(codec.version, "codec version", minimum=1)
        data: JSONDict = {
            "schema_version": STATE_SCHEMA_VERSION,
            "codec": {"id": codec.codec_id, "version": codec.version},
            "run_id": state.run_id,
            "generation": state.generation,
            "population": {
                "individuals": [
                    _encode_individual(item, codec, writer) for item in state.population
                ],
                "metadata": validate_json(dict(state.population.metadata)),
            },
            "generation_settings": state.generation_settings.to_dict(),
            "effective_settings": state.effective_settings.to_dict(),
            "next_individual_id": state.next_individual_id,
            "evaluations": state.evaluations,
            "best": (
                None
                if state.best is None
                else _encode_individual(state.best, codec, writer)
            ),
            "last_improvement_generation": state.last_improvement_generation,
            "statistics": {
                name: _encode_summary(summary)
                for name, summary in state.statistics.items()
            },
            "stop": {"stop": state.stop.stop, "reason": state.stop.reason},
            "phase": state.phase,
            "next_event_sequence": state.next_event_sequence,
        }
        return _object(validate_json(data), "run state")
    except ConfigurationError as exc:
        raise CheckpointError(f"Cannot encode checkpoint state: {exc}") from exc


def _validate_completed[G](state: RunState[G]) -> None:
    _text(state.run_id, "run id")
    _integer(state.generation, "generation")
    _integer(state.next_individual_id, "next individual id")
    _integer(state.evaluations, "evaluations")
    _integer(state.last_improvement_generation, "last improvement generation")
    _integer(state.next_event_sequence, "event sequence", minimum=1)
    if state.last_improvement_generation > state.generation:
        raise CheckpointError("Last improvement cannot lie in a future generation")
    if state.next_event_sequence != state.generation + 1:
        raise CheckpointError("Completed generations require one scientific event each")
    if (
        state.phase not in ("ready", "terminated")
        or (state.phase == "terminated") != state.stop.stop
    ):
        raise CheckpointError("Checkpoint phase and stop decision disagree")
    if type(state.stop.stop) is not bool or (
        state.stop.reason is not None and not isinstance(state.stop.reason, str)
    ):
        raise CheckpointError("Invalid stopping decision")
    if len(state.population) != state.generation_settings.population_size:
        raise CheckpointError("Population does not match its generation settings")
    if not state.population or state.best is None:
        raise CheckpointError(
            "A completed checkpoint needs an evaluated population and best"
        )
    for item in (*state.population.individuals, state.best):
        if type(item.id) is not int or not 0 <= item.id < state.next_individual_id:
            raise CheckpointError("Individual ID is inconsistent with allocation state")
        _integer(item.birth_generation, "birth generation")
        if item.birth_generation > state.generation:
            raise CheckpointError("Invalid individual birth generation")
        if any(
            type(parent) is not int or not 0 <= parent < item.id
            for parent in item.parent_ids
        ):
            raise CheckpointError("Parent IDs must precede their child's allocated ID")
        _number(item.fitness, "fitness")
        _integer(item.evaluation_count, "evaluation count", minimum=1)
        _metrics(validate_json(dict(item.metrics)))
    if any(
        item.evaluation_count != state.generation - item.birth_generation + 1
        for item in state.population
    ):
        raise CheckpointError(
            "Current individuals must be evaluated once per generation of life"
        )
    if (
        state.best.birth_generation > state.last_improvement_generation
        or state.best.evaluation_count
        != state.last_improvement_generation - state.best.birth_generation + 1
    ):
        raise CheckpointError(
            "Historical best is inconsistent with its improvement generation"
        )
    if any(
        _number(item.fitness, "fitness") > _number(state.best.fitness, "best fitness")
        for item in state.population
    ):
        raise CheckpointError(
            "Historical best cannot be worse than the current population"
        )
    if sum(item.evaluation_count for item in state.population) > state.evaluations:
        raise CheckpointError("Population evaluation counts exceed run accounting")
    expected_names = {"fitness"} | {
        name for individual in state.population for name in individual.metrics
    }
    if state.statistics.keys() != expected_names:
        raise CheckpointError("Statistics must summarize exactly the observed metrics")
    for name, summary in state.statistics.items():
        _decode_summary(_encode_summary(summary), len(state.population))
        count = (
            len(state.population)
            if name == "fitness"
            else sum(name in item.metrics for item in state.population)
        )
        if summary.count != count:
            raise CheckpointError(
                "Metric summary count differs from observed contributions"
            )
    fitness_summary = state.statistics.get("fitness")
    if fitness_summary is None or fitness_summary.count != len(state.population):
        raise CheckpointError(
            "Completed state must contain complete fitness statistics"
        )


def decode_state[G](
    value: JSONValue,
    codec: PayloadCodec[G],
    reader: AttachmentReader | None = None,
) -> RunState[G]:
    """Restore explicit state, rejecting incompatible codecs and invalid boundaries."""
    try:
        data = _object(validate_json(value), "run state")
        _keys(
            data,
            {
                "schema_version",
                "codec",
                "run_id",
                "generation",
                "population",
                "generation_settings",
                "effective_settings",
                "next_individual_id",
                "evaluations",
                "best",
                "last_improvement_generation",
                "statistics",
                "stop",
                "phase",
                "next_event_sequence",
            },
            "run state",
        )
        schema = _integer(data["schema_version"], "state schema version", minimum=1)
        if schema != STATE_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(f"Unsupported state schema {schema}")
        codec_data = _object(data["codec"], "codec")
        _keys(codec_data, {"id", "version"}, "codec")
        codec_id = _text(codec_data["id"], "codec id")
        codec_version = _integer(codec_data["version"], "codec version", minimum=1)
        if (codec_id, codec_version) != (codec.codec_id, codec.version):
            raise CheckpointCompatibilityError(
                f"Payload codec {codec_id}@{codec_version} is incompatible with "
                f"{codec.codec_id}@{codec.version}"
            )
        generation = _integer(data["generation"], "generation")
        next_id = _integer(data["next_individual_id"], "next individual id")
        population_data = _object(data["population"], "population")
        _keys(population_data, {"individuals", "metadata"}, "population")
        raw_individuals = population_data["individuals"]
        if not isinstance(raw_individuals, list):
            raise CheckpointError("Population individuals must be a JSON array")
        population = Population(
            individuals=tuple(
                _decode_individual(
                    item, codec, reader, generation=generation, next_id=next_id
                )
                for item in raw_individuals
            ),
            metadata=_object(population_data["metadata"], "population metadata"),
        )
        stopping = _object(data["stop"], "stop decision")
        _keys(stopping, {"stop", "reason"}, "stop decision")
        stop = stopping["stop"]
        reason = stopping["reason"]
        if type(stop) is not bool or (
            reason is not None and not isinstance(reason, str)
        ):
            raise CheckpointError("Invalid stop decision")
        phase = data["phase"]
        if phase not in ("ready", "terminated"):
            raise CheckpointError("Invalid checkpoint phase")
        statistics = _object(data["statistics"], "statistics")
        best = (
            None
            if data["best"] is None
            else _decode_individual(
                data["best"], codec, reader, generation=generation, next_id=next_id
            )
        )
        state = RunState(
            run_id=_text(data["run_id"], "run id"),
            generation=generation,
            population=population,
            generation_settings=EvolutionSettings.from_dict(
                _object(data["generation_settings"], "generation settings")
            ),
            effective_settings=EvolutionSettings.from_dict(
                _object(data["effective_settings"], "effective settings")
            ),
            next_individual_id=next_id,
            evaluations=_integer(data["evaluations"], "evaluations"),
            best=best,
            last_improvement_generation=_integer(
                data["last_improvement_generation"], "last improvement generation"
            ),
            statistics={
                name: _decode_summary(summary, len(population))
                for name, summary in statistics.items()
            },
            stop=StopDecision(stop=stop, reason=reason),
            phase="ready" if phase == "ready" else "terminated",
            next_event_sequence=_integer(data["next_event_sequence"], "event sequence"),
        )
        _validate_completed(state)
        return state
    except ConfigurationError as exc:
        raise CheckpointError(f"Invalid checkpoint state: {exc}") from exc


def encode_metrics(metrics: Mapping[str, MetricSummary]) -> JSONDict:
    """Return detached JSON statistics for generation records and reports."""
    return {name: _encode_summary(summary) for name, summary in metrics.items()}
