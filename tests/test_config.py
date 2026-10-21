"""Configuration boundaries and reproducible managed random state."""

from pathlib import Path
from random import Random
from types import MappingProxyType

import pytest

from xevolution.config import (
    EvolutionSettings,
    ParameterRule,
    RangeParameter,
    RunConfig,
    load_config,
    strict_dumps,
    strict_loads,
    validate_json,
)
from xevolution.errors import CheckpointCompatibilityError, ConfigurationError
from xevolution.randomness import EvaluationContext, RunRandom


class FixedRandom(Random):
    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


@pytest.mark.parametrize(
    "text",
    [
        '{"size": 2, "size": 3}',
        '{"nested": {"x": 1, "x": 2}}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1e999}',
        '[0, {"x": -1e999}]',
        '{"trailing": 1,}',
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_invalid_syntax(text) -> None:
    with pytest.raises(ConfigurationError):
        strict_loads(text)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), (1, 2), {1: "value"}, MappingProxyType({"x": 1})],
)
def test_json_codec_boundary_is_explicitly_json_native(value) -> None:
    with pytest.raises(ConfigurationError):
        validate_json(value)


def test_json_copy_unicode_precision_and_cycles() -> None:
    source = {"unicode": "λ", "values": [1.2345678901234567, 2**100, None, True]}
    copied = validate_json(source)
    assert strict_loads(strict_dumps(copied)) == source
    source["values"].append("later")
    assert len(copied["values"]) == 4
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ConfigurationError, match="cycles"):
        validate_json(cycle)


@pytest.mark.parametrize(
    "changes",
    [
        {"population_size": True},
        {"population_size": 1.0},
        {"population_size": 0},
        {"max_attempts_per_child": False},
        {"max_attempts_per_child": -1},
        {"mutation_probability": True},
        {"mutation_probability": float("nan")},
        {"mutation_probability": float("inf")},
        {"mutation_probability": 10**1000},
        {"mutation_probability": -0.1},
        {"mutation_probability": 1.01},
        {"unexpected": 1},
        {False: 1},
    ],
)
def test_evolution_settings_validate_programmatic_changes(changes) -> None:
    with pytest.raises(ConfigurationError):
        EvolutionSettings.from_dict(changes)


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": 1},
        {"seed": True},
        {"schema_version": True},
        {"schema_version": 200},
        {"experiment": {"unknown": 1}},
        {"experiment": {"factory": 4}},
        {"evolution": {"extra": 1}},
        {"evolution": {"parent_selection": []}},
    ],
)
def test_run_configuration_rejects_bad_core_fields(value) -> None:
    with pytest.raises(ConfigurationError):
        RunConfig.from_dict(value)


def test_configuration_roundtrip_and_file_loading_do_not_import_factory(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        seed=23,
        factory="does_not_exist.application:build_experiment",
        parameters={"target": 64, "nested": {"label": "λ"}},
        evolution=EvolutionSettings(population_size=4, mutation_probability=1.0),
        termination={"max_generations": 10},
    )
    path = tmp_path / "config.json"
    path.write_text(strict_dumps(config.to_dict()), encoding="utf-8")
    assert load_config(path).to_dict() == config.to_dict()
    with pytest.raises(ConfigurationError, match="cannot read"):
        load_config(tmp_path / "missing.json")


def test_ranges_enforce_bounds_mutability_and_checkpointed_values() -> None:
    parameter = RangeParameter(0.4, rule=ParameterRule(0.0, 1.0))
    parameter.set(0.7)
    saved = parameter.state_dict()
    parameter.set(0.1)
    parameter.load_state_dict(saved)
    assert parameter.value == 0.7
    with pytest.raises(ConfigurationError):
        parameter.set(1.1)
    assert parameter.value == 0.7
    fixed = RangeParameter(0.5, rule=ParameterRule(0.0, 1.0, mutable=False))
    fixed.set(0.5)
    with pytest.raises(ConfigurationError, match="immutable"):
        fixed.set(0.6)
    with pytest.raises(ConfigurationError):
        fixed.set(True)
    with pytest.raises(ConfigurationError, match="differs"):
        fixed.load_state_dict(saved)


def test_constant_sampling_respects_exact_bounds() -> None:
    sampled = RangeParameter.sampled(
        FixedRandom(0.3), 0.1, 0.1, rule=ParameterRule(0.1, 0.1)
    )
    assert sampled.value == 0.1
    config = RunConfig(
        sampling={
            "parameters.constant": {"kind": "log_uniform", "low": 0.1, "high": 0.1}
        }
    )
    assert config.resolve(FixedRandom(0.3)).parameters["constant"] == 0.1


def test_uniform_sampling_handles_extreme_finite_bounds() -> None:
    parameter = RangeParameter.sampled(FixedRandom(0.5), -1e308, 1e308)
    assert parameter.value == 0.0


def test_sampled_settings_resolve_once_in_stable_order_before_bound_validation() -> (
    None
):
    sampling = {
        "parameters.rate": {"kind": "log_uniform", "low": 0.01, "high": 10.0},
        "evolution.population_size": {"kind": "integer", "low": 4, "high": 6},
    }
    rule = {"population_size": {"minimum": 4, "maximum": 6, "mutable": True}}
    original = RunConfig(seed=91, sampling=sampling, parameter_rules=rule)
    reverse = RunConfig(
        seed=91, sampling=dict(reversed(sampling.items())), parameter_rules=rule
    )
    random = RunRandom(91)
    evolution_state = random.evolution.getstate()
    resolved = original.resolve(random.sampling)
    assert 4 <= resolved.evolution.population_size <= 6
    assert 0.01 <= resolved.parameters["rate"] <= 10.0
    assert resolved.sampling == {}
    assert random.evolution.getstate() == evolution_state
    assert resolved.to_dict() == reverse.resolve(RunRandom(91).sampling).to_dict()
    saved_random = random.state_dict()
    assert resolved.resolve(random.sampling).to_dict() == resolved.to_dict()
    assert random.state_dict() == saved_random
    assert RunConfig.from_dict(resolved.to_dict()).to_dict() == resolved.to_dict()


@pytest.mark.parametrize(
    "sampling",
    [
        {"invalid.x": {"kind": "integer", "low": 1, "high": 1}},
        {"parameters.deep.x": {"kind": "integer", "low": 1, "high": 1}},
        {"evolution.unknown": {"kind": "integer", "low": 1, "high": 1}},
        {"parameters.x": {"kind": "log_uniform", "low": 0, "high": 1}},
        {"parameters.x": {"kind": "integer", "low": True, "high": 3}},
        {"parameters.x": {"kind": "uniform", "low": 3, "high": 1}},
        {"parameters.x": {"kind": "unknown", "low": 1, "high": 1}},
        {"evolution.mutation_probability": {"kind": "integer", "low": 2, "high": 2}},
    ],
)
def test_sampling_specifications_validate_support_and_effective_values(
    sampling,
) -> None:
    with pytest.raises(ConfigurationError):
        RunConfig(sampling=sampling).resolve(Random(1))


def test_adaptive_updates_enforce_lifetime_rules_without_rewriting_original_settings() -> (
    None
):
    config = RunConfig(
        evolution=EvolutionSettings(population_size=4),
        parameter_rules={
            "population_size": {"minimum": 2, "maximum": 8, "mutable": True},
            "mutation_probability": {"mutable": False},
        },
    )
    next_settings = config.evolution.updated(population_size=8)
    config.validate_update(config.evolution, next_settings)
    assert config.evolution.population_size == 4
    with pytest.raises(ConfigurationError, match="maximum"):
        config.validate_update(
            config.evolution, config.evolution.updated(population_size=9)
        )
    with pytest.raises(ConfigurationError, match="immutable"):
        config.validate_update(
            config.evolution, config.evolution.updated(mutation_probability=0.1)
        )


def test_rng_roundtrip_preserves_gaussian_cache_and_existing_generator_references() -> (
    None
):
    random = RunRandom(998)
    original_evolution = random.evolution
    original_sampling = random.sampling
    random.evolution.gauss(0, 1)
    random.sampling.gauss(0, 1)
    assert random.evolution.getstate()[2] is not None
    saved = strict_loads(strict_dumps(random.state_dict()))
    expected = (
        random.evolution.gauss(0, 1),
        random.evolution.random(),
        random.sampling.gauss(0, 1),
        random.sampling.random(),
    )
    random.load_state_dict(saved)
    assert random.evolution is original_evolution
    assert random.sampling is original_sampling
    actual = (
        random.evolution.gauss(0, 1),
        random.evolution.random(),
        random.sampling.gauss(0, 1),
        random.sampling.random(),
    )
    assert actual == expected


@pytest.mark.parametrize("cache", ["invalid", True, float("nan"), float("inf")])
@pytest.mark.parametrize("stream", ["evolution", "sampling"])
def test_invalid_saved_gaussian_cache_is_rejected_atomically(cache, stream) -> None:
    random = RunRandom(19)
    prior = random.state_dict()
    saved = random.state_dict()
    saved[stream][2] = cache
    with pytest.raises(CheckpointCompatibilityError):
        random.load_state_dict(saved)
    assert random.state_dict() == prior


def test_rng_identity_and_invalid_state_rejection() -> None:
    random = RunRandom(5)
    saved = random.state_dict()
    saved["seed"] = 6
    with pytest.raises(CheckpointCompatibilityError):
        random.load_state_dict(saved)
    saved = random.state_dict()
    saved["sampling"][1] = [1, 2]
    before = random.state_dict()
    with pytest.raises(CheckpointCompatibilityError):
        random.load_state_dict(saved)
    assert random.state_dict() == before
    with pytest.raises(ConfigurationError):
        RunRandom(True)


def test_evaluation_context_inherits_work_identity_and_keeps_streams_separate() -> None:
    random = RunRandom(17)
    before = random.state_dict()
    batch = random.context(4, "one", round=2, replicate=3)
    a = batch.for_individual(7)
    b = random.context(4, "two", individual_id=7, round=2, replicate=3)
    assert a.replicate == 3
    assert a.rng.random() == b.rng.random()
    explicit = batch.for_individual(7, replicate=0)
    assert explicit.replicate == 0
    values = {
        random.context(
            4, "one", individual_id=7, round=round, replicate=replicate
        ).rng.random()
        for round in range(2)
        for replicate in range(3)
    }
    assert len(values) == 6
    assert random.state_dict() == before


@pytest.mark.parametrize(
    "arguments",
    [
        {"generation": True},
        {"round": True},
        {"replicate": -1},
        {"individual_id": False},
        {"root_seed": True},
    ],
)
def test_invalid_evaluation_work_identities(arguments) -> None:
    values = {"generation": 0, "run_id": "run", "root_seed": 1} | arguments
    with pytest.raises(ConfigurationError):
        EvaluationContext(**values)
