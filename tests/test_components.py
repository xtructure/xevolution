"""Component identity, explicit state ownership, and trusted factory composition."""

from __future__ import annotations

import functools
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random
from random import random as global_random_draw

import pytest

from xevolution.config import EvolutionSettings, RunConfig, as_object
from xevolution.engine import Engine, create_engine, resume
from xevolution.errors import CheckpointCompatibilityError, ConfigurationError
from xevolution.experiment import ComponentCatalog, Experiment, compose
from xevolution.models import (
    EvaluationResult,
    Individual,
    JSONDict,
    JSONValue,
    RunState,
    VariationResult,
)
from xevolution.operators import OperatorSpec
from xevolution.protocols import AttachmentReader, AttachmentWriter
from xevolution.randomness import EvaluationContext, RunRandom
from xevolution.registry import ExperimentRegistry
from xevolution.storage import RunStore
from xevolution.termination import AnyOf, MaxGenerations

CALLBACK_EVENTS: list[int] = []


def initialize(rng: Random) -> float:
    return rng.random()


def evaluate(
    individual: Individual[float], context: EvaluationContext
) -> EvaluationResult:
    return EvaluationResult(individual.genotype + 1.0)


def mutate(genotype: float, rng: Random) -> VariationResult[float]:
    return VariationResult(genotype + rng.random())


def basic_experiment() -> Experiment[float]:
    return Experiment(
        initialize=initialize,
        evaluate=evaluate,
        mutations=(OperatorSpec("step", mutate),),
        stop=MaxGenerations(3),
    )


def small_config(**kwargs: JSONValue) -> RunConfig:
    return RunConfig(
        seed=17,
        evolution=EvolutionSettings(population_size=2, mutation_probability=1.0),
        parameters=kwargs,
    )


class SharedCounter:
    """One stateful owner used through different methods or composite slots."""

    def __init__(self) -> None:
        self.calls = 0
        self.restores = (
            0  # Diagnostic only; deliberately excluded from scientific state.
        )
        self.exports = 0  # Diagnostic only; never read by evolutionary callbacks.
        self.history: list[int] = []

    def initialize(self, rng: Random) -> float:
        self.calls += 1
        self.history.append(self.calls)
        return rng.random()

    def evaluate(
        self, individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        self.calls += 1
        self.history.append(self.calls)
        return EvaluationResult(individual.genotype + self.calls)

    def __call__(self, state: RunState[float]) -> bool:
        self.calls += 1
        self.history.append(self.calls)
        return self.calls >= 20

    def state_dict(self) -> JSONValue:
        self.exports += 1
        return {"calls": self.calls, "history": self.history}

    def load_state_dict(self, state: JSONValue) -> None:
        data = as_object(state)
        calls, history = data.get("calls"), data.get("history")
        if type(calls) is not int or not isinstance(history, list):
            raise ConfigurationError("Invalid SharedCounter state")
        if any(type(value) is not int for value in history):
            raise ConfigurationError("Invalid SharedCounter history")
        self.calls = calls
        self.history = [value for value in history if type(value) is int]
        self.restores += 1


class StatefulMutation:
    def __init__(self, calls: int = 0) -> None:
        self.calls = calls

    def __call__(self, genotype: float, rng: Random) -> VariationResult[float]:
        self.calls += 1
        return VariationResult(genotype + self.calls)

    def state_dict(self) -> JSONValue:
        return {"calls": self.calls}

    def load_state_dict(self, state: JSONValue) -> None:
        calls = as_object(state).get("calls")
        if type(calls) is not int:
            raise ConfigurationError("Invalid mutation state")
        self.calls = calls


def test_operator_names_cannot_overwrite_generated_component_slots() -> None:
    first, second, third = StatefulMutation(2), StatefulMutation(3), StatefulMutation(4)
    application = replace(
        basic_experiment(),
        mutations=(
            OperatorSpec("a", first),
            OperatorSpec("a/operation", second),
            OperatorSpec("a~1operation", third),
        ),
    )
    catalog = ComponentCatalog(compose(application, small_config()), RunRandom(17))
    calls = [
        value["calls"]
        for value in catalog.export().values()
        if isinstance(value, dict) and "calls" in value
    ]
    assert sorted(calls) == [2, 3, 4]


def test_application_names_cannot_overwrite_composite_child_slots() -> None:
    first, second = SharedCounter(), SharedCounter()
    first.calls, second.calls = 2, 3
    application = replace(
        basic_experiment(),
        components={
            "team": AnyOf((first,)),
            "team/predicate/0": second,
        },
    )
    catalog = ComponentCatalog(compose(application, small_config()), RunRandom(17))
    calls = [
        value["calls"]
        for value in catalog.export().values()
        if isinstance(value, dict) and "calls" in value
    ]
    assert sorted(calls) == [2, 3]


def test_shared_bound_method_owner_is_exported_and_restored_once() -> None:
    counter = SharedCounter()
    experiment = Experiment(
        initialize=counter.initialize,
        evaluate=counter.evaluate,
        mutations=(OperatorSpec("step", mutate),),
        stop=MaxGenerations(2),
        components={"shared": counter},
    )
    catalog = ComponentCatalog(compose(experiment, small_config()), RunRandom(17))
    counter.calls = 7
    counter.history = [2, 7]
    saved = catalog.export()
    assert set(saved) == {"application/shared", "mutation/step"}
    assert counter.exports == 1
    slot_map = as_object(catalog.identity["slots"])
    assert (
        as_object(slot_map["initialize"])["component"]
        == as_object(slot_map["evaluate"])["component"]
    )
    exported = as_object(saved["application/shared"])
    history = exported["history"]
    assert isinstance(history, list)
    history.append(99)
    assert counter.history == [2, 7]
    catalog.restore(saved)
    assert counter.restores == 1
    assert counter.calls == 7
    assert counter.history == [2, 7, 99]


def test_composite_shared_leaf_is_exported_and_restored_once() -> None:
    counter = SharedCounter()
    experiment = replace(
        basic_experiment(),
        evaluate=counter.evaluate,
        stop=AnyOf((counter, AnyOf((counter, MaxGenerations(3))))),
        components={"shared": counter},
    )
    catalog = ComponentCatalog(compose(experiment, small_config()), RunRandom(17))
    counter.calls = 5
    saved = catalog.export()
    assert set(saved) == {"application/shared", "mutation/step"}
    assert counter.exports == 1
    counter.calls = 0
    catalog.restore(saved)
    assert counter.calls == 5
    assert counter.restores == 1


def test_captured_bound_managed_rng_is_recognized() -> None:
    manager = RunRandom(17)
    draw = manager.evolution.random

    def closure(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        return EvaluationResult(draw())

    ComponentCatalog(
        compose(replace(basic_experiment(), evaluate=closure), small_config()), manager
    )


def test_component_state_keys_are_exact_and_values_are_validated() -> None:
    counter = SharedCounter()
    experiment = replace(basic_experiment(), evaluate=counter.evaluate)
    catalog = ComponentCatalog(compose(experiment, small_config()), RunRandom(17))
    saved = catalog.export()
    with pytest.raises(CheckpointCompatibilityError, match="keys"):
        catalog.restore({})
    with pytest.raises(CheckpointCompatibilityError, match="keys"):
        catalog.restore({**saved, "unexpected": {}})
    owner = next(name for name in saved if "evaluate" in name)
    saved[owner] = {"calls": True, "history": []}
    with pytest.raises(ConfigurationError, match="SharedCounter"):
        catalog.restore(saved)


def test_state_protocol_attributes_must_be_callable() -> None:
    class BrokenState:
        state_dict = "not callable"
        load_state_dict = "not callable"

    application = replace(basic_experiment(), components={"broken": BrokenState()})
    with pytest.raises(ConfigurationError):
        ComponentCatalog(compose(application, small_config()), RunRandom(17))


def test_frozen_wrapper_does_not_hide_an_unregistered_stateful_leaf() -> None:
    @dataclass(frozen=True)
    class Wrapper:
        counter: SharedCounter

        def __call__(
            self, individual: Individual[float], context: EvaluationContext
        ) -> EvaluationResult:
            return self.counter.evaluate(individual, context)

    counter = SharedCounter()
    application = replace(basic_experiment(), evaluate=Wrapper(counter))
    with pytest.raises(ConfigurationError):
        ComponentCatalog(compose(application, small_config()), RunRandom(17))
    registered = replace(application, components={"counter": counter})
    ComponentCatalog(compose(registered, small_config()), RunRandom(17))


def test_hidden_mutable_closure_and_partial_arguments_are_rejected() -> None:
    events: list[int] = []

    def closure(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        events.append(individual.id)
        return EvaluationResult(float(len(events)))

    with pytest.raises(ConfigurationError, match="mutable state"):
        ComponentCatalog(
            compose(replace(basic_experiment(), evaluate=closure), small_config()),
            RunRandom(17),
        )

    def collect(
        events: list[int], individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        events.append(individual.id)
        return EvaluationResult(1.0)

    with pytest.raises(ConfigurationError, match="mutable state"):
        ComponentCatalog(
            compose(
                replace(basic_experiment(), evaluate=functools.partial(collect, [])),
                small_config(),
            ),
            RunRandom(17),
        )


@dataclass(frozen=True)
class GlobalListEvaluator:
    def __call__(
        self, individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        CALLBACK_EVENTS.append(individual.id)
        return EvaluationResult(float(len(CALLBACK_EVENTS)))


class GlobalListMethod(SharedCounter):
    def evaluate(
        self, individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        CALLBACK_EVENTS.append(individual.id)
        return EvaluationResult(float(len(CALLBACK_EVENTS)))


@pytest.mark.parametrize("bound_method", [False, True])
def test_callable_methods_cannot_hide_mutable_globals(bound_method: bool) -> None:
    callback = GlobalListMethod().evaluate if bound_method else GlobalListEvaluator()
    application = replace(basic_experiment(), evaluate=callback)
    with pytest.raises(ConfigurationError):
        ComponentCatalog(compose(application, small_config()), RunRandom(17))


def test_opaque_mutable_capture_is_rejected() -> None:
    events: deque[int] = deque()

    def closure(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        events.append(individual.id)
        return EvaluationResult(float(len(events)))

    application = replace(basic_experiment(), evaluate=closure)
    with pytest.raises(ConfigurationError):
        ComponentCatalog(compose(application, small_config()), RunRandom(17))


def test_immutable_capture_contents_cannot_be_omitted_from_identity() -> None:
    def application_for(target: bytes) -> Experiment[float]:
        def closure(
            individual: Individual[float], context: EvaluationContext
        ) -> EvaluationResult:
            return EvaluationResult(float(target[0]) + individual.genotype)

        return replace(basic_experiment(), evaluate=closure)

    try:
        first = ComponentCatalog(
            compose(application_for(b"a"), small_config()), RunRandom(17)
        )
    except ConfigurationError:
        return  # Explicit rejection is safe when this value type has no identity codec.
    second = ComponentCatalog(
        compose(application_for(b"b"), small_config()), RunRandom(17)
    )
    assert first.identity != second.identity


def test_stateful_closure_capture_requires_explicit_registration() -> None:
    counter = SharedCounter()

    def closure(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        return counter.evaluate(individual, context)

    application = replace(basic_experiment(), evaluate=closure)
    with pytest.raises(ConfigurationError):
        ComponentCatalog(compose(application, small_config()), RunRandom(17))
    registered = replace(application, components={"counter": counter})
    catalog = ComponentCatalog(compose(registered, small_config()), RunRandom(17))
    counter.calls = 4
    counter.history = [4]
    saved = catalog.export()
    assert as_object(saved["application/counter"])["calls"] == 4
    counter.calls = 0
    catalog.restore(saved)
    assert counter.calls == 4


def test_builtin_bound_global_random_capture_is_rejected() -> None:
    def uses_global_random(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        return EvaluationResult(global_random_draw())

    with pytest.raises(ConfigurationError):
        ComponentCatalog(
            compose(
                replace(basic_experiment(), evaluate=uses_global_random), small_config()
            ),
            RunRandom(17),
        )


def test_builtin_bound_method_cannot_hide_a_mutable_container() -> None:
    append = [].append

    def closure(
        individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        append(individual.id)
        return EvaluationResult(1.0)

    with pytest.raises(ConfigurationError):
        ComponentCatalog(
            compose(replace(basic_experiment(), evaluate=closure), small_config()),
            RunRandom(17),
        )


@dataclass(frozen=True)
class FixedEvaluator:
    scale: float

    def __call__(
        self, individual: Individual[float], context: EvaluationContext
    ) -> EvaluationResult:
        return EvaluationResult(self.scale + individual.genotype)


@pytest.mark.parametrize(
    "change", ["configuration", "application_version", "alias_graph"]
)
def test_resume_rejects_changed_composition_and_releases_lock(
    tmp_path: Path, change: str
) -> None:
    counter = SharedCounter()
    original = replace(
        basic_experiment(),
        initialize=counter.initialize,
        evaluate=FixedEvaluator(1.0),
        components={"shared": counter},
    )
    with Engine.create(original, config=small_config(), output=tmp_path):
        pass
    if change == "configuration":
        changed = replace(original, evaluate=FixedEvaluator(2.0))
    elif change == "application_version":
        changed = replace(original, version="different")
    else:
        changed = replace(original, components={"shared": SharedCounter()})
    with pytest.raises(CheckpointCompatibilityError, match="compatibility mismatch"):
        resume(tmp_path, experiment=changed)
    with RunStore(tmp_path) as store:
        store.load(writable=True)


def test_shared_composite_children_cannot_change_aliases_silently(
    tmp_path: Path,
) -> None:
    first, second = SharedCounter(), SharedCounter()
    original = replace(basic_experiment(), stop=AnyOf((first, second)))
    with Engine.create(original, config=small_config(), output=tmp_path):
        pass
    shared = SharedCounter()
    changed = replace(basic_experiment(), stop=AnyOf((shared, shared)))
    with pytest.raises((CheckpointCompatibilityError, ConfigurationError)):
        resume(tmp_path, experiment=changed)


def sampled_factory(config: RunConfig, random: RunRandom) -> Experiment[float]:
    return replace(
        basic_experiment(), evaluate=FixedEvaluator(random.sampling.random())
    )


def test_factory_receives_original_sampling_position_on_resume(tmp_path: Path) -> None:
    config = replace(
        small_config(),
        factory="sampled",
        sampling={"parameters.sample": {"kind": "uniform", "low": 0.0, "high": 1.0}},
    )
    registry = ExperimentRegistry({"sampled": sampled_factory})
    with create_engine(config, registry=registry, output=tmp_path) as engine:
        boundary_random = engine.random.state_dict()
        previous = engine.state
    with resume(tmp_path, registry=registry) as restored:
        assert restored.random.state_dict() == boundary_random
        assert restored.state.generation == 0
        assert restored.state.population[0].fitness == previous.population[0].fitness
        restored.step()


class RestoreDraws(SharedCounter):
    def __init__(self, manager: RunRandom) -> None:
        super().__init__()
        self.manager = manager

    def load_state_dict(self, state: JSONValue) -> None:
        self.manager.evolution.random()
        self.manager.sampling.random()
        super().load_state_dict(state)


def restore_draws_factory(config: RunConfig, random: RunRandom) -> Experiment[float]:
    counter = RestoreDraws(random)
    return replace(
        basic_experiment(), initialize=counter.initialize, evaluate=counter.evaluate
    )


def test_managed_rngs_are_restored_after_component_setup(tmp_path: Path) -> None:
    registry = ExperimentRegistry({"restore_draws": restore_draws_factory})
    config = replace(small_config(), factory="restore_draws")
    with create_engine(config, registry=registry, output=tmp_path) as engine:
        expected_random = engine.random.state_dict()
        expected_fitness = [item.fitness for item in engine.state.population]
    with resume(tmp_path, registry=registry) as restored:
        assert restored.random.state_dict() == expected_random
        assert [item.fitness for item in restored.state.population] == expected_fitness


class MutatingDecodeCodec:
    codec_id = "tests.mutating-decode"
    version = 1

    def __init__(self) -> None:
        self.decodes = 0

    def encode(self, value: float, writer: AttachmentWriter | None = None) -> JSONValue:
        return value

    def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> float:
        self.decodes += 1
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("Expected a numeric payload")
        return float(value)

    def state_dict(self) -> JSONValue:
        return {"decodes": self.decodes}

    def load_state_dict(self, state: JSONValue) -> None:
        decodes = as_object(state).get("decodes")
        if type(decodes) is not int:
            raise ConfigurationError("Invalid codec state")
        self.decodes = decodes


def copy_float(value: float) -> float:
    return value


def test_decode_cannot_mutate_restored_future_component_state(tmp_path: Path) -> None:
    application = replace(
        basic_experiment(), codec=MutatingDecodeCodec(), clone=copy_float
    )
    with Engine.create(application, config=small_config(), output=tmp_path):
        pass
    before = RunStore(tmp_path).load()
    with pytest.raises(CheckpointCompatibilityError):
        resume(tmp_path, experiment=replace(application, codec=MutatingDecodeCodec()))
    assert RunStore(tmp_path).load() == before


def test_lambdas_with_same_qualified_name_have_distinct_code_identity(
    tmp_path: Path,
) -> None:
    evaluators = (
        lambda individual, context: EvaluationResult(individual.genotype + 1.0),
        lambda individual, context: EvaluationResult(individual.genotype + 2.0),
    )
    assert evaluators[0].__qualname__ == evaluators[1].__qualname__
    original = replace(basic_experiment(), evaluate=evaluators[0])
    with Engine.create(original, config=small_config(), output=tmp_path):
        pass
    with pytest.raises(CheckpointCompatibilityError):
        resume(tmp_path, experiment=replace(original, evaluate=evaluators[1]))


def test_registry_resolves_explicit_aliases_and_module_attributes() -> None:
    registry = ExperimentRegistry({"sampled": sampled_factory})
    assert registry.resolve("sampled") is sampled_factory
    assert registry.resolve("examples.arithmetic:build_experiment") is not None
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("sampled", sampled_factory)
    with pytest.raises(ConfigurationError, match="unknown factory"):
        registry.resolve("missing")
    with pytest.raises(ConfigurationError, match="cannot resolve"):
        registry.resolve("examples.arithmetic:no_such_factory")
    with pytest.raises(ConfigurationError, match="not callable"):
        registry.resolve("xevolution:__version__")


def wrong_factory(config: RunConfig, random: RunRandom) -> JSONDict:
    return {"not": "an experiment"}


def test_factory_result_is_validated_before_creating_output(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    registry.register("bad", wrong_factory)
    output = tmp_path / "unused"
    with pytest.raises(ConfigurationError, match="return an Experiment"):
        create_engine(
            replace(small_config(), factory="bad"), registry=registry, output=output
        )
    assert not output.exists()
