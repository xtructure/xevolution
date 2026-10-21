"""Selection, variation, replacement, and stopping behavior."""

from dataclasses import replace
from random import Random

import pytest

from xevolution.config import EvolutionSettings, RunConfig
from xevolution.errors import (
    ConfigurationError,
    OperatorInapplicable,
    ReproductionError,
    RunStateError,
    SelectionError,
)
from xevolution.models import (
    EvaluationResult,
    Individual,
    JSONValue,
    Population,
    RunState,
    StopDecision,
    VariationResult,
)
from xevolution.operators import (
    CloneCrossover,
    CloneMutation,
    OperatorSpec,
    weighted_order,
)
from xevolution.reproduction import reproduce
from xevolution.selection import (
    Elitism,
    Tournament,
    parent_selector,
    replace_all,
    roulette,
    survivor_selector,
    uniform,
    weighted_index,
)
from xevolution.termination import (
    AllOf,
    AnyOf,
    FitnessAtLeast,
    MaxEvaluations,
    MaxGenerations,
    Not,
    Stagnation,
    from_config,
)


class FixedRandom(Random):
    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


class Opaque:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        raise AssertionError("Genotypes must not be compared")

    def __hash__(self) -> int:
        raise AssertionError("Genotypes must not be hashed")

    def __bool__(self) -> bool:
        raise AssertionError("Genotypes must not be truth-tested")


def evaluated() -> Population[Opaque]:
    return Population(
        (
            Individual(5, Opaque([1]), fitness=10.0, evaluation_count=3),
            Individual(9, Opaque([2]), fitness=2.0, evaluation_count=7),
        )
    )


def mutation(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
    return VariationResult(Opaque([*genotype.values, 10]), {"change": "append"})


def make_state(generation: int = 0) -> RunState[Opaque]:
    settings = EvolutionSettings(population_size=2)
    return RunState(
        run_id="test-run",
        generation=generation,
        population=evaluated(),
        generation_settings=settings,
        effective_settings=settings,
        next_individual_id=10,
        evaluations=2 * (generation + 1),
    )


def test_roulette_zero_boundary_never_selects_zero_weight() -> None:
    assert weighted_index([0.0, 3.0, 0.0, 2.0], FixedRandom(0.0)) == 1
    assert weighted_index([0.0, 3.0, 0.0, 2.0], FixedRandom(0.9999999999999999)) == 3


def test_finite_large_weights_do_not_overflow_the_total() -> None:
    assert weighted_index([0.0, 1e308, 1e308], FixedRandom(0.0)) == 1
    assert weighted_index([0.0, 1e308, 1e308], FixedRandom(0.5)) == 2


def test_all_zero_roulette_has_the_documented_uniform_policy() -> None:
    population = Population(tuple(replace(item, fitness=0.0) for item in evaluated()))
    assert roulette(population, Random(42)) == uniform(population, Random(42))


@pytest.mark.parametrize(
    "weights", [[-1.0, 2.0], [float("nan")], [float("inf")], [True], []]
)
def test_bad_weight_sets_fail_before_sampling(weights) -> None:
    with pytest.raises(SelectionError):
        weighted_index(weights, Random(1))


def test_signed_fitness_uses_a_compatible_selector() -> None:
    population = Population(
        tuple(replace(item, fitness=-item.fitness) for item in evaluated())
    )
    with pytest.raises(SelectionError, match="nonnegative"):
        roulette(population, Random(1))
    assert Tournament(3)(population, Random(1)) in {5, 9}
    assert Elitism(1)(population, 2, Random(1)) == (9,)


def test_elitism_uses_ids_for_ties_and_caps_to_smaller_target() -> None:
    population = Population(
        tuple(replace(item, fitness=1.0) for item in reversed(evaluated().individuals))
    )
    assert Elitism(20)(population, 1, Random(1)) == (5,)
    assert Elitism(0)(population, 2, Random(1)) == ()


def test_selectors_require_evaluated_nonempty_populations() -> None:
    for selector in (roulette, uniform, Tournament()):
        with pytest.raises(SelectionError):
            selector(Population(()), Random(1))
        with pytest.raises(SelectionError):
            selector(Population((Individual(1, Opaque([])),)), Random(1))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Tournament(True),
        lambda: Tournament(0),
        lambda: Elitism(-1),
        lambda: parent_selector({"name": "tournament", "size": False}),
        lambda: parent_selector({"name": "roulette", "size": 3}),
        lambda: survivor_selector({"name": "replace_all", "count": 3}),
        lambda: survivor_selector({"name": "missing"}),
        lambda: parent_selector({"name": "uniform", "unexpected": 2}),
    ],
)
def test_selector_configuration_boundaries(factory) -> None:
    with pytest.raises(ConfigurationError):
        factory()


def test_nonsurvivors_remain_parents_and_offspring_have_fresh_bookkeeping() -> None:
    original = evaluated()
    new, births, next_id = reproduce(
        original,
        EvolutionSettings(population_size=3, mutation_probability=1.0),
        generation=4,
        next_id=50,
        rng=Random(1),
        select_parent=lambda population, rng: 9,
        select_survivors=Elitism(1),
        mutations=(OperatorSpec("append", mutation),),
        crossovers=(),
    )
    assert [item.id for item in new] == [5, 50, 51]
    assert new[0] is original[0]
    assert new[1].genotype.values == [2, 10]
    assert new[1].genotype is not original[1].genotype
    assert new[1].parent_ids == (9,)
    assert new[1].fitness is None
    assert new[1].evaluation_count == 0
    assert new[1].age(4) == 0
    assert next_id == 52
    assert [record.individual_id for record in births] == [50, 51]
    assert births[0].operator == "mutation:append"
    assert original[1].genotype.values == [2]
    assert original[1].evaluation_count == 7


def test_mutation_or_crossover_includes_self_crossover() -> None:
    calls: list[str] = []

    def crossover(
        first: Opaque, second: Opaque, rng: Random
    ) -> VariationResult[Opaque]:
        calls.append("crossover")
        assert first is second
        return VariationResult(Opaque(first.values + second.values))

    new, births, _ = reproduce(
        evaluated(),
        EvolutionSettings(population_size=2, mutation_probability=0.0),
        generation=1,
        next_id=10,
        rng=Random(1),
        select_parent=lambda population, rng: 9,
        select_survivors=replace_all,
        mutations=(),
        crossovers=(OperatorSpec("combine", crossover),),
    )
    assert calls == ["crossover", "crossover"]
    assert births[0].parent_ids == (9, 9)
    assert new[0].genotype.values == [2, 2]


def test_finite_failures_reset_operator_pool_and_never_allocate_partial_ids() -> None:
    calls: list[str] = []

    def fails(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
        calls.append("failure")
        raise OperatorInapplicable("no valid child")

    original = evaluated()
    with pytest.raises(ReproductionError, match="after 3 attempts"):
        reproduce(
            original,
            EvolutionSettings(
                population_size=2, mutation_probability=1, max_attempts_per_child=3
            ),
            generation=1,
            next_id=10,
            rng=Random(1),
            select_parent=uniform,
            select_survivors=replace_all,
            mutations=(OperatorSpec("first", fails), OperatorSpec("second", fails)),
            crossovers=(),
        )
    assert len(calls) == 6
    assert [item.id for item in original] == [5, 9]


def test_zero_weight_disabled_and_fallback_order_is_explicit() -> None:
    calls: list[str] = []

    def fails(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
        calls.append("weighted")
        raise OperatorInapplicable()

    def fallback(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
        calls.append("fallback")
        return mutation(genotype, rng)

    def disabled(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
        raise AssertionError("disabled operation was tried")

    _, _, _ = reproduce(
        evaluated(),
        EvolutionSettings(population_size=2, mutation_probability=1),
        generation=1,
        next_id=10,
        rng=Random(1),
        select_parent=uniform,
        select_survivors=replace_all,
        mutations=(
            OperatorSpec("weighted", fails),
            OperatorSpec("disabled", disabled, weight=0),
            OperatorSpec("fallback", fallback, weight=0, fallback=True),
        ),
        crossovers=(),
    )
    assert calls == ["weighted", "fallback", "weighted", "fallback"]


def test_programming_error_is_not_an_evolutionary_retry() -> None:
    calls = 0

    def broken(genotype: Opaque, rng: Random) -> VariationResult[Opaque]:
        nonlocal calls
        calls += 1
        raise ValueError("application bug")

    with pytest.raises(ValueError, match="application bug"):
        reproduce(
            evaluated(),
            EvolutionSettings(population_size=2, mutation_probability=1),
            generation=1,
            next_id=10,
            rng=Random(1),
            select_parent=uniform,
            select_survivors=replace_all,
            mutations=(OperatorSpec("broken", broken),),
            crossovers=(),
        )
    assert calls == 1


@pytest.mark.parametrize("survivors", [(5, 5), (99,), (True,)])
def test_invalid_survivor_ids_fail(survivors) -> None:
    with pytest.raises(ReproductionError):
        reproduce(
            evaluated(),
            EvolutionSettings(population_size=2, mutation_probability=1),
            generation=1,
            next_id=10,
            rng=Random(1),
            select_parent=uniform,
            select_survivors=lambda population, target_size, rng: survivors,
            mutations=(OperatorSpec("append", mutation),),
            crossovers=(),
        )


def test_disabled_enabled_route_fails_configuration_validation() -> None:
    with pytest.raises(ConfigurationError, match="mutation route"):
        reproduce(
            evaluated(),
            EvolutionSettings(population_size=2, mutation_probability=1),
            generation=1,
            next_id=10,
            rng=Random(1),
            select_parent=uniform,
            select_survivors=replace_all,
            mutations=(OperatorSpec("disabled", mutation, weight=0),),
            crossovers=(),
        )


def test_operator_weights_restore_and_pool_names_are_unique() -> None:
    operation = OperatorSpec("append", mutation, weight=3)
    saved = operation.state_dict()
    operation.weight = 0
    assert list(weighted_order((operation,), Random(1))) == []
    operation.load_state_dict(saved)
    assert list(weighted_order((operation,), Random(1))) == [operation]
    with pytest.raises(ConfigurationError, match="Duplicate"):
        list(weighted_order((operation, operation), Random(1)))
    with pytest.raises(ConfigurationError):
        operation.weight = float("inf")
    with pytest.raises(ConfigurationError):
        operation.load_state_dict({"weight": True})


def test_clone_operators_respect_mutable_genotype_ownership() -> None:
    first, second = evaluated().individuals

    def clone(genotype: Opaque) -> Opaque:
        return Opaque(genotype.values.copy())

    child = CloneMutation(clone)(first.genotype, Random(1))
    child.genotype.values.append(8)
    assert first.genotype.values == [1]
    child2 = CloneCrossover(clone, copy_first=0)(
        first.genotype, second.genotype, Random(1)
    )
    assert child2.genotype.values == [2]
    assert child2.genotype is not second.genotype


def test_basic_stopping_generation_and_budget_semantics() -> None:
    assert MaxGenerations(0)(make_state()).stop
    assert not MaxGenerations(2)(make_state(1)).stop
    assert MaxGenerations(2)(make_state(2)).stop
    assert MaxEvaluations(3)(make_state(1)).stop
    assert FitnessAtLeast(10)(make_state()).stop
    assert not FitnessAtLeast(11)(make_state()).stop
    assert not Stagnation(2)(replace(make_state(3), last_improvement_generation=2)).stop
    assert Stagnation(2)(replace(make_state(4), last_improvement_generation=2)).stop


class CountingStop:
    def __init__(self, answer: bool) -> None:
        self.calls = 0
        self.answer = answer

    def __call__(self, state: RunState[Opaque]) -> StopDecision:
        self.calls += 1
        return StopDecision(self.answer)

    def state_dict(self) -> JSONValue:
        return {"calls": self.calls}

    def load_state_dict(self, state: JSONValue) -> None:
        if not isinstance(state, dict) or not isinstance(state.get("calls"), int):
            raise ValueError("bad counter state")
        self.calls = state["calls"]


def test_composites_call_all_children_once_and_restore_nested_state() -> None:
    a, b = CountingStop(True), CountingStop(False)
    condition = AnyOf((a, Not(b)))
    assert condition(make_state()).stop
    assert (a.calls, b.calls) == (1, 1)
    saved = condition.state_dict()
    condition(make_state(1))
    assert (a.calls, b.calls) == (2, 2)
    condition.load_state_dict(saved)
    assert (a.calls, b.calls) == (1, 1)
    assert not AllOf((b, a))(make_state(2)).stop
    assert (a.calls, b.calls) == (2, 2)
    with pytest.raises(RunStateError):
        condition.load_state_dict({"children": []})


def test_composites_restore_state_owned_by_bound_methods() -> None:
    first, second = CountingStop(True), CountingStop(False)
    condition = AnyOf((first.__call__, Not(second.__call__)))
    condition(make_state())
    saved = condition.state_dict()
    condition(make_state(1))
    assert (first.calls, second.calls) == (2, 2)
    condition.load_state_dict(saved)
    assert (first.calls, second.calls) == (1, 1)
    restored_first, restored_second = CountingStop(True), CountingStop(False)
    restored = AnyOf((restored_first.__call__, Not(restored_second.__call__)))
    restored.load_state_dict(saved)
    assert restored(make_state(1)).stop
    assert (restored_first.calls, restored_second.calls) == (2, 2)


@pytest.mark.parametrize("constructor", [AnyOf, AllOf])
def test_composites_validate_programmatic_predicates(constructor) -> None:
    with pytest.raises(ConfigurationError, match="callable"):
        constructor((MaxGenerations(2), False))
    with pytest.raises(ConfigurationError, match="at least one"):
        constructor(iter(()))
    with pytest.raises(ConfigurationError, match="callable"):
        Not(False)


@pytest.mark.parametrize("bound_method", [False, True])
@pytest.mark.parametrize(
    "attributes",
    [("state_dict",), ("load_state_dict",), ("state_dict", "load_state_dict")],
)
def test_composites_reject_noncallable_state_attributes(
    bound_method: bool, attributes: tuple[str, ...]
) -> None:
    class InvalidStateStop:
        def __call__(self, state: RunState[Opaque]) -> bool:
            return False

    predicate = InvalidStateStop()
    for name in attributes:
        setattr(predicate, name, None)
    callback = predicate.__call__ if bound_method else predicate
    with pytest.raises(ConfigurationError, match="both state methods to be callable"):
        AnyOf((MaxGenerations(3), callback))


def test_stopping_configuration_and_validation() -> None:
    condition = from_config(
        {"any": [{"max_generations": 2}, {"fitness_at_least": 11.0}]}
    )
    assert not condition(make_state(1)).stop
    assert condition(make_state(2)).stop
    for invalid in (
        {},
        {"any": []},
        {"max_generations": True},
        {"fitness_at_least": float("inf")},
        {"stagnation": 0},
    ):
        with pytest.raises(ConfigurationError):
            from_config(invalid)


def test_engine_lifecycle_with_arbitrary_payload_and_operational_pause() -> None:
    from xevolution.engine import Engine
    from xevolution.experiment import Experiment

    calls: list[int] = []

    def evaluate(individual, context):
        calls.append(individual.id)
        return EvaluationResult(float(sum(individual.genotype.values)))

    def clone(genotype: Opaque) -> Opaque:
        return Opaque(genotype.values.copy())

    stopping = CountingStop(False)
    stop = AnyOf((stopping, MaxGenerations(2)))
    experiment = Experiment(
        initialize=lambda rng: Opaque([1]),
        evaluate=evaluate,
        mutations=(OperatorSpec("append", mutation),),
        clone=clone,
        stop=stop,
    )
    with Engine.create(
        experiment,
        config=RunConfig(
            seed=99,
            evolution=EvolutionSettings(population_size=4, mutation_probability=1),
        ),
    ) as engine:
        assert engine.state.generation == 0
        assert engine.state.evaluations == 4
        assert stopping.calls == 1
        initial = engine.state.population
        engine.run(until_generation=1)
        assert engine.state.generation == 1
        assert engine.state.phase == "ready"
        assert engine.state.evaluations == 8
        assert stopping.calls == 2
        assert [individual.id for individual in engine.state.population] == [4, 5, 6, 7]
        assert all(individual.age(1) == 0 for individual in engine.state.population)
        assert all(individual.genotype.values == [1] for individual in initial)
        engine.run()
        assert engine.state.generation == 2
        assert engine.state.phase == "terminated"
        assert stopping.calls == 3
        assert len(calls) == 12
        with pytest.raises(StopIteration):
            engine.step()
        assert stopping.calls == 3


def test_adaptive_target_is_saved_for_the_next_generation() -> None:
    from xevolution.engine import Engine
    from xevolution.experiment import Experiment

    def grow(state: RunState[Opaque]) -> EvolutionSettings | None:
        if state.generation == 0:
            return state.effective_settings.updated(population_size=6)
        return None

    experiment = Experiment(
        initialize=lambda rng: Opaque([1]),
        evaluate=lambda individual, context: EvaluationResult(1),
        mutations=(OperatorSpec("append", mutation),),
        clone=lambda genotype: Opaque(genotype.values.copy()),
        stop=MaxGenerations(1),
        adaptive_hooks=(grow,),
    )
    with Engine.create(
        experiment,
        config=RunConfig(
            evolution=EvolutionSettings(population_size=4, mutation_probability=1)
        ),
    ) as engine:
        assert len(engine.state.population) == 4
        assert engine.state.generation_settings.population_size == 4
        assert engine.state.effective_settings.population_size == 6
        engine.step()
        assert len(engine.state.population) == 6
        assert engine.state.generation_settings.population_size == 6
        assert engine.state.evaluations == 10


def test_engine_reevaluates_survivors_and_retains_independent_historical_best() -> None:
    from xevolution.engine import Engine
    from xevolution.experiment import Experiment

    experiment = Experiment(
        initialize=lambda rng: Opaque([1]),
        evaluate=lambda individual, context: EvaluationResult(100 - context.generation),
        mutations=(OperatorSpec("append", mutation),),
        clone=lambda genotype: Opaque(genotype.values.copy()),
        select_survivors=Elitism(1),
        stop=MaxGenerations(1),
    )
    with Engine.create(
        experiment,
        config=RunConfig(
            evolution=EvolutionSettings(population_size=3, mutation_probability=1)
        ),
    ) as engine:
        engine.step()
        assert engine.state.population.by_id[0].evaluation_count == 2
        assert engine.state.population.by_id[0].fitness == 99
        assert engine.state.evaluations == 6
        assert engine.state.best is not None
        assert engine.state.best.fitness == 100
        assert engine.state.last_improvement_generation == 0
        engine.state.population.by_id[0].genotype.values.append(999)
        assert engine.state.best.genotype.values == [1]


def test_engine_rejects_an_unavailable_route_before_initialization() -> None:
    from xevolution.engine import Engine
    from xevolution.experiment import Experiment

    initialized: list[int] = []

    def initialize(rng: Random) -> Opaque:
        initialized.append(1)
        return Opaque([1])

    experiment = Experiment(
        initialize=initialize,
        evaluate=lambda individual, context: EvaluationResult(1),
        mutations=(OperatorSpec("append", mutation),),
        clone=lambda genotype: Opaque(genotype.values.copy()),
        stop=MaxGenerations(1),
    )
    with pytest.raises(ConfigurationError, match="crossover route"):
        Engine.create(experiment, config=RunConfig())
    assert initialized == []


@pytest.mark.parametrize("change", ["settings", "weight"])
def test_adaptation_cannot_accept_an_unavailable_next_route(change: str) -> None:
    from xevolution.engine import Engine
    from xevolution.experiment import Experiment

    operator = OperatorSpec("append", mutation)

    def adapt(state: RunState[Opaque]) -> EvolutionSettings | None:
        if state.generation == 1:
            if change == "settings":
                return state.effective_settings.updated(mutation_probability=0)
            operator.weight = 0
        return None

    experiment = Experiment(
        initialize=lambda rng: Opaque([1]),
        evaluate=lambda individual, context: EvaluationResult(1),
        mutations=(operator,),
        clone=lambda genotype: Opaque(genotype.values.copy()),
        adaptive_hooks=(adapt,),
        stop=MaxGenerations(2),
    )
    with Engine.create(
        experiment,
        config=RunConfig(
            evolution=EvolutionSettings(population_size=2, mutation_probability=1)
        ),
    ) as engine:
        with pytest.raises(ConfigurationError, match="route"):
            engine.step()
        assert engine.state.generation == 0
        assert engine.state.evaluations == 2
        assert engine.state.effective_settings.mutation_probability == 1
        with pytest.raises(RunStateError, match="failed during a transition"):
            engine.step()
