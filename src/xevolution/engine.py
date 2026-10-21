"""Generation coordination and exact continuation at committed boundaries."""

from __future__ import annotations

import hashlib
import logging
import platform
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, cast

from . import __version__
from .config import EvolutionSettings, RunConfig, as_object, load_config, strict_dumps
from .errors import CheckpointCompatibilityError, ConfigurationError, RunStateError
from .evaluation import evaluate_population
from .experiment import ComponentCatalog, Composition, Experiment, compose
from .models import (
    GenerationRecord,
    Individual,
    JSONDict,
    JSONValue,
    LineageRecord,
    Population,
    RunState,
    StopDecision,
)
from .operators import validate_routes
from .randomness import RunRandom
from .registry import ExperimentRegistry
from .reproduction import reproduce
from .serialization import decode_state, encode_state
from .storage import RunStore
from .tracking import make_record, update_statistics

logger = logging.getLogger(__name__)
ENGINE_SCHEMA_VERSION = 1


def _compatibility[G](
    composition: Composition[G],
    catalog: ComponentCatalog,
    construction_random: JSONDict | None,
) -> JSONDict:
    sources: JSONDict = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    experiment = composition.experiment
    return {
        "engine_schema": ENGINE_SCHEMA_VERSION,
        "construction_random": construction_random,
        "package_version": __version__,
        "framework_sources": sources,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "libc": list(platform.libc_ver()),
        },
        "application": {
            "identity": experiment.identity,
            "version": experiment.version,
            "requirements": dict(experiment.compatibility),
        },
        "codec": {
            "id": composition.codec.codec_id,
            "version": composition.codec.version,
        },
        "composition": catalog.identity,
    }


def _check_compatibility(saved: JSONValue, expected: JSONDict) -> None:
    data = as_object(saved, "compatibility")
    if strict_dumps(data) != strict_dumps(expected):
        differences = sorted(
            name
            for name in data.keys() | expected.keys()
            if data.get(name) != expected.get(name)
        )
        raise CheckpointCompatibilityError(
            "checkpoint compatibility mismatch in "
            + ", ".join(differences)
            + "; resume with the recorded application, component configuration, code and Python environment"
        )


class Engine[G]:
    """Own one logical run and, optionally, its single-writer run directory.

    Use ``create`` for an evaluated generation 0 or ``resume`` to load a saved
    generation. ``step`` commits one complete transition. ``run`` stops at a
    configured terminal decision or an operational pause. Failed transitions
    poison this instance: close it and reload the last committed checkpoint.
    """

    def __init__(
        self,
        composition: Composition[G],
        config: RunConfig,
        random: RunRandom,
        store: RunStore | None,
        construction_random: JSONDict | None = None,
    ) -> None:
        self._composition = composition
        self.config = config
        self.random = random
        self._store = store
        self._catalog = (
            ComponentCatalog(composition, random) if store is not None else None
        )
        self._compatibility = (
            _compatibility(composition, self._catalog, construction_random)
            if self._catalog is not None
            else None
        )
        self._state: RunState[G] | None = None
        self.last_record: GenerationRecord | None = None
        self._closed = False
        self._failed = False
        self._pause_requested = False

    @property
    def state(self) -> RunState[G]:
        """The last accepted generation; treat all nested payloads as read-only."""
        if self._state is None:
            raise RunStateError("initial generation has not been accepted")
        return self._state

    @classmethod
    def create(
        cls,
        experiment: Experiment[G],
        *,
        config: RunConfig | None = None,
        output: Path | str | None = None,
        random: RunRandom | None = None,
        _construction_random: JSONDict | None = None,
    ) -> Engine[G]:
        """Initialize, evaluate and commit generation 0 exactly once.

        Programmatic experiments must receive already resolved configuration;
        use ``create_engine`` to resolve sampled config before calling a factory.
        If construction uses managed RNGs, pass that same ``RunRandom`` here.
        """
        if not isinstance(experiment, Experiment):
            raise ConfigurationError("factory must return an Experiment")
        selected = config if config is not None else RunConfig()
        if not isinstance(selected, RunConfig):
            raise ConfigurationError("config must be RunConfig")
        if selected.sampling:
            raise ConfigurationError(
                "resolve sampled configuration before constructing an Experiment; use create_engine"
            )
        selected = RunConfig.from_dict(selected.to_dict())
        manager = random if random is not None else RunRandom(selected.seed)
        if manager.seed != selected.seed:
            raise ConfigurationError("managed random seed differs from configuration")
        store = None if output is None else RunStore(output)
        engine = cls(
            compose(experiment, selected),
            selected,
            manager,
            store,
            _construction_random,
        )
        try:
            run_id = uuid.uuid4().hex  # Operational identity; excluded from seeds.
            if store is not None:
                store.create(selected.to_dict(), run_id)
            size = selected.evolution.population_size
            population = Population(
                tuple(
                    Individual(
                        id=index, genotype=experiment.initialize(manager.evolution)
                    )
                    for index in range(size)
                )
            )
            births = tuple(
                LineageRecord(item.id, 0, (), "initialize") for item in population
            )
            initial = RunState(
                run_id=run_id,
                generation=0,
                population=population,
                generation_settings=selected.evolution,
                effective_settings=selected.evolution,
                next_individual_id=size,
            )
            engine._finalize(initial, births)
            return engine
        except BaseException:
            engine._failed = True
            engine.close()
            raise

    def _require_active(self) -> None:
        if self._closed:
            raise RunStateError("engine is closed")
        if self._failed:
            raise RunStateError(
                "engine failed during a transition; close it and resume the last checkpoint"
            )

    def _finalize(
        self, state: RunState[G], births: tuple[LineageRecord, ...]
    ) -> GenerationRecord:
        application = self._composition.experiment
        evaluated = evaluate_population(
            state.population,
            self.random.context(state.generation, state.run_id),
            evaluate=application.evaluate,
            evaluate_population=application.evaluate_population,
        )
        state = replace(
            state, population=evaluated, evaluations=state.evaluations + len(evaluated)
        )
        state = update_statistics(state, self._composition.clone)
        for hook in application.adaptive_hooks:
            settings = hook(state)
            if settings is not None:
                if not isinstance(settings, EvolutionSettings):
                    raise ConfigurationError(
                        "adaptive hooks must return EvolutionSettings or None"
                    )
                self.config.validate_update(state.effective_settings, settings)
                state = replace(state, effective_settings=settings)
        validate_routes(
            state.effective_settings, application.mutations, application.crossovers
        )
        raw_decision = self._composition.stop(state)
        if type(raw_decision) is bool:
            decision = StopDecision(
                raw_decision, "custom predicate" if raw_decision else None
            )
        elif isinstance(raw_decision, StopDecision):
            decision = raw_decision
        else:
            raise ConfigurationError(
                "stopping predicates must return StopDecision or bool"
            )
        state = replace(
            state,
            stop=decision,
            phase="terminated" if decision.stop else "ready",
            next_event_sequence=state.next_event_sequence + 1,
        )
        record = make_record(state, births)
        if self._store is not None:
            catalog, compatibility = self._catalog, self._compatibility
            if catalog is None or compatibility is None:
                raise RunStateError(
                    "persistent engine is missing its component catalog"
                )
            random_state = self.random.state_dict()
            body: JSONDict = {
                "run_id": state.run_id,
                "config": self.config.to_dict(),
                "state": encode_state(state, self._composition.codec, self._store),
                "random": random_state,
                "components": catalog.export(),
                "compatibility": compatibility,
            }
            if self.random.state_dict() != random_state:
                raise RunStateError("checkpoint encoding consumed managed randomness")
            self._store.commit(body, record)
        # Only expose success after the store's complete commit protocol returns.
        self._state = state
        self.last_record = record
        logger.debug(
            "Completed generation %d with %d members",
            state.generation,
            len(state.population),
        )
        return record

    def step(self) -> GenerationRecord:
        """Advance one generation, or raise StopIteration for a terminal run."""
        self._require_active()
        current = self.state
        if current.phase == "terminated":
            raise StopIteration("run is terminated")
        try:
            settings = current.effective_settings
            composition = self._composition
            population, births, next_id = reproduce(
                current.population,
                settings,
                generation=current.generation + 1,
                next_id=current.next_individual_id,
                rng=self.random.evolution,
                select_parent=composition.select_parent,
                select_survivors=composition.select_survivors,
                mutations=composition.experiment.mutations,
                crossovers=composition.experiment.crossovers,
            )
            pending = replace(
                current,
                generation=current.generation + 1,
                population=population,
                generation_settings=settings,
                next_individual_id=next_id,
            )
            return self._finalize(pending, births)
        except BaseException:
            self._failed = True
            raise

    def run(self, *, until_generation: int | None = None) -> RunState[G]:
        """Run until termination or a completed-generation operational pause.

        An absolute ``until_generation`` never changes the configured stopping
        policy. A requested pause is consumed when this method returns; call
        ``run`` again or resume the ready checkpoint to continue.
        """
        self._require_active()
        if until_generation is not None and (
            type(until_generation) is not int or until_generation < 0
        ):
            raise ConfigurationError("until_generation must be a nonnegative integer")
        try:
            while self.state.phase != "terminated" and not self._pause_requested:
                if (
                    until_generation is not None
                    and self.state.generation >= until_generation
                ):
                    break
                self.step()
            return self.state
        finally:
            self._pause_requested = False

    def request_pause(self) -> None:
        """Request a return from run after any currently executing generation."""
        self._pause_requested = True

    def close(self) -> None:
        """Release the writer lock. Closing never evaluates or commits a generation."""
        if self._store is not None:
            self._store.close()
        self._closed = True

    def __enter__(self) -> Self:
        self._require_active()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _config(value: RunConfig | Path | str) -> RunConfig:
    return value if isinstance(value, RunConfig) else load_config(value)


def _build(
    config: RunConfig, random: RunRandom, registry: ExperimentRegistry | None
) -> Experiment[Any]:
    if config.factory is None:
        raise ConfigurationError(
            "configuration needs experiment.factory; programmatic runs use Engine.create"
        )
    factory = (registry or ExperimentRegistry()).resolve(config.factory)
    experiment = factory(config, random)
    if not isinstance(experiment, Experiment):
        raise ConfigurationError("factory must return an Experiment")
    return experiment


def create_engine(
    config: RunConfig | Path | str,
    *,
    output: Path | str | None = None,
    registry: ExperimentRegistry | None = None,
) -> Engine[Any]:
    """Resolve sampling, construct the configured factory, then commit generation 0."""
    selected = _config(config)
    random = RunRandom(selected.seed)
    selected = selected.resolve(random.sampling)
    construction_random = random.state_dict()
    experiment = _build(selected, random, registry)
    return Engine.create(
        experiment,
        config=selected,
        output=output,
        random=random,
        _construction_random=construction_random,
    )


def resume[G](
    path: Path | str,
    *,
    experiment: Experiment[G] | None = None,
    registry: ExperimentRegistry | None = None,
) -> Engine[G]:
    """Restore the exact committed boundary without repeating its callbacks.

    Supply an explicit experiment for programmatic runs without a factory. The
    restored component set, code and configuration must match the checkpoint.
    Managed generators are restored last, after construction and payload decode.
    """
    store = RunStore(path)
    try:
        body = store.load(writable=True)
        config = RunConfig.from_dict(as_object(body["config"], "saved config"))
        if config.sampling:
            raise CheckpointCompatibilityError(
                "checkpoint config contains unresolved samples"
            )
        random = RunRandom(config.seed)
        saved_compatibility = as_object(body["compatibility"], "compatibility")
        construction_random = saved_compatibility.get("construction_random")
        if construction_random is not None:
            random.load_state_dict(construction_random)
        application = (
            experiment
            if experiment is not None
            else cast(Experiment[G], _build(config, random, registry))
        )
        engine = Engine(
            compose(application, config),
            config,
            random,
            store,
            None
            if construction_random is None
            else as_object(construction_random, "construction randomness"),
        )
        if engine._catalog is None or engine._compatibility is None:
            raise RunStateError("persistent engine is missing its component catalog")
        _check_compatibility(body["compatibility"], engine._compatibility)
        engine._catalog.restore(body["components"])
        state = decode_state(body["state"], engine._composition.codec, store)
        if strict_dumps(engine._catalog.export()) != strict_dumps(body["components"]):
            raise CheckpointCompatibilityError(
                "component restoration or payload decoding changed saved component state; codec decode must preserve future-affecting state"
            )
        if state.run_id != body["run_id"]:
            raise CheckpointCompatibilityError("checkpoint run identities disagree")
        config.validate_update(config.evolution, state.effective_settings)
        config.validate_update(config.evolution, state.generation_settings)
        validate_routes(
            state.effective_settings, application.mutations, application.crossovers
        )
        random.load_state_dict(body["random"])
        engine._state = state
        history = store.read_history()
        if history:
            engine.last_record = GenerationRecord(
                state.generation,
                state.next_event_sequence - 1,
                strict_dumps(history[-1]),
            )
        return engine
    except BaseException:
        store.close()
        raise


def run(
    config: RunConfig | Path | str,
    *,
    output: Path | str | None = None,
    until_generation: int | None = None,
    registry: ExperimentRegistry | None = None,
) -> RunState[Any]:
    """Create, execute and close one factory-configured run."""
    with create_engine(config, output=output, registry=registry) as engine:
        return engine.run(until_generation=until_generation)
