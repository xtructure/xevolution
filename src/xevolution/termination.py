"""Stopping predicates evaluated exactly once at a finalized generation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import ismethod
from math import isfinite
from typing import Any

from .config import validate_json
from .errors import ConfigurationError, RunStateError
from .models import JSONDict, JSONValue, RunState, StopDecision
from .protocols import Stateful, StopPredicate


def _limit(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


def _decision(value: StopDecision | bool) -> StopDecision:
    if isinstance(value, StopDecision):
        return value
    if isinstance(value, bool):
        return StopDecision(value)
    raise RunStateError("A stopping predicate must return bool or StopDecision")


@dataclass(frozen=True, slots=True)
class MaxGenerations:
    """Stop after generation ``limit``; initial evaluation is generation zero."""

    limit: int

    def __post_init__(self) -> None:
        _limit(self.limit, "Maximum generation")

    def __call__[G](self, state: RunState[G]) -> StopDecision:
        reached = state.generation >= self.limit
        return StopDecision(
            reached, f"Maximum generation {self.limit} reached" if reached else None
        )


@dataclass(frozen=True, slots=True)
class MaxEvaluations:
    """Check accepted-result count after a complete batch, not as a hard budget."""

    limit: int

    def __post_init__(self) -> None:
        _limit(self.limit, "Maximum evaluations")

    def __call__[G](self, state: RunState[G]) -> StopDecision:
        reached = state.evaluations >= self.limit
        return StopDecision(
            reached, f"Evaluation threshold {self.limit} reached" if reached else None
        )


@dataclass(frozen=True, slots=True)
class FitnessAtLeast:
    """Stop when the current population reaches the maximized score threshold."""

    threshold: float

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise ConfigurationError("Fitness threshold must be finite and numeric")
        try:
            valid = isfinite(self.threshold)
        except OverflowError as error:
            raise ConfigurationError(
                "Fitness threshold exceeds the finite range"
            ) from error
        if not valid:
            raise ConfigurationError("Fitness threshold must be finite")

    def __call__[G](self, state: RunState[G]) -> StopDecision:
        if not state.population:
            raise RunStateError(
                "Fitness stopping requires a nonempty evaluated population"
            )
        values: list[float] = []
        for individual in state.population:
            fitness = individual.fitness
            if isinstance(fitness, bool) or not isinstance(fitness, (int, float)):
                raise RunStateError("Fitness stopping requires finite evaluated scores")
            try:
                valid = isfinite(fitness)
            except OverflowError as error:
                raise RunStateError("Fitness exceeds the finite range") from error
            if not valid:
                raise RunStateError("Fitness stopping requires finite evaluated scores")
            values.append(fitness)
        reached = max(values) >= self.threshold
        return StopDecision(
            reached, f"Fitness threshold {self.threshold} reached" if reached else None
        )


@dataclass(frozen=True, slots=True)
class Stagnation:
    """Stop after ``generations`` transitions without a strictly improved best.

    Improvement bookkeeping lives in RunState, so this predicate has no hidden
    counter and resumes directly from the saved last-improvement generation.
    """

    generations: int

    def __post_init__(self) -> None:
        _limit(self.generations, "Stagnation window", minimum=1)

    def __call__[G](self, state: RunState[G]) -> StopDecision:
        reached = (
            state.generation - state.last_improvement_generation >= self.generations
        )
        return StopDecision(
            reached,
            f"No fitness improvement for {self.generations} generations"
            if reached
            else None,
        )


def _stateful_predicate[G](predicate: StopPredicate[G]) -> Stateful | None:
    """Find explicitly saved state on a callable or its bound-method owner."""
    if not callable(predicate):
        raise ConfigurationError("Stopping predicates must be callable")
    owner = predicate.__self__ if ismethod(predicate) else predicate
    has_export = callable(getattr(owner, "state_dict", None))
    has_restore = callable(getattr(owner, "load_state_dict", None))
    declares_state = hasattr(owner, "state_dict") or hasattr(owner, "load_state_dict")
    if declares_state and not (has_export and has_restore):
        raise ConfigurationError(
            "Stateful stopping requires both state methods to be callable"
        )
    return owner if isinstance(owner, Stateful) else None


def _save_children[G](predicates: Sequence[StopPredicate[G]]) -> JSONDict:
    states: list[JSONValue] = []
    for predicate in predicates:
        owner = _stateful_predicate(predicate)
        states.append(validate_json(owner.state_dict()) if owner is not None else None)
    return {"children": states}


def _load_children[G](predicates: Sequence[StopPredicate[G]], state: JSONValue) -> None:
    if not isinstance(state, dict) or set(state) != {"children"}:
        raise RunStateError("Composite stopping state must contain 'children'")
    children = state["children"]
    if not isinstance(children, list) or len(children) != len(predicates):
        raise RunStateError("Composite stopping state does not match its children")
    owners = [_stateful_predicate(predicate) for predicate in predicates]
    for owner, child in zip(owners, children, strict=True):
        if owner is None and child is not None:
            raise RunStateError(
                "Stateless stopping predicate has unexpected saved state"
            )
    for owner, child in zip(owners, children, strict=True):
        if owner is not None:
            owner.load_state_dict(child)


@dataclass(frozen=True, slots=True)
class AnyOf[G]:
    """Stop when any child stops; invoke every child once, without short circuit."""

    predicates: tuple[StopPredicate[G], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", tuple(self.predicates))
        if not self.predicates:
            raise ConfigurationError("AnyOf requires at least one stopping predicate")
        for predicate in self.predicates:
            _stateful_predicate(predicate)

    def __call__(self, state: RunState[G]) -> StopDecision:
        decisions = [_decision(predicate(state)) for predicate in self.predicates]
        for decision in decisions:
            if decision.stop:
                return StopDecision(
                    True, decision.reason or "A stopping condition was met"
                )
        return StopDecision()

    def state_dict(self) -> JSONDict:
        return _save_children(self.predicates)

    def load_state_dict(self, state: JSONValue) -> None:
        _load_children(self.predicates, state)


@dataclass(frozen=True, slots=True)
class AllOf[G]:
    """Stop when every child stops, invoking all stateful children every time."""

    predicates: tuple[StopPredicate[G], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", tuple(self.predicates))
        if not self.predicates:
            raise ConfigurationError("AllOf requires at least one stopping predicate")
        for predicate in self.predicates:
            _stateful_predicate(predicate)

    def __call__(self, state: RunState[G]) -> StopDecision:
        decisions = [_decision(predicate(state)) for predicate in self.predicates]
        if all(decision.stop for decision in decisions):
            reasons = [decision.reason for decision in decisions if decision.reason]
            return StopDecision(
                True, "; ".join(reasons) or "All stopping conditions met"
            )
        return StopDecision()

    def state_dict(self) -> JSONDict:
        return _save_children(self.predicates)

    def load_state_dict(self, state: JSONValue) -> None:
        _load_children(self.predicates, state)


@dataclass(frozen=True, slots=True)
class Not[G]:
    """Negate a predicate's decision while preserving its explicit child state."""

    predicate: StopPredicate[G]

    def __post_init__(self) -> None:
        _stateful_predicate(self.predicate)

    def __call__(self, state: RunState[G]) -> StopDecision:
        decision = _decision(self.predicate(state))
        return StopDecision(
            not decision.stop,
            "Negated stopping condition was met" if not decision.stop else None,
        )

    def state_dict(self) -> JSONDict:
        return _save_children((self.predicate,))

    def load_state_dict(self, state: JSONValue) -> None:
        _load_children((self.predicate,), state)


def from_config(config: Mapping[str, JSONValue]) -> StopPredicate[Any]:
    """Resolve one declarative condition or logical composition.

    Application-defined stopping is supplied through Experiment.stop. Empty or
    ambiguous objects are rejected instead of creating a silently endless run.
    """
    if len(config) != 1:
        raise ConfigurationError("A stopping condition requires exactly one key")
    name, value = next(iter(config.items()))
    if name == "max_generations":
        return MaxGenerations(_limit(value, "Maximum generation"))
    if name == "max_evaluations":
        return MaxEvaluations(_limit(value, "Maximum evaluations"))
    if name == "stagnation":
        return Stagnation(_limit(value, "Stagnation window", minimum=1))
    if name == "fitness_at_least":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("Fitness threshold must be finite and numeric")
        return FitnessAtLeast(value)
    if name in {"any", "all"}:
        if not isinstance(value, list) or not value:
            raise ConfigurationError(f"{name!r} requires a nonempty condition array")
        conditions: list[StopPredicate[Any]] = []
        for child in value:
            if not isinstance(child, dict):
                raise ConfigurationError("Each stopping condition must be an object")
            conditions.append(from_config(child))
        return AnyOf(tuple(conditions)) if name == "any" else AllOf(tuple(conditions))
    if name == "not":
        if not isinstance(value, dict):
            raise ConfigurationError("'not' requires one stopping condition object")
        return Not(from_config(value))
    raise ConfigurationError(f"Unknown stopping condition {name!r}")
