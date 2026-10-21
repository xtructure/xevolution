"""Application composition and explicit component checkpoint contracts."""

from __future__ import annotations

import functools
import hashlib
import inspect
import marshal
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import CodeType, MappingProxyType, ModuleType
from typing import cast

from .config import RunConfig, as_object, strict_dumps, validate_json
from .errors import CheckpointCompatibilityError, ConfigurationError
from .models import JSONDict, JSONValue
from .operators import OperatorSpec, validate_routes
from .protocols import (
    AdaptiveHook,
    Clone,
    Crossover,
    IndividualEvaluator,
    Initializer,
    Mutation,
    ParentSelector,
    PayloadCodec,
    PopulationEvaluator,
    Stateful,
    StopPredicate,
    SurvivorSelector,
)
from .randomness import RunRandom
from .selection import parent_selector, survivor_selector
from .serialization import JsonCodec
from .termination import AllOf, AnyOf, Not, from_config


@dataclass(frozen=True, slots=True)
class Experiment[G]:
    """Compose independent application callables, without a base-class hierarchy.

    A population evaluator takes precedence over an individual evaluator. All
    variation operations must leave their input genotypes unchanged. Supply a
    clone for arbitrary payloads or attachment-only codecs; otherwise a codec
    round trip provides independent historical snapshots.

    Stateful collaborators expose ``state_dict`` and ``load_state_dict``. Direct
    collaborators are registered automatically; ``components`` names any shared
    state used indirectly. ``compatibility`` records application data/library
    fingerprints that must match at resume. Factories validate their parameters.
    """

    initialize: Initializer[G]
    evaluate: IndividualEvaluator[G] | None = None
    evaluate_population: PopulationEvaluator[G] | None = None
    mutations: tuple[OperatorSpec[Mutation[G]], ...] = ()
    crossovers: tuple[OperatorSpec[Crossover[G]], ...] = ()
    clone: Clone[G] | None = None
    codec: PayloadCodec[G] | None = None
    select_parent: ParentSelector[G] | None = None
    select_survivors: SurvivorSelector[G] | None = None
    stop: StopPredicate[G] | None = None
    adaptive_hooks: tuple[AdaptiveHook[G], ...] = ()
    components: Mapping[str, object] = field(default_factory=dict)
    identity: str = "application"
    version: str = "1"
    compatibility: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("identity", "version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ConfigurationError(f"experiment {name} must be nonempty text")
        if not callable(self.initialize):
            raise ConfigurationError("experiment initializer must be callable")
        if self.evaluate is None and self.evaluate_population is None:
            raise ConfigurationError(
                "an individual or population evaluator is required"
            )
        for name in (
            "evaluate",
            "evaluate_population",
            "clone",
            "select_parent",
            "select_survivors",
            "stop",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise ConfigurationError(f"experiment {name} must be callable")
        for name in ("mutations", "crossovers", "adaptive_hooks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(not callable(hook) for hook in self.adaptive_hooks):
            raise ConfigurationError("adaptive hooks must be callable")
        if any(not isinstance(name, str) or not name for name in self.components):
            raise ConfigurationError("component keys must be nonempty strings")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(
            self,
            "compatibility",
            MappingProxyType(as_object(validate_json(dict(self.compatibility)))),
        )


@dataclass(frozen=True, slots=True)
class _CodecClone[G]:
    codec: PayloadCodec[G]

    def __call__(self, genotype: G) -> G:
        return self.codec.decode(self.codec.encode(genotype))


@dataclass(frozen=True, slots=True)
class Composition[G]:
    """Resolved runtime collaborators; policies are selected exactly once."""

    experiment: Experiment[G]
    clone: Clone[G]
    codec: PayloadCodec[G]
    select_parent: ParentSelector[G]
    select_survivors: SurvivorSelector[G]
    stop: StopPredicate[G]


def compose[G](experiment: Experiment[G], config: RunConfig) -> Composition[G]:
    """Validate configured policies and resolve defaults before initialization."""
    codec = experiment.codec
    if codec is None:
        # JSON is a dynamic boundary. Arbitrary G needs an application codec
        # when persisted, or an explicit clone for purely in-memory execution.
        codec = cast(PayloadCodec[G], JsonCodec())
    if (
        not isinstance(codec.codec_id, str)
        or not codec.codec_id
        or type(codec.version) is not int
        or codec.version < 1
    ):
        raise ConfigurationError("codec needs a nonempty codec_id and positive version")
    if not callable(codec.encode) or not callable(codec.decode):
        raise ConfigurationError("codec must implement encode and decode")
    validate_routes(config.evolution, experiment.mutations, experiment.crossovers)
    return Composition(
        experiment=experiment,
        clone=experiment.clone if experiment.clone is not None else _CodecClone(codec),
        codec=codec,
        select_parent=experiment.select_parent
        if experiment.select_parent is not None
        else parent_selector(config.parent_selection),
        select_survivors=experiment.select_survivors
        if experiment.select_survivors is not None
        else survivor_selector(config.survival),
        stop=experiment.stop
        if experiment.stop is not None
        else from_config(config.termination),
    )


def _owner(value: object) -> object:
    owner = getattr(value, "__self__", None)
    return owner if owner is not None and not isinstance(owner, ModuleType) else value


def _segment(value: str) -> str:
    """Escape user keys independently of framework composition path segments."""
    return value.replace("~", "~0").replace("/", "~1")


def _source_identity(value: object) -> JSONDict:
    target = (
        value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
    )
    module_name = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", type(target).__qualname__)
    module = sys.modules.get(module_name)
    filename = getattr(module, "__file__", None)
    digest: str | None = None
    if filename is not None:
        path = Path(filename)
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result: JSONDict = {"module": module_name, "name": name, "source_sha256": digest}
    if inspect.isfunction(target):
        # Distinguish functions/lambdas sharing a qualified name. The code is
        # hashed, never restored or executed from a checkpoint. Strip filenames
        # recursively so an identical installed package can move directories.
        def location_free(code: CodeType) -> CodeType:
            return code.replace(
                co_filename="",
                co_consts=tuple(
                    location_free(item) if isinstance(item, CodeType) else item
                    for item in code.co_consts
                ),
            )

        result["code_sha256"] = hashlib.sha256(
            marshal.dumps(location_free(target.__code__))
        ).hexdigest()
    return result


class ComponentCatalog:
    """Stable component identities and explicit state, never object pickling.

    Bound methods sharing an owner are restored once. Frozen dataclasses may
    hold immutable configuration. Other instances carrying data must implement
    the state protocol. Obvious mutable function captures are rejected; Python
    cannot prove arbitrary external code is pure, so the application contract
    still covers globals hidden behind imported modules and external services.
    """

    def __init__[G](self, composition: Composition[G], random: RunRandom) -> None:
        experiment = composition.experiment
        slots: dict[str, object] = {
            "initialize": experiment.initialize,
            "clone": composition.clone,
            "codec": composition.codec,
            "codec/encode": composition.codec.encode,
            "codec/decode": composition.codec.decode,
            "select_parent": composition.select_parent,
            "select_survivors": composition.select_survivors,
            "stop": composition.stop,
        }
        if experiment.evaluate is not None:
            slots["evaluate"] = experiment.evaluate
        if experiment.evaluate_population is not None:
            slots["evaluate_population"] = experiment.evaluate_population
        for pool_name, pool in (
            ("mutation", experiment.mutations),
            ("crossover", experiment.crossovers),
        ):
            for operator in pool:
                key = _segment(operator.name)
                slots[f"{pool_name}/{key}"] = operator
                slots[f"{pool_name}/{key}/operation"] = operator.operation
        for index, hook in enumerate(experiment.adaptive_hooks):
            slots[f"adaptive/{index}"] = hook
        for name, component in experiment.components.items():
            slots[f"application/{_segment(name)}"] = component

        def children(name: str, value: object, active: set[int]) -> None:
            owner = _owner(value)
            if not isinstance(owner, (AnyOf, AllOf, Not)):
                return
            if id(owner) in active:
                raise ConfigurationError("stopping composition contains a cycle")
            nested = (owner.predicate,) if isinstance(owner, Not) else owner.predicates
            for index, child in enumerate(nested):
                child_name = f"{name}/predicate/{index}"
                slots[child_name] = child
                children(child_name, child, active | {id(owner)})

        for name, component in tuple(slots.items()):
            children(name, component, set())
        self._objects: dict[str, object] = {}
        self._names: dict[int, str] = {}
        for name in sorted(slots):
            owner = _owner(slots[name])
            if id(owner) not in self._names:
                self._names[id(owner)] = name
                self._objects[name] = owner
        self._managed = {
            id(random): "run_random",
            id(random.evolution): "evolution_rng",
            id(random.sampling): "sampling_rng",
        }
        self._stateful: dict[str, Stateful] = {}
        for name, component in self._objects.items():
            has_export = callable(getattr(component, "state_dict", None))
            has_restore = callable(getattr(component, "load_state_dict", None))
            declares_state = hasattr(component, "state_dict") or hasattr(
                component, "load_state_dict"
            )
            if declares_state and not (has_export and has_restore):
                raise ConfigurationError(
                    f"component {name} must implement both state methods"
                )
            # Built-in composites own only their child topology. Cataloging the
            # leaf owners avoids restoring shared children more than once.
            if (
                isinstance(component, Stateful)
                and not isinstance(component, (AnyOf, AllOf, Not))
                and id(component) not in self._managed
            ):
                self._stateful[name] = component
        self.identity: JSONDict = {
            "slots": {
                name: self._reference(value) for name, value in sorted(slots.items())
            },
            "objects": {
                name: self._describe(value, set(), root=True)
                for name, value in self._objects.items()
            },
            "callback_code": {},
        }
        callbacks = as_object(self.identity["callback_code"])
        for name, component in sorted(slots.items()):
            function = component
            if not inspect.isfunction(function) and not inspect.ismethod(function):
                if not callable(function):
                    continue
                function = function.__call__
            if inspect.ismethod(function):
                function = function.__func__
            if inspect.isfunction(function):
                callbacks[name] = self._describe(function, set(), root=True)

    def _reference(self, value: object) -> JSONValue:
        result: JSONDict = {"component": self._names[id(_owner(value))]}
        if _owner(value) is not value:
            result["method"] = getattr(value, "__name__", "")
        return result

    def _describe(
        self,
        value: object,
        active: set[int],
        *,
        root: bool = False,
        owned_state: bool = False,
    ) -> JSONValue:
        if value is None or type(value) in (str, int, float, bool):
            return validate_json(value)
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, range):
            return {"range": [value.start, value.stop, value.step]}
        if isinstance(value, frozenset):
            return {
                "frozenset": sorted(
                    (self._describe(child, active) for child in value), key=strict_dumps
                )
            }
        owner = _owner(value)
        if id(owner) in self._managed:
            return {
                "managed": self._managed[id(owner)],
                "method": getattr(value, "__name__", None)
                if owner is not value
                else None,
            }
        if not root and id(_owner(value)) in self._names:
            return self._reference(value)
        if isinstance(value, tuple):
            return [
                self._describe(child, active, owned_state=owned_state)
                for child in value
            ]
        if isinstance(value, ModuleType):
            if value.__name__ == "random":
                raise ConfigurationError(
                    "global random module captured by component; use injected RNGs"
                )
            return {"module": value.__name__}
        if inspect.isclass(value):
            return _source_identity(value)
        if id(value) in active:
            return {"recursive": _source_identity(value)}
        active = active | {id(value)}
        if owner is not value:
            return {
                "method": getattr(value, "__name__", ""),
                "owner": self._describe(owner, active, owned_state=owned_state),
            }
        if inspect.isfunction(value):
            closure = inspect.getclosurevars(value)
            captures = dict(closure.globals) | dict(closure.nonlocals)
            return {
                "callable": _source_identity(value),
                "captures": {
                    name: self._describe(child, active)
                    for name, child in sorted(captures.items())
                },
                "defaults": self._describe(value.__defaults__, active),
                "keyword_defaults": {
                    name: self._describe(child, active)
                    for name, child in sorted((value.__kwdefaults__ or {}).items())
                },
            }
        if isinstance(value, functools.partial):
            return {
                "partial": self._describe(value.func, active),
                "args": self._describe(value.args, active),
                "keywords": {
                    name: self._describe(child, active)
                    for name, child in sorted((value.keywords or {}).items())
                },
            }
        description: JSONDict = {"type": _source_identity(value)}
        if isinstance(value, Stateful):
            if not root and not owned_state:
                raise ConfigurationError(
                    "a captured stateful collaborator must be registered in Experiment.components"
                )
            description["stateful"] = True
            custom = getattr(value, "checkpoint_identity", None)
            if callable(custom):
                description["configuration"] = validate_json(custom())
            if isinstance(value, OperatorSpec):
                description["name"] = value.name
                description["fallback"] = value.fallback
            # Composite stopping policies are frozen dataclasses, and their
            # immutable child composition must match as well as their state.
            if isinstance(value, (AnyOf, AllOf, Not)):
                description["fields"] = {
                    item.name: self._describe(
                        getattr(value, item.name), active, owned_state=True
                    )
                    for item in fields(value)
                }
            return description
        if (
            is_dataclass(value)
            and not isinstance(value, type)
            and getattr(
                getattr(type(value), "__dataclass_params__", None), "frozen", False
            )
        ):
            description["configuration"] = {
                item.name: self._describe(getattr(value, item.name), active)
                for item in fields(value)
            }
            return description
        if (
            isinstance(value, (list, dict, set))
            or getattr(value, "__dict__", {})
            or any(getattr(cls, "__slots__", ()) for cls in type(value).__mro__)
        ):
            raise ConfigurationError(
                f"{type(value).__qualname__} has unregistered mutable state; use a frozen configuration record or implement state_dict/load_state_dict and register shared collaborators"
            )
        if inspect.isbuiltin(value):
            description["builtin"] = value.__name__
            return description
        # Empty ordinary Python instances, such as JsonCodec, have no instance
        # state. Unknown C-extension values can hide mutable storage even when
        # they do not expose __dict__ or __slots__; never guess they are empty.
        if hasattr(value, "__dict__") and type(value).__module__ != "builtins":
            return description
        raise ConfigurationError(
            f"cannot describe captured {type(value).__qualname__}; expose explicit component state or immutable configuration"
        )

    def export(self) -> JSONDict:
        """Export independent JSON state once per shared component owner."""
        return {
            name: validate_json(component.state_dict())
            for name, component in self._stateful.items()
        }

    def restore(self, data: JSONValue) -> None:
        """Restore exactly the registered component set; never add defaults."""
        saved = as_object(data, "component states")
        if saved.keys() != self._stateful.keys():
            raise CheckpointCompatibilityError(
                "checkpoint component-state keys differ from the composition"
            )
        for name, component in self._stateful.items():
            component.load_state_dict(validate_json(saved[name]))
