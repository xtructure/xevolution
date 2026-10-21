"""Strict JSON configuration, effective settings, and bounded parameters."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from random import Random
from types import MappingProxyType

from .errors import ConfigurationError
from .models import JSONDict, JSONValue


def validate_json(value: object) -> JSONValue:
    """Validate JSON-native values and return an independent copy.

    Tuples, arbitrary mappings, custom objects, non-string keys, cycles, and
    non-finite floats are rejected instead of being silently coerced.
    """
    active: set[int] = set()

    def visit(item: object) -> JSONValue:
        if item is None or type(item) in (bool, int, str):
            return item  # type: ignore[return-value]  # Exact primitive types checked.
        if type(item) is float:
            if not math.isfinite(item):
                raise ConfigurationError("JSON numbers must be finite")
            return item
        if isinstance(item, (list, dict)):
            identity = id(item)
            if identity in active:
                raise ConfigurationError("JSON data must not contain cycles")
            active.add(identity)
            try:
                if isinstance(item, list):
                    return [visit(child) for child in item]
                result: JSONDict = {}
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ConfigurationError("JSON object keys must be strings")
                    result[key] = visit(child)
                return result
            finally:
                active.remove(identity)
        raise ConfigurationError(f"not a JSON-native value: {type(item).__name__}")

    try:
        return visit(value)
    except RecursionError as exc:
        raise ConfigurationError("JSON data is nested too deeply") from exc


def as_object(value: JSONValue, label: str = "value") -> JSONDict:
    """Require a JSON object at a documented configuration boundary."""
    if not isinstance(value, dict):
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def strict_loads(text: str) -> JSONValue:
    """Read finite JSON with unique object keys, including overflow checks."""

    def pairs(items: list[tuple[str, JSONValue]]) -> JSONDict:
        result: JSONDict = {}
        for key, value in items:
            if key in result:
                raise ConfigurationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> JSONValue:
        raise ConfigurationError(f"nonstandard JSON numeric constant: {value}")

    try:
        value: object = json.loads(
            text, object_pairs_hook=pairs, parse_constant=constant
        )
    except (ValueError, RecursionError) as exc:
        raise ConfigurationError(f"invalid JSON: {exc}") from exc
    return validate_json(value)


def strict_dumps(value: JSONValue) -> str:
    """Encode canonical compact JSON without lossy or nonstandard values."""
    return json.dumps(
        validate_json(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _only(data: Mapping[str, JSONValue], allowed: set[str], label: str) -> None:
    if any(not isinstance(key, str) for key in data):
        raise ConfigurationError(f"{label} keys must be strings")
    unknown = set(data) - allowed
    if unknown:
        raise ConfigurationError(
            f"unknown {label} fields: {', '.join(sorted(unknown))}"
        )


def _int(value: JSONValue, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: JSONValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigurationError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{label} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class EvolutionSettings:
    """Population-construction settings, validated on every replacement."""

    population_size: int = 100
    mutation_probability: float = 0.5
    max_attempts_per_child: int = 100

    def __post_init__(self) -> None:
        _int(self.population_size, "population_size", minimum=1)
        _int(self.max_attempts_per_child, "max_attempts_per_child", minimum=1)
        value = _number(self.mutation_probability, "mutation_probability")
        if not 0.0 <= value <= 1.0:
            raise ConfigurationError("mutation_probability must be between 0 and 1")
        object.__setattr__(self, "mutation_probability", value)

    def to_dict(self) -> JSONDict:
        return {
            "population_size": self.population_size,
            "mutation_probability": self.mutation_probability,
            "max_attempts_per_child": self.max_attempts_per_child,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> EvolutionSettings:
        _only(
            data,
            {"population_size", "mutation_probability", "max_attempts_per_child"},
            "evolution settings",
        )
        return cls(
            population_size=_int(
                data.get("population_size", 100), "population_size", minimum=1
            ),
            mutation_probability=_number(
                data.get("mutation_probability", 0.5), "mutation_probability"
            ),
            max_attempts_per_child=_int(
                data.get("max_attempts_per_child", 100),
                "max_attempts_per_child",
                minimum=1,
            ),
        )

    def updated(self, **changes: JSONValue) -> EvolutionSettings:
        """Return new validated settings; unknown fields are errors."""
        return self.from_dict(self.to_dict() | changes)


@dataclass(frozen=True, slots=True)
class ParameterRule:
    """Lifetime bounds and mutability for a numeric parameter."""

    minimum: float | None = None
    maximum: float | None = None
    mutable: bool = True

    def __post_init__(self) -> None:
        if type(self.mutable) is not bool:
            raise ConfigurationError("parameter mutable must be a boolean")
        if self.minimum is not None:
            _number(self.minimum, "parameter minimum")
        if self.maximum is not None:
            _number(self.maximum, "parameter maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ConfigurationError("parameter minimum exceeds maximum")

    def validate(self, value: float, *, previous: float | None = None) -> None:
        number = _number(value, "parameter value")
        if self.minimum is not None and number < self.minimum:
            raise ConfigurationError("parameter value is below its lifetime minimum")
        if self.maximum is not None and number > self.maximum:
            raise ConfigurationError("parameter value exceeds its lifetime maximum")
        if previous is not None and not self.mutable and value != previous:
            raise ConfigurationError("parameter is immutable")

    def to_dict(self) -> JSONDict:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mutable": self.mutable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> ParameterRule:
        _only(data, {"minimum", "maximum", "mutable"}, "parameter rule")
        mutable = data.get("mutable", True)
        if type(mutable) is not bool:
            raise ConfigurationError("parameter mutable must be a boolean")
        low, high = data.get("minimum"), data.get("maximum")
        return cls(
            None if low is None else _number(low, "minimum"),
            None if high is None else _number(high, "maximum"),
            mutable,
        )


class RangeParameter:
    """An application-owned numeric value with explicit bounds and saved state.

    Register instances in ``Experiment.components`` when their mutable value
    affects future behavior. Sampling is explicit and uses a supplied generator.
    """

    def __init__(self, value: float, *, rule: ParameterRule | None = None) -> None:
        self.rule = rule if rule is not None else ParameterRule()
        self.rule.validate(value)
        self._value = float(value)

    @property
    def value(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self.rule.validate(value, previous=self._value)
        self._value = float(value)

    @classmethod
    def sampled(
        cls, rng: Random, low: float, high: float, *, rule: ParameterRule | None = None
    ) -> RangeParameter:
        return cls(
            _sample({"kind": "uniform", "low": low, "high": high}, rng), rule=rule
        )

    def state_dict(self) -> JSONValue:
        return {"value": self._value, "rule": self.rule.to_dict()}

    def load_state_dict(self, state: JSONValue) -> None:
        data = as_object(state, "range parameter state")
        _only(data, {"value", "rule"}, "range parameter state")
        saved_rule = ParameterRule.from_dict(as_object(data.get("rule"), "rule"))
        if saved_rule != self.rule:
            raise ConfigurationError("range parameter rule differs from saved rule")
        value = _number(data.get("value"), "range parameter value")
        self.rule.validate(value)
        self._value = value


def _sample(data: Mapping[str, JSONValue], rng: Random) -> float | int:
    _only(data, {"kind", "low", "high"}, "sampling specification")
    kind = data.get("kind", "uniform")
    if kind == "integer":
        low = data.get("low")
        high = data.get("high")
        if type(low) is not int or type(high) is not int or low > high:
            raise ConfigurationError("integer sampling needs ordered integer bounds")
        return rng.randint(low, high)
    low_f = _number(data.get("low"), "sample low")
    high_f = _number(data.get("high"), "sample high")
    if low_f > high_f:
        raise ConfigurationError("sample low exceeds high")
    if kind == "uniform":
        if low_f == high_f:
            return low_f
        fraction = rng.random()
        return _number((1 - fraction) * low_f + fraction * high_f, "sample")
    if kind == "log_uniform":
        if low_f <= 0:
            raise ConfigurationError("log_uniform bounds must be positive")
        if low_f == high_f:
            return low_f
        # Rounding in log/exp can stray just outside the original interval.
        return min(
            high_f, max(low_f, math.exp(rng.uniform(math.log(low_f), math.log(high_f))))
        )
    raise ConfigurationError(f"unknown sampling kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Run construction settings; factories receive a resolved instance.

    ``sampling`` maps ``parameters.NAME`` or ``evolution.NAME`` to a sampling
    specification. It is cleared after one resolution; the effective values
    and advanced sampling RNG are then checkpointed.
    """

    seed: int = 0
    factory: str | None = None
    parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)
    parent_selection: Mapping[str, JSONValue] = field(
        default_factory=lambda: {"name": "roulette"}
    )
    survival: Mapping[str, JSONValue] = field(
        default_factory=lambda: {"name": "replace_all"}
    )
    termination: Mapping[str, JSONValue] = field(default_factory=dict)
    schema_version: int = 1
    sampling: Mapping[str, JSONValue] = field(default_factory=dict)
    parameter_rules: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ConfigurationError("seed must be an integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ConfigurationError("unsupported configuration schema_version")
        if self.factory is not None and (
            not isinstance(self.factory, str) or not self.factory.strip()
        ):
            raise ConfigurationError("factory must be a nonempty reference or None")
        if not isinstance(self.evolution, EvolutionSettings):
            raise ConfigurationError("evolution must be EvolutionSettings")
        for name in (
            "parameters",
            "parent_selection",
            "survival",
            "termination",
            "sampling",
            "parameter_rules",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ConfigurationError(f"{name} must be a mapping")
            clean = as_object(validate_json(dict(value)), name)
            object.__setattr__(self, name, MappingProxyType(clean))
        for name, value in self.parameter_rules.items():
            if name not in self.evolution.to_dict():
                raise ConfigurationError(f"unknown evolution parameter rule: {name}")
            rule = ParameterRule.from_dict(as_object(value, f"rule {name}"))
            if f"evolution.{name}" not in self.sampling:
                rule.validate(_number(self.evolution.to_dict()[name], name))

    def to_dict(self) -> JSONDict:
        evolution = self.evolution.to_dict()
        evolution["parent_selection"] = dict(self.parent_selection)
        evolution["survival"] = dict(self.survival)
        data: JSONDict = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "experiment": {
                "factory": self.factory,
                "parameters": dict(self.parameters),
            },
            "evolution": evolution,
            "termination": dict(self.termination),
        }
        if self.sampling:
            data["sampling"] = dict(self.sampling)
        if self.parameter_rules:
            data["parameter_rules"] = dict(self.parameter_rules)
        return as_object(validate_json(data))

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> RunConfig:
        clean = as_object(validate_json(dict(data)), "configuration")
        _only(
            clean,
            {
                "schema_version",
                "seed",
                "experiment",
                "evolution",
                "termination",
                "sampling",
                "parameter_rules",
            },
            "configuration",
        )
        experiment = as_object(clean.get("experiment", {}), "experiment")
        _only(experiment, {"factory", "parameters"}, "experiment")
        evolution = dict(as_object(clean.get("evolution", {}), "evolution"))
        parent = as_object(
            evolution.pop("parent_selection", {"name": "roulette"}), "parent_selection"
        )
        survival = as_object(
            evolution.pop("survival", {"name": "replace_all"}), "survival"
        )
        factory = experiment.get("factory")
        if factory is not None and not isinstance(factory, str):
            raise ConfigurationError("experiment.factory must be a string")
        seed = clean.get("seed", 0)
        if type(seed) is not int:
            raise ConfigurationError("seed must be an integer")
        return cls(
            seed=seed,
            factory=factory,
            parameters=as_object(experiment.get("parameters", {}), "parameters"),
            evolution=EvolutionSettings.from_dict(evolution),
            parent_selection=parent,
            survival=survival,
            termination=as_object(clean.get("termination", {}), "termination"),
            schema_version=_int(clean.get("schema_version", 1), "schema_version"),
            sampling=as_object(clean.get("sampling", {}), "sampling"),
            parameter_rules=as_object(
                clean.get("parameter_rules", {}), "parameter_rules"
            ),
        )

    def resolve(self, rng: Random) -> RunConfig:
        """Sample initial parameters once in stable path order."""
        if not self.sampling:
            return self
        parameters = as_object(validate_json(dict(self.parameters)))
        evolution = self.evolution.to_dict()
        for path in sorted(self.sampling):
            prefix, separator, name = path.partition(".")
            if not separator or not name or "." in name:
                raise ConfigurationError(
                    "sample paths must be parameters.NAME or evolution.NAME"
                )
            value = _sample(as_object(self.sampling[path], f"sample {path}"), rng)
            if prefix == "parameters":
                parameters[name] = value
            elif prefix == "evolution" and name in evolution:
                evolution[name] = value
            else:
                raise ConfigurationError(f"unsupported sample path: {path}")
        return replace(
            self,
            parameters=parameters,
            evolution=EvolutionSettings.from_dict(evolution),
            sampling={},
        )

    def validate_update(self, old: EvolutionSettings, new: EvolutionSettings) -> None:
        """Enforce configured lifetime rules for adaptive setting updates."""
        before, after = old.to_dict(), new.to_dict()
        for name, value in self.parameter_rules.items():
            rule = ParameterRule.from_dict(as_object(value, f"rule {name}"))
            rule.validate(
                _number(after[name], name), previous=_number(before[name], name)
            )


def load_config(path: Path | str) -> RunConfig:
    """Load a JSON configuration without importing or running its factory."""
    try:
        return RunConfig.from_dict(
            as_object(
                strict_loads(Path(path).read_text(encoding="utf-8")), "configuration"
            )
        )
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
