"""A stateful, opaque-payload application for independent-process acceptance tests."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from random import Random
from typing import cast

from xevolution import (
    EvaluationContext,
    EvaluationResult,
    EvolutionSettings,
    Experiment,
    Individual,
    OperatorSpec,
    ParameterRule,
    RangeParameter,
    RunConfig,
    RunRandom,
    RunState,
    StopDecision,
    VariationResult,
)
from xevolution.config import as_object
from xevolution.models import JSONDict, JSONValue
from xevolution.protocols import AttachmentReader, AttachmentWriter
from xevolution.termination import AnyOf, MaxGenerations


class OpaqueVector:
    """Nested mutable data deliberately lacking generic comparison/truth testing."""

    __hash__ = None

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __eq__(self, other: object) -> bool:
        raise AssertionError("the framework must not compare genotypes")

    def __bool__(self) -> bool:
        raise AssertionError("the framework must not test genotype truthiness")


def clone(value: OpaqueVector) -> OpaqueVector:
    return OpaqueVector(value.values.copy())


@dataclass(frozen=True)
class VectorCodec:
    codec_id: str = "tests.opaque_vector"
    version: int = 1

    def encode(
        self, value: OpaqueVector, writer: AttachmentWriter | None = None
    ) -> JSONValue:
        if writer is None:
            raise ValueError("this fixture requires its binary attachment writer")
        return {
            "payload": writer.put(
                struct.pack("<qq", *value.values), media_type="application/octet-stream"
            )
        }

    def decode(
        self, value: JSONValue, reader: AttachmentReader | None = None
    ) -> OpaqueVector:
        if reader is None:
            raise ValueError("this fixture requires its binary attachment reader")
        reference = as_object(value)["payload"]
        return OpaqueVector(list(struct.unpack("<qq", reader.get(reference))))


class StatefulApplication:
    def __init__(self, random: RunRandom, scale: RangeParameter) -> None:
        # These factory-time draws exercise both construction and future RNG
        # restoration. The source of the application RNG is itself run-owned.
        self.bias = random.sampling.random()
        self.rng = Random(random.evolution.getrandbits(128))
        self.rng.gauss(0, 1)
        self.scale = scale
        self.calls = {
            name: 0 for name in ("initialize", "evaluate", "mutate", "cross", "adapt")
        }
        self.operator: OperatorSpec[object] | None = None

    def checkpoint_identity(self) -> JSONValue:
        return {"factory_bias": self.bias}

    def initialize(self, rng: Random) -> OpaqueVector:
        self.calls["initialize"] += 1
        return OpaqueVector([rng.randrange(-5, 6) + int(rng.gauss(0, 1)), 1])

    def evaluate(
        self, individual: Individual[OpaqueVector], context: EvaluationContext
    ) -> EvaluationResult:
        self.calls["evaluate"] += 1
        if os.environ.get("XEVO_TEST_CRASH_GENERATION") == str(context.generation):
            # Test only: abrupt process death after state has already changed.
            os._exit(73)
        value = sum(individual.genotype.values)
        noise = self.rng.random() + context.rng.gauss(0, 0.001)
        return EvaluationResult(
            -abs(value - 11) + noise * 0.01,
            {"value": float(value), "noise": noise, "scale": self.scale.value},
        )

    def mutate(
        self, genotype: OpaqueVector, rng: Random
    ) -> VariationResult[OpaqueVector]:
        self.calls["mutate"] += 1
        values = genotype.values.copy()
        values[0] += int(rng.gauss(self.scale.value, 2)) + self.rng.randrange(-1, 2)
        return VariationResult(OpaqueVector(values), {"kind": "shift"})

    def cross(
        self, first: OpaqueVector, second: OpaqueVector, rng: Random
    ) -> VariationResult[OpaqueVector]:
        self.calls["cross"] += 1
        return VariationResult(
            OpaqueVector([first.values[0], second.values[1] + rng.randrange(-1, 2)]),
            {"kind": "mix"},
        )

    def adapt(self, state: RunState[OpaqueVector]) -> EvolutionSettings:
        self.calls["adapt"] += 1
        self.scale.set(self.scale.value + 0.25)
        if self.operator is not None:
            self.operator.weight = float(1 + state.generation % 3)
        size = 7 if state.generation == 1 else 6 if state.generation >= 2 else 5
        return state.effective_settings.updated(
            population_size=size,
            mutation_probability=0.7 if state.generation % 2 else 0.4,
        )

    def state_dict(self) -> JSONValue:
        version, internal, gaussian = self.rng.getstate()
        return {"calls": dict(self.calls), "rng": [version, list(internal), gaussian]}

    def load_state_dict(self, value: JSONValue) -> None:
        state = as_object(value)
        self.calls = cast(dict[str, int], dict(as_object(state["calls"])))
        version, internal, gaussian = cast(list[JSONValue], state["rng"])
        self.rng.setstate((version, tuple(cast(list[int], internal)), gaussian))


class WindowStop:
    def __init__(self, limit: int = 7) -> None:
        self.limit = limit
        self.calls = 0
        self.window: list[float] = []

    def checkpoint_identity(self) -> JSONValue:
        return {"limit": self.limit}

    def decide(self, state: RunState[OpaqueVector]) -> StopDecision:
        self.calls += 1
        if state.best is not None and state.best.fitness is not None:
            self.window = (self.window + [state.best.fitness])[-3:]
        stop = self.calls >= self.limit
        return StopDecision(
            stop, "saved stopping counter reached limit" if stop else None
        )

    def state_dict(self) -> JSONValue:
        return {"calls": self.calls, "window": list(self.window)}

    def load_state_dict(self, value: JSONValue) -> None:
        state = as_object(value)
        self.calls = cast(int, state["calls"])
        self.window = cast(list[float], list(cast(list[JSONValue], state["window"])))


def build_experiment(config: RunConfig, random: RunRandom) -> Experiment[OpaqueVector]:
    scale = RangeParameter(1, rule=ParameterRule(0, 10))
    application = StatefulApplication(random, scale)
    mutation = OperatorSpec("shift", application.mutate)
    application.operator = cast(OperatorSpec[object], mutation)
    stopper = WindowStop()
    return Experiment(
        initialize=application.initialize,
        evaluate=application.evaluate,
        mutations=(mutation,),
        crossovers=(OperatorSpec("mix", application.cross),),
        clone=clone,
        codec=VectorCodec(),
        adaptive_hooks=(application.adapt,),
        stop=AnyOf((stopper.decide, MaxGenerations(9))),
        components={"scale": scale},
        identity="tests.stateful_application",
        version="1",
    )


def configuration() -> RunConfig:
    return RunConfig(
        seed=1847,
        factory="resume_application:build_experiment",
        evolution=EvolutionSettings(population_size=5, mutation_probability=0.4),
        parent_selection={"name": "tournament", "size": 3},
        survival={"name": "elitism", "count": 1},
        sampling={
            "parameters.initial_marker": {"kind": "uniform", "low": 2, "high": 4}
        },
        parameter_rules={"population_size": {"minimum": 5, "maximum": 7}},
    )


def summarize_components(components: JSONDict) -> JSONDict:
    """Return only future-affecting counters, independent of canonical key order."""
    return {
        key: value
        for key, value in components.items()
        if isinstance(value, dict) and ("calls" in value or "value" in value)
    }
