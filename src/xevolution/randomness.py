"""Per-run random streams and stable logical evaluation identities."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import cast

from .config import as_object, strict_dumps, validate_json
from .errors import CheckpointCompatibilityError, ConfigurationError
from .models import JSONDict, JSONValue

SEED_DERIVATION_VERSION = 1


def _derive(seed: int, parts: list[JSONValue]) -> int:
    data: JSONDict = {"version": SEED_DERIVATION_VERSION, "seed": seed, "work": parts}
    return int.from_bytes(
        hashlib.sha256(strict_dumps(data).encode("utf-8")).digest(), "big"
    )


def _rng_json(value: object) -> JSONValue:
    if isinstance(value, tuple):
        return [_rng_json(child) for child in value]
    return validate_json(value)


def _rng_tuple(value: JSONValue) -> object:
    if isinstance(value, list):
        return tuple(_rng_tuple(child) for child in value)
    return value


def _validate_generator_state(value: JSONValue) -> tuple[object, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("generator state must have three fields")
    version, internal, gaussian = value
    if type(version) is not int or version != 3:
        raise ValueError("unsupported generator state version")
    if not isinstance(internal, list) or len(internal) != 625:
        raise ValueError("invalid MT19937 state length")
    if any(
        type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in internal[:-1]
    ):
        raise ValueError("invalid MT19937 state word")
    index = internal[-1]
    if type(index) is not int or not 0 <= index <= 624:
        raise ValueError("invalid MT19937 state index")
    if gaussian is not None and (
        isinstance(gaussian, bool)
        or not isinstance(gaussian, (int, float))
        or not math.isfinite(gaussian)
    ):
        raise ValueError("cached Gaussian must be null or a finite number")
    return cast(tuple[object, ...], _rng_tuple(value))


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Randomness for one stable batch or individual evaluation job.

    Access ``rng`` repeatedly to advance the same local stream. Creating another
    context for the same work identity deliberately recreates that stream.
    """

    generation: int
    run_id: str
    root_seed: int
    individual_id: int | None = None
    round: int = 0
    replicate: int = 0
    rng: random.Random = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("generation", "round", "replicate"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ConfigurationError(f"evaluation {name} must be nonnegative")
        if self.individual_id is not None and (
            type(self.individual_id) is not int or self.individual_id < 0
        ):
            raise ConfigurationError("evaluation individual ID must be nonnegative")
        if type(self.root_seed) is not int:
            raise ConfigurationError("evaluation root seed must be an integer")
        identity: list[JSONValue] = [
            "evaluation",
            self.generation,
            self.individual_id,
            self.round,
            self.replicate,
        ]
        object.__setattr__(
            self, "rng", random.Random(_derive(self.root_seed, identity))
        )

    def for_individual(
        self, id: int, *, replicate: int | None = None
    ) -> EvaluationContext:
        return EvaluationContext(
            self.generation,
            self.run_id,
            self.root_seed,
            id,
            self.round,
            self.replicate if replicate is None else replicate,
        )


class RunRandom:
    """Managed evolution and initial-sampling generators plus evaluation seed."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise ConfigurationError("run seed must be an integer")
        self.seed = seed
        self.evolution = random.Random(_derive(seed, ["evolution"]))
        self.sampling = random.Random(_derive(seed, ["sampling"]))
        self.evaluation_seed = _derive(seed, ["evaluation-root"])

    def context(
        self,
        generation: int,
        run_id: str,
        *,
        individual_id: int | None = None,
        round: int = 0,
        replicate: int = 0,
    ) -> EvaluationContext:
        return EvaluationContext(
            generation, run_id, self.evaluation_seed, individual_id, round, replicate
        )

    def state_dict(self) -> JSONDict:
        return {
            "generator": "python.random.Random",
            "derivation_version": SEED_DERIVATION_VERSION,
            "seed": self.seed,
            "evaluation_seed": self.evaluation_seed,
            "evolution": _rng_json(self.evolution.getstate()),
            "sampling": _rng_json(self.sampling.getstate()),
        }

    def load_state_dict(self, data: JSONValue) -> None:
        state = as_object(data, "random state")
        expected = {
            "generator",
            "derivation_version",
            "seed",
            "evaluation_seed",
            "evolution",
            "sampling",
        }
        if (
            set(state) != expected
            or state["generator"] != "python.random.Random"
            or type(state["derivation_version"]) is not int
            or state["derivation_version"] != SEED_DERIVATION_VERSION
        ):
            raise CheckpointCompatibilityError("unsupported random-state format")
        if type(state["seed"]) is not int or state["seed"] != self.seed:
            raise CheckpointCompatibilityError(
                "checkpoint seed differs from the configured seed"
            )
        if (
            type(state["evaluation_seed"]) is not int
            or state["evaluation_seed"] != self.evaluation_seed
        ):
            raise CheckpointCompatibilityError(
                "evaluation seed metadata is inconsistent"
            )
        # Validate using temporary generators before modifying managed objects.
        try:
            evolution, sampling = random.Random(0), random.Random(0)
            evolution.setstate(_validate_generator_state(state["evolution"]))
            sampling.setstate(_validate_generator_state(state["sampling"]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CheckpointCompatibilityError(
                f"invalid saved RNG state: {exc}"
            ) from exc
        self.evolution.setstate(evolution.getstate())
        self.sampling.setstate(sampling.getstate())
