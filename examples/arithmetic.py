"""Evolve bit strings encoding left-to-right integer arithmetic expressions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from xevolution.config import RunConfig
from xevolution.errors import ConfigurationError
from xevolution.experiment import Experiment
from xevolution.models import EvaluationResult, Individual, JSONValue, VariationResult
from xevolution.operators import OperatorSpec
from xevolution.protocols import AttachmentReader, AttachmentWriter
from xevolution.randomness import EvaluationContext, RunRandom

_OPERATORS = ("+", "-", "*", "/")


def _validate_bits(value: object, terms: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 4 + 6 * (terms - 1)
        or set(value) - {"0", "1"}
    ):
        raise ConfigurationError(
            "arithmetic genotype must be a correctly sized bit string"
        )
    return value


@dataclass(frozen=True)
class BitStringCodec:
    """Losslessly encode the application's immutable string representation."""

    terms: int
    codec_id: str = "examples.arithmetic.bits"
    version: int = 1

    def encode(self, value: str, writer: AttachmentWriter | None = None) -> JSONValue:
        return _validate_bits(value, self.terms)

    def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> str:
        return _validate_bits(value, self.terms)


def clone(genotype: str) -> str:
    """Immutable strings can safely share their storage."""
    return genotype


def decode(genotype: str) -> int | None:
    """Evaluate left to right; division truncates toward zero, or returns None."""
    value = int(genotype[:4], 2)
    for offset in range(4, len(genotype), 6):
        operation = int(genotype[offset : offset + 2], 2)
        operand = int(genotype[offset + 2 : offset + 6], 2)
        if operation == 0:
            value += operand
        elif operation == 1:
            value -= operand
        elif operation == 2:
            value *= operand
        elif operand == 0:
            return None
        else:
            # Avoid float conversion and preserve truncation toward zero for
            # negative intermediate results, as in the original Java example.
            value = (abs(value) // operand) * (-1 if value < 0 else 1)
    return value


def expression(genotype: str) -> str:
    """Return a readable expression; its documented evaluation order is left-to-right."""
    tokens = [str(int(genotype[:4], 2))]
    for offset in range(4, len(genotype), 6):
        tokens.extend(
            (
                _OPERATORS[int(genotype[offset : offset + 2], 2)],
                str(int(genotype[offset + 2 : offset + 6], 2)),
            )
        )
    return " ".join(tokens)


@dataclass(frozen=True)
class Arithmetic:
    """Stateless application behavior parameterized by immutable settings."""

    target: int
    terms: int
    mutation_rate: float

    def initialize(self, rng: Random) -> str:
        return "".join(str(rng.randrange(2)) for _ in range(4 + 6 * (self.terms - 1)))

    def evaluate(
        self, individual: Individual[str], context: EvaluationContext
    ) -> EvaluationResult:
        result = decode(individual.genotype)
        if result is None:
            return EvaluationResult(
                0.0, {"valid": 0.0, "complexity": float(self.terms)}
            )
        error = abs(self.target - result)
        return EvaluationResult(
            1.0 / (1 + error),
            {
                "absolute_error": float(error),
                "value": float(result),
                "valid": 1.0,
                "complexity": float(self.terms),
            },
        )

    def mutate(self, genotype: str, rng: Random) -> VariationResult[str]:
        changed: list[JSONValue] = []
        bits = list(genotype)
        for index, bit in enumerate(bits):
            if rng.random() < self.mutation_rate:
                bits[index] = "1" if bit == "0" else "0"
                changed.append(index)
        return VariationResult("".join(bits), {"flipped_positions": changed})

    def crossover(self, first: str, second: str, rng: Random) -> VariationResult[str]:
        point = rng.randrange(1, len(first))
        return VariationResult(first[:point] + second[point:], {"cut_position": point})


def build_experiment(config: RunConfig, random: RunRandom) -> Experiment[str]:
    """Construct behavior without initializing or evaluating a population."""
    parameters = config.parameters
    unknown = set(parameters) - {"target", "terms", "mutation_rate"}
    if unknown:
        raise ConfigurationError(f"unknown arithmetic parameters: {sorted(unknown)}")
    target = parameters.get("target", 64)
    terms = parameters.get("terms", 10)
    rate = parameters.get("mutation_rate", 0.01)
    if isinstance(target, bool) or not isinstance(target, int):
        raise ConfigurationError("arithmetic target must be an integer")
    try:
        target_is_finite = math.isfinite(float(target))
    except OverflowError:
        target_is_finite = False
    if not target_is_finite:
        raise ConfigurationError(
            "arithmetic target must fit a finite floating-point metric"
        )
    if isinstance(terms, bool) or not isinstance(terms, int) or not 1 <= terms <= 32:
        raise ConfigurationError(
            "arithmetic terms must be an integer from 1 through 32"
        )
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not 0 <= rate <= 1
        or not math.isfinite(rate)
    ):
        raise ConfigurationError(
            "arithmetic mutation_rate must be a finite probability"
        )
    application = Arithmetic(target, terms, float(rate))
    return Experiment(
        initialize=application.initialize,
        evaluate=application.evaluate,
        mutations=(OperatorSpec("bit_flip", application.mutate),),
        crossovers=(OperatorSpec("one_point", application.crossover),),
        clone=clone,
        codec=BitStringCodec(terms),
        identity="examples.arithmetic",
        version="1",
    )
