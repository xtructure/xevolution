"""Joint evaluation: candidates share the resource each one chooses."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from random import Random

from xevolution.config import RunConfig
from xevolution.errors import ConfigurationError
from xevolution.experiment import Experiment
from xevolution.models import EvaluationResult, JSONValue, Population, VariationResult
from xevolution.operators import OperatorSpec
from xevolution.protocols import AttachmentReader, AttachmentWriter
from xevolution.randomness import EvaluationContext, RunRandom


@dataclass(frozen=True)
class ResourceCodec:
    """Encode an integer choice without accepting booleans as resource IDs."""

    count: int
    codec_id: str = "examples.shared_resource.choice"
    version: int = 1

    def _validate(self, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self.count
        ):
            raise ConfigurationError(
                "resource genotype must be an integer resource index"
            )
        return value

    def encode(self, value: int, writer: AttachmentWriter | None = None) -> JSONValue:
        return self._validate(value)

    def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> int:
        return self._validate(value)


def clone(genotype: int) -> int:
    return genotype


@dataclass(frozen=True)
class SharedResources:
    """A candidate's score depends on who else chooses the same resource."""

    quality: tuple[float, ...]

    def initialize(self, rng: Random) -> int:
        return rng.randrange(len(self.quality))

    def evaluate_population(
        self, population: Population[int], context: EvaluationContext
    ) -> list[tuple[int, EvaluationResult]]:
        occupancy = Counter(individual.genotype for individual in population)
        return [
            (
                individual.id,
                EvaluationResult(
                    self.quality[individual.genotype] / occupancy[individual.genotype],
                    {
                        "crowding": float(occupancy[individual.genotype]),
                        "resource": float(individual.genotype),
                    },
                ),
            )
            for individual in reversed(population.individuals)
        ]

    def mutate(self, genotype: int, rng: Random) -> VariationResult[int]:
        choice = rng.randrange(len(self.quality))
        return VariationResult(
            choice, {"previous_resource": genotype, "resource": choice}
        )

    def crossover(self, first: int, second: int, rng: Random) -> VariationResult[int]:
        selected = rng.randrange(2)
        return VariationResult((first, second)[selected], {"selected_parent": selected})


def build_experiment(config: RunConfig, random: RunRandom) -> Experiment[int]:
    """Construct a batch-only evaluator; no individual scoring hook is supplied."""
    unknown = set(config.parameters) - {"resource_quality"}
    if unknown:
        raise ConfigurationError(
            f"unknown shared-resource parameters: {sorted(unknown)}"
        )
    raw = config.parameters.get("resource_quality", [1.0, 1.8, 3.0, 5.0])
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError("resource_quality must be a nonempty list")
    quality: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float) or item <= 0:
            raise ConfigurationError(
                "resource qualities must be positive finite numbers"
            )
        try:
            number = float(item)
        except OverflowError as error:
            raise ConfigurationError("resource qualities must be finite") from error
        if not math.isfinite(number):
            raise ConfigurationError("resource qualities must be finite")
        quality.append(number)
    application = SharedResources(tuple(quality))
    return Experiment(
        initialize=application.initialize,
        evaluate_population=application.evaluate_population,
        mutations=(OperatorSpec("choose_resource", application.mutate),),
        crossovers=(OperatorSpec("inherit_resource", application.crossover),),
        clone=clone,
        codec=ResourceCodec(len(quality)),
        identity="examples.shared_resources",
        version="1",
    )
