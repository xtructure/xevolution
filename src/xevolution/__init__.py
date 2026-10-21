"""Representation-independent evolutionary computation for Python."""

__version__ = "0.1.1"

from .config import (
    EvolutionSettings,
    ParameterRule,
    RangeParameter,
    RunConfig,
    load_config,
)
from .engine import Engine, create_engine, resume, run
from .errors import (
    CheckpointCompatibilityError,
    CheckpointError,
    ConfigurationError,
    EvaluationError,
    OperatorInapplicable,
    ReproductionError,
    RunStateError,
    SelectionError,
    XevolutionError,
)
from .experiment import Experiment
from .models import (
    EvaluationResult,
    GenerationRecord,
    Individual,
    Population,
    RunState,
    StopDecision,
    VariationResult,
)
from .operators import OperatorSpec
from .randomness import EvaluationContext, RunRandom
from .registry import ExperimentRegistry

__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "ConfigurationError",
    "Engine",
    "EvaluationContext",
    "EvaluationError",
    "EvaluationResult",
    "EvolutionSettings",
    "Experiment",
    "ExperimentRegistry",
    "GenerationRecord",
    "Individual",
    "OperatorInapplicable",
    "OperatorSpec",
    "ParameterRule",
    "Population",
    "RangeParameter",
    "ReproductionError",
    "RunConfig",
    "RunRandom",
    "RunState",
    "RunStateError",
    "SelectionError",
    "StopDecision",
    "VariationResult",
    "XevolutionError",
    "__version__",
    "create_engine",
    "load_config",
    "resume",
    "run",
]
