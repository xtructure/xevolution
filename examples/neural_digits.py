"""Evolve small Digits classifiers using validation accuracy as the sole fitness.

Each architecture receives fresh weights and the same fixed training recipe.
The bundled 8x8 images are scaled by their known pixel maximum, 16. Search
never scores the held-out test split; ``digits_report`` owns that final step.
Install the optional ``neural`` dependency group to run this example.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import threadpoolctl  # type: ignore[import-untyped]
from sklearn import datasets, model_selection, neural_network  # type: ignore[import-untyped]

from xevolution.config import (
    RunConfig,
    as_object,
    strict_dumps,
    strict_loads,
    validate_json,
)
from xevolution.errors import (
    CheckpointCompatibilityError,
    ConfigurationError,
    OperatorInapplicable,
)
from xevolution.experiment import Experiment
from xevolution.models import (
    EvaluationResult,
    Individual,
    JSONDict,
    JSONValue,
    VariationResult,
)
from xevolution.operators import OperatorSpec
from xevolution.protocols import AttachmentReader, AttachmentWriter
from xevolution.randomness import EvaluationContext, RunRandom

APPLICATION_ID = "xevolution.examples.neural_digits"
APPLICATION_VERSION = "1"
WIDTHS = (8, 16, 32, 64)
ACTIVATIONS = ("relu", "tanh")
SAMPLE_COUNT = 1797
INPUT_COUNT = 64
CLASS_COUNT = 10

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"digits {name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"digits {name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise ConfigurationError(f"digits {name} must be finite") from error
    if not math.isfinite(number):
        raise ConfigurationError(f"digits {name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class Architecture:
    """An immutable sequence of one to three hidden layers and one activation."""

    hidden_layers: tuple[int, ...]
    activation: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_layers, tuple)
            or not 1 <= len(self.hidden_layers) <= 3
            or any(
                type(width) is not int or width not in WIDTHS
                for width in self.hidden_layers
            )
        ):
            raise ConfigurationError(
                "digits hidden_layers must be a tuple of 1..3 widths from 8, 16, 32, 64"
            )
        if not isinstance(self.activation, str) or self.activation not in ACTIVATIONS:
            raise ConfigurationError("digits activation must be 'relu' or 'tanh'")

    def to_dict(self) -> JSONDict:
        """Return an independent JSON representation for reports and the codec."""
        return {
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureCodec:
    """Encode only explicit architecture fields, with no trained model payload."""

    codec_id: ClassVar[str] = "xevolution.examples.digits_architecture"
    version: ClassVar[int] = 1

    def encode(
        self, value: Architecture, writer: AttachmentWriter | None = None
    ) -> JSONValue:
        if not isinstance(value, Architecture):
            raise ConfigurationError("digits codec requires an Architecture")
        return value.to_dict()

    def decode(
        self, value: JSONValue, reader: AttachmentReader | None = None
    ) -> Architecture:
        payload = as_object(validate_json(value), "digits architecture")
        if set(payload) != {"hidden_layers", "activation"}:
            raise ConfigurationError(
                "digits architecture has missing or unknown fields"
            )
        layers, activation = payload["hidden_layers"], payload["activation"]
        if not isinstance(layers, list) or not isinstance(activation, str):
            raise ConfigurationError(
                "digits architecture requires layer array and activation"
            )
        return Architecture(
            tuple(_integer(width, "layer width", minimum=1) for width in layers),
            activation,
        )


@dataclass(frozen=True, slots=True)
class DigitsSettings:
    """Fixed data partition and training recipe; the search evolves neither."""

    train_size: int = 1000
    validation_size: int = 400
    test_size: int = 397
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    alpha: float = 0.0001
    split_seed: int = 17
    training_seed: int = 23

    def __post_init__(self) -> None:
        for name in ("train_size", "validation_size", "test_size"):
            _integer(getattr(self, name), name, minimum=CLASS_COUNT)
        if self.train_size + self.validation_size + self.test_size != SAMPLE_COUNT:
            raise ConfigurationError("digits split sizes must sum to 1797")
        _integer(self.epochs, "epochs", minimum=1)
        _integer(self.batch_size, "batch_size", minimum=1)
        for name in ("split_seed", "training_seed"):
            if _integer(getattr(self, name), name) >= 2**32:
                raise ConfigurationError(f"digits {name} must be below 2**32")
        rate = _number(self.learning_rate, "learning_rate")
        alpha = _number(self.alpha, "alpha")
        if rate <= 0 or alpha < 0:
            raise ConfigurationError("digits learning_rate must be > 0 and alpha >= 0")
        object.__setattr__(self, "learning_rate", rate)
        object.__setattr__(self, "alpha", alpha)

    def to_dict(self) -> JSONDict:
        """Return all effective application parameters, including defaults."""
        return {
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "test_size": self.test_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "split_seed": self.split_seed,
            "training_seed": self.training_seed,
        }

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, JSONValue]) -> DigitsSettings:
        """Validate the complete namespace rather than silently ignoring typos."""
        supplied = as_object(validate_json(dict(parameters)), "digits parameters")
        defaults = cls().to_dict()
        unknown = supplied.keys() - defaults.keys()
        if unknown:
            raise ConfigurationError(f"unknown digits parameters: {sorted(unknown)}")
        values = defaults | supplied
        return cls(
            train_size=_integer(
                values["train_size"], "train_size", minimum=CLASS_COUNT
            ),
            validation_size=_integer(
                values["validation_size"], "validation_size", minimum=CLASS_COUNT
            ),
            test_size=_integer(values["test_size"], "test_size", minimum=CLASS_COUNT),
            epochs=_integer(values["epochs"], "epochs", minimum=1),
            batch_size=_integer(values["batch_size"], "batch_size", minimum=1),
            learning_rate=_number(values["learning_rate"], "learning_rate"),
            alpha=_number(values["alpha"], "alpha"),
            split_seed=_integer(values["split_seed"], "split_seed"),
            training_seed=_integer(values["training_seed"], "training_seed"),
        )


@dataclass(frozen=True, slots=True, eq=False)
class DigitsData:
    """Prepared read-only arrays and epoch orders, reproducible from settings.

    Test arrays are carried for the final report, but training and evaluation
    use only their named training and validation counterparts.
    """

    settings: DigitsSettings
    x_train: FloatArray = field(repr=False)
    y_train: IntArray = field(repr=False)
    x_validation: FloatArray = field(repr=False)
    y_validation: IntArray = field(repr=False)
    x_test: FloatArray = field(repr=False)
    y_test: IntArray = field(repr=False)
    train_indices: IntArray = field(repr=False)
    validation_indices: IntArray = field(repr=False)
    test_indices: IntArray = field(repr=False)
    epoch_orders: tuple[IntArray, ...] = field(repr=False)
    fingerprint: str


def _arrays_digest(*arrays: FloatArray | IntArray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        # Fix byte order and describe boundaries to avoid ambiguous concatenation.
        canonical = np.asarray(array, dtype="<f8" if array.dtype.kind == "f" else "<i8")
        digest.update(str(canonical.shape).encode("ascii") + b"\0")
        digest.update(canonical.dtype.str.encode("ascii") + b"\0")
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def prepare_data(settings: DigitsSettings) -> DigitsData:
    """Load local Digits data, stratify fixed splits, and freeze epoch orders.

    The split uses two stratified ``train_test_split`` calls with the same
    explicit split seed. Epoch orders use their own PCG64 generator; they do
    not depend on architecture size or on model-initialization random draws.
    """
    if not isinstance(settings, DigitsSettings):
        raise ConfigurationError("digits data requires DigitsSettings")
    raw_x, raw_y = datasets.load_digits(return_X_y=True)
    features = np.asarray(raw_x, dtype=np.float64)
    labels = np.asarray(raw_y, dtype=np.int64)
    if features.shape != (SAMPLE_COUNT, INPUT_COUNT) or labels.shape != (SAMPLE_COUNT,):
        raise ConfigurationError("the bundled Digits dataset has an unexpected shape")
    raw_digest = _arrays_digest(features, labels)
    features = features / 16.0
    indices = np.arange(SAMPLE_COUNT, dtype=np.int64)
    train_indices, remaining = model_selection.train_test_split(
        indices,
        train_size=settings.train_size,
        test_size=settings.validation_size + settings.test_size,
        stratify=labels,
        random_state=settings.split_seed,
    )
    validation_indices, test_indices = model_selection.train_test_split(
        remaining,
        train_size=settings.validation_size,
        test_size=settings.test_size,
        stratify=labels[remaining],
        random_state=settings.split_seed,
    )
    order_rng = np.random.Generator(np.random.PCG64(settings.training_seed))
    orders = tuple(
        np.asarray(order_rng.permutation(settings.train_size), dtype=np.int64)
        for _ in range(settings.epochs)
    )
    partitions = (train_indices, validation_indices, test_indices)
    feature_splits = tuple(features[part] for part in partitions)
    label_splits = tuple(labels[part] for part in partitions)
    for array in (*partitions, *feature_splits, *label_splits, *orders):
        array.flags.writeable = False
    description: JSONDict = {
        "dataset": "sklearn.datasets.load_digits",
        "raw_data_sha256": raw_digest,
        "preprocessing": "float64 pixels divided by 16; no fitted transform",
        "partition_method": "two stratified train_test_split calls; training split first",
        "epoch_order_method": "numpy.Generator(PCG64(training_seed)).permutation per epoch",
        "partition_and_order_sha256": _arrays_digest(*partitions, *orders),
    }
    fingerprint = hashlib.sha256(strict_dumps(description).encode("utf-8")).hexdigest()
    return DigitsData(
        settings,
        feature_splits[0],
        label_splits[0],
        feature_splits[1],
        label_splits[1],
        feature_splits[2],
        label_splits[2],
        train_indices,
        validation_indices,
        test_indices,
        orders,
        fingerprint,
    )


def _check_data(settings: DigitsSettings, data: DigitsData) -> None:
    if not isinstance(settings, DigitsSettings) or not isinstance(data, DigitsData):
        raise ConfigurationError(
            "digits training requires DigitsSettings and DigitsData"
        )
    if data.settings != settings:
        raise ConfigurationError("digits data was prepared with different settings")


def train_model(
    architecture: Architecture, settings: DigitsSettings, data: DigitsData
) -> neural_network.MLPClassifier:
    """Fit fresh weights for exactly the configured epochs, using one BLAS thread.

    ``partial_fit`` performs one epoch and retains Adam state between calls.
    No validation or test data participates in training or early stopping.
    The integer initialization seed is shared across candidate evaluations;
    precomputed epoch permutations are independent of initialization draws.
    """
    _check_data(settings, data)
    if not isinstance(architecture, Architecture):
        raise ConfigurationError("digits training requires an Architecture")
    model = neural_network.MLPClassifier(
        hidden_layer_sizes=architecture.hidden_layers,
        activation=architecture.activation,
        solver="adam",
        alpha=settings.alpha,
        batch_size=min(settings.batch_size, settings.train_size),
        learning_rate_init=settings.learning_rate,
        random_state=settings.training_seed,
        shuffle=False,
        early_stopping=False,
        max_iter=1,
        n_iter_no_change=settings.epochs + 1,
        tol=0.0,
    )
    classes = np.arange(CLASS_COUNT, dtype=np.int64)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for order in data.epoch_orders:
            model.partial_fit(data.x_train[order], data.y_train[order], classes=classes)
    # scikit-learn can catch a KeyboardInterrupt internally. Never accept a
    # partially trained candidate as though it completed the requested recipe.
    if model.t_ != settings.epochs * settings.train_size:
        raise RuntimeError("digits training stopped before completing all epochs")
    return model


def score_accuracy(
    model: neural_network.MLPClassifier, features: FloatArray, targets: IntArray
) -> tuple[float, int]:
    """Return exact correct/total accuracy and the integer number correct."""
    if targets.ndim != 1 or len(targets) == 0 or len(features) != len(targets):
        raise ConfigurationError(
            "digits accuracy requires equal nonempty feature/label rows"
        )
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        predictions = model.predict(features)
    correct = int(np.count_nonzero(predictions == targets))
    return correct / len(targets), correct


def parameter_count(architecture: Architecture) -> int:
    """Count all weights and biases, as a descriptive metric only."""
    widths = (INPUT_COUNT, *architecture.hidden_layers, CLASS_COUNT)
    return sum(
        (before + 1) * after
        for before, after in zip(widths[:-1], widths[1:], strict=True)
    )


def compatibility_fingerprint(settings: DigitsSettings, data: DigitsData) -> JSONDict:
    """Describe data, recipe, code, and numerical environment required for replay.

    Paths and current ambient thread counts are excluded. Backend identities
    are recorded under the same one-thread BLAS limit used for training.
    This is a conservative compatibility check, not a cross-platform guarantee.
    """
    _check_data(settings, data)
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        backends = [
            as_object(
                validate_json(
                    {
                        key: pool.get(key)
                        for key in (
                            "user_api",
                            "internal_api",
                            "prefix",
                            "version",
                            "threading_layer",
                            "architecture",
                            "num_threads",
                        )
                    }
                )
            )
            for pool in threadpoolctl.threadpool_info()
            if pool.get("user_api") == "blas"
        ]
    return {
        "settings": settings.to_dict(),
        "data_sha256": data.fingerprint,
        "training_recipe": {
            "version": 1,
            "input_features": INPUT_COUNT,
            "classes": CLASS_COUNT,
            "dtype": "float64",
            "solver": "MLPClassifier Adam; one partial_fit call per epoch",
            "initialization": "sklearn RandomState(training_seed), fresh for every model",
            "epoch_order": "separate numpy PCG64(training_seed) permutations",
            "shuffle": False,
            "early_stopping": False,
            "effective_batch_size": min(settings.batch_size, settings.train_size),
            "blas_threads": 1,
            "fitness": "validation correct_count / validation_size",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "threadpoolctl",
            )
        },
        "blas_backends": sorted(backends, key=strict_dumps),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
            "byteorder": sys.byteorder,
        },
        "application_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


class DigitsEvaluator:
    """Evaluate validation accuracy, with an explicit saved model-fit counter.

    Arrays and epoch orders are reconstructed and fingerprinted during setup.
    No fitted models or random generators survive an evaluation. ``model_fits``
    is an application audit counter; the framework separately owns accepted
    evaluation accounting. Failed, uncommitted work may repeat after a crash.
    """

    def __init__(self, settings: DigitsSettings, data: DigitsData) -> None:
        self.settings = settings
        self.data = data
        self.model_fits = 0
        self._identity_json = strict_dumps(compatibility_fingerprint(settings, data))

    def __call__(
        self, individual: Individual[Architecture], context: EvaluationContext
    ) -> EvaluationResult:
        # Training deliberately ignores per-individual context RNGs: a genotype
        # receives the same recipe regardless of its ID, generation, or run seed.
        architecture = individual.genotype
        model = train_model(architecture, self.settings, self.data)
        accuracy, correct = score_accuracy(
            model, self.data.x_validation, self.data.y_validation
        )
        self.model_fits += 1
        return EvaluationResult(
            accuracy,
            {
                "accuracy": accuracy,
                "correct_count": float(correct),
                "validation_size": float(self.settings.validation_size),
                "parameter_count": float(parameter_count(architecture)),
                "depth": float(len(architecture.hidden_layers)),
                "epochs": float(self.settings.epochs),
            },
        )

    def checkpoint_identity(self) -> JSONDict:
        """Return an independent copy of the immutable prepared recipe identity."""
        return as_object(strict_loads(self._identity_json))

    def state_dict(self) -> JSONDict:
        """Export replayable application state; no arrays or trained weights."""
        return {
            "schema_version": 1,
            "compatibility": self.checkpoint_identity(),
            "model_fits": self.model_fits,
        }

    def load_state_dict(self, state: JSONValue) -> None:
        """Validate recipe identity before restoring the successful-fit counter."""
        saved = as_object(validate_json(state), "digits evaluator state")
        if set(saved) != {"schema_version", "compatibility", "model_fits"}:
            raise ConfigurationError(
                "digits evaluator state has missing or unknown fields"
            )
        if type(saved["schema_version"]) is not int or saved["schema_version"] != 1:
            raise CheckpointCompatibilityError(
                "unsupported digits evaluator state version"
            )
        if saved["compatibility"] != self.checkpoint_identity():
            raise CheckpointCompatibilityError(
                "digits evaluator recipe compatibility mismatch"
            )
        self.model_fits = _integer(saved["model_fits"], "model_fits")


def initialize_architecture(rng: Random) -> Architecture:
    """Choose a valid depth, each width, and activation from the run-owned RNG."""
    return Architecture(
        tuple(rng.choice(WIDTHS) for _ in range(rng.randint(1, 3))),
        rng.choice(ACTIVATIONS),
    )


def clone_architecture(genotype: Architecture) -> Architecture:
    """Immutable architectures can safely share their storage."""
    return genotype


@dataclass(frozen=True, slots=True)
class ResizeLayer:
    """Change one hidden layer to a different allowed width."""

    def __call__(
        self, genotype: Architecture, rng: Random
    ) -> VariationResult[Architecture]:
        index = rng.randrange(len(genotype.hidden_layers))
        previous = genotype.hidden_layers[index]
        width = rng.choice(tuple(value for value in WIDTHS if value != previous))
        layers = (
            genotype.hidden_layers[:index]
            + (width,)
            + genotype.hidden_layers[index + 1 :]
        )
        return VariationResult(
            Architecture(layers, genotype.activation),
            {"layer": index, "before": previous, "after": width},
        )


@dataclass(frozen=True, slots=True)
class AddLayer:
    """Insert one layer, signalling inapplicability at the three-layer limit."""

    def __call__(
        self, genotype: Architecture, rng: Random
    ) -> VariationResult[Architecture]:
        if len(genotype.hidden_layers) == 3:
            raise OperatorInapplicable(
                "digits architecture already has three hidden layers"
            )
        index = rng.randrange(len(genotype.hidden_layers) + 1)
        width = rng.choice(WIDTHS)
        layers = (
            genotype.hidden_layers[:index] + (width,) + genotype.hidden_layers[index:]
        )
        return VariationResult(
            Architecture(layers, genotype.activation), {"layer": index, "width": width}
        )


@dataclass(frozen=True, slots=True)
class RemoveLayer:
    """Remove one layer, signalling inapplicability at the one-layer limit."""

    def __call__(
        self, genotype: Architecture, rng: Random
    ) -> VariationResult[Architecture]:
        if len(genotype.hidden_layers) == 1:
            raise OperatorInapplicable(
                "digits architecture must retain one hidden layer"
            )
        index = rng.randrange(len(genotype.hidden_layers))
        layers = genotype.hidden_layers[:index] + genotype.hidden_layers[index + 1 :]
        return VariationResult(
            Architecture(layers, genotype.activation),
            {"layer": index, "width": genotype.hidden_layers[index]},
        )


@dataclass(frozen=True, slots=True)
class ChangeActivation:
    """Switch the shared hidden-layer activation between ReLU and tanh."""

    def __call__(
        self, genotype: Architecture, rng: Random
    ) -> VariationResult[Architecture]:
        activation = "tanh" if genotype.activation == "relu" else "relu"
        return VariationResult(
            Architecture(genotype.hidden_layers, activation),
            {"before": genotype.activation, "after": activation},
        )


@dataclass(frozen=True, slots=True)
class ArchitectureCrossover:
    """Inherit a parent's depth and combine widths at corresponding positions."""

    def __call__(
        self, first: Architecture, second: Architecture, rng: Random
    ) -> VariationResult[Architecture]:
        parents = (first, second)
        depth_parent = rng.randrange(2)
        depth = len(parents[depth_parent].hidden_layers)
        sources = tuple(
            rng.choice(
                tuple(
                    index
                    for index, parent in enumerate(parents)
                    if position < len(parent.hidden_layers)
                )
            )
            for position in range(depth)
        )
        layers = tuple(
            parents[source].hidden_layers[position]
            for position, source in enumerate(sources)
        )
        activation_parent = rng.randrange(2)
        return VariationResult(
            Architecture(layers, parents[activation_parent].activation),
            {
                "depth_parent": depth_parent,
                "width_parents": list(sources),
                "activation_parent": activation_parent,
            },
        )


def build_experiment(config: RunConfig, random: RunRandom) -> Experiment[Architecture]:
    """Compose the demo, preparing data without training or accessing test scores."""
    settings = DigitsSettings.from_parameters(config.parameters)
    data = prepare_data(settings)
    evaluator = DigitsEvaluator(settings, data)
    return Experiment(
        initialize=initialize_architecture,
        evaluate=evaluator,
        mutations=(
            OperatorSpec("resize", ResizeLayer()),
            OperatorSpec("add", AddLayer()),
            OperatorSpec("remove", RemoveLayer()),
            OperatorSpec("activation", ChangeActivation()),
        ),
        crossovers=(OperatorSpec("combine", ArchitectureCrossover()),),
        clone=clone_architecture,
        codec=ArchitectureCodec(),
        identity=APPLICATION_ID,
        version=APPLICATION_VERSION,
        compatibility=evaluator.checkpoint_identity(),
    )
