"""Optional CPU neural example tests, including real fresh-process restoration.

The default dependency group does not install the numerical stack. Import
guards precede the example import, so core-only test collection skips this
module without requiring NumPy, scikit-learn, or threadpoolctl.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from pathlib import Path
from random import Random

import pytest

from xevolution.config import RunConfig, as_object, strict_dumps, strict_loads
from xevolution.errors import (
    CheckpointCompatibilityError,
    ConfigurationError,
    OperatorInapplicable,
)
from xevolution.models import Individual, JSONDict
from xevolution.randomness import RunRandom
from xevolution.storage import RunStore

pytestmark = pytest.mark.neural
np = pytest.importorskip("numpy", reason="install the optional neural dependency group")
pytest.importorskip("sklearn", reason="install the optional neural dependency group")
pytest.importorskip(
    "threadpoolctl", reason="install the optional neural dependency group"
)
neural = import_module("examples.neural_digits")

PROJECT = Path(__file__).resolve().parents[1]
FAST_PARAMETERS: JSONDict = {
    "train_size": 100,
    "validation_size": 100,
    "test_size": 1597,
    "epochs": 2,
    "batch_size": 64,
    "learning_rate": 0.001,
    "alpha": 0.0001,
    "split_seed": 17,
    "training_seed": 23,
}


def fast_settings():
    return neural.DigitsSettings.from_parameters(FAST_PARAMETERS)


def small_config() -> RunConfig:
    return RunConfig.from_dict(
        {
            "schema_version": 1,
            "seed": 43,
            "experiment": {
                "factory": "examples.neural_digits:build_experiment",
                "parameters": dict(FAST_PARAMETERS),
            },
            "evolution": {
                "population_size": 3,
                "mutation_probability": 0.5,
                "parent_selection": {"name": "tournament", "size": 2},
                "survival": {"name": "elitism", "count": 1},
            },
            "termination": {"max_generations": 2},
        }
    )


@pytest.fixture(scope="module")
def prepared_data():
    return neural.prepare_data(fast_settings())


@pytest.mark.parametrize(
    ("hidden_layers", "activation"),
    [
        ((), "relu"),
        ((8, 8, 8, 8), "relu"),
        ((0,), "relu"),
        ((12,), "relu"),
        ((128,), "relu"),
        ((True,), "relu"),
        ((8.0,), "relu"),
        ((8,), "sigmoid"),
        ((8,), True),
    ],
)
def test_architecture_rejects_invalid_shape_or_activation(
    hidden_layers: tuple, activation: object
) -> None:
    with pytest.raises(ConfigurationError):
        neural.Architecture(hidden_layers, activation)


def test_architecture_codec_round_trip_is_strict_and_immutable() -> None:
    architecture = neural.Architecture((8, 32, 64), "tanh")
    codec = neural.ArchitectureCodec()
    encoded = codec.encode(architecture)
    restored = codec.decode(encoded)
    assert restored == architecture
    assert isinstance(restored.hidden_layers, tuple)
    with pytest.raises(FrozenInstanceError):
        restored.activation = "relu"
    encoded = as_object(encoded)
    with pytest.raises(ConfigurationError):
        codec.decode({**encoded, "unexpected": 1})
    for malformed in (None, [], "architecture"):
        with pytest.raises(ConfigurationError):
            codec.decode(malformed)


def test_variation_preserves_parents_and_architecture_bounds() -> None:
    first = neural.Architecture((16, 32), "relu")
    second = neural.Architecture((8, 64, 16), "tanh")
    original = (
        first.hidden_layers,
        first.activation,
        second.hidden_layers,
        second.activation,
    )
    operations = (
        neural.ResizeLayer(),
        neural.AddLayer(),
        neural.RemoveLayer(),
        neural.ChangeActivation(),
    )
    for seed in range(12):
        for operation in operations:
            child = operation(first, Random(seed)).genotype
            assert 1 <= len(child.hidden_layers) <= 3
            assert all(width in (8, 16, 32, 64) for width in child.hidden_layers)
            assert child.activation in ("relu", "tanh")
        child = neural.ArchitectureCrossover()(first, second, Random(seed)).genotype
        assert 1 <= len(child.hidden_layers) <= 3
        assert all(width in (8, 16, 32, 64) for width in child.hidden_layers)
        assert child.activation in ("relu", "tanh")
    assert (
        first.hidden_layers,
        first.activation,
        second.hidden_layers,
        second.activation,
    ) == original


def test_layer_operations_signal_inapplicability_at_bounds() -> None:
    with pytest.raises(OperatorInapplicable):
        neural.RemoveLayer()(neural.Architecture((8,), "relu"), Random(0))
    with pytest.raises(OperatorInapplicable):
        neural.AddLayer()(neural.Architecture((8, 16, 32), "relu"), Random(0))


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": 1},
        {"epochs": 0},
        {"epochs": True},
        {"epochs": 1.5},
        {"batch_size": 0},
        {"batch_size": True},
        {"learning_rate": 0.0},
        {"learning_rate": float("inf")},
        {"learning_rate": True},
        {"alpha": -0.1},
        {"alpha": float("nan")},
        {"alpha": False},
        {"split_seed": True},
        {"training_seed": -1},
        {"train_size": 99},
        {"train_size": 1, "test_size": 1696},
        {"validation_size": 1, "test_size": 1696},
        {"train_size": True},
    ],
)
def test_settings_validate_programmatic_inputs(changes: JSONDict) -> None:
    with pytest.raises(ConfigurationError):
        neural.DigitsSettings.from_parameters(FAST_PARAMETERS | changes)


def test_data_partitions_are_disjoint_complete_stratified_and_repeatable(
    prepared_data,
) -> None:
    data = prepared_data
    partitions = (
        set(data.train_indices.tolist()),
        set(data.validation_indices.tolist()),
        set(data.test_indices.tolist()),
    )
    assert tuple(map(len, partitions)) == (100, 100, 1597)
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(range(1797))
    for name in ("x_train", "x_validation", "x_test"):
        features = getattr(data, name)
        assert features.shape[1] == 64
        assert np.all((0 <= features) & (features <= 1))
        assert not features.flags.writeable
    for name in ("y_train", "y_validation", "y_test"):
        assert set(getattr(data, name).tolist()) == set(range(10))
    assert len(data.epoch_orders) == fast_settings().epochs
    assert all(
        sorted(order.tolist()) == list(range(100)) for order in data.epoch_orders
    )
    repeated = neural.prepare_data(fast_settings())
    assert repeated.fingerprint == data.fingerprint
    assert np.array_equal(repeated.train_indices, data.train_indices)
    assert all(
        np.array_equal(first, second)
        for first, second in zip(repeated.epoch_orders, data.epoch_orders, strict=True)
    )


class ForbiddenTestData:
    def __array__(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("Architecture evaluation must not use held-out test data")

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"Architecture evaluation accessed test data attribute {name}"
        )

    def __len__(self) -> int:
        raise AssertionError("Architecture evaluation must not inspect test data")

    def __getitem__(self, item: object) -> object:
        raise AssertionError("Architecture evaluation must not index test data")


def test_training_is_repeatable_and_fitness_is_validation_accuracy_only(
    prepared_data,
) -> None:
    settings = fast_settings()
    protected = replace(
        prepared_data, x_test=ForbiddenTestData(), y_test=ForbiddenTestData()
    )
    architecture = neural.Architecture((8,), "relu")
    first = neural.train_model(architecture, settings, protected)
    second = neural.train_model(architecture, settings, protected)
    for before, after in zip(first.coefs_, second.coefs_, strict=True):
        assert np.array_equal(before, after)
    for before, after in zip(first.intercepts_, second.intercepts_, strict=True):
        assert np.array_equal(before, after)
    assert first.t_ == settings.epochs * settings.train_size
    expected_accuracy, expected_correct = neural.score_accuracy(
        first, protected.x_validation, protected.y_validation
    )
    evaluator = neural.DigitsEvaluator(settings, protected)
    context = RunRandom(11).context(0, "neural-test", individual_id=0)
    initial = evaluator(Individual(0, architecture), context)
    repeated = evaluator(
        Individual(99, architecture, birth_generation=7),
        RunRandom(12).context(7, "different-run", individual_id=99),
    )
    assert initial == repeated
    assert 0.0 <= initial.fitness <= 1.0
    assert initial.fitness == expected_accuracy == expected_correct / 100
    assert initial.metrics == {
        "accuracy": expected_accuracy,
        "correct_count": float(expected_correct),
        "validation_size": 100.0,
        "parameter_count": 610.0,
        "depth": 1.0,
        "epochs": 2.0,
    }
    assert evaluator.model_fits == 2


def test_evaluator_state_restores_counts_and_rejects_changed_data(
    prepared_data,
) -> None:
    settings = fast_settings()
    evaluator = neural.DigitsEvaluator(settings, prepared_data)
    saved = evaluator.state_dict()
    saved["model_fits"] = 7
    evaluator.load_state_dict(saved)
    assert evaluator.model_fits == 7
    assert evaluator.state_dict() == saved
    saved["model_fits"] = 99
    assert evaluator.model_fits == 7

    changed = neural.DigitsEvaluator(
        settings, replace(prepared_data, fingerprint="different-dataset-content")
    )
    with pytest.raises(CheckpointCompatibilityError, match="compatibility mismatch"):
        changed.load_state_dict(evaluator.state_dict())
    assert changed.model_fits == 0

    mismatched_recipe = replace(settings, epochs=settings.epochs + 1)
    with pytest.raises(ConfigurationError, match="different settings"):
        neural.train_model(
            neural.Architecture((8,), "relu"), mismatched_recipe, prepared_data
        )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"model_fits": True}, ConfigurationError),
        ({"model_fits": -1}, ConfigurationError),
        ({"model_fits": 1.5}, ConfigurationError),
        ({"unexpected": 1}, ConfigurationError),
        ({"schema_version": 0}, CheckpointCompatibilityError),
        ({"schema_version": True}, CheckpointCompatibilityError),
        ({"compatibility": {}}, CheckpointCompatibilityError),
    ],
)
def test_evaluator_rejects_invalid_saved_state_without_changing_count(
    prepared_data, changes: JSONDict, error: type[Exception]
) -> None:
    evaluator = neural.DigitsEvaluator(fast_settings(), prepared_data)
    before = evaluator.state_dict()
    with pytest.raises(error):
        evaluator.load_state_dict(before | changes)
    assert evaluator.state_dict() == before


DRIVER = """
import sys
from xevolution import create_engine, resume
mode, directory, config, pause = sys.argv[1:]
engine = create_engine(config, output=directory) if mode == 'fresh' else resume(directory)
with engine:
    engine.run(until_generation=None if pause == '-' else int(pause))
"""


def execute(
    mode: str, directory: Path, configuration: Path, pause: int | None = None
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT)
    environment["PYTHONHASHSEED"] = "17" if mode == "fresh" else "91"
    return subprocess.run(
        [
            sys.executable,
            "-c",
            DRIVER,
            mode,
            str(directory),
            str(configuration),
            "-" if pause is None else str(pause),
        ],
        cwd=directory.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def snapshot(path: Path) -> tuple[JSONDict, list[JSONDict]]:
    with RunStore(path) as store:
        body = store.load()
        records = store.read_history()
    body.pop("run_id")
    as_object(body["state"]).pop("run_id")
    for record in records:
        record.pop("run_id")
    return body, records


@pytest.fixture(scope="module")
def complete_neural_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("neural-complete")
    configuration = root / "config.json"
    configuration.write_text(strict_dumps(small_config().to_dict()), encoding="utf-8")
    output = root / "run"
    result = execute("fresh", output, configuration)
    assert result.returncode == 0, result.stderr
    return output, configuration


@pytest.mark.parametrize("pause", [0, 1])
def test_new_process_resume_matches_all_state_and_history(
    tmp_path: Path, complete_neural_run: tuple[Path, Path], pause: int
) -> None:
    complete, configuration = complete_neural_run
    interrupted = tmp_path / "interrupted"
    result = execute("fresh", interrupted, configuration, pause)
    assert result.returncode == 0, result.stderr
    body, records = snapshot(interrupted)
    state = as_object(body["state"])
    assert state["generation"] == pause
    assert state["phase"] == "ready"
    assert state["evaluations"] == 3 * (pause + 1)
    assert len(records) == pause + 1
    result = execute("resume", interrupted, configuration)
    assert result.returncode == 0, result.stderr
    assert snapshot(interrupted) == snapshot(complete)
    body, records = snapshot(interrupted)
    state = as_object(body["state"])
    assert state["generation"] == 2
    assert state["phase"] == "terminated"
    assert state["evaluations"] == 9
    assert [record["sequence"] for record in records] == [0, 1, 2]
    components = as_object(body["components"])
    evaluator_state = next(
        value
        for value in components.values()
        if isinstance(value, dict) and "model_fits" in value
    )
    assert evaluator_state["model_fits"] == 9


def test_corrupt_neural_compatibility_is_rejected_before_training(
    tmp_path: Path, complete_neural_run: tuple[Path, Path]
) -> None:
    _, configuration = complete_neural_run
    directory = tmp_path / "incompatible"
    result = execute("fresh", directory, configuration, 0)
    assert result.returncode == 0, result.stderr
    checkpoint = directory / "checkpoint.json"
    envelope = as_object(strict_loads(checkpoint.read_text(encoding="utf-8")))
    body = as_object(envelope["body"])
    compatibility = as_object(body["compatibility"])
    application = as_object(compatibility["application"])
    requirements = as_object(application["requirements"])
    requirements["changed_numerical_environment"] = "incompatible"
    envelope["body_sha256"] = hashlib.sha256(
        strict_dumps(body).encode("utf-8")
    ).hexdigest()
    checkpoint.write_text(strict_dumps(envelope) + "\n", encoding="utf-8")
    checkpoint_before = checkpoint.read_bytes()
    journal_before = (directory / "history.jsonl").read_bytes()
    result = execute("resume", directory, configuration)
    assert result.returncode != 0
    assert "compatibility mismatch" in result.stderr
    assert checkpoint.read_bytes() == checkpoint_before
    assert (directory / "history.jsonl").read_bytes() == journal_before
