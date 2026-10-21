"""Held-out reporting stays separate from committed neural search history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import digits_report
from xevolution.config import EvolutionSettings, RunConfig
from xevolution.engine import create_engine
from xevolution.errors import CheckpointCompatibilityError, RunStateError
from xevolution.storage import RunStore

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
digits = pytest.importorskip("examples.neural_digits")
pytestmark = pytest.mark.neural


def neural_config(*, generations: int = 0) -> RunConfig:
    return RunConfig(
        seed=71,
        factory="examples.neural_digits:build_experiment",
        parameters={
            "train_size": 100,
            "validation_size": 100,
            "test_size": 1597,
            "epochs": 2,
            "batch_size": 32,
        },
        evolution=EvolutionSettings(population_size=2, mutation_probability=1.0),
        termination={"max_generations": generations},
    )


@pytest.fixture
def finished_run(tmp_path: Path) -> Path:
    directory = tmp_path / "finished"
    with create_engine(neural_config(), output=directory) as engine:
        assert engine.state.phase == "terminated"
    return directory


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_report_reproduces_validation_then_tests_two_fixed_models_read_only(
    finished_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with (finished_run / "history.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":')
    before = snapshot(finished_run)
    original_prepare = digits.prepare_data
    original_train = digits.train_model
    original_score = digits.score_accuracy
    prepared = []
    fitted = []
    scored = []

    def prepare(settings):
        data = original_prepare(settings)
        prepared.append(data)
        return data

    def train(architecture, settings, data):
        fitted.append(architecture.to_dict())
        return original_train(architecture, settings, data)

    def score(model, features, labels):
        data = prepared[-1]
        if features is data.x_validation:
            scored.append("validation")
        elif features is data.x_test:
            scored.append("test")
        else:
            raise AssertionError("unexpected scoring partition")
        return original_score(model, features, labels)

    monkeypatch.setattr(digits, "prepare_data", prepare)
    monkeypatch.setattr(digits, "train_model", train)
    monkeypatch.setattr(digits, "score_accuracy", score)
    first = digits_report.build_report(finished_run)
    second = digits_report.build_report(finished_run)

    assert first == second
    assert scored == ["validation", "validation", "test", "test"] * 2
    assert len(fitted) == 4
    assert fitted[1] == fitted[3] == {"hidden_layers": [32], "activation": "relu"}
    assert first["baseline"]["parameter_count"] == 2410
    assert (
        first["selected"]["validation"]["accuracy"]
        == (first["search"]["saved_best_validation_accuracy"])
    )
    assert first["report_evaluations"]["fresh_model_fits"] == 2
    assert first["report_evaluations"]["test_evaluations"] == 2
    assert first["report_evaluations"]["selection_uses_test"] is False
    for model in ("selected", "baseline"):
        score_record = first[model]["test"]
        assert score_record["total"] == 1597
        assert score_record["accuracy"] == score_record["correct"] / 1597
    assert snapshot(finished_run) == before


def test_ready_run_fails_before_data_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "paused"
    with create_engine(neural_config(generations=1), output=directory) as engine:
        assert engine.state.phase == "ready"
    before = snapshot(directory)

    def unexpected(_settings):
        raise AssertionError("a ready run must not load data for final reporting")

    monkeypatch.setattr(digits, "prepare_data", unexpected)
    with pytest.raises(RunStateError, match="finished run"):
        digits_report.build_report(directory)
    assert snapshot(directory) == before


def test_wrong_application_fails_before_data_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "arithmetic"
    config = RunConfig(
        factory="examples.arithmetic:build_experiment",
        evolution=EvolutionSettings(population_size=2),
        termination={"max_generations": 0},
    )
    with create_engine(config, output=directory):
        pass

    def unexpected(_settings):
        raise AssertionError("an unrelated application must not load Digits data")

    monkeypatch.setattr(digits, "prepare_data", unexpected)
    with pytest.raises(CheckpointCompatibilityError, match="neural_digits"):
        digits_report.build_report(directory)


def test_incompatible_numerical_environment_fails_before_training(
    finished_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fingerprint = digits.compatibility_fingerprint

    def incompatible(settings, data):
        fingerprint = original_fingerprint(settings, data)
        fingerprint["changed_environment"] = True
        return fingerprint

    def unexpected(*_args):
        raise AssertionError("incompatible reporting must not train a model")

    monkeypatch.setattr(digits, "compatibility_fingerprint", incompatible)
    monkeypatch.setattr(digits, "train_model", unexpected)
    with pytest.raises(CheckpointCompatibilityError, match="compatibility differs"):
        digits_report.build_report(finished_run)


def test_validation_mismatch_fails_before_test_scoring_or_baseline_fit(
    finished_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with RunStore(finished_run) as store:
        best_fitness = store.load()["state"]["best"]["fitness"]
    original_prepare = digits.prepare_data
    original_train = digits.train_model
    prepared = []
    fitted = []

    def prepare(settings):
        data = original_prepare(settings)
        prepared.append(data)
        return data

    def train(architecture, settings, data):
        fitted.append(architecture)
        return original_train(architecture, settings, data)

    def changed_score(_model, features, labels):
        assert features is prepared[-1].x_validation, "test must not be scored"
        return (1.0, len(labels)) if best_fitness == 0.0 else (0.0, 0)

    monkeypatch.setattr(digits, "prepare_data", prepare)
    monkeypatch.setattr(digits, "train_model", train)
    monkeypatch.setattr(digits, "score_accuracy", changed_score)
    with pytest.raises(CheckpointCompatibilityError, match="did not reproduce"):
        digits_report.build_report(finished_run)
    assert len(fitted) == 1


def test_cli_writes_json_outside_the_run_and_prints_accuracy_counts(
    finished_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "reports" / "digits.json"
    before = snapshot(finished_run)
    assert digits_report.main([str(finished_run), "--output", str(destination)]) == 0
    captured = capsys.readouterr()
    assert "Held-out test:" in captured.out
    assert "/1597 (" in captured.out
    assert "Fixed baseline: [32] relu; 2410 trainable parameters" in captured.out
    assert not captured.err
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["report_type"] == "digits_held_out_evaluation"
    assert snapshot(finished_run) == before


def test_cli_rejects_output_inside_run_before_training(
    finished_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(_directory):
        raise AssertionError("invalid report destination must fail before training")

    monkeypatch.setattr(digits_report, "build_report", unexpected)
    output = finished_run / "report.json"
    assert digits_report.main([str(finished_run), "--output", str(output)]) == 1
    assert "outside the run directory" in capsys.readouterr().err
    assert not output.exists()
