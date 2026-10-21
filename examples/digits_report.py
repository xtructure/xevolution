"""Fresh held-out evaluation of a finished Digits architecture-search run.

This companion reads a committed checkpoint without resuming it. It first
reproduces the selected architecture's recorded validation accuracy, then
evaluates that model and a fixed baseline on the held-out test partition. The
two fresh fits and their test results are outside the evolutionary history.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from xevolution.config import RunConfig, as_object, strict_dumps
from xevolution.errors import (
    CheckpointCompatibilityError,
    ConfigurationError,
    RunStateError,
    XevolutionError,
)
from xevolution.models import JSONDict, JSONValue
from xevolution.serialization import decode_state
from xevolution.storage import RunStore


def _accuracy_record(accuracy: float, correct: int, total: int) -> JSONDict:
    if (
        type(correct) is not int
        or type(total) is not int
        or total <= 0
        or not 0 <= correct <= total
        or isinstance(accuracy, bool)
        or not isinstance(accuracy, int | float)
        or not math.isfinite(accuracy)
        or accuracy != correct / total
    ):
        raise ConfigurationError("accuracy must agree with finite correct/total counts")
    return {
        "accuracy": float(accuracy),
        "percent": 100.0 * accuracy,
        "correct": correct,
        "total": total,
    }


def build_report(run_directory: Path | str) -> JSONDict:
    """Evaluate two fixed architectures without writing or resuming the run.

    A terminal committed checkpoint and matching application/data/numerical
    compatibility fingerprints are required. The historical best is selected
    solely by its saved validation fitness. Its fresh validation score must
    reproduce that fitness exactly before either model is scored on test data.

    Both models use the original training partition, fixed recipe, and seed.
    Repeated calls repeat two fits and two test evaluations; they do not select
    another candidate or add measurements to the authoritative run history.
    """
    with RunStore(run_directory) as store:
        body = store.load()
        raw_state = as_object(body["state"], "checkpoint state")
        if raw_state.get("phase") != "terminated":
            raise RunStateError(
                "Digits reporting requires a finished run; resume the ready run "
                "until its configured stopping policy terminates it"
            )

        # Defer optional numerical imports until the completed-run guard. This
        # also keeps --help usable without installing the neural dependency group.
        try:
            from examples import neural_digits as digits
        except ModuleNotFoundError as error:
            if error.name in {"numpy", "scipy", "sklearn", "threadpoolctl"}:
                raise ConfigurationError(
                    "Digits reporting needs the neural dependencies; run "
                    "uv sync --locked --group neural"
                ) from error
            raise

        compatibility = as_object(body["compatibility"], "checkpoint compatibility")
        application = as_object(
            compatibility.get("application"), "application identity"
        )
        if (
            application.get("identity") != digits.APPLICATION_ID
            or application.get("version") != digits.APPLICATION_VERSION
        ):
            raise CheckpointCompatibilityError(
                "checkpoint is not a compatible neural_digits application run"
            )
        codec = digits.ArchitectureCodec()
        if strict_dumps(compatibility.get("codec")) != strict_dumps(
            {"id": codec.codec_id, "version": codec.version}
        ):
            raise CheckpointCompatibilityError(
                "Digits architecture codec does not match"
            )
        config = RunConfig.from_dict(as_object(body["config"], "saved configuration"))
        if config.sampling:
            raise CheckpointCompatibilityError(
                "saved configuration contains unresolved samples"
            )
        state = decode_state(body["state"], codec, store)
        if state.run_id != body["run_id"]:
            raise CheckpointCompatibilityError("checkpoint run identities disagree")
        config.validate_update(config.evolution, state.generation_settings)
        config.validate_update(config.evolution, state.effective_settings)
        best = state.best
        if best is None or best.fitness is None:
            raise CheckpointCompatibilityError(
                "finished run has no evaluated historical best"
            )

    settings = digits.DigitsSettings.from_parameters(config.parameters)
    data = digits.prepare_data(settings)
    current_fingerprint = digits.compatibility_fingerprint(settings, data)
    saved_fingerprint = application.get("requirements")
    if strict_dumps(saved_fingerprint) != strict_dumps(current_fingerprint):
        raise CheckpointCompatibilityError(
            "Digits report compatibility differs from the saved run: use the "
            "recorded application source, data splits, training settings, and "
            "numerical environment"
        )

    selected_architecture = best.genotype
    baseline_architecture = digits.Architecture((32,), "relu")
    selected_model = digits.train_model(selected_architecture, settings, data)
    selected_accuracy, selected_correct = digits.score_accuracy(
        selected_model, data.x_validation, data.y_validation
    )
    selected_validation = _accuracy_record(
        selected_accuracy, selected_correct, len(data.y_validation)
    )
    if selected_accuracy != best.fitness:
        raise CheckpointCompatibilityError(
            "fresh validation accuracy did not reproduce the checkpoint best "
            f"({selected_accuracy!r} versus {best.fitness!r}); check application "
            "compatibility and numerical determinism. Test data were not scored"
        )

    baseline_model = digits.train_model(baseline_architecture, settings, data)
    baseline_accuracy, baseline_correct = digits.score_accuracy(
        baseline_model, data.x_validation, data.y_validation
    )
    baseline_validation = _accuracy_record(
        baseline_accuracy, baseline_correct, len(data.y_validation)
    )

    # Both architectures and all fitting decisions are fixed before these calls.
    selected_test_accuracy, selected_test_correct = digits.score_accuracy(
        selected_model, data.x_test, data.y_test
    )
    baseline_test_accuracy, baseline_test_correct = digits.score_accuracy(
        baseline_model, data.x_test, data.y_test
    )
    return {
        "schema_version": 1,
        "report_type": "digits_held_out_evaluation",
        "run_id": state.run_id,
        "completed_generation": state.generation,
        "search": {
            "objective": "validation_accuracy",
            "accepted_evaluations": state.evaluations,
            "historical_best_id": best.id,
            "best_improvement_generation": state.last_improvement_generation,
            "saved_best_validation_accuracy": best.fitness,
            "stop_reason": state.stop.reason,
        },
        "partitions": {
            "training": len(data.y_train),
            "validation": len(data.y_validation),
            "test": len(data.y_test),
        },
        "training_settings": settings.to_dict(),
        "compatibility": current_fingerprint,
        "report_evaluations": {
            "fresh_model_fits": 2,
            "test_evaluations": 2,
            "training_partition": "original training split",
            "selection_uses_test": False,
            "recorded_in_evolutionary_history": False,
        },
        "selected": {
            "architecture": selected_architecture.to_dict(),
            "parameter_count": digits.parameter_count(selected_architecture),
            "validation": selected_validation,
            "test": _accuracy_record(
                selected_test_accuracy, selected_test_correct, len(data.y_test)
            ),
        },
        "baseline": {
            "architecture": baseline_architecture.to_dict(),
            "parameter_count": digits.parameter_count(baseline_architecture),
            "validation": baseline_validation,
            "test": _accuracy_record(
                baseline_test_accuracy, baseline_test_correct, len(data.y_test)
            ),
        },
    }


def _score_text(value: JSONValue) -> str:
    score = as_object(value, "report accuracy")
    # These typed fields were produced by _accuracy_record, not external input.
    percent = cast(float, score["percent"])
    return f"{score['correct']}/{score['total']} ({percent:.2f}%)"


def _render(report: JSONDict) -> str:
    search = as_object(report["search"], "report search")
    partitions = as_object(report["partitions"], "report partitions")
    lines = [
        "Digits MLP held-out report",
        f"Run {report['run_id']}; completed generation {report['completed_generation']}",
        f"Search objective: validation accuracy; {search['accepted_evaluations']} accepted evaluations",
        f"Historical best: individual {search['historical_best_id']}",
        "Original data splits: "
        f"{partitions['training']} training, {partitions['validation']} validation, "
        f"{partitions['test']} test",
        "",
    ]
    for key, label in (
        ("selected", "Selected architecture"),
        ("baseline", "Fixed baseline"),
    ):
        model = as_object(report[key], f"report {key}")
        architecture = as_object(model["architecture"], "report architecture")
        lines.extend(
            (
                f"{label}: {architecture['hidden_layers']} {architecture['activation']}; "
                f"{model['parameter_count']} trainable parameters",
                f"  Validation: {_score_text(model['validation'])}",
                f"  Held-out test: {_score_text(model['test'])}",
                "",
            )
        )
    lines.extend(
        (
            "The selected validation score reproduces the checkpoint's historical best.",
            "These are two fresh fits using the original fixed recipe and training seed.",
            "Each fixed architecture was scored on test data once; neither used test for selection.",
            "This report is a separate evaluation outside the evolutionary history.",
        )
    )
    return "\n".join(lines)


def _output_path(run_directory: Path, output: Path) -> Path:
    destination = output.resolve()
    if destination.is_relative_to(run_directory.resolve()):
        raise ConfigurationError("report output must be outside the run directory")
    if destination.is_dir():
        raise ConfigurationError("--output must name a JSON file, not a directory")
    return destination


def _write_json(destination: Path, report: JSONDict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".digits-report-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(strict_dumps(report) + "\n")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a readable report and optionally write strict JSON outside the run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--output", type=Path, help="JSON report file outside the run directory"
    )
    args = parser.parse_args(argv)
    try:
        destination = (
            None
            if args.output is None
            else _output_path(args.run_directory, args.output)
        )
        report = build_report(args.run_directory)
        if destination is not None:
            _write_json(destination, report)
        print(_render(report))
        if destination is not None:
            print(f"\nJSON report: {destination}")
        return 0
    except (XevolutionError, OSError) as error:
        print(f"digits_report: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("digits_report: report interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
