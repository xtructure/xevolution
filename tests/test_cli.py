"""Exercise public commands in fresh Python processes and on real run files."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def invoke(*arguments: str, cwd: Path = PROJECT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "xevolution", *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def config_file(tmp_path: Path, *, example: str = "batch_problem") -> Path:
    document = json.loads(
        (PROJECT / "examples" / f"{example}.json").read_text(encoding="utf-8")
    )
    document["evolution"]["population_size"] = 8
    document["termination"] = {"max_generations": 3}
    path = tmp_path / "parameters.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def successful(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_run_pause_read_only_commands_and_resume(tmp_path: Path) -> None:
    config = config_file(tmp_path)
    directory = tmp_path / "run"
    started = successful(
        invoke(
            "run",
            "--config",
            str(config),
            "--output",
            str(directory),
            "--pause-after-generation",
            "0",
        )
    )
    assert started["generation"] == 0
    assert started["phase"] == "ready"
    assert started["evaluations"] == 8

    # Read-only commands must use the committed prefix without repairing a tail.
    checkpoint_before = (directory / "checkpoint.json").read_bytes()
    history = directory / "history.jsonl"
    with history.open("ab") as stream:
        stream.write(b'{"uncommitted":')
    history_before = history.read_bytes()
    # The example factory is unavailable outside the checkout, so these also
    # verify that inspection and reporting do not reconstruct application code.
    inspected = successful(invoke("inspect", str(directory), cwd=tmp_path))
    assert inspected["generation"] == 0
    assert inspected["phase"] == "ready"
    assert inspected["best"] is not None

    reports = tmp_path / "reports"
    successful(
        invoke("analyze", str(directory), "--output", str(reports), cwd=tmp_path)
    )
    report_before = (reports / "analysis.json").read_bytes()
    successful(
        invoke("analyze", str(directory), "--output", str(reports), cwd=tmp_path)
    )
    assert (reports / "analysis.json").read_bytes() == report_before
    assert json.loads(report_before)["generation_count"] == 1
    assert (directory / "checkpoint.json").read_bytes() == checkpoint_before
    assert history.read_bytes() == history_before

    resumed = successful(invoke("resume", str(directory)))
    assert resumed["run_id"] == started["run_id"]
    assert resumed["generation"] == 3
    assert resumed["phase"] == "terminated"
    assert resumed["evaluations"] == 32
    committed_history = history.read_bytes()
    assert b"uncommitted" not in committed_history
    checkpoint_after = (directory / "checkpoint.json").read_bytes()

    # A terminal run must not reevaluate, append history, or replace its checkpoint.
    assert successful(invoke("resume", str(directory))) == resumed
    assert history.read_bytes() == committed_history
    assert (directory / "checkpoint.json").read_bytes() == checkpoint_after
    successful(invoke("analyze", str(directory)))
    report = json.loads(
        (directory / "reports" / "analysis.json").read_text(encoding="utf-8")
    )
    assert report["generation_count"] == 4
    assert report["latest_generation"] == 3
    assert report["individuals"]
    assert report["edges"]


def test_fresh_run_refuses_existing_output(tmp_path: Path) -> None:
    config = config_file(tmp_path)
    output = tmp_path / "run"
    successful(invoke("run", "--config", str(config), "--output", str(output)))
    before = (output / "checkpoint.json").read_bytes()
    result = invoke("run", "--config", str(config), "--output", str(output))
    assert result.returncode != 0
    assert result.stderr.strip()
    assert (output / "checkpoint.json").read_bytes() == before


def test_arithmetic_example_uses_the_public_cli(tmp_path: Path) -> None:
    config = config_file(tmp_path, example="arithmetic")
    output = tmp_path / "arithmetic"
    result = successful(invoke("run", "--config", str(config), "--output", str(output)))
    assert result["generation"] == 3
    assert 0 <= result["best_fitness"] <= 1
    inspected = successful(invoke("inspect", str(output)))
    assert inspected["config"]["experiment"]["factory"] == (
        "examples.arithmetic:build_experiment"
    )


def test_batch_example_has_joint_fitness_and_stable_result_ids() -> None:
    program = textwrap.dedent(
        """
        from examples.batch_problem import SharedResources
        from xevolution.models import Individual, Population
        from xevolution.randomness import RunRandom
        problem = SharedResources((6.0, 2.0))
        context = RunRandom(4).context(0, 'joint-test')
        single = Population((Individual(1, 0),))
        shared = Population((Individual(1, 0), Individual(2, 0)))
        assert problem.evaluate_population(single, context)[0][1].fitness == 6.0
        results = problem.evaluate_population(shared, context)
        assert [item[0] for item in results] == [2, 1]
        assert all(result.fitness == 3.0 for _, result in results)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "document", ['{"schema_version":1,"schema_version":1}', '{"seed":1e999}']
)
def test_bad_configuration_is_a_command_error(tmp_path: Path, document: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(document, encoding="utf-8")
    result = invoke("run", "--config", str(path), "--output", str(tmp_path / "run"))
    assert result.returncode != 0
    assert "xevolution:" in result.stderr
    assert not (tmp_path / "run" / "checkpoint.json").exists()


def test_entry_points_work_outside_the_checkout(tmp_path: Path) -> None:
    module_result = invoke("--version", cwd=tmp_path)
    assert module_result.returncode == 0, module_result.stderr
    executable = Path(sys.executable).parent / (
        "xevolution.exe" if os.name == "nt" else "xevolution"
    )
    script_result = subprocess.run(
        [str(executable), "--version"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert script_result.returncode == 0, script_result.stderr
    assert module_result.stdout == script_result.stdout


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal/pipe coordination")
def test_sigint_finishes_generation_zero_before_pausing(tmp_path: Path) -> None:
    # A pipe synchronizes the test with evaluation; no timing sleeps or flaky
    # polling are needed. The first signal arrives before create_engine returns.
    (tmp_path / "pause_application.py").write_text(
        textwrap.dedent(
            """
            import sys
            from xevolution.experiment import Experiment
            from xevolution.models import EvaluationResult, VariationResult
            from xevolution.operators import OperatorSpec
            def initialize(rng):
                return 'payload'
            def clone(value):
                return value
            def mutate(value, rng):
                return VariationResult(value)
            def evaluate_population(population, context):
                print('EVALUATION_READY', file=sys.stderr, flush=True)
                sys.stdin.readline()
                return [(individual.id, EvaluationResult(1.0)) for individual in population]
            def build_experiment(config, random):
                return Experiment(initialize=initialize, clone=clone,
                    evaluate_population=evaluate_population,
                    mutations=(OperatorSpec('copy', mutate),),
                    identity='pause-test', version='1')
            """
        ),
        encoding="utf-8",
    )
    config = tmp_path / "pause.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 2,
                "experiment": {"factory": "pause_application:build_experiment"},
                "evolution": {"population_size": 4, "mutation_probability": 1.0},
                "termination": {"max_generations": 3},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xevolution",
            "run",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr, selectors.EVENT_READ)
            assert selector.select(timeout=15), "application did not reach evaluation"
        line = process.stderr.readline()
        assert line.strip() == "EVALUATION_READY", line
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(input="\n", timeout=15)
        assert process.returncode == 0, stderr
        summary = json.loads(stdout)
        assert summary["generation"] == 0
        assert summary["phase"] == "ready"
        assert summary["evaluations"] == 4
        assert (output / "checkpoint.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
