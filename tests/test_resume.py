"""Decisive continuation tests compare full scientific state in new processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from xevolution.config import strict_dumps
from xevolution.storage import RunStore

FIXTURES = Path(__file__).parent / "fixtures"
DRIVER = """
import sys
from xevolution import create_engine, resume
from resume_application import configuration
mode, path, pause = sys.argv[1:]
engine = create_engine(configuration(), output=path) if mode == 'fresh' else resume(path)
with engine:
    engine.run(until_generation=None if pause == '-' else int(pause))
"""


def execute(
    mode: str,
    directory: Path,
    pause: int | None = None,
    *,
    crash_generation: int | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(FIXTURES)
    environment["PYTHONHASHSEED"] = "17" if mode == "fresh" else "91"
    environment.pop("XEVO_TEST_CRASH_GENERATION", None)
    if crash_generation is not None:
        environment["XEVO_TEST_CRASH_GENERATION"] = str(crash_generation)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            DRIVER,
            mode,
            str(directory),
            "-" if pause is None else str(pause),
        ],
        cwd=directory.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def snapshot(directory: Path) -> tuple[dict, list[dict], dict[str, bytes]]:
    with RunStore(directory) as store:
        body = store.load()
        history = store.read_history()
    # Only operational run IDs differ. Every scientific field, payload ref,
    # RNG tuple/cache, component state, setting and journal event is compared.
    body.pop("run_id")
    body["state"].pop("run_id")
    for record in history:
        record.pop("run_id")
    artifacts = {
        path.name: path.read_bytes() for path in (directory / "artifacts").glob("*.bin")
    }
    return body, history, artifacts


@pytest.mark.parametrize("pause", [0, 3])
def test_independent_process_resume_matches_complete_uninterrupted_run(
    tmp_path: Path, pause: int
) -> None:
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    result = execute("fresh", uninterrupted)
    assert result.returncode == 0, result.stderr
    result = execute("fresh", interrupted, pause)
    assert result.returncode == 0, result.stderr
    paused, history, _ = snapshot(interrupted)
    assert paused["state"]["generation"] == pause
    assert paused["state"]["phase"] == "ready"
    assert len(history) == pause + 1
    if pause == 0:
        assert paused["random"]["evolution"][2] is not None
    result = execute("resume", interrupted)
    assert result.returncode == 0, result.stderr
    assert snapshot(interrupted) == snapshot(uninterrupted)
    body, history, artifacts = snapshot(interrupted)
    assert body["state"]["generation"] == 6
    assert body["state"]["phase"] == "terminated"
    assert len(history) == 7
    assert artifacts
    application = next(
        value
        for value in body["components"].values()
        if isinstance(value, dict) and "rng" in value
    )
    assert application["calls"]["initialize"] == 5
    assert application["calls"]["adapt"] == 7
    assert application["calls"]["evaluate"] == body["state"]["evaluations"]
    stopping = next(
        value
        for value in body["components"].values()
        if isinstance(value, dict) and "window" in value
    )
    assert stopping["calls"] == 7
    assert len(stopping["window"]) == 3


def test_abrupt_death_during_generation_does_not_commit_partial_work(
    tmp_path: Path,
) -> None:
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    result = execute("fresh", uninterrupted)
    assert result.returncode == 0, result.stderr
    result = execute("fresh", interrupted, crash_generation=2)
    assert result.returncode == 73, result.stderr
    body, history, _ = snapshot(interrupted)
    assert body["state"]["generation"] == 1
    assert len(history) == 2
    # Simulate a complete-looking but uncommitted tail independently of the
    # interrupted evaluation. Writable recovery must remove it before append.
    with (interrupted / "history.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":"partial next generation"}\n')
    result = execute("resume", interrupted)
    assert result.returncode == 0, result.stderr
    assert snapshot(interrupted) == snapshot(uninterrupted)


def test_terminal_resume_is_read_only_for_scientific_files(tmp_path: Path) -> None:
    directory = tmp_path / "terminal"
    result = execute("fresh", directory)
    assert result.returncode == 0, result.stderr
    before = {
        name: (directory / name).read_bytes()
        for name in ("checkpoint.json", "history.jsonl", "config.json")
    }
    result = execute("resume", directory)
    assert result.returncode == 0, result.stderr
    assert before == {name: (directory / name).read_bytes() for name in before}


def test_resume_rejects_changed_application_compatibility(tmp_path: Path) -> None:
    directory = tmp_path / "incompatible"
    result = execute("fresh", directory, 0)
    assert result.returncode == 0, result.stderr
    # Create a self-consistent envelope carrying incompatible application data;
    # this is distinct from testing byte corruption (covered by storage tests).
    checkpoint = directory / "checkpoint.json"
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["body"]["compatibility"]["application"]["version"] = "different"
    import hashlib

    envelope["body_sha256"] = hashlib.sha256(
        strict_dumps(envelope["body"]).encode("utf-8")
    ).hexdigest()
    checkpoint.write_text(strict_dumps(envelope) + "\n", encoding="utf-8")
    result = execute("resume", directory)
    assert result.returncode != 0
    assert "compatibility mismatch" in result.stderr
