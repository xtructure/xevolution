"""Command-line run, resume, inspection, and scientific report projections."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from xevolution import __version__
from xevolution.config import as_object, strict_dumps
from xevolution.engine import create_engine, resume
from xevolution.errors import XevolutionError
from xevolution.models import JSONDict, RunState
from xevolution.storage import RunStore
from xevolution.tracking import analyze


def _nonnegative_integer(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xevolution", description="Generic, headless evolutionary computation."
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--verbose", action="store_true", help="show diagnostic logging"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("run", help="create and execute a fresh run")
    start.add_argument("--config", type=Path, required=True, help="JSON configuration")
    start.add_argument("--output", type=Path, required=True, help="fresh run directory")

    restore = commands.add_parser("resume", help="continue the recorded checkpoint")
    restore.add_argument("run_directory", type=Path)
    for command in (start, restore):
        command.add_argument(
            "--pause-after-generation",
            type=_nonnegative_integer,
            metavar="N",
            help="pause at completed generation N without changing stopping policies",
        )

    inspect = commands.add_parser("inspect", help="read committed checkpoint metadata")
    inspect.add_argument("run_directory", type=Path)
    report = commands.add_parser("analyze", help="export committed scientific history")
    report.add_argument("run_directory", type=Path)
    report.add_argument(
        "--output", type=Path, help="report directory (default: RUN_DIR/reports)"
    )
    return parser


@dataclass
class _PauseRequest:
    """Delay the first operational interrupt until the next generation boundary."""

    requested: bool = False
    callback: Callable[[], None] | None = None

    def attach(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        if self.requested:
            callback()

    def handle(self, signum: int, frame: FrameType | None) -> None:
        if self.requested:
            # A second request remains an immediate escape from an application
            # evaluation that does not return. No partial generation is committed.
            raise KeyboardInterrupt
        self.requested = True
        if self.callback is not None:
            self.callback()


@contextmanager
def _pause_signals(request: _PauseRequest) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request.handle)
    signal.signal(signal.SIGTERM, request.handle)
    try:
        yield
    finally:
        if previous_int is not None:
            signal.signal(signal.SIGINT, previous_int)
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


def _run_summary(state: RunState[Any]) -> JSONDict:
    """Summarize a dynamically resolved application's type-erased run state."""
    return {
        "run_id": state.run_id,
        "generation": state.generation,
        "phase": state.phase,
        "population_size": len(state.population),
        "evaluations": state.evaluations,
        "best_id": None if state.best is None else state.best.id,
        "best_fitness": None if state.best is None else state.best.fitness,
        "stop_reason": state.stop.reason,
    }


def _execute(args: argparse.Namespace) -> int:
    request = _PauseRequest()
    with _pause_signals(request):
        engine = (
            create_engine(args.config, output=args.output)
            if args.command == "run"
            else resume(args.run_directory)
        )
        try:
            request.attach(engine.request_pause)
            state = engine.run(until_generation=args.pause_after_generation)
            summary = _run_summary(state)
        finally:
            engine.close()
    print(strict_dumps(summary))
    return 0


def _inspect(path: Path) -> int:
    with RunStore(path) as store:
        body = store.load()
    state = as_object(body["state"], "checkpoint state")
    summary: JSONDict = {
        "run_id": body["run_id"],
        "checkpoint": str(path / "checkpoint.json"),
        "config": body["config"],
        "compatibility": body["compatibility"],
    }
    for name in (
        "generation",
        "phase",
        "evaluations",
        "next_individual_id",
        "next_event_sequence",
        "generation_settings",
        "effective_settings",
        "statistics",
        "best",
        "last_improvement_generation",
        "stop",
    ):
        summary[name] = state.get(name)
    print(strict_dumps(summary))
    return 0


def _write_report(path: Path, report: JSONDict) -> None:
    """Replace a rebuildable report without exposing a partial JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".analysis-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(strict_dumps(report) + "\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _analyze(path: Path, output: Path | None) -> int:
    with RunStore(path) as store:
        records = store.read_history()
    report = analyze(records)
    destination = (output if output is not None else path / "reports") / "analysis.json"
    _write_report(destination, report)
    print(
        strict_dumps(
            {
                "report": str(destination),
                "generation_count": report["generation_count"],
                "latest_generation": report["latest_generation"],
            }
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return an exit code; only entry points exit the process."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        if args.command in {"run", "resume"}:
            return _execute(args)
        if args.command == "inspect":
            return _inspect(args.run_directory)
        return _analyze(args.run_directory, args.output)
    except (XevolutionError, OSError) as error:
        print(f"xevolution: {error}", file=sys.stderr)
        if args.verbose:
            logging.getLogger(__name__).exception("Command failed")
        return 1
    except KeyboardInterrupt:
        print(
            "xevolution: interrupted; resume will validate the last published checkpoint",
            file=sys.stderr,
        )
        return 130
