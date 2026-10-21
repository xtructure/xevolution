"""Single-writer generation commits, verified journal recovery, and attachments.

The active checkpoint defines the committed history prefix. A read-only reader
never repairs files; a writer acquires an OS lock and removes uncommitted tail
before appending. Local POSIX filesystems receive file and directory fsyncs;
other platforms retain atomic publication and process-crash recovery, without
claiming power-loss durability for directory changes.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

from .config import as_object, strict_dumps, strict_loads, validate_json
from .errors import CheckpointCompatibilityError, CheckpointError, ConfigurationError
from .models import GenerationRecord, JSONDict, JSONValue

CHECKPOINT_SCHEMA_VERSION = 1
ATTACHMENT_MARKER = "__xevolution_attachment__"


def _object(value: JSONValue, label: str) -> JSONDict:
    try:
        return as_object(value, label)
    except ConfigurationError as exc:
        raise CheckpointError(str(exc)) from exc


def _keys(data: JSONDict, expected: set[str], label: str) -> None:
    if data.keys() != expected:
        raise CheckpointError(f"Invalid fields in {label}")


def _integer(value: JSONValue, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CheckpointError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: JSONValue, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointError(f"{label} must be a nonempty string")
    return value


def _digest(value: JSONValue, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CheckpointError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _json_bytes(value: JSONValue) -> bytes:
    return strict_dumps(value).encode("utf-8")


def _read_json(path: Path) -> JSONDict:
    try:
        return _object(strict_loads(path.read_text(encoding="utf-8")), path.name)
    except (OSError, UnicodeError, ConfigurationError) as exc:
        raise CheckpointError(f"Cannot read {path.name}: {exc}") from exc


def _sync_directory(path: Path) -> None:
    """Synchronize local POSIX directory entries; fail if that guarantee fails."""
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _lock(stream: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(stream: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RunStore:
    """Own a run's files without constructing application code.

    ``create`` and ``load(writable=True)`` retain an exclusive advisory lock
    until ``close``. Entering a context alone and read-only methods create no
    files. After any failed write, close and reopen the store to discover the
    authoritative checkpoint before continuing; replacement may already have
    published the new checkpoint even when its final synchronization failed.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._writer: BinaryIO | None = None
        self._failed = False
        self._run_id: str | None = None
        self._offset = 0
        self._next_sequence = 0
        self._generation = -1
        self._journal_hash = hashlib.sha256()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release writer ownership; this never deletes checkpoint or lock files."""
        stream = self._writer
        self._writer = None
        if stream is not None:
            try:
                _unlock(stream)
            finally:
                stream.close()
        self._failed = False
        self._run_id = None

    def _acquire(self) -> None:
        if self._writer is not None:
            raise CheckpointError("This store already owns a writer session")
        try:
            stream = (self.path / ".writer.lock").open("a+b")
        except OSError as exc:
            raise CheckpointError(
                f"Cannot open writer lock in {self.path}: {exc}"
            ) from exc
        try:
            _lock(stream)
        except OSError as exc:
            stream.close()
            raise CheckpointError(
                f"Run directory already has a writer: {self.path}"
            ) from exc
        self._writer = stream
        self._failed = False

    def _require_writer(self) -> None:
        if self._writer is None or self._run_id is None:
            raise CheckpointError("Create or load a writable run before saving")
        if self._failed:
            raise CheckpointError(
                "Previous write failed; close and reopen this run to recover"
            )

    def create(self, config: JSONDict, run_id: str) -> None:
        """Create an empty run directory and retain its writer lock.

        A directory containing only a prior lock file is empty for this purpose;
        all other existing contents are protected from replacement.
        """
        if self._writer is not None:
            raise CheckpointError("This store already owns a writer session")
        try:
            resolved = _object(validate_json(config), "configuration")
            identity = _text(run_id, "run id")
            missing_directories: list[Path] = []
            candidate = self.path
            while not candidate.exists():
                missing_directories.append(candidate)
                candidate = candidate.parent
            self.path.mkdir(parents=True, exist_ok=True)
            for directory in reversed(missing_directories):
                _sync_directory(directory.parent)
            self._acquire()
            if any(entry.name != ".writer.lock" for entry in self.path.iterdir()):
                raise CheckpointError(f"Run directory is not empty: {self.path}")
            _atomic_write(self.path / "config.json", _json_bytes(resolved) + b"\n")
            with (self.path / "history.jsonl").open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(self.path)
            _sync_directory(self.path.parent)
            self._run_id = identity
            self._offset = 0
            self._next_sequence = 0
            self._generation = -1
            self._journal_hash = hashlib.sha256()
        except (OSError, ConfigurationError) as exc:
            self.close()
            raise CheckpointError(f"Cannot create run directory: {exc}") from exc
        except CheckpointError:
            self.close()
            raise

    def _validate_body(
        self, body: JSONDict, *, run_id: str, generation: int, next_sequence: int
    ) -> None:
        _keys(
            body,
            {"run_id", "config", "state", "random", "components", "compatibility"},
            "checkpoint body",
        )
        for name in ("config", "random", "components", "compatibility"):
            _object(body[name], name)
        if body.get("run_id") != run_id:
            raise CheckpointError("Checkpoint body and journal run identities disagree")
        state = _object(body.get("state"), "checkpoint state")
        if (
            state.get("run_id") != run_id
            or _integer(state.get("generation"), "state generation") != generation
            or _integer(state.get("next_event_sequence"), "state event sequence")
            != next_sequence
        ):
            raise CheckpointError("Checkpoint state and committed journal disagree")

    def _validate_record(
        self, record: JSONDict, *, run_id: str, generation: int, sequence: int
    ) -> None:
        if (
            record.get("type") != "generation"
            or _integer(record.get("schema_version"), "record schema", minimum=1) != 1
            or record.get("run_id") != run_id
            or _integer(record.get("generation"), "record generation") != generation
            or _integer(record.get("sequence"), "record sequence") != sequence
        ):
            raise CheckpointError(
                "Invalid run identity or ordering in committed history"
            )

    def commit(self, body: JSONDict, record: GenerationRecord) -> None:
        """Durably publish one finalized generation after its journal and attachments.

        Validation precedes mutation. A write failure poisons this writer session
        because the checkpoint might already be visible; reopening validates and
        recovers whichever complete checkpoint is present.
        """
        self._require_writer()
        run_id = self._run_id
        if run_id is None:
            raise CheckpointError("Writer has no run identity")
        try:
            checkpoint_body = _object(validate_json(body), "checkpoint body")
            payload = _object(strict_loads(record.data_json), "generation record")
            generation = self._generation + 1
            if (
                type(record.generation) is not int
                or type(record.sequence) is not int
                or record.generation != generation
                or record.sequence != self._next_sequence
            ):
                raise CheckpointError(
                    "A commit must contain the next generation exactly once"
                )
            self._validate_record(
                payload,
                run_id=run_id,
                generation=generation,
                sequence=self._next_sequence,
            )
            self._validate_body(
                checkpoint_body,
                run_id=run_id,
                generation=generation,
                next_sequence=self._next_sequence + 1,
            )
            self._validate_attachments(checkpoint_body)
            line = _json_bytes(payload) + b"\n"
            next_hash = self._journal_hash.copy()
            next_hash.update(line)
            next_offset = self._offset + len(line)
            envelope: JSONDict = {
                "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
                "body": checkpoint_body,
                "body_sha256": hashlib.sha256(_json_bytes(checkpoint_body)).hexdigest(),
                "journal": {
                    "run_id": run_id,
                    "offset": next_offset,
                    "sha256": next_hash.hexdigest(),
                    "next_sequence": self._next_sequence + 1,
                    "generation": generation,
                },
            }
            encoded_checkpoint = _json_bytes(envelope) + b"\n"
        except ConfigurationError as exc:
            raise CheckpointError(f"Invalid checkpoint encoding: {exc}") from exc
        try:
            with (self.path / "history.jsonl").open("r+b") as stream:
                stream.seek(0, os.SEEK_END)
                if stream.tell() != self._offset:
                    raise CheckpointError("Journal changed since writer recovery")
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_write(self.path / "checkpoint.json", encoded_checkpoint)
        except (OSError, CheckpointError) as exc:
            self._failed = True
            raise CheckpointError(
                "Generation commit failed; close and reopen to recover the active checkpoint. "
                f"Publication may have completed before final synchronization: {exc}"
            ) from exc
        self._offset = next_offset
        self._generation = generation
        self._next_sequence += 1
        self._journal_hash = next_hash

    def _snapshot(self) -> tuple[JSONDict, JSONDict, bytes, list[JSONDict]]:
        envelope = _read_json(self.path / "checkpoint.json")
        _keys(
            envelope,
            {"checkpoint_schema", "body", "body_sha256", "journal"},
            "checkpoint",
        )
        schema = _integer(envelope["checkpoint_schema"], "checkpoint schema", minimum=1)
        if schema != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"Unsupported checkpoint schema {schema}"
            )
        body = _object(envelope["body"], "checkpoint body")
        expected_body_digest = _digest(envelope["body_sha256"], "body digest")
        if hashlib.sha256(_json_bytes(body)).hexdigest() != expected_body_digest:
            raise CheckpointError("Checkpoint body integrity verification failed")
        journal = _object(envelope["journal"], "journal position")
        _keys(
            journal,
            {"run_id", "offset", "sha256", "next_sequence", "generation"},
            "journal position",
        )
        run_id = _text(journal["run_id"], "journal run id")
        offset = _integer(journal["offset"], "committed byte offset", minimum=1)
        next_sequence = _integer(
            journal["next_sequence"], "next event sequence", minimum=1
        )
        generation = _integer(journal["generation"], "completed generation")
        if next_sequence != generation + 1:
            raise CheckpointError(
                "Checkpoint must commit every generation beginning at zero"
            )
        expected_digest = _digest(journal["sha256"], "journal digest")
        self._validate_body(
            body, run_id=run_id, generation=generation, next_sequence=next_sequence
        )
        try:
            with (self.path / "history.jsonl").open("rb") as stream:
                prefix = stream.read(offset)
        except OSError as exc:
            raise CheckpointError(f"Cannot read committed history: {exc}") from exc
        if len(prefix) != offset or not prefix.endswith(b"\n"):
            raise CheckpointError("Committed history is missing or truncated")
        if hashlib.sha256(prefix).hexdigest() != expected_digest:
            raise CheckpointError("Committed history integrity verification failed")
        lines = prefix.splitlines()
        if len(lines) != next_sequence:
            raise CheckpointError("Committed history count differs from checkpoint")
        records: list[JSONDict] = []
        try:
            for sequence, line in enumerate(lines):
                record = _object(
                    strict_loads(line.decode("utf-8")), "generation record"
                )
                self._validate_record(
                    record, run_id=run_id, generation=sequence, sequence=sequence
                )
                records.append(record)
        except (UnicodeError, ConfigurationError) as exc:
            raise CheckpointError(f"Invalid committed history: {exc}") from exc
        self._validate_attachments(body)
        return body, journal, prefix, records

    def load(self, *, writable: bool = False) -> JSONDict:
        """Verify and return the active body, optionally preparing an exclusive writer.

        Writable recovery removes every byte past the committed offset and
        synchronizes that truncation before allowing another append. Missing or
        corrupt committed bytes are errors in both modes.
        """
        if writable:
            self._acquire()
        try:
            body, journal, prefix, _ = self._snapshot()
            if writable:
                with (self.path / "history.jsonl").open("r+b") as stream:
                    stream.truncate(len(prefix))
                    stream.flush()
                    os.fsync(stream.fileno())
                self._run_id = _text(journal["run_id"], "journal run id")
                self._offset = len(prefix)
                self._next_sequence = _integer(
                    journal["next_sequence"], "event sequence"
                )
                self._generation = _integer(journal["generation"], "generation")
                self._journal_hash = hashlib.sha256(prefix)
            return body
        except OSError as exc:
            if writable:
                self.close()
            raise CheckpointError(f"Cannot recover writable history: {exc}") from exc
        except CheckpointError:
            if writable:
                self.close()
            raise

    def read_history(self) -> list[JSONDict]:
        """Read one verified committed prefix without touching files or app objects."""
        return self._snapshot()[3]

    def put(
        self, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> JSONDict:
        """Durably save immutable bytes and return a portable integrity reference."""
        self._require_writer()
        if not isinstance(data, bytes):
            raise CheckpointError("Attachments must be bytes")
        if not isinstance(media_type, str) or not media_type:
            raise CheckpointError("Attachment media type must be a nonempty string")
        digest = hashlib.sha256(data).hexdigest()
        reference: JSONDict = {
            ATTACHMENT_MARKER: {
                "version": 1,
                "sha256": digest,
                "size": len(data),
                "media_type": media_type,
            }
        }
        directory = self.path / "artifacts"
        target = directory / f"{digest}.bin"
        try:
            if directory.is_symlink():
                raise CheckpointError(
                    "Attachment directory must not be a symbolic link"
                )
            if target.exists() or target.is_symlink():
                self.get(reference)
                return reference
            if not directory.exists():
                directory.mkdir()
                _sync_directory(self.path)
            _atomic_write(target, data)
        except (OSError, CheckpointError) as exc:
            self._failed = True
            raise CheckpointError(f"Attachment write failed: {exc}") from exc
        return reference

    def get(self, reference: JSONValue) -> bytes:
        """Read a referenced attachment only after size and SHA-256 verification."""
        value = _object(reference, "attachment reference")
        _keys(value, {ATTACHMENT_MARKER}, "attachment reference")
        metadata = _object(value[ATTACHMENT_MARKER], "attachment metadata")
        _keys(
            metadata, {"version", "sha256", "size", "media_type"}, "attachment metadata"
        )
        version = _integer(metadata["version"], "attachment version", minimum=1)
        if version != 1:
            raise CheckpointCompatibilityError(
                f"Unsupported attachment version {version}"
            )
        digest = _digest(metadata["sha256"], "attachment digest")
        size = _integer(metadata["size"], "attachment size")
        _text(metadata["media_type"], "attachment media type")
        directory = self.path / "artifacts"
        target = directory / f"{digest}.bin"
        try:
            if directory.is_symlink() or target.is_symlink():
                raise CheckpointError("Required attachments must not be symbolic links")
            with target.open("rb") as stream:
                if os.fstat(stream.fileno()).st_size != size:
                    raise CheckpointError(f"Attachment size mismatch: {digest}")
                data = stream.read()
        except OSError as exc:
            raise CheckpointError(
                f"Cannot read required attachment {digest}: {exc}"
            ) from exc
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise CheckpointError(f"Attachment integrity verification failed: {digest}")
        return data

    def _validate_attachments(self, value: JSONValue) -> None:
        if isinstance(value, dict):
            if ATTACHMENT_MARKER in value:
                self.get(value)
            else:
                for nested in value.values():
                    self._validate_attachments(nested)
        elif isinstance(value, list):
            for nested in value:
                self._validate_attachments(nested)
