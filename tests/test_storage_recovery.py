"""Real-file commit, recovery, explicit codec, and writer ownership contracts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from xevolution import storage
from xevolution.config import (
    EvolutionSettings,
    RunConfig,
    as_object,
    strict_dumps,
    strict_loads,
)
from xevolution.errors import (
    CheckpointCompatibilityError,
    CheckpointError,
    ConfigurationError,
)
from xevolution.models import (
    Individual,
    JSONDict,
    JSONValue,
    LineageRecord,
    Population,
    RunState,
)
from xevolution.protocols import AttachmentReader, AttachmentWriter, PayloadCodec
from xevolution.serialization import JsonCodec, decode_state, encode_state
from xevolution.storage import RunStore
from xevolution.tracking import make_record, update_statistics


def state_at(generation: int = 0) -> RunState[JSONValue]:
    settings = EvolutionSettings(population_size=1, mutation_probability=1.0)
    item = Individual[JSONValue](
        id=generation,
        genotype={"value": generation},
        birth_generation=generation,
        parent_ids=() if generation == 0 else (generation - 1,),
        fitness=float(generation),
        evaluation_count=1,
    )
    state = RunState(
        run_id="storage-test",
        generation=generation,
        population=Population((item,)),
        generation_settings=settings,
        effective_settings=settings,
        next_individual_id=generation + 1,
        next_event_sequence=generation + 1,
        evaluations=generation + 1,
    )
    return update_statistics(
        state, lambda value: JsonCodec().decode(JsonCodec().encode(value))
    )


def checkpoint_body[G](
    state: RunState[G], codec: PayloadCodec[G], writer: AttachmentWriter | None = None
) -> JSONDict:
    return {
        "run_id": state.run_id,
        "config": RunConfig(evolution=state.generation_settings).to_dict(),
        "state": encode_state(state, codec, writer),
        "random": {},
        "components": {},
        "compatibility": {},
    }


def commit_generation(store: RunStore, generation: int) -> JSONDict:
    state = state_at(generation)
    body = checkpoint_body(state, JsonCodec())
    birth = LineageRecord(
        generation,
        generation,
        () if generation == 0 else (generation - 1,),
        "initialize" if generation == 0 else "mutate",
    )
    store.commit(body, make_record(state, (birth,)))
    return body


def test_initial_generation_round_trip_and_read_only_lifetime(tmp_path: Path) -> None:
    path = tmp_path / "run"
    with RunStore(path) as store:
        assert not path.exists()
        store.create({}, "storage-test")
        body = commit_generation(store, 0)
    before = {
        entry.name: entry.read_bytes() for entry in path.iterdir() if entry.is_file()
    }
    with RunStore(path) as reader:
        restored = reader.load()
        records = reader.read_history()
    assert restored == body
    assert [record["generation"] for record in records] == [0]
    state = decode_state(restored["state"], JsonCodec())
    assert encode_state(state, JsonCodec()) == body["state"]
    assert {
        entry.name: entry.read_bytes() for entry in path.iterdir() if entry.is_file()
    } == before


def test_read_only_missing_run_creates_nothing(tmp_path: Path) -> None:
    path = tmp_path / "missing"
    with pytest.raises(CheckpointError, match="Cannot read checkpoint"):
        RunStore(path).load()
    assert not path.exists()


def test_create_refuses_populated_directory(tmp_path: Path) -> None:
    existing = tmp_path / "keep.txt"
    existing.write_text("user data", encoding="utf-8")
    with RunStore(tmp_path) as store:
        with pytest.raises(CheckpointError, match="not empty"):
            store.create({}, "storage-test")
    assert existing.read_text(encoding="utf-8") == "user data"


def test_recovery_truncates_tail_before_next_append(tmp_path: Path) -> None:
    with RunStore(tmp_path) as writer:
        writer.create({}, "storage-test")
        commit_generation(writer, 0)
    journal = tmp_path / "history.jsonl"
    prefix = journal.read_bytes()
    tail = b'{"partial":true}\n\xffbroken'
    with journal.open("ab") as stream:
        stream.write(tail)
    assert len(RunStore(tmp_path).read_history()) == 1
    assert journal.read_bytes() == prefix + tail
    with RunStore(tmp_path) as writer:
        writer.load(writable=True)
        assert journal.read_bytes() == prefix
        commit_generation(writer, 1)
    history = RunStore(tmp_path).read_history()
    assert [record["generation"] for record in history] == [0, 1]
    assert [record["sequence"] for record in history] == [0, 1]


@pytest.mark.parametrize("damage", ["missing", "truncated", "changed"])
def test_corrupt_committed_history_is_not_repaired(tmp_path: Path, damage: str) -> None:
    with RunStore(tmp_path) as writer:
        writer.create({}, "storage-test")
        commit_generation(writer, 0)
    journal = tmp_path / "history.jsonl"
    data = journal.read_bytes()
    if damage == "missing":
        journal.unlink()
    elif damage == "truncated":
        journal.write_bytes(data[:-4])
    else:
        journal.write_bytes(data.replace(b'"fitness":0.0', b'"fitness":9.0', 1))
    before = journal.read_bytes() if journal.exists() else None
    for writable in (False, True):
        with RunStore(tmp_path) as reader:
            with pytest.raises(CheckpointError):
                reader.load(writable=writable)
    assert (journal.read_bytes() if journal.exists() else None) == before


@pytest.mark.parametrize("failure_point", ["journal_fsync", "checkpoint_replace"])
def test_failed_commit_recovers_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    writer = RunStore(tmp_path)
    writer.create({}, "storage-test")
    first = commit_generation(writer, 0)
    original_fsync = os.fsync
    original_replace = os.replace

    def fail_fsync(descriptor: int) -> None:
        raise OSError("injected journal fsync failure")

    def fail_replace(source: str | Path, target: str | Path) -> None:
        if Path(target).name == "checkpoint.json":
            raise OSError("injected checkpoint publication failure")
        original_replace(source, target)

    if failure_point == "journal_fsync":
        monkeypatch.setattr(storage.os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(CheckpointError, match="commit failed"):
        commit_generation(writer, 1)
    with pytest.raises(CheckpointError, match="Previous write failed"):
        commit_generation(writer, 1)
    monkeypatch.setattr(storage.os, "fsync", original_fsync)
    monkeypatch.setattr(storage.os, "replace", original_replace)
    writer.close()
    assert RunStore(tmp_path).load() == first
    with RunStore(tmp_path) as recovered:
        recovered.load(writable=True)
        commit_generation(recovered, 1)
    assert len(RunStore(tmp_path).read_history()) == 2


def test_failure_after_replacement_does_not_claim_old_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = RunStore(tmp_path)
    writer.create({}, "storage-test")
    commit_generation(writer, 0)
    original = storage._sync_directory

    def fail_directory_sync(path: Path) -> None:
        raise OSError("injected post-publication directory sync failure")

    monkeypatch.setattr(storage, "_sync_directory", fail_directory_sync)
    with pytest.raises(CheckpointError, match="Publication may have completed"):
        commit_generation(writer, 1)
    monkeypatch.setattr(storage, "_sync_directory", original)
    writer.close()
    with RunStore(tmp_path) as recovered:
        body = recovered.load(writable=True)
        assert as_object(body["state"])["generation"] == 1
        commit_generation(recovered, 2)
    assert len(RunStore(tmp_path).read_history()) == 3


def test_checkpoint_schema_and_body_integrity_are_verified(tmp_path: Path) -> None:
    with RunStore(tmp_path) as writer:
        writer.create({}, "storage-test")
        commit_generation(writer, 0)
    path = tmp_path / "checkpoint.json"
    original = path.read_text(encoding="utf-8")
    envelope = as_object(strict_loads(original))
    envelope["checkpoint_schema"] = 999
    path.write_text(strict_dumps(envelope), encoding="utf-8")
    with pytest.raises(CheckpointCompatibilityError, match="schema"):
        RunStore(tmp_path).load()
    envelope = as_object(strict_loads(original))
    as_object(envelope["body"])["run_id"] = "different"
    path.write_text(strict_dumps(envelope), encoding="utf-8")
    with pytest.raises(CheckpointError, match="body integrity"):
        RunStore(tmp_path).load()


def test_writer_lock_excludes_another_store_and_process(tmp_path: Path) -> None:
    with RunStore(tmp_path) as owner:
        owner.create({}, "storage-test")
        commit_generation(owner, 0)
        with pytest.raises(CheckpointError, match="already has a writer"):
            RunStore(tmp_path).load(writable=True)
        script = """import sys
from xevolution.storage import RunStore
from xevolution.errors import CheckpointError
try:
    RunStore(sys.argv[1]).load(writable=True)
except CheckpointError:
    sys.exit(0)
sys.exit(1)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert RunStore(tmp_path).load()["run_id"] == "storage-test"
    script = """import os, sys
from xevolution.storage import RunStore
store = RunStore(sys.argv[1])
store.load(writable=True)
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with RunStore(tmp_path) as successor:
        successor.load(writable=True)


class BytesCodec:
    codec_id = "tests.bytes"
    version = 1

    def encode(self, value: bytes, writer: AttachmentWriter | None = None) -> JSONValue:
        if writer is None:
            raise CheckpointError("Bytes codec needs an attachment writer")
        return writer.put(value, media_type="application/x-test-bytes")

    def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> bytes:
        if reader is None:
            raise CheckpointError("Bytes codec needs an attachment reader")
        return reader.get(value)


def test_binary_codec_attachments_round_trip_and_missing_data_fails(
    tmp_path: Path,
) -> None:
    template = state_at()
    individual = Individual(0, b"\x00\xffpayload", fitness=0.0, evaluation_count=1)
    state = RunState(
        run_id=template.run_id,
        generation=0,
        population=Population((individual,)),
        generation_settings=template.generation_settings,
        effective_settings=template.effective_settings,
        next_individual_id=1,
        evaluations=1,
        best=replace(individual),
        statistics=template.statistics,
        next_event_sequence=1,
    )
    codec = BytesCodec()
    with RunStore(tmp_path) as writer:
        writer.create({}, "storage-test")
        body = checkpoint_body(state, codec, writer)
        writer.commit(
            body, make_record(state, (LineageRecord(0, 0, (), "initialize"),))
        )
    with RunStore(tmp_path) as reader:
        loaded = reader.load()
        restored = decode_state(loaded["state"], codec, reader)
        assert restored.population[0].genotype == individual.genotype
        assert restored.best is not None
        assert restored.best.genotype == individual.genotype
    artifacts = list((tmp_path / "artifacts").iterdir())
    assert len(artifacts) == 1  # Shared immutable content is written once.
    artifacts[0].write_bytes(b"\x00\xffpayloae")
    with pytest.raises(CheckpointError, match="integrity"):
        RunStore(tmp_path).load()
    artifacts[0].unlink()
    with pytest.raises(CheckpointError, match="required attachment"):
        RunStore(tmp_path).load()


def test_attachment_reference_cannot_choose_a_path(tmp_path: Path) -> None:
    reference: JSONDict = {
        storage.ATTACHMENT_MARKER: {
            "version": 1,
            "sha256": "../../outside",
            "size": 1,
            "media_type": "application/octet-stream",
        }
    }
    with pytest.raises(CheckpointError, match="SHA-256"):
        RunStore(tmp_path).get(reference)


def test_codec_and_state_validation_reject_loss_or_invalid_boundaries() -> None:
    codec = JsonCodec()
    value: JSONValue = {"numbers": [0, 2**70, 1.5], "none": None}
    cloned = codec.decode(codec.encode(value))
    assert cloned == value
    assert cloned is not value
    for malformed in ((1, 2), object(), float("inf")):
        with pytest.raises(ConfigurationError):
            codec.encode(malformed)
    state = state_at()
    with pytest.raises(CheckpointError, match="event sequence"):
        encode_state(replace(state, next_event_sequence=False), codec)
    with pytest.raises(CheckpointError, match="one scientific event"):
        encode_state(replace(state, next_event_sequence=2), codec)
    data = encode_state(state, codec)
    data["codec"] = {"id": "other", "version": 1}
    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        decode_state(data, codec)
    data = encode_state(state, codec)
    as_object(data["statistics"])["fitness"] = {
        "count": 1,
        "minimum": 0.0,
        "mean": 2.0,
        "maximum": 1.0,
    }
    with pytest.raises(CheckpointError, match="inconsistent"):
        decode_state(data, codec)


def test_edited_journal_position_cannot_hide_a_missing_generation(
    tmp_path: Path,
) -> None:
    with RunStore(tmp_path) as writer:
        writer.create({}, "storage-test")
        commit_generation(writer, 0)
        commit_generation(writer, 1)
    journal_path = tmp_path / "history.jsonl"
    final_line = journal_path.read_bytes().splitlines(keepends=True)[1]
    journal_path.write_bytes(final_line)
    checkpoint = tmp_path / "checkpoint.json"
    envelope = as_object(strict_loads(checkpoint.read_text(encoding="utf-8")))
    journal = as_object(envelope["journal"])
    journal["offset"] = len(final_line)
    journal["sha256"] = hashlib.sha256(final_line).hexdigest()
    checkpoint.write_text(strict_dumps(envelope), encoding="utf-8")
    with pytest.raises(CheckpointError, match="count differs"):
        RunStore(tmp_path).load()
