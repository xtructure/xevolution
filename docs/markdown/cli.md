# Command-line reference

The `xevolution` console entry point and `python -m xevolution` expose the same
commands. From a source checkout, use `uv run --locked python -m xevolution` so
the repository's example factories are importable.

All commands below run from the source project root, beside `pyproject.toml`.
The [examples overview](examples.md) links to complete guides for arithmetic,
shared resources, and Digits. For a Digits run or resume, retain `--group neural`
so uv includes its optional numerical dependencies:

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits.json --output runs/digits
uv run --locked --group neural python -m xevolution resume runs/digits
```

The separate final-evaluation command is
`uv run --locked --group neural python -m examples.digits_report runs/digits`.
It trains two models and scores test data; the core `analyze` command below
only projects existing history. See the [Digits guide](digits_demo.md) before
using that report. When switching among example commands, keep `--group neural`
or use `--all-groups` if you want the optional stack to remain installed.

## Create a fresh run

```bash
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
```

The output directory must be new or empty. A populated directory is an error,
not an instruction to resume. Configuration is validated, the application
factory is resolved, generation 0 is initialized and evaluated, and subsequent
generations execute until stopped or paused.

## Pause and resume

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/paused --pause-after-generation 2
uv run --locked python -m xevolution resume runs/paused
```

`--pause-after-generation N` is an operational limit for this invocation. It
does not change the recorded evolutionary stopping policy. A limit of 0 commits
the evaluated initial population and returns. On resume a limit at or below the
saved generation returns without another transition.

The first `Ctrl-C` (`SIGINT`) or `SIGTERM` requests a graceful pause after the
current generation completes and commits. It does not cancel an application
evaluation already in progress. A second signal interrupts immediately; the
next resume validates the last published checkpoint. Applications with long
evaluations decide whether to provide their own cancellation or partial-work
mechanism.

Resume reads only the explicit `checkpoint.json`, restores recorded settings,
components, random states, and observations, and repairs only an uncommitted
journal tail. Saved-generation evaluation and stopping are not repeated. If a
configured stopping predicate already terminated the run, ordinary resume
leaves it terminated. Changing saved policies is not part of these commands.

Successful `run` and `resume` print one JSON summary containing run identity,
generation, phase, population size, accepted-evaluation count, historical best
ID/fitness, and stopping reason.

## Inspect committed state

```bash
uv run --locked python -m xevolution inspect runs/arithmetic
```

Inspection validates the checkpoint and its committed data, then prints JSON
settings, progress, statistics, best record, stopping state, and compatibility
metadata. It does not import the application factory, evolve candidates, or
truncate an uncommitted history tail. Large genotypes may appear as codec
payloads or attachment references in the best record.

## Build analysis exports

```bash
uv run --locked python -m xevolution analyze runs/arithmetic --output runs/arithmetic/reports
```

The output defaults to `RUN_DIR/reports` when `--output` is omitted. The command
replaces `analysis.json`, a deterministic projection containing generation
summaries/settings/stopping, individuals with recorded births and observations,
and parent-to-child genealogy edges. Ancestors remain available after leaving
the population.

Only the checkpoint's committed journal prefix contributes. Repeated analysis
over the same prefix produces the same logical report and does not append
scientific records, invoke evaluators, or consume evolutionary randomness.
Reports are rebuildable outputs; checkpoint durability does not depend on them.

## Run directory

| Path | Purpose |
| --- | --- |
| `config.json` | Resolved starting configuration |
| `checkpoint.json` | Latest authoritative state and committed journal reference |
| `history.jsonl` | Structured scientific records; an uncommitted tail may exist after a failure |
| `artifacts/` | Immutable codec payload attachments, when used |
| `reports/analysis.json` | Rebuildable default analysis export |

Storage may also use writer-lock or temporary files for coordination. These do
not replace the active checkpoint as the source of truth.

## Diagnostics

```bash
uv run --locked python -m xevolution --help
uv run --locked python -m xevolution --version
uv run --locked python -m xevolution --verbose inspect runs/arithmetic
```

`--verbose` is a global option, placed before the subcommand. Structured results
go to standard output; command errors and diagnostic logging go to standard
error. The CLI returns 0 on success, 1 for handled run/configuration/storage
errors, 2 for argument-parsing errors, and 130 for an immediate interrupt.
Unexpected application programming exceptions remain failures rather than
being treated as operator inapplicability.

See [checkpointing](checkpointing.md) for the exact saved boundary, recovery,
compatibility, and external-side-effect limits.
