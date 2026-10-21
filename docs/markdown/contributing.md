# Contributor and coding-agent guidelines

This is a documentation copy of the repository-root `AGENTS.md`, which remains
the instruction file for agents working on the source project. Repository paths
below are relative to that project root. See [the package overview](overview.md)
for an introduction and [documentation maintenance](building_docs.md) for the
Markdown/HTML workflow.

This file guides coding agents working anywhere in this repository. Paths are relative to the repository root. The [port specification](port_specification.md) defines the required behavior; this file defines the working practices and recommended implementation stack. Keep both consistent when requirements change.

Inspect the checkout before acting. The specification describes a target architecture, so a named module, test, or command may not have been implemented yet. Do not report planned functionality as existing or verified functionality.

## 1. Project overview and scope

Port the generic Java `xevolution` framework into a standalone, headless Python library for evolutionary computation. Preserve its useful capabilities, correct the identified algorithmic defects, and make application extension straightforward. The reviewed Java baseline is [xtructure/xnet-public at commit 509c673e502f0c27daa6b91f39cc298404b59ae2](https://github.com/xtructure/xnet-public/tree/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution).

The agreed scope is:

- Python 3.12 and later, with an installable `src/xevolution/` package.
- Arbitrary application-defined genotypes, individual evaluation, and genuine whole-population evaluation.
- One finite scalar objective, maximized by default, with extensible parent and survivor selection.
- Initialization, mutation, crossover, population replacement, termination, statistics, historical results, genealogy, and headless analysis.
- Versioned JSON configuration and explicit Python checkpoint state.
- Reliable continuation from completed-generation checkpoints under the specified state and determinism contracts.
- Necessary Xnet utility behavior incorporated into this package using suitable Python facilities.

NEAT, the graphical interface, the operator workbench, and compatibility with Java files or random trajectories are outside this port. Do not introduce a built-in simulator, distributed scheduler, GPU requirement, or multiobjective engine without a requirement for that work. Applications can supply specialized capabilities through the extension contracts.

## 2. Recommended technology stack

| Area | Project choice |
| --- | --- |
| Language | Python `>=3.12`; use syntax and standard-library APIs available at the minimum supported version. |
| Environment and dependency management | `uv`, a repository-local `.venv`, `pyproject.toml`, and a committed `uv.lock`. |
| Core runtime | Standard library initially. Prefer `dataclasses`, `typing`, `collections.abc`, `random`, `math`, `json`, `pathlib`, `hashlib`, `logging`, and `argparse` as appropriate. |
| Packaging | Hatchling build backend, with `src/xevolution` explicitly selected for the wheel and `py.typed` included. Build through `uv build`. |
| Testing | `pytest`, deterministic fixtures, temporary run directories, and subprocess-based resume tests. |
| Formatting and linting | Ruff formatter and linter. |
| Static typing | mypy, with strict checking for the library. |
| Persistence | Versioned JSON checkpoints and JSON Lines scientific history; application codecs for payloads. |
| Documentation | Markdown, executable examples, public API docstrings, and this repository-level instruction file. |
| CI | GitHub Actions using the same environment and check commands as local development. Test Python 3.12 and each newer version advertised as supported. |

Keep the initial runtime dependency list empty. Testing, linting, typing, and build tools are development/build dependencies. Numerical libraries belong to applications or justified optional integrations. Avoid adding a framework or dependency for behavior the standard library already handles clearly.

## 3. Required uv environment policy

**All Python dependencies must be installed in virtual environments created and managed by `uv`. Runtime, development, testing, documentation, and build dependencies are all covered by this requirement.**

Use the repository's `.venv` for normal development and execution. Temporary build-isolation or packaging-test environments are permitted when `uv` creates and manages them. Never install project dependencies into a system interpreter, a user-wide Python installation, or an unrelated environment. The standalone `uv` executable itself is tooling, not a project Python dependency.

- Run project Python commands and development tools through `uv run`.
- Before dependency-changing commands, verify that `UV_PROJECT_ENVIRONMENT` is unset or resolves to this repository's `.venv`. Correct an unexpected override first; do not use `--active` to select an unrelated environment.
- Declare dependencies in `pyproject.toml`; commit the corresponding `uv.lock` changes.
- Use `uv add`, `uv remove`, and `uv sync` for ordinary dependency management. Do not maintain an unrecorded collection of manually installed packages.
- Do not use global `pip`, `pip install --user`, `sudo pip`, or `uv pip --system`. Do not use `uvx` or globally installed tools to bypass the project's locked development tools.
- Do not redirect the project environment to a system or shared interpreter. Configure the editor to use this repository's `.venv`.
- Do not commit `.venv`, caches, build output, generated run directories, or large checkpoint payloads.

`uv` maintains the project environment and supports dependency groups; keep the normal developer tools in the `dev` group. See the official [project workflow][uv-projects] and [dependency documentation][uv-dependencies].

### Initial project bootstrap

Only when the Python project does not yet exist, create its package skeleton and `pyproject.toml` with `requires-python = ">=3.12"`, a Hatchling build-system declaration, the console entry point, and the tool settings below. Do not reinitialize or overwrite an existing project.

Use Python 3.12 as the initial development baseline:

```bash
uv python install 3.12
uv python pin 3.12
uv add --group dev pytest ruff mypy
uv sync --group dev
```

The `uv add` command creates or updates the lockfile. Commit `pyproject.toml`, `uv.lock`, and `.python-version`. If editing dependency declarations directly during an intentional dependency change, run `uv lock` and review its diff.

### Existing checkout

Run from the repository root:

```bash
uv sync --locked --group dev
uv run --locked --group dev python -c "import sys; from pathlib import Path; assert sys.prefix != sys.base_prefix; assert Path(sys.prefix).resolve() == Path('.venv').resolve(); print(sys.executable)"
```

The interpreter should belong to the repository's `.venv`. An explicit environment activation is unnecessary for these commands. `--locked` makes stale or missing lock state an error instead of silently updating it. Resolve the actual dependency mismatch; do not remove the flag merely to make a check pass. See [locking and syncing][uv-sync].

Hatchling belongs in `[build-system].requires`, with backend `hatchling.build`. Let `uv` manage isolated build dependencies. Do not assume that `uv.lock` alone pins every isolated build requirement; use deliberate build constraints when reproducible distribution builds are needed. See [uv builds][uv-build] and [Hatchling package selection][hatch-build].

## 4. Repository and module boundaries

Follow the shallow module layout in the specification. Organize by responsibility, not by Java class hierarchy.

| Location | Ownership |
| --- | --- |
| `models.py`, `protocols.py`, `errors.py` | Shared records, extension contracts, and focused exceptions. No dependency on concrete engine, storage, or CLI implementations. |
| `config.py`, `randomness.py` | Validated settings, effective configuration, managed random streams, and stable evaluation contexts. |
| `evaluation.py`, `selection.py`, `operators.py`, `reproduction.py`, `termination.py` | Evolutionary policies and calculations. Framework bookkeeping has one explicit owner. |
| `engine.py`, `experiment.py`, `registry.py` | Lifecycle coordination, component composition, and explicit factory resolution. |
| `serialization.py` | Conversion between typed state and versioned representations. |
| `storage.py` | File operations, journal positions, checkpoint publication, and recovery. |
| `tracking.py` | Scientific statistics, historical snapshots, genealogy, and report projections. |
| `cli.py`, `__main__.py`, `__init__.py` | Command boundaries and a small public import API. |
| `examples/`, `tests/`, `docs/` | Application examples, behavioral verification, and user/developer documentation. |

The Python-module rows above are under `src/xevolution/`. The engine must not import example applications. Application behavior enters through experiment composition and explicit factories. Keep optional application imports out of package initialization.

Documentation prose lives in `docs/markdown/`; `docs/html/` is generated with
`docs/build_docs.py`. Update Markdown first, keep all three example guides and
the overview consistent with actual behavior, and verify local links and heading
fragments after rebuilding HTML. The documentation builder uses the standard
library and the standalone Pandoc executable.

Do not recreate Xnet as a dependency. Port needed utility behavior into the module that owns it; avoid a miscellaneous `utils.py` that accumulates unrelated responsibilities.

## 5. Python coding practices

### API and type design

- Prefer small functions, explicit collaborators, and cohesive records. Use classes for meaningful state and behavior.
- Use `typing.Protocol` and well-typed callables for extension points. Applications should not need to inherit from framework base classes. Avoid interface/abstract/implementation triples, routine builders, getter/setter wrappers, service locators, and one-class-per-file layouts.
- Annotate public APIs and core internal interfaces. Preserve genotype generics across component boundaries. Use modern built-in annotations such as `list[T]` and `T | None`, with Python 3.12-compatible generic syntax.
- Prefer `Sequence`, `Mapping`, or other appropriate abstract interfaces for inputs when their operations suffice. Do not expose mutable coordinator collections unnecessarily.
- Validate untrusted or application-provided values at boundaries, then work with typed records. Type annotations and runtime protocol checks do not validate result contents. Keep `Any`, casts, and type-checker suppressions narrow and explain them. [Python typing documentation][python-typing]
- Use keyword-only parameters where several booleans, probabilities, or optional policies would make positional calls unclear. Avoid hidden behavior encoded in loosely related flags.

### Data ownership and arbitrary genotypes

- Keep genotype type `G` unconstrained. Framework code must not assume it supports hashing, equality, ordering, truth testing, length, indexing, or numerical-array operations.
- Identify individuals through explicit framework IDs. Do not let dataclass-generated equality, ordering, or hashing traverse genotype payloads. Configure those methods deliberately and use explicit selection keys.
- Use `default_factory` for mutable dataclass defaults; never use shared mutable function defaults.
- Prefer frozen records for retained observations and value-like configuration where appropriate, but remember that a frozen wrapper does not freeze a nested genotype. `dataclasses.replace()` is not a cloning policy. Use application cloning/codecs for independent offspring and historical snapshots. [Dataclass semantics][python-dataclasses]
- Mutators and crossovers must leave parents unchanged and return independent children unless the shared data is explicitly immutable. Fresh offspring receive fresh IDs and evaluation bookkeeping.
- Avoid blanket deep-copying of arbitrary large objects. Define ownership and copy only what the application contract requires. Keep large payloads out of routine representations and log messages.

### Readability, errors, and resources

- Use `snake_case` for functions/modules, `PascalCase` for classes, and clear domain names. Let Ruff enforce formatting; prefer readable control flow over clever expressions.
- Document public contracts, units, side effects, ownership, failure modes, and determinism requirements. Comments should explain decisions and invariants.
- Raise focused exceptions with useful context and preserve causes with exception chaining. Catch only the defined operator-inapplicability condition for evolutionary retries; unexpected exceptions must remain failures.
- Do not use `assert` to validate configuration, evaluator results, or checkpoint data. Assertions are appropriate for internal invariants and tests.
- Use `pathlib`, context managers, and explicit UTF-8 text encoding. Keep scientific persistence in storage; components must not write their own checkpoint files.
- Use `logging.getLogger(__name__)` in library modules. Configure logging and process exit only at the application/CLI boundary. Importing the package must not start work, consume randomness, or create run files.
- Profile a concrete bottleneck before optimizing. Preserve stable ordering and state ownership when changing performance-sensitive code.

## 6. Evolutionary invariants

Preserve these behaviors when implementing or refactoring; the specification supplies the full algorithms and defaults:

- Generation 0 is the evaluated initial population. Newborn age is zero; derive age from birth generation.
- Allocate IDs with saved monotonic counters. Never reconstruct the next ID from current survivors.
- Represent unevaluated fitness explicitly, such as `None`; zero is valid fitness.
- Keep parent selection and survivor selection separate. The default parent pool includes the entire previous population, including nonsurvivors.
- Default to complete replacement and reevaluate the entire resulting population, including survivors. Mutation probability selects mutation **or** crossover for each child; it is not a second mutation stage after crossover.
- Use the whole-population evaluator directly when configured. Otherwise adapt the individual evaluator. Providing both must not evaluate candidates twice.
- Validate the complete evaluation result set before applying scores or accounting: every expected ID exactly once, no unexpected IDs, and finite fitness values. The framework owns accepted-evaluation counts.
- Signed fitness is valid. Direct roulette weighting requires nonnegative weights; all-zero weights select uniformly. Negative weights require an error or an explicitly configured transformation, never a silent fallback.
- Bound reproduction attempts. Try positive-weight operators without replacement within an attempt, then explicit fallbacks. Zero-weight operators are disabled in the weighted pool; explicitly declared fallbacks remain available. Unexpected errors are not retry signals.
- Keep historical snapshots independent of live state. Missing metrics are not zero; means use each metric's contributing count. Equal fitness with a different ID is not an improvement.
- Resolve sampled settings before constructing components. Preserve `generation_settings` separately from `effective_settings` prepared for the next transition. Adaptive hooks must not rewrite finalized observations or population data.

## 7. Determinism and checkpoint discipline

**Checkpoint compatibility and exact continuation are core behavior, not optional conveniences.** Read the specification's checkpoint section before changing any stateful component, generation ordering, randomness, or storage behavior.

- Inject run-owned random generators and stable evaluation contexts. Do not use module-global randomness, wall time, process IDs, unordered set iteration, worker-completion order, or Python's randomized `hash()` for scientific decisions.
- Separate evolution and evaluation randomness. Persist complete generator states and versioned seed-derivation metadata, not only the initial seed.
- Preserve deterministic population, operator, tie-break, result-application, and numerical-accumulation order. Derive job randomness from stable logical work identities.
- Every component retaining future-affecting state must expose explicit export/restoration. Include adaptive policies, evaluator state, stopping windows, caches that affect behavior, and application RNGs. Hidden mutable globals and closures are not a checkpoint mechanism.
- Commit only after complete evaluation, validated result application, tracking, adaptive updates, and one stopping decision. There must be no jobs in flight. When persistent storage is enabled, checkpoint generation 0 and every later completed generation by default. Preserve in-memory execution without a run directory or file I/O.
- Persist ordered population payloads, identities/counters, both settings snapshots, RNGs, historical state, component states, termination state, journal position/integrity, and schema/codec/application compatibility information.
- Restoration reconstructs components, restores explicit state, and installs saved RNG state after setup. Do not rerun initialization, saved-generation evaluation, adaptive updates, or stateful stopping.
- A graceful operational pause leaves the checkpoint ready unless a configured stopping predicate terminates the run. Ordinary resume must honor a saved terminal decision.
- One coordinator commits: durable attachments first, then append/synchronize history, then write/synchronize a temporary checkpoint and atomically replace the active one. Complete required directory synchronization before claiming power-loss durability.
- The checkpoint identifies the committed journal prefix. Before writable resume, truncate or reset the journal to the saved committed byte offset; new appends must begin there. Read-only inspection may ignore the uncommitted tail. Missing or corrupt committed data is an error, not permission to reevaluate saved candidates.
- Encoding and post-commit reporting must not mutate evolutionary state or consume its randomness. Reports must be reproducible from committed records.

Exact continuation assumes compatible application code, data, state, and numerical execution. Mid-generation external work may repeat after a crash; applications own any required idempotency. Do not equate support for several Python versions with exact checkpoint portability between them.

## 8. JSON and numerical boundaries

- Reject unsupported schema versions, unknown core configuration fields, duplicate object keys, invalid types, and out-of-range values. Application factories validate their own parameter namespaces.
- Validate programmatic entry points as well as JSON input. Explicitly reject booleans where integer IDs/counts or numeric probabilities/fitness are required.
- Reject non-finite values after parsing, including overflow such as `1e999`, and encode strict JSON with `allow_nan=False`. Rejecting only the `NaN` and `Infinity` tokens is insufficient.
- Keep configuration, runtime records, and serialized payloads distinct. Use explicit codecs; do not treat `dataclasses.asdict()`, `__dict__`, `pickle`, or arbitrary object-graph serialization as the persistence design.
- Preserve tuple-based RNG state, application payload semantics, numeric precision, and string-key requirements explicitly. Include schema and codec versions; validate before accepting restored state.
- Make weight normalization and summaries numerically sound. Finite inputs can still produce invalid aggregates; validate results and use stable arithmetic where appropriate.

## 9. Tool configuration and normal checks

Keep tool settings in `pyproject.toml`. Recommended starting configuration:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.12"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--import-mode=importlib"]

[tool.hatch.build.targets.wheel]
packages = ["src/xevolution"]
```

This is a tool-configuration excerpt, not a complete `pyproject.toml`. Merge it with package metadata, build requirements, dependency groups, and the `xevolution = "xevolution.cli:main"` script entry. See the official [Ruff][ruff], [mypy][mypy], and [pytest][pytest] guidance.

Once the corresponding implementation exists, the standard checks are:

```bash
uv sync --locked --group dev
uv run --locked --group dev ruff format --check .
uv run --locked --group dev ruff check .
uv run --locked --group dev mypy src/xevolution
uv run --locked --group dev python -m pytest
uv build
```

For formatting edits, use `uv run --locked --group dev ruff format` with the affected paths. Do not disable checks broadly or add blanket suppressions to hide failures. Use focused tests while developing; run the relevant broader checks before completing code changes. Documentation-only changes need proportionate documentation validation.

CI should pin a reviewed `uv` version, record its selected Python versions, and use the committed lockfile. Each matrix job must select its interpreter explicitly through the `python-version` input of `astral-sh/setup-uv` or the `UV_PYTHON` environment variable. Do not rely on `PATH` alone to override the committed `.python-version`. See [uv's CI guidance][uv-ci]. Keep dependency upgrades intentional and separate from unrelated behavior changes when practical.

## 10. Testing expectations

Test observable contracts and regressions, not private helper structure. Use small deterministic applications; do not require GPUs, network services, or long evolutionary runs for the default suite.

- Cover population size, parent/survivor distinctions, selection boundaries, finite retries, identity, age, evaluation coverage/accounting, and stopping.
- Include mutable nested genotypes and custom payloads that cannot be generically compared or hashed. Verify parent independence and historical snapshot integrity.
- Test both evaluation hooks, including jointly dependent population evaluation and invalid/duplicate/missing result identities.
- Compare uninterrupted execution with restoration in a **new process** at generation 0 and a later generation. Include stateful components and stopping, custom codecs, and failure during an uncommitted generation. Compare complete logical state and committed scientific history, not just the best fitness.
- Exercise real temporary files and targeted failure injection around journal and checkpoint operations. Test uncommitted-tail recovery separately from missing/corrupt committed history. Do not mock away the persistence layer being verified.
- Test that repeated analysis does not duplicate scientific records or alter evolutionary state.
- Avoid flaky statistical assertions and wall-clock sleeps. Use controlled random inputs and explicit boundary cases where practical.

For packaging changes, build a wheel and smoke-test it outside the checkout in a disposable `uv`-created environment. Install that exact wheel using `uv pip` with the disposable interpreter explicitly selected, then invoke that interpreter directly. Verify import, the console entry point, `python -m xevolution`, and included `py.typed`/resources. A project-mode `uv run` that returns to the development environment would invalidate this check.

Use the specification's acceptance criteria as the completion target. Add tests for changed behavior and concrete risks; avoid arbitrary coverage quotas or tests that merely repeat implementation details.

## 11. Agent workflow and completion

1. Read the relevant specification sections, existing code, tests, and any instructions scoped to the files you will change. Preserve unrelated user work.
2. Identify the behavior and state ownership affected. Use reasonable judgment for routine implementation choices within the requested scope.
3. Implement a cohesive change. Keep documentation and examples synchronized with public behavior, configuration, and checkpoint formats. Record deliberate Java behavior corrections in `docs/markdown/porting_notes.md`.
4. Verify the changed contract with the appropriate checks. Do not weaken the contract to make an existing test pass or claim unrun checks succeeded.
5. Report what changed, why, what was verified, and any remaining limitations. Distinguish implementation, planned work, and environmental blockers clearly.

For the initial port, follow the specification's sequence: in-memory evolution, reliable continuation, then headless tools and documentation. Establish state ownership and the completed-generation boundary before applications depend on the framework.

## Reference links

The project policies above are specific to this port. Consult these primary references for tool behavior and Python semantics; version decisions belong in the repository configuration and lockfile.

[uv-projects]: https://docs.astral.sh/uv/guides/projects/
[uv-dependencies]: https://docs.astral.sh/uv/concepts/projects/dependencies/
[uv-sync]: https://docs.astral.sh/uv/concepts/projects/sync/
[uv-build]: https://docs.astral.sh/uv/concepts/projects/build/
[uv-ci]: https://docs.astral.sh/uv/guides/integration/github/
[hatch-build]: https://hatch.pypa.io/latest/config/build/
[python-typing]: https://docs.python.org/3.12/library/typing.html
[python-dataclasses]: https://docs.python.org/3.12/library/dataclasses.html
[ruff]: https://docs.astral.sh/ruff/configuration/
[mypy]: https://mypy.readthedocs.io/en/stable/config_file.html
[pytest]: https://docs.pytest.org/en/stable/explanation/goodpractices.html
