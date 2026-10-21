# Verification report: xevolution 0.1.1

This is the preserved **Python release verification**, recorded before the
documentation was reorganized into Markdown and HTML editions. Its wheel,
source, and lockfile hashes describe that tested release. New documentation
build and link checks are recorded separately in the
[documentation build guide](building_docs.md); they do not replace or rerun
the release checks below.

Verified on **2026-10-21 (UTC)** using **CPython 3.12.14 on Linux x86-64**.
This report records executed local checks for version 0.1.1. The accompanying
[verification.json](verification.json) contains the test counts, environment,
wheel identity, source hashes, and complete Digits measurements. See the
[Digits walkthrough](digits_demo.md) for installation and run instructions.

## Executed release checks

All three test runs below completed with zero failures and zero errors.

| Test environment | Passed | Skipped | Elapsed time |
| --- | ---: | ---: | ---: |
| Source checkout, development and neural dependencies | 246 | 0 | 11.96 s |
| Installed wheel, without numerical dependencies | 196 | 2 modules | 4.36 s |
| Installed wheel, with neural dependencies | 246 | 0 | 13.87 s |

The two collection skips are the optional neural example and reporting test
modules. The core suite runs successfully without NumPy, scikit-learn, SciPy,
or threadpoolctl. With the optional group installed, all 50 additional neural
and reporting tests run.

Ruff formatting passed for 47 Python files, and Ruff linting was clean.
Strict mypy passed for 21 files: the core package and both neural example
modules. `uv build --offline` produced the version 0.1.1 wheel and source
distribution. Both complete Digits presets and their final reports ran through
the public command-line interfaces; their measured results appear below.

### Interpreter scope and prior verification

The current release was exercised locally on CPython 3.12.14 only. Version
**0.1.0** had previously passed its then-complete 196-test suite on CPython
3.12.14, 3.13.16, 3.14.8, and the 3.15.0rc3 release candidate. Those historical
checks predate this release's neural example and do not establish that the
0.1.1 suite has run on those other interpreters.

The included [reference GitHub Actions workflow](resources/ci.yml) configures
core jobs for Python 3.12, 3.13, 3.14, and the named 3.15 release-candidate
preview, plus neural jobs for Python 3.12, 3.13, and 3.14. **Hosted CI was not
executed for this delivery.** macOS and Windows were not exercised locally.
Support for several Python versions does not imply that checkpoints can be
transferred between them with identical continuation.

## Exact installed-wheel verification

The built wheel was installed into a disposable environment created with uv.
The test working directory, outside the original checkout, contained copies
of `tests/`, `examples/`, and `pyproject.toml`, with no `src/` package. Tests
ran through that environment's interpreter directly. They first ran without
the optional numerical dependencies and then with the neural dependencies
installed. Both runs exercised the installed core wheel.

The checks confirmed:

- `import xevolution` resolved inside that environment's `site-packages`.
- The wheel contained `py.typed` and declared no core runtime dependencies.
- The console entry point and `python -m xevolution` passed both `--help` and
  `--version`.
- All 196 core tests passed without the numerical stack; all 246 tests passed
  once the neural dependencies were present.
- All 20 package files, including `py.typed`, were byte-identical between the
  delivered source package and the tested wheel.

The tested wheel is `xevolution-0.1.1-py3-none-any.whl`, with SHA-256:

```text
f8f498f503f4742303ffde8c006c3352bd99812ccffdadc017244c1cfd8bd9a6
```

Example applications are included in the source distribution and project
archive. They are not installed inside the core wheel; the copied examples
above provide application factories for the installed-wheel tests.

## Digits CPU measurements

### Method and numerical environment

Measurements used the Linux environment above, with glibc 2.39, NumPy 2.5.4,
SciPy 1.18.1, scikit-learn 1.9.1, and threadpoolctl 3.7.0. The recorded OpenBLAS
backends reported versions `0.3.31.dev` and `0.3.34.106.0`, both with the
`SkylakeX` dispatch architecture. Each BLAS backend was limited to one thread
during numerical work. This backend label is not a measurement of a particular
laptop's processor performance.

For each fixed architecture below, three fresh complete evaluator calls were
timed sequentially with a monotonic performance counter. Data and epoch orders
were prepared beforehand. Each call included model construction, training,
validation prediction, accuracy calculation, and returned metrics. There was
no warmup fit, fitted-model cache, or concurrent project test run. Imports,
evaluator setup, and checkpoint I/O are outside these per-candidate timings;
data preparation was timed separately at about **0.014 seconds**.

These calls used the normal recipe: 1,000 training images, 400 validation
images, 30 epochs, batch size 128, learning rate 0.001, regularization 0.0001,
split seed 17, and training seed 23. The architectures were fixed timing probes
measured independently of the two evolutionary presets.

| Hidden layers and activation | Parameters | Median of 3 calls | Largest observed call | Validation accuracy on all 3 calls |
| --- | ---: | ---: | ---: | ---: |
| `[32]`, ReLU | 2,410 | 0.105 s | 0.105 s | 86.75% |
| `[64, 64, 64]`, ReLU | 13,130 | 0.217 s | 0.223 s | 94.75% |
| `[64, 64, 64]`, tanh | 13,130 | 0.210 s | 0.235 s | 95.00% |

The largest observed time is the largest of three measurements, not a runtime
bound. These results describe this execution environment; they do not
guarantee a time limit on another machine or under another workload.

### Complete preset runs

Each preset was run once with its recorded evolutionary seed of 42. Search
timing covered a fresh CLI process, including interpreter/library startup,
data preparation, every candidate evaluation, and generation checkpoint I/O.
The report was timed as a separate fresh process, including startup, its two
new model fits, validation/test scoring, and JSON output. Dependency
installation was outside both timers.

| Preset | Evaluated generations | Candidate fits | Epochs per fit | Search time | Additional report time |
| --- | --- | ---: | ---: | ---: | ---: |
| Quick | 0 through 3 | 24 | 10 | 2.64 s | 1.45 s |
| Normal | 0 through 5 | 48 | 30 | 7.95 s | 1.38 s |

Generation 0 and reevaluated elites are included in the candidate-fit counts.
The report's two fits are separate from evolutionary evaluation accounting.
It reproduced each selected architecture's saved validation accuracy before
scoring the selected model and fixed baseline on the 397 held-out test images.

| Preset and model | Hidden layers and activation | Validation correct / total | Test correct / total |
| --- | --- | --- | --- |
| Quick: selected | `[64, 64]`, tanh | 354 / 400 (88.50%) | 360 / 397 (90.68%) |
| Quick: fixed baseline | `[32]`, ReLU | 260 / 400 (65.00%) | 267 / 397 (67.25%) |
| Normal: selected | `[64, 32, 64]`, tanh | 378 / 400 (94.50%) | 385 / 397 (96.98%) |
| Normal: fixed baseline | `[32]`, ReLU | 347 / 400 (86.75%) | 366 / 397 (92.19%) |

Each baseline used the same training partition and epoch budget as its preset.
Test scores did not select the architectures. These are demonstration results
from one fixed split and one evolutionary seed per preset, not an estimate of
expected search quality across seeds or datasets. Short training budgets affect
which architectures perform well; the measurements do not establish general
architecture-search superiority. Complete unrounded timings, fingerprints, and
report fields are retained under `digits_demo` in
[verification.json](verification.json).

## Checkpoint and acceptance evidence

`tests/test_resume.py` launches independent processes and compares uninterrupted
execution with restoration at generations 0 and 3. It compares the complete
logical checkpoint body, committed scientific history, and attachment bytes,
excluding only operational run IDs. Processes use different `PYTHONHASHSEED`
values. The application exercises opaque mutable genotypes, explicit binary
codecs and cloning, shared stateful components, application RNG state, managed
RNG Gaussian caches, sampled construction settings, adaptive operator weights,
changing population sizes, elitism, and a stateful stopping window.

A separate process exits abruptly during generation 2. Recovery restores
generation 1, removes an injected uncommitted journal tail, and reaches the same
final state and history. Other tests confirm that saved-generation evaluation,
adaptive callbacks, and stopping are not repeated, and that terminal resume
does not change scientific files.

The neural suite adds real small MLP fits and fresh-process restoration from
generations 0 and 1, comparing all logical state and scientific history. It
checks deterministic repeated training, strict architecture/settings/state
validation, parent independence, disjoint data splits, and a sentinel that
fails if search accesses test arrays. Reporting tests verify the saved
validation score, two fixed test evaluations, repeatable reports, and unchanged
run files.

| Specification cases | Executed evidence |
| --- | --- |
| A01–A02: installation and evaluation hooks | Exact-wheel checks, arithmetic and joint population evaluation, and CLI tests. |
| A03–A04: population construction and ownership | Parent/survivor separation, monotonic IDs, age, full reevaluation, and next-transition settings. |
| A05–A06: corrected algorithms | Signed/all-zero/extreme weights, explicit fallbacks, bounded retries, route validation, and programming-error propagation. |
| A07: retained observations | Independent best clones, immutable journal records, and per-metric contributing counts. |
| A08–A09: restoration boundary | New-process comparisons, saved callback/stopping state, and abrupt mid-generation exit. |
| A10–A11: persistence and recovery | Real files, writer exclusion, synchronization/replacement failure injection, tail truncation, and corrupt/missing history or attachments. |
| A12: repeatable analysis | Repeated analysis without duplicate generations, births, or genealogy edges. |
| A13: complete explicit state | RNG/configuration restoration, original factory construction state, shared-component alias topology, codec identity, and incompatibility rejection. |
| A14: distinct CLI operations | Fresh-output protection, explicit/terminal resume, inspection without application imports, analysis, and a real signal during generation-0 evaluation. |

Component tests also reject obvious hidden mutable or unregistered stateful
captures, unmanaged RNG references, and payload decoding that changes restored
component state. Exact continuation still depends on complete application
state and compatible code, data, and numerical execution. It does not promise
exactly-once external side effects for uncommitted work or numerical equality
across platforms. See [checkpointing.md](checkpointing.md) and
[extensions.md](extensions.md) for the application contracts.

## Tools and reproduction

The verified development tools were uv 0.12.23, pytest 9.1.1, Ruff 0.17.1, and
mypy 2.4.0. All Python dependencies were installed in environments managed by
uv. To repeat the complete source checks from the project root:

```bash
uv sync --locked --group dev --group neural
uv run --locked --group dev --group neural ruff format --check .
uv run --locked --group dev --group neural ruff check .
uv run --locked --group dev --group neural mypy src/xevolution examples/neural_digits.py examples/digits_report.py
uv run --locked --group dev --group neural python -m pytest
uv build
```

The executed build used `uv build --offline` after uv had populated its isolated
build cache. Build requirements are declared in `pyproject.toml`; the lockfile
records project and development dependency resolution. Isolated build
dependencies are managed by uv and are not claimed to be fully pinned by
`uv.lock`. Full preset and pause/resume commands are in
[digits_demo.md](digits_demo.md).

The generic port follows the included specification and Java baseline commit
`509c673e502f0c27daa6b91f39cc298404b59ae2`. Source mappings, deliberate corrections,
and retained licensing are described in [porting_notes.md](porting_notes.md).
NEAT, graphical/workbench components, and Java file/run compatibility remain
outside the agreed port.
