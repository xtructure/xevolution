# xevolution

`xevolution` is a headless Python library for evolutionary computation with
application-defined genotypes and evaluation. It provides explicit population
and identity management, mutation and crossover, extensible selection, one
scalar objective, scientific history and genealogy, and continuation from
completed-generation checkpoints.

The package targets Python 3.12 and later and initially uses only the Python
standard library at runtime. Applications may use numerical libraries,
simulators, or their own execution backends behind the evaluation hooks.

## Development setup

Use a repository-local environment managed by `uv`. From this directory:

```bash
uv python install 3.12
uv sync --locked --group dev
uv run --locked --group dev python -c "import sys; print(sys.executable)"
```

The interpreter should be this repository's `.venv`. Keep
`UV_PROJECT_ENVIRONMENT` unset or pointing at that directory; do not select an
unrelated active environment. Project dependencies belong in `pyproject.toml`
and the committed `uv.lock`.

## Run an example

From the repository root:

```bash
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run --locked python -m xevolution inspect runs/arithmetic
uv run --locked python -m xevolution analyze runs/arithmetic --output runs/arithmetic/reports
```

The arithmetic application evolves immutable bit strings that decode to an
arithmetic expression. It supplies initialization, variation, scoring, and a
codec. The shared-resource example demonstrates a genuinely joint population
evaluator:

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-resources
```

See [examples/README.md](examples/README.md) for the representations and complete
commands. These example modules are part of the source checkout. An installed
application supplies its own importable experiment factory; the core package
never imports examples automatically.

### Evolve a small neural network

The Digits example evolves the hidden-layer sizes and activation of a small
multilayer perceptron. Its fitness is simply the fraction of validation images
classified correctly. Every candidate is trained from scratch on CPU with the
same data, training seed, and epoch budget. The dataset ships with scikit-learn;
there is no separate dataset download.

Install the optional numerical dependencies in this project's uv environment:

```bash
uv sync --locked --group dev --group neural
uv run --locked --group neural python -m xevolution run --config examples/neural_digits_quick.json --output runs/digits-quick
uv run --locked --group neural python -m examples.digits_report runs/digits-quick
```

For the normal preset, use `examples/neural_digits.json` and a fresh output
directory. It performs 48 candidate evaluations with 30 training epochs each;
the quick preset performs 24 evaluations with 10 epochs each. The companion
report compares the selected architecture with a fixed 32-neuron baseline on
the held-out test set. See the [Digits walkthrough](docs/digits_demo.md) for the
genotype, fitness, timing guidance, and exact continuation contract.

## Continue a saved run

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/paused --pause-after-generation 2
uv run --locked python -m xevolution resume runs/paused
```

The pause option stops this invocation after generation 2 is committed, while
preserving the configured generation-12 stopping policy. Resume restores
`checkpoint.json` and its committed scientific history. It does not reinitialize
or reevaluate the saved population. A run that already reached its configured
stopping condition remains terminated under ordinary resume. The first
`Ctrl-C` also requests a pause after the current generation completes.

Exact continuation requires compatible application code, input data, numerical
execution, explicit component state, and managed randomness. Work inside an
uncommitted generation may repeat after a crash. Read the
[checkpoint contract](docs/checkpointing.md) before relying on external side
effects or moving saved runs between environments.

## Core design

- Arbitrary genotype payloads are identified through framework IDs, not genotype
  comparison or hashing. Applications define cloning and serialization.
- Individual and whole-population evaluators are distinct hooks. A configured
  population hook runs directly and can compute jointly dependent fitness.
- Parent and survivor selection are independent. Nonsurvivors remain eligible
  parents during the transition that replaces them.
- Every completed population is fully evaluated. Signed finite fitness is
  supported; direct roulette weights must be nonnegative.
- Mutation-or-crossover routing, finite operator retries, deterministic ordering,
  explicit random state, and saved component state support reliable recovery.
- Analysis is a projection of committed records and does not evaluate candidates
  or alter evolutionary state.

## A minimal in-memory experiment

This complete example needs no configuration file or run directory:

```python
from random import Random

from xevolution.config import EvolutionSettings, RunConfig
from xevolution.engine import Engine
from xevolution.experiment import Experiment
from xevolution.models import EvaluationResult, Individual, VariationResult
from xevolution.operators import OperatorSpec
from xevolution.randomness import EvaluationContext


def initialize(rng: Random) -> str:
    return "".join(str(rng.randrange(2)) for _ in range(12))


def evaluate(
    individual: Individual[str], context: EvaluationContext
) -> EvaluationResult:
    return EvaluationResult(float(individual.genotype.count("1")))


def mutate(genotype: str, rng: Random) -> VariationResult[str]:
    index = rng.randrange(len(genotype))
    bit = "0" if genotype[index] == "1" else "1"
    return VariationResult(genotype[:index] + bit + genotype[index + 1 :])


def clone(genotype: str) -> str:
    return genotype  # Strings are immutable.


experiment = Experiment(
    initialize=initialize,
    evaluate=evaluate,
    mutations=(OperatorSpec("flip_one_bit", mutate),),
    clone=clone,
)
config = RunConfig(
    seed=3,
    evolution=EvolutionSettings(population_size=20, mutation_probability=1.0),
    termination={"max_generations": 8},
)
with Engine.create(experiment, config=config) as engine:
    result = engine.run()
    print(result.generation, result.best.fitness if result.best is not None else None)
```

For persistence, supply `output=` and a codec appropriate to the genotype.
Configured factories make reconstruction automatic; an explicitly constructed
programmatic experiment must be supplied again when resuming a run without a
recorded factory.

## Checks and build

```bash
uv run --locked --group dev ruff format --check .
uv run --locked --group dev ruff check .
uv run --locked --group dev mypy src/xevolution
uv run --locked --group dev python -m pytest
uv build
```

To include the optional neural example's tests:

```bash
uv sync --locked --group dev --group neural
uv run --locked --group dev --group neural python -m pytest
```

Without the `neural` group, its test modules are skipped. The core library and
the arithmetic/shared-resource examples still need no numerical dependencies.

The [CI workflow](.github/workflows/ci.yml) is configured to select stable Python
3.12, 3.13, and 3.14, plus a separately named Python 3.15.0rc3 preview job. It pins
`uv` to 0.12.23 and runs these locked checks for each interpreter. The preview
matches the Python 3.15 runtime available through that uv distribution; it does
not establish compatibility with the final Python 3.15.0 release. The workflow
also installs the exact built wheel into a disposable `uv`-created environment
outside the checkout and checks imports, the typing marker, and both command
entry points.

A separate neural-example matrix selects Python 3.12, 3.13, and 3.14 and
installs the `neural` group. Hosted workflow configuration is distinct from
the checks actually executed for this release, recorded below.

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Application extension contracts](docs/extensions.md)
- [JSON configuration](docs/configuration.md)
- [Checkpointing and recovery](docs/checkpointing.md)
- [Command-line reference](docs/cli.md)
- [Digits neural architecture demo](docs/digits_demo.md)
- [Java mapping and deliberate corrections](docs/porting_notes.md)
- [Executed verification and acceptance coverage](docs/verification.md)
- [Full port specification](xevolution_python_port_spec.md)
- [Coding-agent instructions](AGENTS.md)

The port covers generic Java `xevolution` and the necessary Xnet utility
behavior. NEAT, graphical interfaces, the operator workbench, and compatibility
with legacy Java files are outside its scope.

The port preserves the source project's LGPL-3.0-or-later license. See
[LICENSE.txt](LICENSE.txt), the accompanying
[GPL license text](LICENSE-GPL-3.0.txt), and [NOTICE.txt](NOTICE.txt) for the
retained source notice.
