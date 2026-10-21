# Example projects

The three examples use the same evolutionary engine with different
representations and evaluation tasks. Start with arithmetic to see the basic
contracts, continue with shared resources to understand joint evaluation, and
use Digits to explore architecture evolution with a training step inside the
evaluator. The [package overview](overview.md) introduces the common lifecycle.

## Choose an example

| Example | Genotype | Fitness | Evaluation hook | Main lesson |
| --- | --- | --- | --- | --- |
| [Arithmetic expressions](arithmetic_demo.md) | An immutable string of bits encoding operands and operators | `1 / (1 + absolute_error)`; invalid division receives zero | Individual | Separate a representation from its decoded behavior, then vary and score it |
| [Shared resources](batch_demo.md) | An integer identifying a resource | Resource quality divided by current occupancy | Whole population | A candidate's score can depend on the other candidates present |
| [Digits MLP](digits_demo.md) | Hidden-layer widths and an activation function | Fraction of validation images classified correctly | Individual | Evolve architecture while ordinary training learns each candidate's weights |

Each application provides initialization, mutation, crossover, cloning, and an
explicit genotype codec. The engine supplies IDs, evaluation accounting,
selection, generations, stopping, genealogy, and checkpoints. The examples are
application code in the source project; importing the core package does not
import them or start a run.

## Environment and commands

Run commands from the full xevolution project root, beside `pyproject.toml`
and `examples/`. The standalone documentation archive does not contain an
installed application environment. Use the repository-local `.venv` managed by
uv, with `UV_PROJECT_ENVIRONMENT` unset or pointing to that directory.

The arithmetic and shared-resource applications need only the standard library
at runtime:

```bash
uv python install 3.12
uv sync --locked --group dev
```

Digits uses an optional numerical dependency group:

```bash
uv sync --locked --group dev --group neural
```

Keep `--group neural` on Digits run, resume, and reporting commands so uv
includes the required numerical dependencies. All three applications run on
CPU. The core wheel retains an empty runtime dependency list.

### Arithmetic: a compact first run

```bash
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run --locked python -m xevolution inspect runs/arithmetic
uv run --locked python -m xevolution analyze runs/arithmetic --output reports/arithmetic
```

The supplied preset seeks the integer value 64 using expressions with ten
operands. It uses 100 individuals, direct roulette parent selection, complete
replacement, and an equal probability of choosing mutation or crossover. A
mutation operation then considers each bit for flipping with probability 0.01.
Those two probabilities serve different purposes.

The stopping policy ends the run on an exact solution or at completed generation
200. It can stop at generation 0 if an initial candidate is already exact.
Expression evaluation is left to right, and integer division truncates toward
zero. The [full guide](arithmetic_demo.md) explains why a printed expression
must be interpreted using those rules.

### Shared resources: a population that affects its own scores

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-resources
uv run --locked python -m xevolution inspect runs/shared-resources
uv run --locked python -m xevolution analyze runs/shared-resources --output reports/shared-resources
```

The preset contains 48 individuals choosing among four resources with qualities
`[1.0, 1.8, 3.0, 5.0]`. It evaluates generations 0 through 12, giving 624 accepted
individual results. Each generation uses one joint evaluator call to calculate
occupancy and score the whole population.

If five candidates choose the resource with quality 5.0, each receives fitness
1.0. If ten choose it, each receives 0.5. This makes the example useful for
understanding why fitness cannot always be cached by genotype or calculated
without the surrounding population. See the [shared-resource guide](batch_demo.md)
for operators, metrics, and interpretation of historical best observations.

### Digits: evolve architecture, train weights, measure accuracy

Start with the quick preset:

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits_quick.json --output runs/digits-quick
uv run --locked --group neural python -m examples.digits_report runs/digits-quick
```

Or use the normal preset:

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits.json --output runs/digits
uv run --locked --group neural python -m examples.digits_report runs/digits --output reports/digits.json
```

The genotype allows one to three hidden layers with 8, 16, 32, or 64 neurons per
layer, and ReLU or tanh activation. Each candidate gets a fresh model, fixed
training recipe, and the same split and training seeds. Accuracy on 400
validation images is the sole evolutionary fitness; final test images stay
outside selection.

| Preset | Population | Evaluated generations | Epochs per candidate | Search evaluations |
| --- | ---: | --- | ---: | ---: |
| Quick | 6 | 0 through 3 | 10 | 24 |
| Normal | 8 | 0 through 5 | 30 | 48 |

The companion report requires a terminated run. It trains the selected
architecture and a fixed 32-neuron ReLU baseline afresh with the original recipe,
reproduces the selected architecture's saved validation score, and evaluates
each model on the held-out test set once. These two fits are separate from
evolutionary history. See the [Digits walkthrough](digits_demo.md) for full
details and [verification](verification.md) for measured runtime and results.

## Settings and results shared by all three

Application parameters live under `experiment.parameters` in each JSON
configuration. The top-level `evolution` and `termination` sections control the
framework. Changing a configuration is a way to start a new experiment; resume
restores the recorded configuration instead of applying edits made afterward.

All supplied presets use one finite scalar fitness and maximize it. Extra
metrics are observations, not additional objectives. Generation 0 is included
in evaluation counts, and every new population is evaluated in full. In the
Digits presets this includes the retained elite, whose architecture is trained
afresh in each generation.

The core commands have the same meanings for every example:

| Command | Work performed |
| --- | --- |
| `run` | Create and execute a fresh run using a new or empty output directory |
| `resume` | Continue from an existing completed-generation checkpoint |
| `inspect` | Read settings, state, statistics, and compatibility metadata |
| `analyze` | Export committed generations, observations, and genealogy without evaluating candidates |

`digits_report` is an additional application command with an explicit training
and test-scoring step. It is not another name for the core `analyze` operation.

To try pause and resume with the shared-resource example:

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-paused --pause-after-generation 2
uv run --locked python -m xevolution resume runs/shared-paused
```

The pause commits generation 2 and leaves the configured generation-12 stopping
policy intact. The Digits guide gives equivalent commands with its optional
dependency group. Exact continuation requires compatible code, data, numerical
execution, and explicit component state; the full contract is in
[checkpointing and recovery](checkpointing.md).

## Use an example as a starting point

Choose arithmetic when developing a new representation and simple independent
scoring. Choose shared resources when the evaluator needs interactions or a
common population context. Choose Digits when evaluation constructs and trains
a temporary model, and the genotype should describe only the choices that
evolution changes.

Keep the framework's bookkeeping in the engine. Define your genotype's ownership
and codec, validate application parameters in the factory, use the supplied
random generators, and save every component field that affects future results.
The [extension guide](extensions.md) describes those contracts; the
[configuration reference](configuration.md) and [CLI reference](cli.md) cover
the reusable controls.

Return to the [documentation index](index.md).
