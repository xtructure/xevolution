# Example applications

These are small application factories using the public library. They are not
imported by the engine and are not installed inside the `xevolution` wheel.
Run the commands below from the repository root, where the `examples` package
is importable through `python -m`.

## Setup

```bash
uv python install 3.12
uv sync --locked --group dev
```

Use the repository's `.venv`; keep `UV_PROJECT_ENVIRONMENT` unset or pointing to
it. No numerical or simulation package is needed for the arithmetic and
shared-resource examples. The Digits example uses the optional `neural` group.

## Arithmetic expressions

```bash
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run --locked python -m xevolution inspect runs/arithmetic
uv run --locked python -m xevolution analyze runs/arithmetic --output runs/arithmetic/reports
```

An immutable bit string encodes four-bit unsigned numbers and two-bit arithmetic
operators. Expressions are evaluated left to right. Integer division truncates
toward zero. Division by zero yields an invalid expression with fitness zero;
it does not use a non-finite sentinel.

For valid expressions the fitness is `1 / (1 + abs(target - value))`, so the
exact target has fitness 1. The application returns optional measurements such
as error and expression complexity rather than asking the engine to infer them.
Mutation flips independently sampled bits; crossover joins two strings at one
point. The immutable representation allows cloning to return the same string
value safely.

The application namespace accepts `target` (an integer), `terms` (1 through 32),
and `mutation_rate` (a finite probability from 0 through 1). Unknown parameters
are rejected. These are separate from the framework's core evolution settings.

## Joint shared-resource evaluation

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-resources
uv run --locked python -m xevolution inspect runs/shared-resources
uv run --locked python -m xevolution analyze runs/shared-resources --output runs/shared-resources/reports
```

Each candidate is an integer identifying a resource. A resource's quality is
shared among all population members choosing it:

```text
fitness = resource_quality[choice] / number_of_candidates_with_that_choice
```

Changing another candidate's choice can therefore change this candidate's
fitness. The example uses the whole-population hook directly and does not
provide an independent scoring loop. Results are returned in reverse population
order to show that the coordinator associates them by ID and applies them in
its own deterministic order.

The `resource_quality` application parameter is a nonempty list of positive,
finite numbers. Mutation chooses a resource using the supplied random
generator; crossover chooses one parent's resource. The initial configuration
ends at generation 12, which means generation 0 plus twelve transitions.

## Digits: evolve a small MLP

```bash
uv sync --locked --group dev --group neural
uv run --locked --group neural python -m xevolution run --config examples/neural_digits_quick.json --output runs/digits-quick
uv run --locked --group neural python -m examples.digits_report runs/digits-quick
```

`neural_digits.py` evolves an immutable description such as
`{"hidden_layers": [32, 16], "activation": "relu"}`. Each number counts individual
neurons in that hidden layer. Mutation changes a width, adds/removes a layer,
or changes the activation; crossover combines parental descriptions. Every
candidate is trained afresh. Fitness is validation accuracy from 0 to 1;
training loss and parameter count do not change the score.

The fixed stratified split has 1,000 training, 400 validation, and 397 final test
examples. Data is loaded from scikit-learn's bundled 8-by-8 Digits dataset and
scaled by the known pixel maximum of 16. Test predictions are made only by the
separate final report after the evolutionary run has terminated.

| Preset | Population | Last generation | Epochs per fit | Candidate fits |
| --- | ---: | ---: | ---: | ---: |
| `neural_digits_quick.json` | 6 | 3 | 10 | 24 |
| `neural_digits.json` | 8 | 5 | 30 | 48 |

Generation 0 and retained survivors are included in those counts. The report
adds two fresh fits: the selected architecture and the fixed 32-neuron ReLU
baseline. It uses the same training split and budget, then reports each model's
test accuracy without using test accuracy to choose the winner.

To pause and continue the normal preset:

```bash
uv run --locked --group neural python -m xevolution run --config examples/neural_digits.json --output runs/digits-paused --pause-after-generation 2
uv run --locked --group neural python -m xevolution resume runs/digits-paused
uv run --locked --group neural python -m examples.digits_report runs/digits-paused --output reports/digits-paused.json
```

Keep `--group neural` on commands that construct or restore the application.
The optional report path is outside the scientific run directory. Read-only
inspection and history analysis use the ordinary core commands:

```bash
uv run --locked --group neural python -m xevolution inspect runs/digits-paused
uv run --locked --group neural python -m xevolution analyze runs/digits-paused --output reports/digits-history
```

See [the walkthrough](../docs/digits_demo.md) for the training protocol,
configuration fields, and the limits of short-budget architecture comparisons.

## Saved runs

```bash
uv run --locked python -m xevolution resume runs/shared-resources
```

Ordinary resume honors the saved stopping decision. If the example already
finished, it remains finished; `resume` does not extend its generation limit.
Use a new output directory for a new run or a changed policy.

The installed console command and `python -m xevolution` share a CLI. The
`python -m` form above also makes source-checkout factories easy to import.
For an installed application, package its factory in an importable module and
record that module's `package.module:factory` reference in configuration.
