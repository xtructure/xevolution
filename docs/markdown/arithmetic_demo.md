# Arithmetic expression search

This example evolves bit strings that describe integer arithmetic expressions.
The task is to find an expression whose value matches a target, such as 64.
It is a compact introduction to the division of work in xevolution: the
application defines the representation, operators, and scoring rule; the
framework manages populations, selection, identity, history, and checkpoints.

Start with the [package overview](overview.md) if evolutionary runs are new to
you. The [examples guide](examples.md) compares this application with the
[shared-resource](batch_demo.md) and [Digits](digits_demo.md) examples.

## Run the supplied preset

The application lives in `examples/arithmetic.py`, and its configuration is
`examples/arithmetic.json`. Run the following from the project root containing
`pyproject.toml` and `examples/`, rather than from the documentation directory:

```bash
uv python install 3.12
uv sync --locked --group dev
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run --locked python -m xevolution inspect runs/arithmetic
uv run --locked python -m xevolution analyze runs/arithmetic --output reports/arithmetic
```

All dependencies belong to the repository's uv-managed `.venv`. Keep
`UV_PROJECT_ENVIRONMENT` unset or pointing to that directory. This example uses
the Python standard library and needs no numerical packages, dataset, or GPU.
The Python example module is distributed with the source project; it is not
installed inside the core xevolution wheel.

The run command requires a new or empty output directory. Its JSON summary
reports the completed generation, accepted-evaluation count, historical best
fitness, and stopping reason. The analysis command writes
`reports/arithmetic/analysis.json`.

## What the genotype represents

A genotype is an immutable string containing only `0` and `1`. The first four
bits encode an unsigned integer from 0 to 15. Each subsequent six-bit group
contains a two-bit operator followed by another four-bit integer.

| Operator bits | Operation |
| --- | --- |
| `00` | Addition, `+` |
| `01` | Subtraction, `-` |
| `10` | Multiplication, `*` |
| `11` | Integer division, `/` |

The `terms` parameter counts the numbers in an expression. There are
`terms - 1` operators and `4 + 6 * (terms - 1)` bits. The supplied ten-term
configuration therefore uses 58-bit genotypes. The number of terms remains
fixed throughout a run.

The decoder evaluates operations **from left to right**. Division truncates
toward zero, including when an intermediate result is negative. For example,
`1 - 8 / 3` is evaluated as `(1 - 8) / 3`, producing `-2`. A zero divisor makes
the entire expression invalid.

### A small worked example

For three terms, the bit string `0111001001100100` decodes as follows:

| Bits | Meaning |
| --- | --- |
| `0111` | Start with 7 |
| `00 1001` | Add 9 |
| `10 0100` | Multiply by 4 |

The result is `(7 + 9) * 4 = 64`. You can inspect this representation with the
application's helper functions:

```python
from examples.arithmetic import decode, expression

bits = "0111001001100100"
print(expression(bits))  # 7 + 9 * 4
print(decode(bits))  # 64
```

The text returned by `expression()` does not add parentheses. Read it using
the example's left-to-right rule; feeding that text to Python's `eval()` would
apply different precedence rules. The example's decoder never uses `eval()`.

## How fitness is calculated

The individual evaluation hook decodes one candidate and calculates:

```text
absolute_error = abs(target - decoded_value)
fitness = 1 / (1 + absolute_error)
```

Higher fitness is better. An exact match scores 1; an error of 1 scores 0.5;
an error of 3 scores 0.25. An expression containing division by zero scores 0.
Its invalidity is an ordinary evaluation outcome, with no infinite or missing
fitness value.

The evaluator also returns these measurements:

| Metric | Meaning |
| --- | --- |
| `valid` | 1 for a valid expression; 0 for division by zero |
| `value` | Decoded integer value, recorded as a float; valid expressions only |
| `absolute_error` | Distance from the target; valid expressions only |
| `complexity` | Number of terms, recorded for both valid and invalid expressions |

Complexity is descriptive and does not affect selection. Every candidate in
this example has the same configured term count. Because invalid candidates
omit `value` and `absolute_error`, the framework's summaries calculate those
means only from candidates that supplied the metric. Missing values do not
become zeros.

## Initialization, mutation, and crossover

Initialization samples each bit with equal probability using the run-owned
random generator. Mutation visits every bit and flips it independently with
the application's `mutation_rate`. It can leave the string unchanged when no
bit is selected. Such a child still receives a fresh framework identity and
is evaluated normally.

Crossover chooses a cut between two bits, takes the prefix from the first
parent, and the suffix from the second. A cut may fall inside a number or
operator field. The child retains the required bit-string length, and every
two-bit/four-bit field still decodes to an operator/operand. Division by zero
is handled later by evaluation.

Parents are unchanged. The clone function may return the same string because
strings are immutable. The explicit `BitStringCodec` checks the alphabet and
configured length when encoding or restoring a genotype.

Two probability settings have different jobs:

| Setting | Supplied value | Effect |
| --- | ---: | --- |
| `evolution.mutation_probability` | 0.5 | Choose mutation or crossover for each child |
| `experiment.parameters.mutation_rate` | 0.01 | Chance of flipping each bit after the mutation route is chosen |

The framework does not apply an additional mutation stage after crossover.
See [configuration](configuration.md) and [extension contracts](extensions.md)
for the general routing and operator rules.

## Configuration

The application's namespace is `experiment.parameters`:

| Parameter | Default and supplied value | Accepted values |
| --- | ---: | --- |
| `target` | 64 | An integer representable as a finite floating-point metric |
| `terms` | 10 | An integer from 1 through 32 |
| `mutation_rate` | 0.01 | A finite number from 0 through 1 |

Booleans are not accepted as numeric parameters. Unknown parameter names are
errors. Some targets cannot be represented by a particular term count, so a
bounded stopping policy remains useful even when a target score is configured.

The supplied framework settings are:

| Setting | Value |
| --- | --- |
| Evolutionary seed | 12345 |
| Population size | 100 |
| Parent selection | Roulette using the nonnegative fitness values |
| Survivor policy | Complete replacement |
| Maximum attempts per child | 100 |
| Termination | Stop at generation 200, or earlier when fitness reaches 1 |

Generation 0 is the evaluated initial population. An exact target can already
appear there, allowing the run to finish after just 100 accepted evaluations.
A run reaching generation 200 would evaluate 201 populations, or 20,100
candidates. Complete replacement gives every child a new ID, while genealogy
and retained history preserve its relationships to earlier generations.

During documentation verification, the supplied preset reached the exact
target in generation 0, with 100 accepted evaluations and fitness 1. If your
run finishes immediately with that result, the target was already present in
the initial population. The longer-run variation below makes the generation
cycle easier to observe.

## Read the results and observe a longer run

Look at `best_fitness` in the run summary and the `best` record from `inspect`.
A value of 1 means that the target was reached. The inspected best record also
contains the bit-string payload and its `value` and `absolute_error` metrics
when valid. The generic CLI reports the stored representation; application
helpers such as `expression()` can turn a bit string into readable arithmetic.

The analysis JSON contains generation summaries, individual birth and
observation records, and genealogy edges. The `flipped_positions` and
`cut_position` details supplied by the operators explain how children were
created. Analysis does not decode or evaluate the candidates again.

For a demonstration that always includes several generations, save a copy of
the preset as `examples/arithmetic_progress.json` and replace its `termination`
object with:

```json
{"max_generations": 20}
```

This runs for the chosen generation budget even if an exact expression appears
early. You can then pause that new run and continue it:

```bash
uv run --locked python -m xevolution run --config examples/arithmetic_progress.json --output runs/arithmetic-progress --pause-after-generation 2
uv run --locked python -m xevolution resume runs/arithmetic-progress
```

Resume restores the saved configuration and completed-generation state. It does
not use later edits to the original JSON file, and it does not extend a run
that has already met its stopping condition. See [checkpointing](checkpointing.md)
for compatibility and recovery requirements and [the CLI reference](cli.md)
for operational pause behavior.

## Useful next steps

Change the target, bit-mutation rate, or population size in a new configuration
and compare the recorded histories. To evolve expression length itself, extend
the genotype, operators, and codec together; changing only the decoder would
leave inconsistent ownership and persistence rules.

The [shared-resource example](batch_demo.md) introduces scores that depend on
the other population members. The [Digits example](digits_demo.md) keeps the
individual hook but makes evaluation train a neural network before scoring it.

