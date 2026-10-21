# xevolution: what it does and how it works

`xevolution` is a Python library for searching a space of candidate solutions
through evolutionary computation. You describe a candidate, define how to
evaluate it, and provide ways to create variations. The library maintains a
population, selects parents, produces new candidates, evaluates them, and keeps
the history needed to understand and continue the search.

The same engine can search for an arithmetic expression, distribute competing
candidates across shared resources, or choose the architecture of a neural
network. Those are the [three included examples](examples.md). The framework
does not need to know what their genotypes mean.

The package is headless: it runs through a Python API or command line without
a graphical interface. It supports Python 3.12 and later. Its core has no
runtime dependencies outside the standard library. Application code can use
numerical libraries, simulators, or other services through the evaluation
contracts. The Digits example adds its numerical dependencies through an
optional uv dependency group.

## The central ideas

An evolutionary run repeatedly asks two questions: how well does each current
candidate perform, and which variations should be tried next? The vocabulary
below separates the application problem from the engine's bookkeeping.

| Term | Meaning in xevolution | Example |
| --- | --- | --- |
| Genotype | The application-defined description that operators change | An immutable bit string, an integer resource choice, or an MLP architecture |
| Individual | A genotype wrapped with a unique framework ID, birth generation, parent IDs, and observations | Individual 37 carries a three-layer neural architecture |
| Population | An ordered collection of individuals evaluated together | Eight candidate architectures in the normal Digits preset |
| Evaluation | Application work that measures a candidate or a whole population | Decode an expression, calculate resource sharing, or train and validate a network |
| Fitness | One finite scalar used as the optimization objective | Validation accuracy of `0.945` |
| Metric | An additional named numerical observation | Parameter count, expression error, or resource crowding |
| Mutation | An operation that produces a child from one parent genotype | Resize one neural layer |
| Crossover | An operation that produces a child from two parent genotypes | Combine the layer descriptions of two architectures |
| Generation | A completed population and its accepted observations | Generation 0 is the evaluated initial population |

A genotype is not necessarily the object that evaluation executes. In the
Digits example it describes layers and activation; the evaluator constructs a
network and trains its weights. In the arithmetic example, the evaluator
decodes a bit string before calculating a score. Any such decoding or model
construction belongs to the application.

The built-in comparison policies and historical-best tracking treat larger
fitness as better. Finite negative values and zero are valid. To minimize a
cost, an application can return its negative or another explicitly chosen
quality score. Choose a compatible selector: tournament selection accepts
signed fitness, while direct roulette selection requires nonnegative weights.
See [configuration](configuration.md) for the supported policies.

Metrics do not silently become extra objectives. For example, the Digits
application records parameter count but ranks candidates by accuracy alone.
An unevaluated fitness is represented separately from zero, and missing metrics
remain missing in statistical summaries.

## What the application supplies

An `Experiment` composes the application-specific pieces. Small functions,
callable objects, and immutable configuration records work together through
typed structural contracts; an application does not need a framework base
class hierarchy.

| Application contribution | Responsibility |
| --- | --- |
| Initializer | Create a genotype using the run-owned random generator |
| Individual or population evaluator | Return fitness and optional metrics |
| Mutation and/or crossover operators | Produce valid children while leaving parents unchanged |
| Clone policy | Preserve independent offspring and retained observations, or share explicitly immutable data |
| Payload codec | Describe how genotype data is encoded and restored for checkpoints |
| Optional policies and hooks | Customize parent selection, survival, stopping, and adaptive updates |
| Explicit component state | Preserve any application information that affects later behavior |

The engine owns IDs, population order, generation numbers, evaluation counts,
birth records, and the completed-generation boundary. It does not require
arbitrary genotypes to support comparison, hashing, indexing, or truth testing.
Two individuals can carry identical genotypes and still have distinct IDs and
separate evaluation histories.

JSON configuration names an importable experiment factory, such as
`examples.neural_digits:build_experiment`. The factory receives validated core
configuration, resolved settings, and the managed random streams. It validates
its own application parameters and constructs an `Experiment`. This makes the
same application reconstructible during resume.
The [extension guide](extensions.md) explains the exact callable signatures,
ownership rules, codecs, and state contracts.

## What happens in one run?

Generation 0 is meaningful work. On a fresh run the engine resolves settings,
constructs components, initializes genotypes, assigns IDs, and evaluates the
initial population. It then performs the same finalization steps used by later
generations: validation, statistics, adaptive updates, stopping, and optional
checkpoint publication.

Each later transition follows this order:

1. **Choose survivors.** The configured survivor policy decides which previous
   individuals, if any, remain in the next population.
2. **Produce the remaining members.** Parents are selected from the complete
   previous population. Each required child follows either the mutation route
   or the crossover route. Operators receive the managed random generator and
   return a new genotype; the framework assigns its ID and parent links.
3. **Evaluate the assembled population.** All members are evaluated, including
   retained survivors. The engine validates the complete set of results before
   accepting fitness values or incrementing accepted-evaluation counts.
4. **Record observations and prepare the next transition.** Statistics and
   independent historical-best snapshots are updated. Registered adaptive
   hooks may update their explicit state and settings for the next transition.
5. **Decide whether to stop.** The stopping policy runs once against the
   finalized state. Its decision and any stateful stopping window are retained.
6. **Commit the completed generation.** If persistence is enabled, scientific
   history and the checkpoint are committed before the engine exposes success.

Parent selection and survival answer different questions. An individual that
will not survive can still contribute offspring to the transition replacing
it. Conversely, retaining an elite preserves its identity and genotype, but
does not reuse its old fitness without reevaluation.

The `mutation_probability` setting chooses mutation **or** crossover for each
child. It does not request a second mutation stage after crossover. Operators
can declare a defined inapplicability condition, and reproduction uses explicit
fallbacks and finite retry budgets. Unexpected programming errors remain
failures.

The default survivor policy replaces the entire population. Other supplied
policies include elitism, and applications can supply custom policies. The
[architecture guide](architecture.md) describes the modules responsible for
each stage; [configuration](configuration.md) describes settings and stopping
conditions.

## Independent and whole-population evaluation

Use an individual evaluator when a candidate's score can be calculated on its
own. Arithmetic expressions and the Digits networks use this hook. The engine
adapts it into a complete population evaluation and provides a stable evaluation
context for each individual.

Use a population evaluator when scoring needs the population together. In the
[shared-resource example](batch_demo.md), a candidate receives its chosen
resource's quality divided by the number of candidates choosing that resource.
The same genotype can receive different fitness when its neighbors change.

A population evaluator returns results associated with individual IDs. Every
expected ID must occur exactly once, with no extra IDs and no non-finite scores.
Results are applied in the engine's stable population order. If both hooks are
supplied, the population hook takes precedence; candidates are not evaluated
twice.

These hooks define what the application computes. They do not imply that the
core schedules parallel workers or supplies a simulator. An application can
place such work behind its evaluator while satisfying the result and state
contracts.

## History, checkpoints, and continuation

A persistent run contains versioned JSON configuration and checkpoint state,
a JSON Lines scientific history, and any codec attachments. History records
population observations, statistics, births, parent links, operator details,
settings, and stopping decisions. The recorded historical best is the best
fitness observation seen; for a population-dependent problem, interpreting
that observation also requires its evaluation context.

The core `inspect` and `analyze` commands read committed results. Analysis
reconstructs summaries and genealogy without evaluating candidates or advancing
the run. The Digits companion report is a distinct application command that
explicitly performs two fresh training runs for its final comparison.

Checkpoints describe completed generations. They preserve population payloads,
IDs and counters, current and next-transition settings, random generator
states, historical state, application component state, stopping state, and the
committed history boundary. Resume reconstructs the application and restores
that state without reinitializing or reevaluating the saved generation.

Exact continuation requires compatible code, data, numerical execution, and
complete future-affecting state. An interrupted, uncommitted generation may
run again after a crash. A saved terminal decision stays terminal under normal
resume; an operational pause leaves the configured stopping policy intact.
Read [checkpointing and recovery](checkpointing.md) before relying on
continuation for an application with external side effects or stateful
evaluators.

## Run an example

Use the full project's root directory, containing `pyproject.toml`, `uv.lock`,
and `examples/`. The documentation archive can be read independently, but the
commands need the source project. All Python dependencies belong in a local
virtual environment managed by uv. Keep `UV_PROJECT_ENVIRONMENT` unset or
pointing to this project's `.venv`.

```bash
uv python install 3.12
uv sync --locked --group dev
uv run --locked python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run --locked python -m xevolution inspect runs/arithmetic
uv run --locked python -m xevolution analyze runs/arithmetic --output reports/arithmetic
```

Use a new or empty output directory for a fresh run. The arithmetic preset
stops when it finds its target or reaches its generation limit, so a successful
initial population can end the run at generation 0. The [arithmetic
guide](arithmetic_demo.md) explains its encoding and fitness.

To embed a configured application in Python, the same engine is available
through the public API:

```python
from xevolution import create_engine

with create_engine("examples/batch_problem.json") as engine:
    result = engine.run()
    best_fitness = None if result.best is None else result.best.fitness
    print(result.generation, best_fitness)
```

This call reads the configuration file and keeps run state in memory because
no output directory was supplied. Add `output="runs/shared-resources"` to
`create_engine` when a persistent run is wanted. Applications can also construct
`Experiment` and `RunConfig` directly; see [extensions](extensions.md).

## Where to go next

| Goal | Read next |
| --- | --- |
| Choose a working example | [Examples overview](examples.md) |
| Understand representation, decoding, and simple scoring | [Arithmetic expressions](arithmetic_demo.md) |
| Evaluate interactions between candidates | [Shared resources](batch_demo.md) |
| Train a model inside fitness evaluation | [Digits MLP architecture evolution](digits_demo.md) |
| Change populations, selection, operators, or stopping | [JSON configuration](configuration.md) |
| Run, pause, inspect, resume, or export results | [Command-line reference](cli.md) |
| Build an application | [Application extension contracts](extensions.md) |
| Understand library internals | [Architecture and ownership](architecture.md) |
| Design reliable saved runs | [Checkpointing and recovery](checkpointing.md) |

Return to the [documentation index](index.md).
