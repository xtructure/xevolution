# Shared resources and population evaluation

This example models candidates choosing among resources whose quality must be
shared. A candidate receives a larger share when fewer population members
choose the same resource. Its fitness therefore depends on both its own
choice and the choices made by the rest of the population.

The example demonstrates xevolution's **whole-population evaluation hook**.
The framework passes the complete population to the application, which computes
all the related scores together and returns them with individual IDs. The
[package overview](overview.md) explains where evaluation fits into a run;
the [examples guide](examples.md) compares the three applications.

## Run the supplied preset

The application is `examples/batch_problem.py`, configured by
`examples/batch_problem.json`. Run these commands from the project root
containing `pyproject.toml` and the `examples/` package:

```bash
uv python install 3.12
uv sync --locked --group dev
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-resources
uv run --locked python -m xevolution inspect runs/shared-resources
uv run --locked python -m xevolution analyze runs/shared-resources --output reports/shared-resources
```

The project uses a local virtual environment managed by uv. Keep
`UV_PROJECT_ENVIRONMENT` unset or pointing to its `.venv`. This application
uses only the Python standard library; it needs no external dataset or
numerical package. The example factory is supplied by the source project,
rather than installed inside the core wheel.

The output run directory must be new or empty. The analysis command writes
`reports/shared-resources/analysis.json`. It reads the committed history and
does not run another population evaluation.

## Genotype and fitness

Each genotype is one integer: a zero-based index into the configured
`resource_quality` list. The preset contains four resources:

| Resource index | Quality |
| --- | ---: |
| 0 | 1.0 |
| 1 | 1.8 |
| 2 | 3.0 |
| 3 | 5.0 |

For each resource, the application counts how many population members chose
it. Every candidate at that resource receives the same share:

```text
fitness = resource_quality[choice] / occupancy[choice]
```

The highest-quality resource is not automatically the best choice in a
particular population. For example, quality 5 shared by ten candidates gives
each fitness 0.5; quality 1.8 shared by two gives each fitness 0.9.

Each result also contains two metrics:

| Metric | Meaning |
| --- | --- |
| `resource` | The candidate's chosen resource index, recorded as a float |
| `crowding` | The number of candidates sharing that resource, including itself |

All configured qualities are positive and finite. Every evaluated candidate
contributes to the count at its own resource, so the denominator is at least
one. Resources with no occupants contribute no individual evaluations.

### A small worked example

Consider two resources with qualities `[6.0, 2.0]` and three candidates. These
qualities are chosen to make the arithmetic clear; they differ from the
four-resource preset.

| Candidate label | First choice | First fitness | Choice after B moves | New fitness |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 6 / 2 = 3 | 0 | 6 / 1 = 6 |
| B | 0 | 6 / 2 = 3 | 1 | 2 / 2 = 1 |
| C | 1 | 2 / 1 = 2 | 1 | 2 / 2 = 1 |

Only B changed its choice, yet all three scores changed. A's unchanged
genotype cannot be assigned a correct score without knowing the population.
The letters label two hypothetical population arrangements. They are separate
from resource indices and literal framework IDs; an actual evolutionary child
receives a new integer ID.

## How the application uses the framework

`build_experiment()` supplies `evaluate_population=application.evaluate_population`.
It does not supply the individual `evaluate` hook. For every generation the
population evaluator:

1. Counts resource occupancy using the entire population.
2. Calculates each candidate's share and descriptive metrics.
3. Returns one `(individual_id, EvaluationResult)` pair for every individual.

The implementation deliberately returns these pairs in reverse population
order. xevolution associates them by explicit ID, validates complete coverage,
and applies accepted results in its deterministic population order. Applications
must not assume that the position of a result identifies its candidate.

This is joint scoring performed once for the complete population. The
framework does not also call an individual evaluator or count accepted
evaluations twice. A population of 48 still contributes 48 accepted individual
evaluations to the framework's counter, even though one population-hook call
produced all 48 results. See [evaluation contracts](extensions.md) for result
validation and the choice between the two hooks.

## Initialization and variation

Initialization chooses a resource uniformly with the run-owned random
generator. Mutation also chooses uniformly from all resource indices, so it
can select the current resource again. Crossover chooses one of the two
parents' resource indices with equal probability. This representation has one
gene; crossover therefore inherits a complete parental choice.

The mutation result records `previous_resource` and `resource` in its lineage
details. Crossover records `selected_parent`, either 0 or 1. Offspring receive
fresh framework IDs even when their resource choice matches a parent's.

An integer is immutable, so cloning can return the existing value safely.
`ResourceCodec` encodes the integer directly and validates its range on
restoration. A boolean is rejected as a resource ID despite Python's treatment
of booleans as a kind of integer.

## Configuration and evaluation counts

The application accepts one parameter under `experiment.parameters`:

| Parameter | Default and supplied value | Validation |
| --- | --- | --- |
| `resource_quality` | `[1.0, 1.8, 3.0, 5.0]` | A nonempty list of positive finite numbers; booleans are rejected |

The list's length determines the number of resources and the legal genotype
indices. Unknown application parameters are errors. Resource quality does
not evolve; it defines the environment in which resource choices evolve.

The preset uses the following framework settings:

| Setting | Value |
| --- | --- |
| Evolutionary seed | 24680 |
| Population size | 48 |
| Parent selection | Roulette using the current population's fitness values |
| Survivor policy | Complete replacement |
| Mutation probability | 0.75; otherwise choose crossover |
| Maximum attempts per child | 20 |
| Termination | Maximum generation 12 |

Generation 0 is the evaluated initial population. With no earlier stopping
predicate, a complete preset run evaluates generations 0 through 12: **13
population-hook calls and 624 accepted individual evaluations**. Parents are
selected from the entire previous population, using its accepted scores. Every
new generation is evaluated as a complete population so that scores reflect
the new occupancy.

For general settings, bounds, and available selection policies, see
[JSON configuration](configuration.md).

## Interpret history in its population context

The run summary and `inspect` output distinguish the historical best record
from the current population's statistics. Here, historical best means **the
largest individual share observed in its recorded population context**. It
does not mean that the winning resource index has a permanent fitness or will
produce that score in a different population.

Keep the associated generation and `crowding` metric when interpreting an
observation. A historical-best candidate may have left the population, and
the same resource can score differently in later generations. The analysis
JSON preserves generation membership, individual observation histories, and
genealogy so that these relationships can be examined together.

Population mean fitness also needs care. Summing the shares of all occupants
of one resource recovers that resource's quality. Consequently:

```text
mean_fitness = sum(qualities of occupied resources) / population_size
```

When all four preset resources are occupied, the mean is approximately
`10.8 / 48 = 0.225`, regardless of how those 48 candidates are distributed
among them. The allocation can change substantially while the mean stays
constant. Inspect occupancy and individual shares alongside summary fitness.
The example illustrates coupled evaluation; it does not establish a unique
globally optimal allocation.

## Pause, continue, and change the experiment

This preset has a fixed generation limit, making it convenient for seeing
checkpoint continuation. Use a fresh directory for the paused run:

```bash
uv run --locked python -m xevolution run --config examples/batch_problem.json --output runs/shared-resources-paused --pause-after-generation 2
uv run --locked python -m xevolution inspect runs/shared-resources-paused
uv run --locked python -m xevolution resume runs/shared-resources-paused
```

The pause commits generation 2 after three population evaluations, accounting
for 144 accepted individual evaluations. Resume continues at the next
transition and retains the configured generation-12 limit. It does not
rescore the saved generation. Resuming an already terminated run leaves it
terminated.

Save changed qualities, population size, or evolutionary seed in a new
configuration and start a new output directory. Resume reconstructs the saved
configuration; editing the original JSON does not alter an existing run.
Compatible code and explicit random/component state remain part of the
[checkpoint contract](checkpointing.md). The [CLI reference](cli.md) describes
graceful signals and recovery.

To adapt the example to shared constraints, competition, or a coupled
simulation, keep the population hook and replace its scoring model. If each
candidate can be scored independently, the [arithmetic](arithmetic_demo.md)
and [Digits](digits_demo.md) examples show the simpler individual hook.

