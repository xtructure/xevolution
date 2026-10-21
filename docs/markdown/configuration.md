# JSON configuration

Configuration selects application behavior and supplies its parameters. The
framework validates core fields, and the application factory validates its own
namespace. Arbitrary Python object graphs are not represented declaratively.

## Envelope

```json
{
  "schema_version": 1,
  "seed": 12345,
  "experiment": {
    "factory": "examples.arithmetic:build_experiment",
    "parameters": {"target": 64, "terms": 10, "mutation_rate": 0.01}
  },
  "evolution": {
    "population_size": 100,
    "mutation_probability": 0.5,
    "parent_selection": {"name": "roulette"},
    "survival": {"name": "replace_all"},
    "max_attempts_per_child": 100
  },
  "termination": {
    "any": [
      {"max_generations": 200},
      {"fitness_at_least": 1.0}
    ]
  }
}
```

`experiment.factory` is an explicit `package.module:factory` reference or a name
registered by a programmatic application. The CLI requires a resolvable factory.
Programmatic `Engine.create()` can receive an already constructed `Experiment`
and a `RunConfig` whose factory is absent.

## Defaults and policies

| Setting | Default | Validation and meaning |
| --- | --- | --- |
| `schema_version` | `1` | Only the supported integer version is accepted |
| `seed` | `0` | Integer; booleans are rejected |
| `population_size` | `100` | Positive integer; resulting completed population size |
| `mutation_probability` | `0.5` | Finite value in `[0, 1]`; mutation **or** crossover route per child |
| `max_attempts_per_child` | `100` | Positive integer; finite budget after operator/fallback exhaustion |
| `parent_selection` | `{"name":"roulette"}` | Full previous population is eligible |
| `survival` | `{"name":"replace_all"}` | Existing individuals retained before producing children |

Supported parent selectors:

- `{"name": "roulette"}` uses fitness directly as nonnegative weight. All-zero
  weights select uniformly; negative weights raise an error.
- `{"name": "uniform"}` samples members uniformly.
- `{"name": "tournament", "size": 3}` compares finite signed fitness among a
  positive number of sampled candidates. Size defaults to 3.

Supported survivor selectors:

- `{"name": "replace_all"}` retains none.
- `{"name": "elitism", "count": 1}` retains the best eligible candidates, up
  to the configured count and target population size. Count defaults to 1 and
  must be nonnegative. `elitist` is also accepted as a policy-name alias.

Ties use stable individual IDs. Parent sampling permits replacement and
self-crossover. A custom selection policy is supplied through `Experiment`.

## Stopping

Each stopping object contains exactly one predicate or combinator:

| Object | Meaning |
| --- | --- |
| `{"max_generations": N}` | Stop at completed generation `N`; generation 0 is initialization |
| `{"max_evaluations": N}` | Stop after a complete generation reaches the accepted-result threshold |
| `{"fitness_at_least": F}` | Stop when current population fitness reaches finite threshold `F` |
| `{"stagnation": N}` | Stop after `N` transitions without a strictly better historical fitness |
| `{"any": [A, B]}` | Stop when any child condition stops |
| `{"all": [A, B]}` | Stop when all child conditions stop |
| `{"not": A}` | Negate one child condition |

Generation and evaluation limits are nonnegative integers; a stagnation window
is positive. `any` and `all` require nonempty arrays. Composite predicates
evaluate each child once per completed generation rather than short-circuiting
away stateful updates.

The evaluation threshold is a boundary check, not a hard mid-batch budget. The
last population may cross it. An experiment must provide a stopping policy
either here or as `Experiment.stop`; an empty declaration does not silently
create an endless run.

Ordinary resume preserves the saved policy and terminal decision. The CLI's
`--pause-after-generation` option is operational and does not change these
predicates.

## Seeded initial sampling

Optional `sampling` maps simple paths to sampling specifications:

```json
{
  "sampling": {
    "parameters.mutation_rate": {"kind": "log_uniform", "low": 0.001, "high": 0.1},
    "evolution.population_size": {"kind": "integer", "low": 40, "high": 80}
  }
}
```

Add this field to the complete envelope. Supported kinds are `uniform`
(default), `integer` with inclusive integer bounds, and `log_uniform` with
positive bounds. Paths are `parameters.NAME` or `evolution.NAME`; arbitrary
nested traversal is not supported.

Sampling uses a managed stream and sorted path order before component
construction. Resolved values are validated, recorded, and restored. Resume
does not resample them.

## Bounds and explicit changes

Optional `parameter_rules` declares lifetime rules for core evolution settings:

```json
{
  "parameter_rules": {
    "population_size": {"minimum": 20, "maximum": 200, "mutable": true},
    "mutation_probability": {"minimum": 0.0, "maximum": 1.0, "mutable": false}
  }
}
```

Rules apply to starting settings and registered adaptive updates. Invalid values
are rejected rather than clamped. A hook returns, for example,
`state.effective_settings.updated(population_size=120)`; that schedules the next
population and preserves the completed generation's settings snapshot.

Applications can use `ParameterRule` and `RangeParameter` from `xevolution.config`
for their own bounded numeric state. Register a future-affecting mutable
`RangeParameter` in `Experiment.components`; its value and rule are explicit
checkpoint state.

## Strict boundaries

Unknown core fields, duplicate JSON object keys, unsupported versions, invalid
types, booleans masquerading as numeric settings, and non-finite numbers are
errors. Overflowing input such as `1e999` is rejected along with nonstandard
`NaN`/`Infinity` tokens. Programmatic settings receive equivalent validation.

Run directories record resolved starting configuration in `config.json`.
Checkpoints retain both current-generation settings and the next effective
settings. Loading configuration itself does not import or execute the factory;
experiment construction is explicit.
