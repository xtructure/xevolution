# xevolution Python Port: Architecture and Implementation Specification

This is a documentation copy of the repository-root `xevolution_python_port_spec.md`.
It defines the agreed port contract and target design. For implemented behavior,
start with the [package overview](overview.md), [example guides](examples.md),
and [release verification](verification.md). Repository paths in the specification
remain relative to the source project root.

**Version:** 1.0  
**Date:** October 20, 2026  
**Target:** Python 3.12 and later  
**Status:** Specification based on the agreed project outline; implementation has not begun  
**Java baseline:** [`xtructure/xnet-public`, commit `509c673e502f0c27daa6b91f39cc298404b59ae2`][J1]

## Contents

1. [Purpose and agreed scope](#1-purpose-and-agreed-scope)
2. [Design principles](#2-design-principles)
3. [Repository layout](#3-repository-layout)
4. [Python modules and interactions](#4-python-modules-and-interactions)
5. [Data model and ownership](#5-data-model-and-ownership)
6. [Application extension contracts](#6-application-extension-contracts)
7. [Generation lifecycle and algorithms](#7-generation-lifecycle-and-algorithms)
8. [JSON configuration](#8-json-configuration)
9. [Randomness and deterministic execution](#9-randomness-and-deterministic-execution)
10. [Checkpoint and recovery contract](#10-checkpoint-and-recovery-contract)
11. [Tracking, genealogy, and reporting](#11-tracking-genealogy-and-reporting)
12. [Command-line interface and run directories](#12-command-line-interface-and-run-directories)
13. [Java functionality and utility migration](#13-java-functionality-and-utility-migration)
14. [Examples and documentation](#14-examples-and-documentation)
15. [Verification and acceptance criteria](#15-verification-and-acceptance-criteria)
16. [Implementation sequence](#16-implementation-sequence)
17. [Source references](#17-source-references)

## 1. Purpose and agreed scope

Create a standalone, headless Python evolutionary-computation library that preserves the useful functionality of Java `xevolution`, corrects identified defects, and supports a broad range of application-defined representations and evaluators.

The architecture must support experiments through Python code and a command-line interface. It must retain population statistics, historical results, and genealogy independently of any graphical interface.

The Java repository separates the generic evolution framework from NEAT, simulation, and batch-simulation implementations. Its `xevolution` module declares `xutil` as its direct sibling dependency. This port follows that boundary and incorporates the required utility behavior into the new Python package. [J2], [J3]

### 1.1 Agreed decisions

| Topic | Required scope |
| --- | --- |
| Evolution framework | Port the generic `xevolution` functionality. |
| NEAT | Exclude the `xneat` implementation from this port. Preserve extension points for future applications. |
| Graphical interface | Exclude all GUI components, visualization callbacks, screenshot tools, and the unfinished operator workbench. |
| Java interoperability | No compatibility with Java configuration files, population files, serialized classes, or historical runs is required. |
| Configuration | Use versioned JSON configuration instead of XML. |
| Application assumptions | Support arbitrary genotype data and application-defined evaluation. Do not require a numerical-array, neural-network, or simulation representation. |
| Evaluation | Preserve both individual and whole-population hooks. |
| Optimization | Use one scalar objective, with higher fitness preferred. Keep parent and survivor selection extensible. |
| Algorithm fidelity | Preserve intended evolutionary behavior while correcting defects and making ambiguous policies explicit. Reproducing Java random trajectories is not required. |
| Checkpoints | Support continuation from a committed, completed generation, subject to explicit application-state and determinism contracts. |
| Other Xnet utilities | Reimplement necessary behavior inside `xevolution`, using Python facilities where appropriate. |

This specification does not require built-in distributed execution, GPU execution, a simulation engine, multiobjective optimization, or resumption inside an arbitrary external evaluation. Applications may provide such capabilities behind the evaluation contracts.

### 1.2 Requirement language

**Must** identifies required behavior for the initial implementation. **Should** identifies a recommended implementation choice. **May** identifies an optional capability or a choice left to the implementer. Illustrative signatures describe contracts; they are not complete source-code implementations.

The package name in this specification is `xevolution`. No distribution has been published or installed as part of preparing this document.

## 2. Design principles

### 2.1 Organize by responsibility

Use a shallow package containing cohesive modules. Java interface, abstract-base-class, and implementation triples should generally become a small record, a function, or a typed protocol. Introduce a subpackage only when the implementation grows enough to justify it.

Stateful objects remain appropriate for components that actually own state. The goal is clear ownership and composition, rather than eliminating classes.

### 2.2 Separate domain calculations from bookkeeping

Application components create genotypes, calculate scores, choose candidates, and perform domain-specific variation. Framework code owns individual identities, evaluation accounting, population assembly, generation numbering, provenance, and durable commits.

### 2.3 Make data and execution independent

The core must not require NumPy, PyTorch, a particular graph library, a simulator, or an executor. Applications can use these behind their hooks. A whole-population evaluator may perform jointly dependent calculations; it need not be equivalent to independent individual evaluations.

### 2.4 Keep persistence out of genetic objects

Individuals, populations, selectors, and operators must not open checkpoint files or write reports directly. Serialization and storage own those operations. This permits in-memory use through scripts, notebooks, and tests without creating a run directory.

### 2.5 Make reproducibility an explicit contract

Run-owned randomness, stable work identities, deterministic ordering, component state, and checkpoint boundaries must be designed together. A seed alone is insufficient to describe the current position of a running experiment.

### 2.6 Keep the initial runtime small

The initial core should use the Python standard library. Use a project-local virtual environment managed with `uv`, and commit the dependency lockfile. `uv` provides the project environment and records resolved dependency versions in `uv.lock`. [P1]

Use type annotations and `typing.Protocol` for extension contracts. Protocols allow structural typing without requiring application classes to inherit from framework classes. Runtime validation of configuration, returned results, and saved state remains necessary; annotations do not enforce values during execution. [P2]

## 3. Repository layout

Use an installable `src/` layout. Keeping the import package beneath `src/` helps tests exercise the installed distribution and exposes packaging mistakes that direct imports from a repository root can conceal. [P3]

All paths in this document are relative to the new repository root unless explicitly identified as Java source references.

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | Package metadata; Python requirement `>=3.12`; build configuration; development dependencies; the `xevolution` console entry point; test and code-quality configuration. |
| `uv.lock` | Resolved development dependency versions. |
| `README.md` | Installation, a minimal runnable experiment, and links to extension and checkpoint documentation. |
| `AGENTS.md` | Repository instructions for coding agents, including component ownership, randomness, checkpoint invariants, and behavioral verification. |
| `src/xevolution/` | Installable library containing the modules in Section 4. |
| `src/xevolution/py.typed` | Package typing marker. |
| `examples/` | Small application examples and their JSON configurations. |
| `tests/` | Behavioral, integration, and recovery tests. |
| `tests/fixtures/` | Small configuration, checkpoint, and application-payload fixtures. |
| `docs/markdown/` | Package overview, three example guides, architecture, extension, configuration, checkpoint, CLI, porting, and verification documentation. |
| `docs/html/` | Cross-linked HTML generated from the Markdown sources, with local reading assets. |
| `docs/build_docs.py` | Offline documentation builder and local link/fragment validation. |
| `.github/workflows/ci.yml` | Installation, tests, and code-quality checks on supported Python versions. |

The generated local environment and experiment output directories must be excluded from version control. The package must be tested on Python 3.12 and the newer Python versions it advertises as supported. Supporting several Python versions does not imply that an exact-resume checkpoint can move between them without compatibility checks.

## 4. Python modules and interactions

All modules below live directly under `src/xevolution/`. There are 19 Python modules, including the two package entry-point modules.

### 4.1 Shared data and extension contracts

| Module | Responsibility | Interactions |
| --- | --- | --- |
| `models.py` | Defines shared records such as `Individual`, `Population`, `EvaluationResult`, `VariationResult`, `RunState`, and immutable generation and lineage records. | Provides the data exchanged by evaluation, reproduction, tracking, and persistence. Contains no file I/O or application-specific calculations. |
| `protocols.py` | Defines contracts for initialization, cloning, optional decoding, individual evaluation, population evaluation, parent selection, survival selection, mutation, crossover, termination, payload codecs, and stateful components. | Allows applications to supply functions or compatible objects. Behavior modules depend on these contracts rather than concrete application implementations. |
| `config.py` | Loads and validates JSON configuration, applies defaults, and represents effective settings with typed records. Supports explicit bounds, optional seeded parameter sampling, and validated setting changes where needed. | Supplies configuration to experiment construction and execution. Coordinates sampling with `randomness.py`; effective values are included in run state and checkpoints. |
| `randomness.py` | Owns per-run random generators, exports and restores their complete states, and derives reproducible evaluation contexts from stable work identities. | Supplies randomness to initialization, selection, variation, and evaluation. Its persistent state is saved by storage. |
| `errors.py` | Defines focused exceptions for configuration errors, invalid evaluation results, inapplicable operators, exhausted reproduction attempts, and checkpoint failures or incompatibility. | Lets reproduction distinguish expected operator inapplicability from programming failures; lets the CLI report errors without changing library behavior. |

### 4.2 Evolutionary behavior

| Module | Responsibility | Interactions |
| --- | --- | --- |
| `evaluation.py` | Adapts individual evaluation to population evaluation, invokes custom population hooks, validates returned results, and owns evaluation bookkeeping. | Receives population data and evaluation context from the engine. Associates results with stable IDs and applies accepted scores and metrics in a defined order. |
| `selection.py` | Implements distinct parent-selection and survivor-selection contracts. Includes roulette, uniform, and tournament parent selection, complete replacement, and simple elitist survival. | Reproduction uses parent selection; generation assembly uses survivor decisions. Selectors consume evaluated population data and explicit random context when needed. |
| `operators.py` | Represents mutation and crossover operations, weighted operator pools, explicit fallbacks, recoverable inapplicability, and structured operation provenance. Provides baseline cloning operations. | Application operators produce genotype data and details. Reproduction selects and invokes them, then assigns offspring identity and lineage. |
| `reproduction.py` | Produces the required offspring using parent selection, mutation-or-crossover choice, operator selection, finite retry budgets, and offspring construction. | Reads the complete previous population and survivor decisions. Uses selection, operators, and run-owned ID allocation; returns a complete offspring set or raises an error. |
| `termination.py` | Provides generation, evaluation-count, fitness-threshold, and stagnation predicates, plus logical composition and custom predicate support. | Reads finalized run state. Any persistent counters or windows are exposed for checkpointing. |
| `engine.py` | Coordinates fresh initialization, one generation, repeated execution, registered adaptive updates, termination, and generation commits. Exposes incremental execution and a convenience run interface. | Delegates to reproduction, evaluation, tracking, termination, and an optional store. Exposes completed generation records to callers. |

Parent selection and survivor selection remain different contracts even though they initially share a module. One chooses who reproduces; the other chooses which existing individuals remain.

### 4.3 Experiment construction, persistence, and tools

| Module | Responsibility | Interactions |
| --- | --- | --- |
| `experiment.py` | Defines an `Experiment` composition holding application initialization, evaluation, variation, selection, stopping, cloning, and codec collaborators. Provides application-factory construction helpers. | Connects validated settings with application code and produces the components used by the engine. |
| `registry.py` | Resolves configured short names or explicit `package.module:factory` references into experiment or component factories. | Used for fresh construction and restoration. Resolves behavior from recorded identities and settings; it does not serialize live Python object graphs. |
| `serialization.py` | Converts framework records, random state, component state, and application payloads to versioned representations and restores them. Includes JSON-native codecs and a contract for other payload formats. | Called by storage. Uses explicit application codecs, preserves required types and numerical precision, and reconstructs values such as tuple-based random state correctly. |
| `storage.py` | Owns run directories, checkpoint loading, atomic checkpoint replacement, the structured journal, attachment references, and committed-state recovery. | Receives finalized state and generation records from the engine. Calls serialization and provides committed history to inspection and analysis. |
| `tracking.py` | Computes generation statistics, maintains independent historical-best snapshots, records births and provenance, queries genealogy, and builds analysis projections. | Consumes finalized populations and variation records. Provides information used by stopping and persistence; reconstructs reports from committed history without evolution. |
| `cli.py` | Implements `run`, `resume`, `inspect`, and `analyze`; handles arguments, operational logging, and user-facing errors. | Calls configuration, registry, experiment construction, engine, storage, and tracking. Interpreter exit is confined to the command boundary. |
| `__init__.py` | Exposes a small public API containing principal records and experiment/run/resume entry points. | Provides convenient imports. Importing the package must not configure logging, instantiate an experiment, or consume randomness. |
| `__main__.py` | Delegates `python -m xevolution` to `cli.main()`. | Shares the implementation of the installed `xevolution` command. |

### 4.4 Dependency and ownership rules

Shared records and protocols must not depend on concrete execution, storage, or CLI implementations. Application code is introduced through experiment construction; the core engine must not import example applications.

`serialization.py` owns representation conversion. `storage.py` owns durability, journal positions, and file operations. `tracking.py` owns the interpretation of scientific records. These responsibilities must not be duplicated across modules.

Operational logging uses the standard logging package. Scientific history uses structured generation records; it must not be recovered by parsing diagnostic log messages.

## 5. Data model and ownership

### 5.1 Principal records

Use `Individual[G]` for the framework wrapper around arbitrary genotype data of type `G`. This corresponds to the role of Java `Genome<D>`. A genotype may be a string, graph, array, tensor, or application-defined object. [J4]

| Record | Required meaning |
| --- | --- |
| `Individual[G]` | Stable individual ID; genotype; birth generation; parent IDs; latest fitness or unevaluated status; additional metrics; accepted-evaluation count; optional metadata. |
| `Population[G]` | Deterministically ordered individuals, lookup by ID, and optional population metadata. Enforces unique IDs. |
| `EvaluationResult` | One finite scalar fitness and named supplementary measurements. Underlying simulations, training steps, or other effort can be reported as additional measurements. |
| `VariationResult[G]` | A newly produced genotype and structured operation details. It does not allocate framework IDs or update the population. |
| `RunState[G]` | Current completed generation and phase; population; settings used for that generation and effective settings for the next transition; ID and evaluation counters; historical bests; improvement information; termination status; other framework state affecting continuation. |
| `GenerationRecord` | An immutable account of a finalized generation: population membership identifiers, measurements needed by reporting, births, operation provenance, and relevant state changes. |

Small additional context or decision records, such as `EvaluationContext` and `StopDecision`, may be added to `models.py` when they clarify a public contract. They do not require separate modules.

### 5.2 Identity, age, and evaluation status

- Individual IDs must be unique within a run. Cross-run references use run identity together with individual ID.
- Allocate individual IDs from an explicit monotonic run-state counter. Never reconstruct the counter from current survivors.
- Genotype equality must not determine individual uniqueness or population membership.
- An unevaluated individual has `fitness=None`, or an equivalently explicit unevaluated state. Zero is a valid fitness.
- Generation 0 is the initial evaluated population. An individual born in generation `g` has age zero in that generation; age is derived as current generation minus birth generation.
- Offspring receive fresh identity and evaluation bookkeeping. Copying a parent must not copy its evaluation count, fitness validity, or unrelated transient attributes into the new individual.

### 5.3 Mutable genotype data

Freezing the wrapper does not freeze a nested genotype. Operators must leave parent data unchanged and return an independent child, unless sharing immutable data is explicitly valid.

Applications must supply a cloning operation when their representation needs one. The framework must not assume that a shallow copy is sufficient, nor require a universal deep copy of every large payload.

Historical best records and retained scientific observations must be independent of mutable live individuals. Use the application clone/codec contract to preserve the required genotype snapshot, and copy associated metadata. A later fitness update must not change an earlier generation's recorded result.

### 5.4 Metrics and metadata

Keep the primary scalar objective separate from supplementary numeric metrics and nonnumeric metadata. Complexity is an optional application-provided measurement; the framework must not infer it from the representation.

Every individual must have valid fitness before it participates in selection. Supplementary metrics may be absent. Statistics must report the number of contributing observations and must not turn missing values into zero. If no observations exist, represent the statistic as absent or `null`, consistently.

## 6. Application extension contracts

### 6.1 Required separation of responsibilities

| Extension | Application responsibility | Framework responsibility |
| --- | --- | --- |
| Initialization | Produce initial genotype data from parameters and explicit random context. | Assign IDs, create individual records, and validate the initial population. |
| Cloning | Copy genotype data according to its representation and ownership rules. | Use cloning where offspring or retained snapshots require independence. |
| Decoding | Optionally convert genotype data to an evaluable phenotype. | Permit composition inside evaluation without requiring a dummy decoder. |
| Individual evaluation | Return fitness and measurements for one individual. | Validate the result and maintain accepted-evaluation bookkeeping. |
| Population evaluation | Evaluate the requested population jointly or in batches, returning results associated with requested IDs. | Validate coverage and identities and apply accepted results deterministically. |
| Parent selection | Identify parents from the supplied evaluated population. | Validate selected IDs and invoke variation. |
| Survivor selection | Identify existing individuals that remain. | Validate survivors and assemble the next population. |
| Mutation/crossover | Return newly produced genotype data and provenance, or report inapplicability. | Assign offspring IDs, parentage, birth generation, and fresh evaluation state. |
| Termination | Decide whether finalized run state satisfies a stopping policy. | Invoke the policy once per completed generation and record its decision and state. |
| Payload codec | Encode and decode application data without loss of required semantics. | Record codec identity/version and coordinate durable storage. |

### 6.2 Individual and population evaluation

The conceptual interfaces are:

```python
evaluate(individual, context) -> EvaluationResult
evaluate_population(population, context) -> results_associated_with_individual_ids
```

An application may provide either or both hooks. When a population hook is configured, the engine uses it directly. Otherwise, a standard adapter invokes the individual hook. Supplying both must not cause duplicate evaluation.

A population hook is allowed to evaluate candidates jointly, comparatively, or in a shared environment. It need not produce the same results as an independent loop. The Java application examples already use an override of whole-population evaluation through an application-specific batch system. [J5]

Before updating population state, the evaluation coordinator must verify that all requested IDs have exactly one accepted result, no unexpected IDs occur, and every fitness value is finite. A sequence-based result API must detect duplicate IDs; a mapping-based API must enforce complete expected key coverage.

The framework increments accepted-evaluation counts consistently. One accepted fitness result counts as one fitness evaluation. An application performing several simulations or training iterations to produce that result reports those quantities separately, unless a separately specified evaluation-round policy schedules multiple fitness evaluations.

Applications control their own execution mechanism behind the hook. The initial engine need not create process pools or remote workers. No outstanding jobs may remain at a generation checkpoint boundary.

### 6.3 Stateful components

Components retaining information that affects future results must provide explicit state export and restoration, conceptually:

```python
state_dict() -> JSONValue
load_state_dict(state: JSONValue) -> None
```

The component composition must identify each stateful component with a stable key. Stateless functions require no state payload, but must depend only on their explicit inputs and controlled external conditions. Hidden mutable globals or closures must not be silently treated as checkpointable state.

Adaptive operators, selectors, evaluators, stopping windows, decoding state, and future-affecting caches are included. A deterministic derived cache may be discarded and recomputed only when doing so cannot change randomness, counters, budgets, or subsequent behavior.

Observation-only reporting hooks run after commit and must not mutate evolutionary state. An application hook that changes future behavior is a registered stateful component and must run at a defined point before the checkpoint.

## 7. Generation lifecycle and algorithms

### 7.1 Fresh initialization

1. Load and validate the configuration envelope and establish run-owned randomness.
2. Resolve defaults and any sampled initial settings once, validate the resulting effective settings, and retain their values.
3. Construct experiment components from those settings, create initial genotype data, and assign IDs.
4. Evaluate the initial population and validate the complete result set.
5. Apply results, initialize tracking, perform registered adaptive updates, and evaluate the stopping policy once.
6. Finalize generation 0 and commit its checkpoint and scientific records when persistent storage is enabled.

A crash before the generation-0 checkpoint requires a fresh initialization. There is no earlier committed state to restore.

### 7.2 A generation transition

Starting with completed generation `g`, snapshot its effective settings as the settings to use for generation `g + 1`:

1. Select survivors from the evaluated previous population. The default retains none.
2. Determine the number of children needed to reach the configured population size.
3. Select parents from the complete previous population, including individuals not selected to survive.
4. Produce each child through mutation or crossover, according to the configured mutation probability.
5. Assemble survivors and children into the population for `g + 1`.
6. Evaluate the complete resulting population, including survivors, and validate all results.
7. Apply results in the defined order; update accounting, statistics, historical records, and registered adaptive state.
8. Evaluate termination exactly once and retain both its decision and any updated predicate state.
9. Commit the finalized generation and expose its immutable record to callers.

The Java implementation explicitly keeps genomes marked for removal eligible as parents and evaluates the resulting population after replacement. These intended semantics are preserved. [J6]

No partial offspring population or partially applied evaluation batch may be committed as a completed generation. Failure before commit stops the transition; recovery uses the last committed state. The framework does not promise automatic in-memory rollback of arbitrary application objects after a hook fails.

Post-evaluation adaptive hooks may update future policy and component state, but must not rewrite the population or accepted observations being finalized. Changes to population-construction and evaluation settings take effect on the next transition. Preserve a `generation_settings` snapshot for the completed generation separately from the `effective_settings` prepared for the next one. For example, increasing the target size from 100 to 120 leaves the completed population at 100 and schedules 120 for its successor.

### 7.3 Default policies

| Policy | Initial behavior |
| --- | --- |
| Population size | Positive configured target size; default 100. |
| Survivor policy | Complete replacement by default; simple elitist survival is also available. |
| Parent pool | Entire evaluated previous population. |
| Parent sampling | With replacement; self-crossover is permitted. |
| Variation route | Mutation or crossover for each child; default mutation probability 0.5. |
| Evaluation | Evaluate the full resulting population every generation. |
| Objective | Maximize one finite scalar fitness. |
| Parent selector | Roulette by default; uniform and tournament selectors are available. |
| Termination | Explicit configured predicates, including logical combinations. |

Mutation probability is the probability of choosing the mutation route. It is not an independent post-crossover mutation probability. Population size and mutation-probability defaults match the base Java configuration. [J7]

### 7.4 Selection and numerical validity

The data model accepts finite signed fitness. Roulette requires finite nonnegative selection weights. With the default direct fitness-to-weight policy, a negative score is an error that explains the need for a compatible selector or an explicit weight transformation.

An all-zero roulette population selects uniformly as a documented policy. Do not silently switch to uniform selection because one score is negative. Tournament selection compares signed scalar scores directly. Any future transformation of scores into weights must be explicit because it can change selection pressure.

Reject NaN, infinity, invalid weights, and empty parent pools before sampling. Define stable tie-breaking by ID where ordering is required. Equal fitness must not count as an improvement solely because a different ID wins a tie.

### 7.5 Bounded operator attempts

For one offspring attempt:

1. Choose the variation route and parent or parents.
2. Try positive-weight operators in weighted order without replacement.
3. Try any explicitly registered fallbacks in their documented order.
4. If all fail with a recoverable inapplicability result, retry with new parents subject to a finite per-child attempt budget.
5. If the budget is exhausted, raise a descriptive reproduction error before committing the generation.

Zero-weight operators are disabled in the weighted pool. A fallback must be declared explicitly. This replaces Java's implicit availability of zero-weight operators after positive choices are exhausted.

Only a specific inapplicability exception or result is recoverable. Unexpected programming errors propagate. Validate configuration so that every route with nonzero probability has an operator or declared fallback. Zero-probability routes may have empty pools.

The implementation must not retain a partly exhausted operator pool across unrelated offspring attempts. The existing Java reproduction loop can retry indefinitely when no operator succeeds; the Python attempt budget corrects that behavior. [J8], [J9]

### 7.6 Stopping and limits

Generation 0 does not count as an evolutionary transition. A maximum-generation limit of `N` stops no later than the completion of generation `N`, unless another predicate stops earlier.

Evaluation-count predicates check accepted fitness evaluations at completed-generation boundaries. A threshold can be crossed while completing a population; this is not a hard mid-batch spending cap. Exact pre-operation resource budgeting is outside the initial execution contract.

Persist stateful stopping windows, counters, and the resulting `StopDecision`. Restoration must not evaluate the stopping policy again for the saved generation. A checkpoint whose saved decision is terminated remains terminated under ordinary `resume`.

## 8. JSON configuration

### 8.1 Configuration envelope

JSON describes settings and component identities; Python application code provides behavior. A proposed complete arithmetic example is:

```json
{
  "schema_version": 1,
  "seed": 12345,
  "experiment": {
    "factory": "examples.arithmetic:build_experiment",
    "parameters": {
      "target": 64
    }
  },
  "evolution": {
    "population_size": 100,
    "mutation_probability": 0.5,
    "parent_selection": {
      "name": "roulette"
    },
    "survival": {
      "name": "replace_all"
    },
    "max_attempts_per_child": 100
  },
  "termination": {
    "any": [
      {
        "max_generations": 200
      },
      {
        "fitness_at_least": 1.0
      }
    ]
  }
}
```

For this example, the application can define fitness as `1 / (1 + absolute_error)`, making 1.0 the exact-target score. Operator definitions and application-specific parameters are constructed by `examples.arithmetic:build_experiment`. The general configuration does not require a universal declarative schema for arbitrary Python objects.

### 8.2 Validation and effective settings

Configuration handling must:

- Reject unsupported schema versions, duplicate JSON object keys, unknown core fields, invalid types, and out-of-range settings.
- Reject non-finite numbers. Python's JSON implementation accepts some nonstandard numeric values by default, so strict encoding and decoding must be configured explicitly. [P4]
- Validate application parameters through the application factory or its registered validator.
- Resolve defaults and optional sampled values once, then record the effective settings.
- Preserve bounds and explicit mutability/update policies where the application uses them. Ordinary invalid values must not be silently clamped.
- Make runtime parameter changes explicit, validated, and recorded before the next commit. Save both the completed generation's settings snapshot and the effective settings prepared for the next transition, as defined in Section 7.2.
- Reconstruct resumed runs from saved settings and component identities rather than current defaults.

Input configuration and Python objects may both be supported as entry points. Programmatic callers must receive equivalent validation of core invariants.

### 8.3 Factory resolution

An explicit application-supplied registry may map short names to factories. An explicit import reference such as `package.module:build_experiment` may also be used. Automatic scanning of installed packages is unnecessary for the initial version.

Factories construct compatible components from validated settings. Stored identities, component schema versions, and relevant implementation fingerprints support compatibility checking on restoration. Component construction must be separate from executing the fresh-run initializer.

## 9. Randomness and deterministic execution

### 9.1 Persistent random state

Use per-run generators rather than module-global random calls. Preserve separate evolution and evaluation randomness so additional evaluation work does not silently change the variation stream.

Save the complete state of every persistent generator. Python's `Random.getstate()` and `setstate()` support this; the saved object includes details such as any cached Gaussian value. Saving only the original seed is insufficient. [P5]

Record generator identity and any seed-derivation version. Restoration must preserve required types rather than treating every JSON array as an interchangeable Python list.

### 9.2 Stable work identities

Individual evaluation identities should include generation, individual ID, evaluation round, and replicate where applicable. Whole-population evaluators receive a stable batch identity and ordered population view.

Derive evaluation seeds from a saved root seed and a stable, versioned encoding of the logical work identity. Do not derive them from worker numbers, completion order, wall-clock time, or Python's process-dependent `hash()`.

Do not use a randomly generated run ID in seed derivation if two fresh runs with the same root seed and inputs are intended to reproduce one another. Run IDs may still identify journal records and external idempotency keys.

### 9.3 Ordering

Population iteration, operator ordering, tie-breaking, ID assignment, and result application must be deterministic. Numerical summaries must use a defined accumulation order.

Parallel evaluators return results to the coordinator, which applies them in that order. Invariance to worker count additionally requires application independence and numerically compatible execution. A joint population evaluator may depend on grouping, so grouping is part of its policy and state.

### 9.4 Application responsibilities

Application-owned random generators must either expose persistent state or initialize each independent job from its supplied context. The application must also restore any simulator, optimizer, curriculum, cache, or environment state that influences future results.

Exact continuation is conditional on compatible code, data, numerical settings, and execution environment. Python documents that many higher-level random algorithms can change across versions. Numerical libraries can impose additional limits; PyTorch, for example, does not promise identical results across releases, platforms, or CPU/GPU execution simply from matching seeds. [P5], [P6]

## 10. Checkpoint and recovery contract

### 10.1 Required guarantee

**Given compatible application code and execution conditions, and components satisfying the explicit state and randomness contracts, restoring a committed checkpoint continues the same logical evolutionary run.**

The guarantee concerns genotypes, population membership, fitness observations, selection and variation behavior, identities, counters, stopping behavior, and committed scientific history. Process IDs, diagnostic timestamps, downtime, and console-output timing may differ.

Default persistent execution must checkpoint after every completed generation, including generation 0. In-memory execution may omit persistence entirely.

### 10.2 Exact saved boundary

A checkpoint represents the point after all of the following:

1. All scheduled evaluations for the generation have finished.
2. Results have been validated and applied in deterministic order.
3. Population membership, counters, metrics, genealogy records, and historical bests agree.
4. Registered adaptive updates have run exactly once.
5. The stopping policy has run exactly once, and its updated state and decision have been retained.

It represents the point before any work for the next generation. There are no outstanding jobs or partially applied mutations. Its phase is ready for the next generation or terminated.

Restoration must not repeat initialization, evaluate the saved generation, repeat its adaptive updates, rerun its stateful stopping policy, or replay mutating callbacks. A ready checkpoint at generation `g` proceeds to `g + 1`; a terminated checkpoint remains terminated.

### 10.3 Required checkpoint contents

| State | Reason |
| --- | --- |
| Run and journal identity; completed generation; phase; stop decision and reason | Identifies the authoritative run and its next permitted transition. |
| Ordered population; stable IDs; complete genotype payloads; observations; birth and parent information | Restores the actual candidates and deterministic iteration order. |
| Next individual, evaluation, and event identifiers | Prevents reuse and changes in logical work identities. |
| Accepted-evaluation counts and logical budgets | Preserves stopping and adaptation behavior. |
| Completed-generation settings, next-transition effective settings, and resolved component identities | Preserves the meaning of the saved population and prevents default changes or resampling from altering continuation. |
| All persistent RNG states and evaluation-seed metadata | Restores the current position of each random stream. |
| Historical-best snapshots, last-improvement generation, and policy-dependent archives or windows | Preserves information absent from the current population. |
| State of all future-affecting application components | Restores adaptive probabilities, retained evaluation state, and meaningful caches. |
| Committed journal byte offset, next event sequence, and integrity information | Keeps durable history aligned with the checkpoint. |
| Checkpoint schema, codec IDs/versions, application identities, and relevant runtime/environment fingerprints | Supports explicit compatibility checks. |

### 10.4 Serialization and attachments

Checkpoint encoding must not mutate the run, advance its random streams, or trigger evaluation. Application state methods and codecs have the same restriction.

JSON-native payloads are supported initially. The codec contract must permit future binary attachments without requiring large arrays or models to be expanded into JSON lists. An attachment reference must identify its format/version and integrity information. Required attachments must be durable before the checkpoint that references them is committed.

Components are reconstructed from identities and settings, then their explicit state is restored. Install saved persistent RNG states after reconstruction so setup work cannot advance the restored streams. Components must retain references to the managed generators that receive those states.

Do not make arbitrary closure or entire Python object-graph serialization the default mechanism. If a configured component or payload cannot satisfy its checkpoint contract, fail clearly before claiming a durable checkpoint.

### 10.5 Single-writer commit protocol

One coordinator owns commits for a run. Workers return evaluation results and do not append to the authoritative journal or replace checkpoints.

The local-file implementation must:

1. Finalize and validate generation state in memory; prepare its immutable scientific records and required payload representations.
2. Make required immutable attachments durable, if any.
3. Append the generation's records to `history.jsonl`; flush and synchronize the journal.
4. Write a temporary checkpoint in the same filesystem as the active checkpoint. Include the committed journal byte offset, identity, integrity information, next event sequence, and finalized state.
5. Flush and synchronize the temporary checkpoint, then atomically replace `checkpoint.json`. Synchronize affected directory entries where supported when establishing the documented local durability guarantee.
6. Report commit success and generate optional projections from committed records.

Python exposes `flush()`/`os.fsync()` for buffered-file synchronization and `os.replace()` for atomic replacement when successful. Replacement may fail across filesystems, so temporary and final files must share a filesystem. [P7]

Atomic publication and power-loss durability are separate milestones. After successful replacement, readers in the running system observe the new checkpoint; an ordinary process crash can recover it because its journal prefix and payloads were synchronized first. A machine or power failure before directory synchronization may recover the old or new checkpoint on filesystems supporting this protocol. Recovery must validate whichever committed checkpoint is present. For generation 0, there may be no surviving checkpoint yet. Do not claim power-loss durability until the required synchronization has succeeded; document a narrower guarantee on platforms that cannot provide it.

The checkpoint is the authority for the committed journal prefix. A failed write must not be reported as a successful commit. Retrying a write must not rerun generation finalization or create duplicate committed events; an implementation may instead stop with the preceding checkpoint still authoritative.

### 10.6 Restoration and incomplete writes

On `resume`:

1. Load the explicit active checkpoint and validate its schema, identities, and compatibility requirements.
2. Verify the required journal prefix and attachments.
3. Ignore or remove bytes after the saved journal offset before any new append. They belong to uncommitted work.
4. Fail if committed history or required attachments are missing or corrupt. Do not repair scientific state by reevaluating saved individuals.
5. Reconstruct components, restore all state, and install saved random states.
6. Honor the saved stop decision and phase, then continue only if the saved run is ready.

If the process failed during the next generation, work after the last commit may repeat using the same logical identities and random contexts. No committed generation is reevaluated.

A checkpoint copied to another location is a bundle: checkpoint state plus its required journal prefix and attachments. A JSON manifest alone is resumable only when it contains or can resolve everything required.

### 10.7 Limits of the guarantee

- External calls may repeat if they completed before a crash but their results were not committed. Applications requiring deduplication must use an external idempotency or result-retrieval mechanism.
- Resuming inside a long evaluation requires that evaluator to preserve its own partial computation. Model training may require model, optimizer, scheduler, data-position, and application RNG state.
- Nondeterministic numerical operations or changing external data can change future results even when framework state is restored correctly.
- A time-based policy may restore an active-runtime budget excluding downtime, but cannot generally promise the same final generation because execution durations can change.
- Arbitrary external side effects and console messages do not inherit exactly-once semantics from local journal commits.

These limits must be documented as application contracts rather than reasons to block the generic implementation. Completed-generation continuation is the required initial capability.

## 11. Tracking, genealogy, and reporting

### 11.1 Scientific records

Retain headless population analysis and genealogy. The Java tools already derive population summaries and genealogy from saved populations independently of the graphical interface. [J10]

The structured history must support:

- Population membership identifiers and generation-level fitness summaries.
- Supplementary metric summaries, including contributing counts.
- Accepted evaluation observations needed by the supplied reports.
- Birth generation, parent IDs, operator identity, and operation details.
- Independent historical-best snapshots and last-improvement information.
- Relevant effective-setting changes and stopping decisions.

Each event has a stable run/generation/sequence identity. Store parent links as the authoritative lineage relationship; derive child indexes from them rather than maintaining two independently mutable directions.

An individual's evaluation observations may change over its lifetime. Its birth record must not treat first-observed fitness as permanent fitness.

### 11.2 Statistics and historical bests

Compute min, mean, and max for fitness and supplied numeric metrics, with observation counts. Use a stable tie-breaker for presentation and selection ordering. A best-fitness improvement requires a strictly better fitness value under the configured objective comparison; identity alone does not establish improvement.

Maintain independent historical-best state, including candidates no longer in the population. Do not retain references that allow live mutation to rewrite earlier records. This corrects problematic coupling in the Java statistics implementation. [J11]

### 11.3 Report projections

Only committed records contribute to durable scientific reports. `analyze` must be idempotent: running it repeatedly over the same committed prefix produces the same logical output.

JSON exports are required. CSV summaries may be provided as a convenience. Report generation must not consume evolutionary randomness, alter component state, or invoke evaluation.

Operational progress messages may differ after recovery. The authoritative scientific history must not duplicate committed generations or births.

## 12. Command-line interface and run directories

### 12.1 Commands

| Command | Behavior |
| --- | --- |
| `xevolution run --config CONFIG --output RUN_DIR` | Creates a fresh run, resolves the experiment, initializes generation 0, and evolves. Refuses to silently reuse an existing populated run directory. |
| `xevolution resume RUN_DIR` | Restores `checkpoint.json` and the recorded component composition, repairs only uncommitted journal tail as specified, and continues from the saved phase. |
| `xevolution inspect RUN_DIR` | Displays effective settings, progress, best results, and checkpoint metadata without evolving the population. |
| `xevolution analyze RUN_DIR --output REPORT_DIR` | Builds population summaries and genealogy exports from committed history without evaluating candidates. |

`python -m xevolution` must expose the same commands. CLI functions return exit codes or raise errors handled at the command boundary. Library code must not terminate the interpreter or reconfigure an application's logging.

Ordinary resume uses saved effective settings. It must not silently accept changed defaults or change an evolutionary policy. Explicitly changing a saved run's policies would be a separately identified continuation/fork operation, outside the initial four-command interface.

### 12.2 Run directory contents

| Path | Contents |
| --- | --- |
| `config.json` | Resolved starting configuration. Current effective settings are also recorded in checkpoints and relevant journal events. |
| `checkpoint.json` | Latest authoritative checkpoint. |
| `history.jsonl` | Structured generation, measurement, membership, and lineage records. The checkpoint identifies its committed prefix. |
| `artifacts/` | Immutable payload attachments supplied by codecs, when used. |
| `reports/` | Rebuildable analysis exports. |

Inspection and analysis read a specific committed prefix. They must not treat uncommitted journal tail or the most recently modified arbitrary file as authoritative.

A graceful stop request should finish the current generation and save at the defined boundary. An abrupt failure may require replaying work since the last commit. Failure to save must be reported as a failure to make the new state durable.

An operational pause, such as a graceful user interrupt, must not by itself set the evolutionary stop decision to terminated. Save a ready checkpoint unless the configured stopping policy has independently finished the run. This distinction allows ordinary `resume` after a planned pause.

## 13. Java functionality and utility migration

### 13.1 Main functionality mapping

| Java area | Python destination |
| --- | --- |
| `genetics` interfaces and implementations | Shared records in `models.py`; initialization, cloning, and optional decoding contracts in `protocols.py`; application composition in `experiment.py`. |
| Evolution and reproduction strategies | `engine.py`, `reproduction.py`, and the experiment composition. |
| Evaluation strategies and genome decoders | `evaluation.py` plus application-supplied evaluation/decoding functions or objects. |
| Survival filters | Survivor-selection contract and policies in `selection.py`. |
| Operators and operator selectors | `operators.py`, `selection.py`, and the finite attempt workflow in `reproduction.py`. |
| Evolution configuration and stopping conditions | `config.py`, `registry.py`, and `termination.py`. |
| Experiment and population-analysis tools | `cli.py`, `experiment.py`, `storage.py`, and `tracking.py`. |
| Population statistics and genealogy data | `tracking.py`, immutable records in `models.py`, and the committed journal. |
| GUI package | Excluded by agreed scope. |

### 13.2 Xnet utility behavior

| Java utility or pattern | Python replacement |
| --- | --- |
| `XId` and ID allocation | Explicit run and individual identity plus saved counters in models/run state. |
| `XValId`, `ValueMap`, and primitive type dispatch | Typed core fields, named metric and metadata mappings, and boundary validation. |
| `XIdObjectManager` | Ordinary mappings for lookup and genealogy queries in `tracking.py`. |
| `RandomUtil` | `randomness.py`, with selection policies in `selection.py` and operator ordering in `operators.py`. |
| Configuration builders, fields, ranges, and vetting | Typed settings and explicit validation/sampling/update behavior in `config.py`. |
| `Condition` and combinators | Callable stopping predicates and logical composition in `termination.py`. |
| XML formats and readers/writers | Versioned codecs in `serialization.py` and durable operations in `storage.py`. |
| `XLogger` | Standard logging configured by the application or CLI. |
| `XOption` subclasses | CLI argument handling. |
| Collection builders and transforms | Collection literals, comprehensions, iterators, and functions where used. |
| `WormField` lazy construction | Explicit experiment construction and local caching only when needed. |
| Validation hierarchies | Focused checks and exceptions in `errors.py`. |

The Java configuration machinery supports initial ranges, lifetime bounds, and mutability rules. Preserve any such behavior used by the port through explicit operations; its inheritance hierarchy is not required. [J12]

### 13.3 Documented behavior changes

`docs/markdown/porting_notes.md` must identify at least these changes:

1. JSON and explicit checkpoint state replace Java XML and class-name serialization.
2. Fresh run and resume are separate paths; resume does not reinitialize the population. [J14]
3. Operator attempts are finite, and fallback operators are explicit.
4. Invalid or incompatible roulette weights fail clearly; negative fitness no longer silently disables selection pressure. [J13]
5. Individual IDs come from saved allocation state.
6. Newborn age is zero, derived from birth generation.
7. Missing metrics are excluded from each metric's denominator and reported with contributing counts.
8. Historical records are independent snapshots.
9. Evaluation bookkeeping has one framework owner.
10. Scientific history and checkpoints share a committed-generation boundary.

## 14. Examples and documentation

### 14.1 Example modules

| Module | Purpose and interactions |
| --- | --- |
| `examples/__init__.py` | Makes example factories importable when running from the repository. |
| `examples/arithmetic.py` | Demonstrates a string-based representation, initialization, optional decoding, individual scoring, mutation, crossover, and a JSON-native codec through the public experiment API. |
| `examples/batch_problem.py` | Demonstrates whole-population evaluation, stable result identities, and a joint evaluation contract. It must use the population hook directly. |

Provide `examples/arithmetic.json` and `examples/batch_problem.json`, plus `examples/README.md` with setup and commands. Examples should remain small and independent of NEAT or the excluded simulation subsystems.

The approved Digits extension adds `examples/neural_digits.py`, normal and quick
JSON presets, and `examples/digits_report.py`. It evolves 1–3 hidden layers with
8, 16, 32, or 64 neurons per layer and ReLU/tanh activation. Fitness is solely
validation accuracy, maximized, with no loss or model-size penalty. Each
candidate receives a fresh CPU-trained MLP with a fixed training budget and
recorded data and training seeds. The default stratified split is 1,000 training,
400 validation, and 397 final test examples. The final test partition is excluded
from evolution and used by the separate report only after termination. Numerical
libraries belong to an optional `neural` uv dependency group; core runtime
dependencies remain empty. This application must satisfy the same completed-
generation checkpoint and explicit-state contracts as other applications. See
`docs/markdown/digits_demo.md` for the implemented application contract and presets.

A repository-root development invocation may use:

```bash
uv run python -m xevolution run --config examples/arithmetic.json --output runs/arithmetic
uv run python -m xevolution resume runs/arithmetic
uv run python -m xevolution inspect runs/arithmetic
uv run python -m xevolution analyze runs/arithmetic --output runs/arithmetic/reports
```

`resume` continues only if the saved run is ready for further evolution. A run that already satisfied its stopping policy is inspected or analyzed without further evolution.

### 14.2 Documentation files

| File | Required contents |
| --- | --- |
| `docs/markdown/index.md` | Documentation home and reading routes. |
| `docs/markdown/overview.md` | What the package does, its central concepts, application responsibilities, and generation lifecycle. |
| `docs/markdown/examples.md` | Comparison and entry points for all three examples. |
| `docs/markdown/arithmetic_demo.md` | Bit-string encoding, decoding, scoring, operators, configuration, and commands. |
| `docs/markdown/batch_demo.md` | Shared-resource scoring, genuine whole-population evaluation, contextual fitness, configuration, and commands. |
| `docs/markdown/digits_demo.md` | MLP architecture, fixed training recipe, accuracy fitness, presets, reporting, and continuation. |
| `docs/markdown/building_docs.md` | Markdown maintenance, HTML generation, and documentation validation. |
| `docs/markdown/architecture.md` | Module responsibilities, data ownership, generation lifecycle, and dependency boundaries. |
| `docs/markdown/extensions.md` | Application factory, genotype ownership/cloning, individual and population evaluation, custom selection/operators, component state, and codecs. |
| `docs/markdown/configuration.md` | JSON envelope, built-in policies, defaults, validation, sampled/effective settings, and stopping semantics. |
| `docs/markdown/checkpointing.md` | Exact saved boundary, required state, commit/recovery protocol, bundle contents, compatibility, and application limits. |
| `docs/markdown/cli.md` | Commands, run directory contents, stopping, resume, inspection, and analysis. |
| `docs/markdown/porting_notes.md` | Java-to-Python mapping, excluded areas, and deliberate corrections or behavior changes. |

Generate the HTML edition under `docs/html/` from these Markdown sources. Local navigation, page links, and heading fragments must work after extracting a documentation-only archive. Include all reading assets locally and keep application commands explicitly relative to the source project root.

`AGENTS.md` must direct coding agents to preserve these contracts, keep application assumptions outside the engine, use explicit run-owned randomness, and verify meaningful end-to-end behavior. It should specify the project-local environment and normal build/test commands once implemented.

## 15. Verification and acceptance criteria

### 15.1 Test modules

| Module | Primary coverage |
| --- | --- |
| `tests/conftest.py` | Reusable deterministic components, stateful test applications, payload codecs, and temporary run stores. |
| `tests/test_evolution.py` | Population size, survivor/parent behavior, mutation-or-crossover routing, identity, finite retries, generation/age semantics, and stopping. |
| `tests/test_evaluation.py` | Individual adaptation, genuine population evaluation, result association and validation, accepted-evaluation accounting, and deterministic application order. |
| `tests/test_tracking.py` | Historical snapshots, missing-metric statistics, genealogy after ancestors leave the population, and idempotent analysis. |
| `tests/test_resume.py` | Uninterrupted execution versus new-process restoration, including stateful extensions and custom codecs. |
| `tests/test_storage_recovery.py` | Failures around journal writes, attachment preparation, and checkpoint replacement; missing/corrupt committed data; safe uncommitted-tail recovery. |
| `tests/test_cli.py` | Fresh run, explicit resume, inspection, analysis, errors, and the installed entry points. |

Tests should verify observable contracts rather than mirror helper implementations. Model, configuration, numerical, and serialization checks may be grouped into these behavioral suites or split into focused modules if needed.

### 15.2 Decisive continuation test

Execute the same deterministic experiment in two ways:

1. Run continuously from generation 0 through generation `N`.
2. Run through generation `K`, commit, terminate the process, restore in a new process, and continue through generation `N`.

Compare complete logical state: ordered populations, genotypes, fitness, metrics, IDs, counters, effective settings, RNG states, component states, historical bests, stopping decisions, and committed scientific records. Normalize only intentionally variable diagnostic metadata such as timestamps and fresh-run identity.

Repeat with `K = 0`, with a later `K`, and with a failure during an uncommitted generation. Include a stateful evaluator or operator and a stateful stopping predicate, so repeating finalization or losing hidden state is observable.

### 15.3 Required acceptance cases

| ID | Acceptance criterion |
| --- | --- |
| A01 | The package installs and imports on Python 3.12 without requiring a neural-network or simulation library. |
| A02 | Both examples run through the public application API; the batch example exercises whole-population evaluation directly. |
| A03 | Survivors are chosen separately from parents; the full previous population remains the default parent pool. |
| A04 | Every committed generation has valid fitness for all members, unique IDs, and the target population size recorded in that generation's settings snapshot. Adaptive setting updates affect the next transition. |
| A05 | Negative, non-finite, and all-zero weight cases follow documented policies without an infinite loop or silent loss of selection pressure. |
| A06 | Unusable operators exhaust a finite budget and raise a useful error; no partial generation is committed. |
| A07 | Modifying or reevaluating live individuals cannot alter earlier history or retained best snapshots. |
| A08 | Restoring generation 0 or a later generation reproduces the uninterrupted logical run in a new process under the determinism contract. |
| A09 | Saved-generation initialization, evaluation, adaptive updates, and stateful stopping are not repeated after restoration. |
| A10 | Recovery never exposes a partial committed generation. Process-failure visibility and power-loss durability follow Section 10.5; the new commit is durably guaranteed only after the required synchronization completes. |
| A11 | Uncommitted journal tail can be recovered safely; missing or corrupt committed history or attachments fails explicitly. |
| A12 | Repeated analysis of the same committed prefix does not duplicate generations or genealogy records. |
| A13 | Effective settings, component identities, codec versions, and complete random state are restored rather than inferred from current defaults. |
| A14 | The CLI distinguishes fresh creation, resume, inspection, and analysis and never silently resumes an arbitrary newest file. |

## 16. Implementation sequence

### Increment 1: In-memory evolution

Implement shared records and contracts, validated programmatic settings, initialization, explicit randomness, selection, operators, reproduction, evaluation, tracking needed by stopping, and the incremental engine. Complete the arithmetic example and core behavioral tests.

The engine must already expose the intended completed-generation boundary, even before durable storage is implemented.

### Increment 2: Reliable continuation

Implement state export/restoration, codecs, effective-configuration persistence, the journal/checkpoint commit protocol, compatibility checks, and recovery. Complete new-process equivalence tests and failure-injection cases before relying on saved experiments.

Checkpointing belongs early enough that ownership mistakes can still be corrected without disrupting a large set of applications.

### Increment 3: Headless tools and documentation

Complete JSON-driven factory construction, command-line commands, the whole-population example, analysis exports, packaging checks, and extension documentation. Verify both examples through the public CLI.

### Initial implementation is complete when

All agreed capabilities are implemented, required acceptance cases pass, the documented command interface works from an installed package, and the checkpoint documentation accurately states both its guarantees and application responsibilities. Changes beyond the agreed scope should be proposed separately rather than silently incorporated into the core.

## 17. Source references

The Java review was static source-and-test inspection of the pinned commit below. The legacy Java application was not built or executed. Architectural decisions and Python requirements in this document are recommendations for the new implementation, not claims that those features already exist in the repository.

### Java baseline and implementation evidence

- **J1:** [Reviewed repository commit][J1].
- **J2:** [Xnet repository overview and subproject boundaries][J2].
- **J3:** [xevolution Maven dependencies][J3].
- **J4:** [Generic Java Genome contract][J4].
- **J5:** [Application-level whole-population evaluation example][J5].
- **J6:** [Java generation lifecycle][J6].
- **J7:** [Base evolution configuration and defaults][J7].
- **J8:** [Reproduction, parent selection, and retry loop][J8].
- **J9:** [Operator selection and reset behavior][J9].
- **J10:** [Headless population and genealogy data tracking][J10].
- **J11:** [Population statistics and historical-extrema handling][J11].
- **J12:** [Range-based configuration parameter behavior][J12].
- **J13:** [Java weight normalization][J13].
- **J14:** [Java fresh-start and population-loading behavior][J14].

### Python and tooling references

- **P1:** [uv project structure, environments, and lockfiles][P1].
- **P2:** [Python 3.12 typing and protocols][P2].
- **P3:** [Python Packaging User Guide: src layout][P3].
- **P4:** [Python 3.12 JSON encoding and decoding][P4].
- **P5:** [Python 3.12 random state and reproducibility][P5].
- **P6:** [PyTorch reproducibility limitations][P6].
- **P7:** [Python 3.12 file synchronization and replacement operations][P7].

[J1]: https://github.com/xtructure/xnet-public/commit/509c673e502f0c27daa6b91f39cc298404b59ae2
[J2]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/README.md
[J3]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/pom.xml#L15-L20
[J4]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/genetics/Genome.java#L30-L68
[J5]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/demos/src/main/java/com/xtructure/xnet/demos/oned/OneDEvaluationStrategyImpl.java#L94-L147
[J6]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractEvolutionStrategy.java#L243-L265
[J7]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/config/EvolutionConfiguration.java#L35-L62
[J8]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractReproductionStrategy.java#L156-L265
[J9]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/operator/impl/AbstractOperatorSelecter.java#L90-L183
[J10]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/tool/data/DataTracker.java#L150-L238
[J11]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/genetics/impl/AbstractPopulation.java#L537-L579
[J12]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xutil/src/main/java/com/xtructure/xutil/config/AbstractRangeXParameter.java#L66-L111
[J13]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xutil/src/main/java/com/xtructure/xutil/RandomUtil.java#L562-L578
[J14]: https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractEvolutionExperiment.java#L123-L169
[P1]: https://docs.astral.sh/uv/concepts/projects/layout/
[P2]: https://docs.python.org/3.12/library/typing.html#typing.Protocol
[P3]: https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
[P4]: https://docs.python.org/3.12/library/json.html
[P5]: https://docs.python.org/3.12/library/random.html
[P6]: https://docs.pytorch.org/docs/2.14/notes/randomness.html
[P7]: https://docs.python.org/3.12/library/os.html
