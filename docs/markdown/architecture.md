# Architecture

`xevolution` is a generic, headless evolutionary-computation library. Application
code supplies the representation and problem; the framework coordinates its
evolution, identities, measurements, history, and continuation.

## Module ownership

All library modules live under `src/xevolution/`.

| Module | Owns | Main interactions |
| --- | --- | --- |
| `models.py` | Shared individuals, populations, results, run state, and generation records | Data exchanged by policies, the engine, tracking, and serialization |
| `protocols.py` | Structural contracts for application components | Applications implement functions or compatible objects without inheriting framework classes |
| `errors.py` | Focused configuration, evaluation, reproduction, and checkpoint errors | Components distinguish operator inapplicability from unexpected failures |
| `config.py` | Strict JSON input, validated settings, sampling, and explicit updates | Supplies construction settings and separate completed/next-generation settings |
| `randomness.py` | Managed random streams, evaluation contexts, and stable work identities | Initialization, variation, selection, evaluation, and state restoration |
| `evaluation.py` | Hook routing, complete result validation, deterministic application, and accepted counts | Engine supplies the population; applications return observations |
| `selection.py` | Distinct parent and survivor policies | Reproduction chooses parents; population assembly retains survivors |
| `operators.py` | Weighted operations, fallbacks, and cloning operations | Supplies variation behavior and provenance to reproduction |
| `reproduction.py` | Finite offspring attempts and framework birth bookkeeping | Reads the complete previous population and produces independent children |
| `termination.py` | Stopping predicates and logical composition | Reads finalized state once per generation; exposes persistent predicate state |
| `engine.py` | Fresh initialization, transitions, adaptive updates, stopping, and commits | Coordinates the policy modules and an optional store |
| `experiment.py` | Explicit composition of an application's collaborators | Connects initialization, clone/codec, evaluator, operators, and policies |
| `registry.py` | Named and explicit Python factory resolution | Reconstructs application behavior from recorded identities |
| `serialization.py` | Versioned conversion of typed state and application payloads | Engine prepares state for storage and decodes restored records |
| `storage.py` | Files, attachments, journal offsets, atomic publication, and recovery | Commits prepared state and exposes only committed scientific history |
| `tracking.py` | Statistics, independent historical snapshots, genealogy, and report projections | Consumes finalized observations; supplies historical data to stopping |
| `cli.py` | Command parsing, operational logging, and user-facing process errors | Calls public experiment/engine/storage operations |
| `__main__.py` | `python -m xevolution` entry point | Delegates to the same CLI as the console script |
| `__init__.py` | Small public import surface | Does not import examples or perform run-time work on import |

Shared models and protocols do not depend on the concrete engine or storage.
Application behavior enters through composition. There is no miscellaneous
utility module or runtime dependency on the old Xnet packages.

## Data ownership

The framework owns individual IDs, population order, generation numbers,
accepted-evaluation counts, provenance, and the commit boundary. An application
owns genotype semantics, cloning, decoding, domain evaluation, and any private
state retained by its components.

IDs determine individual identity. The engine does not compare, hash, index,
measure, or truth-test arbitrary genotype payloads. A frozen wrapper does not
make a nested list, array, or graph immutable: operators must leave parents
unchanged, and retained historical snapshots use the application's cloning
contract.

Numeric metrics supplement one scalar fitness. They do not create multiple
objectives. Population summaries include each metric's contributing count;
missing observations remain missing.

## Completed-generation lifecycle

Fresh execution resolves settings, constructs components, creates genotypes,
assigns IDs, and evaluates generation 0. A later transition:

1. Captures the effective settings to use for the new generation.
2. Selects survivors and determines how many children are needed.
3. Selects parents from the **complete** previous population.
4. Produces children through mutation or crossover with finite retry budgets.
5. Assembles the new population and evaluates all members, including survivors.
6. Validates all result identities and scores before applying observations.
7. Updates statistics, independent historical snapshots, and registered adaptive
   components.
8. Evaluates stopping once and retains the decision and predicate state.
9. Commits the generation if persistence is enabled, then exposes its record.

The population hook takes precedence when supplied. Otherwise the coordinator
adapts the individual hook. A population hook can calculate jointly dependent
fitness; the engine does not replace it with an independent loop.

An adaptive update changes `effective_settings` for the next transition. The
completed population retains its own `generation_settings` snapshot, so changing
the next target population size cannot rewrite what just happened.

In-memory execution follows the same lifecycle without opening a run directory.
Failure inside a transition may have mutated application state; the library does
not promise general in-memory rollback. Durable recovery starts from the last
committed generation.

## Scientific records and operational messages

Scientific history records generation membership, evaluation observations,
statistics, births, parent links, operations, settings, and stopping decisions.
Its committed prefix is the authority for reproducible analysis. Genealogy child
indexes are derived from recorded parent links.

Logging is for progress and diagnostics. It is not parsed to reconstruct
scientific results, and its timing may differ after recovery. The CLI configures
logging; importing or embedding the library does not.

See [extensions](extensions.md) for application contracts,
[configuration](configuration.md) for settings, and
[checkpointing](checkpointing.md) for durability and restoration.
