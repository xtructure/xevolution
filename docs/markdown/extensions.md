# Application extension contracts

Applications provide ordinary functions or compatible objects. No framework
base class is required. Core bookkeeping stays in the framework, and arbitrary
genotype payloads stay under application control.

## Constructing an experiment

A configured factory has this signature:

```python
def build_experiment(config: RunConfig, random: RunRandom) -> Experiment[G]: ...
```

Use `config.parameters` for application settings and `config.evolution` for
resolved core settings. Validate your parameter namespace, including unknown
keys. The factory constructs collaborators without creating the initial
population or performing evaluation; restoration also calls the factory.

`Experiment` accepts these collaborators:

| Field | Contract |
| --- | --- |
| `initialize` | `(rng: Random) -> G`; produce one initial genotype |
| `clone` | `(genotype: G) -> G`; preserve independent ownership where required |
| `evaluate` | `(individual: Individual[G], context: EvaluationContext) -> EvaluationResult` |
| `evaluate_population` | `(population: Population[G], context: EvaluationContext) -> Mapping[int, EvaluationResult]` or an iterable of `(id, result)` pairs |
| `mutations` | Tuple of named `OperatorSpec` entries with `(genotype, rng) -> VariationResult[G]` operations |
| `crossovers` | Tuple of named entries with `(first, second, rng) -> VariationResult[G]` operations |
| `select_parent` | `(population, rng) -> individual_id` |
| `select_survivors` | `(population, target_size, rng) -> Sequence[individual_id]` |
| `stop` | `(finalized_state) -> StopDecision` or `bool` |
| `adaptive_hooks` | Tuple of `(state) -> EvolutionSettings | None` hooks |
| `codec` | A `PayloadCodec[G]` with explicit identity, version, encode, and decode |
| `components` | Stable named additional stateful collaborators |
| `identity`, `version` | Application compatibility identity and version |
| `compatibility` | Explicit JSON fingerprints for application data or external library versions |

Absent selection and stopping collaborators are constructed from configuration.
A stopping policy must be supplied by the experiment or configuration. The
default JSON codec is suitable only for genuinely JSON-native payloads. Supply a
custom codec for tuples, arrays, bytes, graphs, or application objects.

## Genotype and observation ownership

The engine does not assume that a genotype supports equality, ordering, hashing,
truth testing, length, or indexing. Individuals are identified by their explicit
IDs. Application operators may of course use operations supported by their own
representation.

An operator must leave both parents unchanged and return an independent child,
unless sharing explicitly immutable data is safe. The framework assigns the
offspring ID, birth generation, parents, fitness validity, and evaluation count.
Do not copy a parent's wrapper and carry those fields into a child.

A frozen dataclass does not freeze nested payloads. Supply a clone that handles
your representation correctly. Historical bests use it to retain independent
genotype snapshots; metadata and measurements are preserved separately. For an
immutable string or integer, `clone(value): return value` is valid.

Optional decoding belongs naturally inside the evaluator. The arithmetic
example composes a `decode()` function with scoring and does not need a dummy
decoder object in the engine.

## Evaluation

Return one finite scalar fitness and an optional mapping of named finite numeric
metrics. Zero is a valid score. Signed scores are valid; choose a selector that
supports them. Metadata and encoded payloads have separate JSON contracts and
should not be placed into numeric metrics.

If `evaluate_population` is supplied, it runs directly, even if `evaluate` is
also present. Otherwise the framework adapts the individual hook. A population
hook may score candidates jointly or compare them in a shared environment; it
need not match independent scoring.

The result set must contain each requested ID exactly once and no other IDs.
Sequence results can arrive in arbitrary order; the coordinator checks coverage
and duplicates and applies all accepted results in deterministic population
order. It validates the complete batch before applying scores or incrementing
counts. One accepted result counts as one fitness evaluation; report underlying
simulations or training steps as supplementary metrics.

An individual evaluation receives `context.rng`. A batch can derive stable job
contexts with `context.for_individual(id, replicate=0)`. These logical contexts
are independent of worker completion order. Applications own their execution
backend and must return with no work in flight at the checkpoint boundary.

See [the shared-resource example](batch_demo.md), whose score depends on the
number of other candidates choosing the same resource. Its implementation is
`examples/batch_problem.py` in the source project. The [arithmetic](arithmetic_demo.md)
and [Digits](digits_demo.md) guides illustrate the individual hook.

## Variation and selection

Register operators with stable names:

```python
OperatorSpec("domain_mutation", mutate, weight=2.0)
OperatorSpec("copy_fallback", CloneMutation(clone), fallback=True)
```

Import these helpers from `xevolution.operators`. Weighted entries are tried
without replacement within an attempt; weight zero disables weighted selection.
Fallbacks are an explicit separate designation and run in registration order
after weighted choices, including an explicitly declared fallback whose weight
is zero. A new attempt gets a fresh pool and new parent choices. The configured
per-child budget bounds retries.

Raise `xevolution.errors.OperatorInapplicable` only when an operation cannot
apply to the selected parent(s). Unexpected exceptions remain failures, not
signals to silently try another operation. Return operation details in
`VariationResult.details`; the framework records stable parent and operator
identities.

`OperatorSpec.weight` is a validated mutable value and is checkpointed. Changes
that affect future behavior must occur in registered adaptive/stateful logic,
not observation-only reporting callbacks.

Custom selectors return framework IDs from the supplied population. Parent
selection sees the full previous population. Survivor selection independently
chooses retained individuals, whose fitness will be reevaluated in the resulting
population.

## Stateful components and adaptation

Any collaborator retaining future-affecting information implements:

```python
def state_dict(self) -> JSONValue: ...


def load_state_dict(self, state: JSONValue) -> None: ...
```

State export is observational and returns an explicit JSON-native description.
Restoration validates its schema and contents. Core collaborators exposing this
contract are tracked automatically; use `Experiment.components` for additional
shared state such as an evaluator environment, curriculum, or bounded parameter.
Keep registration keys stable. Stateless functions may close over immutable
configuration values, but mutable closure cells and global state must not hide
future-affecting information.

Frozen dataclasses are suitable for immutable component configuration, as the
examples demonstrate. Their nested fields must also satisfy the configuration
or registered-component contracts; a frozen wrapper does not make a captured
list immutable. A `Stateful` object reached only through a function capture must
be registered in `Experiment.components`, even when it exposes the state methods.
Direct collaborators are registered automatically, and bound methods sharing
one registered owner are restored only once.

A stateful component may additionally implement `checkpoint_identity() ->
JSONValue` to report immutable configuration that must match during restoration,
such as an algorithm revision or a dataset digest. This optional identity is
compared before loading mutable state; it does not replace `state_dict()`.
Identity and state export must be observational and return strict JSON values.

An adaptive hook receives finalized observations and may return new validated
settings for the **next** transition. It may update its own explicitly saved
state. It must not mutate the finalized population or its observations. The
checkpoint records both `generation_settings` and the next
`effective_settings`.

Stopping runs once after these updates for each completed generation. Stateful
custom predicates must save their windows and counters. Restoration honors the
recorded decision instead of invoking the predicate again.

## Codecs and attachments

A payload codec defines `codec_id: str`, `version: int`, and:

```python
def encode(self, value: G, writer: AttachmentWriter | None = None) -> JSONValue: ...


def decode(self, value: JSONValue, reader: AttachmentReader | None = None) -> G: ...
```

Encode required semantics explicitly. The built-in `JsonCodec` rejects lossy
coercions; it is not a generic object-graph serializer. Large or binary payloads
can use `writer.put(bytes, media_type=...)` and `reader.get(reference)`. Codecs
must reject a missing attachment context when they require one. Storage owns
attachment integrity, ordering, and durability.

Codec calls and component state export must not consume evolutionary randomness
or mutate application state. For configured factories, restoration first
reinstalls the saved pre-construction random snapshot and calls the factory with
resolved parameters. Reconstructed components then receive their explicit
state, payloads are decoded, and the completed-generation managed random states
are installed last. Keep references to the supplied managed generators rather
than replacing them during restoration.

For application changes that alter state interpretation or future behavior,
update compatibility identities/versions deliberately. See
[checkpointing](checkpointing.md) for the full contract.
