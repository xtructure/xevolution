# Checkpoints and continuation

A checkpoint restores the same logical evolutionary run when application code,
input data, component state, and numerical execution satisfy the documented
contracts. Saving only an initial seed or a list of genotypes is insufficient.

## Saved boundary

Persistent runs checkpoint generation 0 and every completed generation. The
saved boundary follows complete evaluation, validation and application of all
results, scientific tracking, adaptive updates, and one stopping decision. No
evaluation jobs remain in flight. The next action is either a transition to the
next generation or no further evolution because the run is terminated.

Restoration does not rerun initialization, saved-generation evaluation, adaptive
updates, or a stateful stopping predicate. A saved terminal decision remains
terminal under ordinary `resume`. An operational pause requests a stop at the
boundary and leaves a ready checkpoint unless an evolutionary predicate has
independently terminated the run.

## Required state

Checkpoints preserve ordered individuals and genotype payloads; IDs, births, and
parentage; fitness and other observations; allocation and evaluation counters;
completed-generation and next-transition settings; historical bests and
improvement information; stopping decisions; component identities and state;
all managed random states; codec and schema versions; and the identity,
integrity, and byte length of the committed journal prefix.

The evolution and evaluation random streams are separate. Stable evaluation
contexts derive work identities from logical generation/individual/round/
replicate information, not worker order, process IDs, wall time, Python's
randomized hash, or a fresh run identifier.

Factory-based runs also save the original managed random state immediately
after initial parameter sampling and before factory construction, under
`compatibility.construction_random`. Resume installs that snapshot before
reconstructing the factory with resolved settings, so random choices made while
constructing collaborators can be reproduced. After component-state restoration
and genotype decoding, the saved completed-generation random state is installed
last. Initialization and evaluation still do not repeat. A programmatic run
without a recorded factory requires an explicitly supplied, compatible
`Experiment` when resumed.

Stateful application objects must export and restore every value affecting
future results. This includes adaptive probabilities, stopping windows,
application generators, simulator or optimizer state, and meaningful caches.
Stateless functions need no payload, but hidden mutable globals or closures do
not become checkpointable automatically.

## Serialization

Framework and application records use explicit codecs and versioned JSON.
Required genotype semantics must survive encode/decode. Encoding and state
export are observational: they must not mutate the run, advance its random
generators, or evaluate candidates.

JSON object keys are strings. Strict boundaries reject duplicate keys, unsupported
versions, invalid types, and non-finite numbers, including values that overflow
during parsing. Booleans are not numeric settings or fitness values. Tuple-based
random state is reconstructed deliberately rather than passed back as nested
JSON lists.

The payload contract allows immutable attachments so application data need not
always expand into JSON arrays. Attachment references carry format/version and
integrity information. Referenced attachments must be durable before their
checkpoint becomes authoritative.

Storage writes attachment bytes to `artifacts/<sha256>.bin`. It reserves the
JSON object key `__xevolution_attachment__` for references containing version,
SHA-256 digest, byte size, and media type. Use `writer.put()` and `reader.get()`
instead of constructing these references or writing files from a codec.

## Commit order and visibility

One coordinator writes a run. Workers return results rather than write history
or checkpoints themselves. A commit proceeds in this order:

1. Finalize state and prepare independent scientific records and payloads.
2. Make required immutable attachments durable.
3. Append scientific records to `history.jsonl`, flush, and synchronize the file.
4. Write and synchronize a temporary checkpoint in the run directory. It names
   the committed history byte offset and integrity information.
5. Atomically replace `checkpoint.json` and synchronize the directory entries
   required by the platform's local-file durability protocol.
6. Report success; any subsequent reports read committed records.

Atomic replacement establishes visibility to readers in the running system.
Power-loss durability additionally depends on successful file and directory
synchronization and the filesystem's guarantees. Failure before directory
synchronization may leave an older or newer checkpoint after a machine failure;
recovery validates whichever checkpoint survived. Generation 0 has no preceding
checkpoint to recover if its initial publication did not survive.

These are local-file guarantees. A remote or unusual filesystem can have weaker
semantics. A failure to save is reported as a failure to make the new generation
durable, not as a successful evolutionary commit.

The implementation synchronizes directory entries on POSIX systems. On other
platforms it supplies atomic checkpoint publication and process-crash recovery,
but does not claim power-loss durability for directory changes. A writable
session holds the run's writer lock until `close()`; use engine/store context
managers to release it reliably.

## Recovery and journal tails

The active checkpoint determines the authoritative journal prefix. An abrupt
failure may leave extra bytes in `history.jsonl` after that prefix. Read-only
inspection and analysis ignore them. A resumed writer truncates or resets the
journal to the saved committed offset **before** appending new records.

Missing or corrupt committed history or attachments is an error. Recovery does
not invent replacement observations or reevaluate candidates from an already
committed generation. Temporary checkpoint files and arbitrary recently modified
files do not supersede `checkpoint.json`.

A portable run is a bundle: its checkpoint, required journal prefix, and any
referenced attachments, together with compatible application code and data.
Copying a checkpoint manifest alone does not necessarily create a resumable run.

## Application and environment limits

- Work in an interrupted, uncommitted generation can repeat. External calls
  needing deduplication must use application-level idempotency or result lookup.
- Continuation inside an arbitrary long evaluation is not supplied by the
  framework. A training evaluator may need model, optimizer, scheduler,
  data-position, and application random state of its own.
- Changing external data or using nondeterministic numerical operations can
  change future results even when engine state restores exactly.
- Supporting multiple Python versions is not a promise of identical trajectories
  or checkpoint portability between those versions.
- Operational messages and their timestamps need not match across executions.

Compatibility checks must succeed before a resumed run claims exact
continuation. Keep the application's factory, codec identifiers/versions, and
registered state keys stable; use a new compatibility version when semantics
change.

The current checks compare framework source fingerprints, package and schema
versions, Python implementation/version and platform information, application
identity/version and explicit `compatibility` entries, codec identity/version,
and the registered component composition. A source edit or a Python patch
upgrade can therefore require the original environment to resume a saved run.
These checks are deliberately strict; `inspect` and `analyze` can still read
committed records without loading the application factory.
