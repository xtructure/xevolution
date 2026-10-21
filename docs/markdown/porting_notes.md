# Porting notes

The source baseline is the generic Java [`xevolution` subproject at commit
`509c673e502f0c27daa6b91f39cc298404b59ae2`](https://github.com/xtructure/xnet-public/tree/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution).
The implementation follows the repository's
[port specification](port_specification.md).

## Scope and mapping

| Java responsibility | Python owner |
| --- | --- |
| Genome, population, attributes, and identity | `models.py`, with explicit IDs, typed core fields, and named metrics |
| Genetics factories and genome decoders | Application initialization, cloning, codecs, and optional decoding composed in `experiment.py` |
| Evolution strategy | `engine.py` |
| Evaluation strategies | `evaluation.py` and application evaluation hooks |
| Parent selection and survival filters | Separate policies in `selection.py` |
| Mutation, crossover, and operator selection | `operators.py` and `reproduction.py` |
| Configuration, ranges, and stopping conditions | `config.py` and `termination.py` |
| XML persistence | Explicit versioned representations in `serialization.py` and durable file operations in `storage.py` |
| Population statistics and genealogy tools | `tracking.py` and the `inspect`/`analyze` commands |
| Logging and command-line option wrappers | Python `logging` and `argparse` |

Needed `xutil` behavior is incorporated where it belongs. Collection builders,
getter/setter wrappers, lazy-construction holders, and Java interface/abstract/
implementation hierarchies are not reproduced as parallel Python hierarchies.

NEAT, simulation implementations, batch-simulation infrastructure, all graphical
interfaces, screenshot tools, and the unfinished operator workbench are excluded.
Application evaluators may use their own simulators, numerical libraries, or
execution backends. The framework itself has no such runtime dependencies.

## Intended algorithmic behavior retained

- Generation 0 is the evaluated initial population.
- Parent selection and survivor selection are separate. The complete previous
  population remains eligible for parent selection, including nonsurvivors.
- Complete replacement is the default. Every member of the resulting population,
  including a retained survivor, is evaluated each generation.
- The mutation probability chooses mutation **or** crossover for one offspring.
  It does not schedule a second mutation stage after crossover.
- The objective is one finite scalar fitness, maximized by default. Additional
  named measurements support analysis without becoming extra objectives.
- Whole-population evaluation is a first-class application hook. It may depend
  jointly on all candidates rather than reproduce a loop of independent scores.

The [Java generation lifecycle](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractEvolutionStrategy.java#L243-L265)
and the [application batch-evaluation example](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/demos/src/main/java/com/xtructure/xnet/demos/oned/OneDEvaluationStrategyImpl.java#L94-L147)
establish these extension and lifecycle boundaries.

## Deliberate corrections and changes

1. **Explicit JSON state replaces Java serialization.** Configuration,
   checkpoints, and scientific history have versioned representations. Application
   codecs preserve genotype semantics. Java files and Java random trajectories
   are not compatibility targets.
2. **Fresh creation and restoration are separate operations.** The Java runner
   [loads the most recently modified XML file in its output directory](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractEvolutionExperiment.java#L154-L169).
   Python resumes only the explicit authoritative checkpoint and does not
   initialize or reevaluate its saved population.
3. **Reproduction is bounded.** Positive-weight operators are tried without
   replacement within an attempt, followed by explicit fallbacks. Each child has
   a finite attempt budget. Only the defined inapplicability condition is
   recoverable; programming failures propagate.
4. **Selection weights are validated.** Direct roulette weights must be finite
   and nonnegative. All-zero weights select uniformly. Negative fitness is valid
   data but requires a compatible selector or an explicit weight transformation;
   it does not silently remove selection pressure.
5. **Saved counters allocate identities.** Removed individuals cannot cause an
   ID to be reused after restoration. Genotype comparison and hashing do not
   determine identity.
6. **Newborn age is zero.** Age is the current generation minus birth generation,
   rather than a separately mutable field.
7. **Missing measurements remain missing.** Each metric's statistics include its
   own contributing count. Missing values are not replaced by zero or divided by
   the full population size.
8. **Historical observations are independent.** Retained best individuals and
   scientific records cannot be rewritten by later mutation or reevaluation of
   live candidates. Equal fitness under a different ID is not an improvement.
9. **The framework owns accepted-evaluation accounting.** Evaluators return
   results. The coordinator validates complete ID coverage before applying any
   scores or incrementing counts.
10. **Scientific history shares the checkpoint boundary.** Only the committed
    journal prefix contributes to reports. A resumed writer discards an
    uncommitted tail before appending; missing or corrupt committed data is an
    error.

The [original reproduction loop](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/evolution/impl/AbstractReproductionStrategy.java#L156-L265),
[operator-selection reset behavior](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/operator/impl/AbstractOperatorSelecter.java#L90-L183),
[weight normalization](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xutil/src/main/java/com/xtructure/xutil/RandomUtil.java#L562-L578),
and [population statistics](https://github.com/xtructure/xnet-public/blob/509c673e502f0c27daa6b91f39cc298404b59ae2/xnet/xevolution/src/main/java/com/xtructure/xevolution/genetics/impl/AbstractPopulation.java#L537-L579)
are the reference points for the corresponding corrections.

Exact continuation is a contract for compatible application code, data, state,
and numerical execution. It does not imply cross-version or cross-platform
numerical equivalence, or exactly-once external side effects. See
[checkpointing](checkpointing.md).
