# xevolution documentation

`xevolution` is a Python library for evolutionary computation. An application
defines its candidates, evaluation, and variation operators; xevolution manages
the population, selection, identities, history, and checkpoint continuation.
This documentation covers **version 0.1.1**, including all three example
applications.

## Start here

If you are new to the package, read the [overview](overview.md) for the central
concepts and a walkthrough of one evolutionary run. Then choose an example
from the [examples guide](examples.md). Each example has its own explanation
of the genotype, fitness, operators, configuration, commands, and results.

| Example | What it demonstrates | Guide |
| --- | --- | --- |
| Arithmetic expression search | Evolve a bit string, decode its expression, and score its distance from a target | [Arithmetic expressions](arithmetic_demo.md) |
| Shared-resource choices | Evaluate the whole population when each candidate's score depends on other candidates | [Shared resources](batch_demo.md) |
| Digits neural architecture search | Evolve hidden layers and activation, train a fresh MLP, and use validation accuracy as fitness | [Digits MLP](digits_demo.md) |

The arithmetic and shared-resource examples use only the Python standard
library. The Digits example uses the optional `neural` dependency group and
runs on CPU. Every Python dependency belongs in a project-local virtual
environment managed by uv.

## Use the library

| Question | Documentation |
| --- | --- |
| What does xevolution do, and how does a run work? | [Package overview](overview.md) |
| Which example should I start with? | [Three examples](examples.md) |
| How do I change population size, selection, operators, or stopping? | [JSON configuration](configuration.md) |
| How do I run, pause, resume, inspect, or analyze an experiment? | [Command-line reference](cli.md) |
| How do I supply my own genotype and evaluator? | [Application extension contracts](extensions.md) |
| Which module owns each responsibility? | [Architecture and ownership](architecture.md) |
| What must be saved for reliable continuation? | [Checkpointing and recovery](checkpointing.md) |

Commands in these guides run from the **source project root**, containing
`pyproject.toml`, `uv.lock`, and `examples/`. This documentation archive can be
read independently; it does not contain the installable package or application
source files. Use it alongside the full xevolution Python project.

## Understand the port and its verification

The generic framework was ported from the Java Xnet `xevolution` sub-project.
The new design uses Python composition, typed records, explicit JSON state,
and application-defined extension contracts. NEAT, the graphical interface,
the operator workbench, and compatibility with old Java runs are outside the
agreed scope.

| Reference | Contents |
| --- | --- |
| [Porting notes](porting_notes.md) | Java-to-Python mappings, utility migration, and deliberate algorithm corrections |
| [Python release verification](verification.md) | Executed 0.1.1 checks, installed-wheel tests, checkpoint tests, and measured Digits results |
| [Full port specification](port_specification.md) | Agreed behavior, module design, algorithms, persistence, and acceptance criteria |
| [Contributor and agent guidelines](contributing.md) | Repository practices, uv environment policy, typing, testing, and state ownership |
| [Maintain and rebuild these docs](building_docs.md) | The Markdown/HTML layout, offline HTML generation, validation, and source maintenance |

## Read either edition

The Markdown edition starts at `docs/markdown/index.md`; the HTML edition
starts at `docs/html/index.html`. The HTML edition includes local navigation,
section links, highlighted code, and a search index. All reading assets are
bundled, so opening it does not require a web server or an internet connection.
External source references remain ordinary links to their original websites.

Both editions share the same Markdown content. Edit the Markdown and rebuild
the HTML using the [documentation build instructions](building_docs.md).

The bundle retains the project's [license text](resources/LICENSE.txt),
[accompanying GPL text](resources/LICENSE-GPL-3.0.txt), and
[source notice](resources/NOTICE.txt).

