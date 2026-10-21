# Maintain and rebuild the documentation

The documentation has two editions generated from one set of Markdown sources.
The HTML pages are ready to read by opening `docs/html/index.html` in a browser.
Reading requires no Python environment, web server, external stylesheet, or
downloaded font.

## Directory layout

| Path within `docs/` | Purpose |
| --- | --- |
| `README.md` | Entry instructions for the downloadable documentation bundle |
| `markdown/index.md` | Markdown documentation home |
| `markdown/*.md` | Authoritative prose for the package, examples, and references |
| `markdown/verification.json` | Preserved machine-readable evidence from the verified Python 0.1.1 release |
| `markdown/resources/` | Reference copies of the CI workflow and licensing notices |
| `html/index.html` | HTML documentation home; open this after extracting the ZIP |
| `html/*.html` | HTML pages generated from the corresponding Markdown files |
| `html/assets/` | Local stylesheet, optional navigation/search code, and generated search index |
| `build_docs.py` | Standard-library Python driver for Pandoc conversion and link validation |
| `build_report.json` | Results and source identities recorded by the documentation build |
| `browser_verification.json` | Browser checks recorded for the delivered documentation edition |

The main documentation pages are flat within each format directory. For
example, the Markdown link `[checkpoints](checkpointing.md)` becomes an HTML
link to `checkpointing.html`. Section fragments are retained. Supporting
resources are copied into the HTML tree at the same relative locations.

## Edit the source

Change the relevant file under `docs/markdown/`, then rebuild. A correction to
the HTML alone will be replaced on the next build. Use ordinary GitHub-flavored
Markdown: headings, paragraphs, lists, fenced code blocks, and tables.

Keep code fences labeled with their language, for example `python`, `bash`, or
`json`. Use relative links between documentation pages. Link to another section
using its heading fragment, and let the build validate that the target exists.
Avoid links that depend on files outside the documentation bundle.

When describing source code, name its path relative to the project root, such
as `examples/batch_problem.py`. A documentation-only archive cannot resolve a
clickable link to that external source file. Link to the relevant example guide
when the explanation is what the reader needs.

When adding a page, also add it to the grouped page catalog in `build_docs.py`
and link to it from [the documentation home](index.md) or an appropriate guide.
This keeps navigation and the generated search index complete.

The [port specification](port_specification.md) and
[coding guidelines](contributing.md) are reference copies of the repository's
`xevolution_python_port_spec.md` and `AGENTS.md`. Update the repository versions
first when their requirements change, then refresh these copies and preserve
their documentation introduction and local links. The reference workflow is a
snapshot of `.github/workflows/ci.yml`.

## Build HTML

Put this `docs/` directory in the xevolution source project. Use its
repository-local virtual environment managed by uv, and keep
`UV_PROJECT_ENVIRONMENT` unset or pointing to that `.venv`.

Install the standalone [Pandoc command-line tool](https://pandoc.org/installing.html)
using the method appropriate to your operating system. The documentation build
was developed with **Pandoc 3.1.3 and Python 3.12.14**. The Python driver uses
only the standard library; it adds no Python package dependency to xevolution.
Pandoc is a separate executable, not a Python dependency.

From the source project root:

```bash
uv sync --locked --group dev
pandoc --version
uv run --locked python docs/build_docs.py
```

If you already use the Digits example and want to retain its numerical stack,
include all configured dependency groups on the same commands:

```bash
uv sync --locked --all-groups
uv run --locked --all-groups python docs/build_docs.py
```

After a successful build, open `docs/html/index.html`. Keep the whole `html/`
directory together when copying it, including `assets/` and supporting files.
Keep its sibling `markdown/` directory too if you want the HTML pages' Markdown
source links to remain available.

The builder regenerates HTML and local assets. It checks local file links,
heading fragments, and duplicate HTML IDs; a missing document or broken local
target is a build failure. External reference websites are not downloaded by
the builder. The generated search works from local files and does not contact
a service.

To validate that the checked-in HTML still matches the Markdown and generator,
without replacing the generated files, run:

```bash
uv run --locked python docs/build_docs.py --check
```

Add `--all-groups` after `--locked` when retaining the Digits dependencies. The
check creates a temporary build, validates it, and compares the output files
byte for byte with the existing HTML tree. Use the same Pandoc version when
checking a previously generated edition.

## Check content as well as links

A working link does not establish that an API example is correct. When editing
commands or behavior, compare the guide with current code and configurations.
Run a small relevant example where the change warrants it. For documentation
changes, inspect representative HTML pages at desktop and narrow widths and
check long tables and code blocks.

The documentation release added full arithmetic and shared-resource guides,
an introductory package overview, an examples comparison, and this maintenance
guide. The actual arithmetic and shared-resource commands were checked again,
including analysis and shared-resource pause/resume. The Digits results remain
the measured results from the previously verified Python release.

The [release verification report](verification.md) and its JSON preserve that
Python release's original tests, timing measurements, and artifact identities.
Documentation edits do not imply that the full application test suite, wheel
build, or hosted CI has run again. Documentation build results are recorded
separately in `docs/build_report.json`.

The delivered HTML was also opened directly from disk in Chromium at desktop
(1440 × 1000) and phone (390 × 844) viewport sizes. All 16 pages were checked
for a visible title, current-page navigation, working search initialization,
and horizontal page overflow. Interaction checks covered local search, keyboard
controls, section links, mobile navigation, and reading/navigation with
JavaScript disabled. No browser errors or network requests occurred during
these checks. Their results are retained in `docs/browser_verification.json`;
the normal builder does not rerun those browser checks.

Return to the [documentation home](index.md) or the [package overview](overview.md).

