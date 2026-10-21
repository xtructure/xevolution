# xevolution documentation bundle

This directory contains Markdown and HTML documentation for xevolution 0.1.1.

- **Read in a browser:** open [html/index.html](html/index.html) after extracting
  the ZIP. The HTML edition works offline and needs no web server.
- **Read or edit Markdown:** start at [markdown/index.md](markdown/index.md).
- **Rebuild the HTML:** follow [markdown/building_docs.md](markdown/building_docs.md).

The docs include a package overview; full guides for arithmetic expressions,
shared-resource population evaluation, and Digits MLP architecture evolution;
configuration, CLI, extension, architecture, and checkpoint references; and the
port specification, coding guidelines, and preserved release-verification data.

The application commands run from the full xevolution source project, beside
`pyproject.toml` and `examples/`. This ZIP contains documentation and its build
driver, not the installable library or example source code.

Keep `markdown/` and `html/` together to retain all local cross-links and HTML
links to their Markdown source pages. Edit Markdown first; HTML is generated.
All Python dependencies used with the project must be installed in its local
virtual environment managed by uv.

When replacing an older `docs/` directory, update project links from
`docs/<page>.md` to `docs/markdown/<page>.md`, or use the corresponding
`docs/html/<page>.html` browser link. Both editions now have an `index` page.

