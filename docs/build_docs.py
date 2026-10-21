"""Build and validate the offline xevolution documentation with Pandoc.

Markdown is the authoritative content. The builder adds local navigation,
presentation, and optional search without requiring a Python package beyond
the standard library. Run through the project's uv environment.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

VERSION = "0.1.1"
GROUPS = (
    (
        "Start here",
        (
            ("index", "Documentation home"),
            ("overview", "Package overview"),
            ("examples", "Choose an example"),
        ),
    ),
    (
        "Example projects",
        (
            ("arithmetic_demo", "Arithmetic expressions"),
            ("batch_demo", "Shared resources"),
            ("digits_demo", "Digits MLP"),
        ),
    ),
    (
        "Reference",
        (
            ("configuration", "JSON configuration"),
            ("cli", "Command line"),
            ("extensions", "Application extensions"),
            ("architecture", "Architecture & ownership"),
            ("checkpointing", "Checkpoints & recovery"),
        ),
    ),
    (
        "Project",
        (
            ("porting_notes", "Porting notes"),
            ("verification", "Release verification"),
            ("port_specification", "Port specification"),
            ("contributing", "Contributing"),
            ("building_docs", "Building these docs"),
        ),
    ),
)

STYLE = r"""
:root{--navy:#112c40;--teal:#087c80;--teal-soft:#e6f3f1;--ink:#203747;--muted:#627583;--line:#dce5e8;--paper:#fff;--back:#f4f7f8;--header-height:76px;--sidebar-width:246px;--mono:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace}
*{box-sizing:border-box}html{scroll-padding-top:100px}body{margin:0;background:var(--back);color:var(--ink);font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}a{color:var(--teal);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#075b60}a:focus-visible,button:focus-visible,input:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #21aaa7;outline-offset:4px}[hidden]{display:none!important}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.skip-link{position:fixed;top:8px;left:12px;z-index:100;background:white;padding:10px 18px;border-radius:6px;transform:translateY(-150%)}.skip-link:focus{transform:translateY(0)}
.site-header{position:sticky;top:0;z-index:40;background:var(--navy);color:white;box-shadow:0 1px 0 #ffffff12}.header-inner{min-height:var(--header-height);display:flex;align-items:center;justify-content:space-between;gap:24px;padding:12px 28px}.brand{display:flex;align-items:center;gap:13px;color:white;text-decoration:none;flex-shrink:0}.brand:hover{color:white}.brand-mark{width:35px;height:35px;display:grid;place-items:center;background:#27b1aa;border-radius:9px;font-size:24px;font-weight:800;line-height:1;color:#102d40}.brand-name{font-size:21px;line-height:1.1;letter-spacing:-.5px;font-weight:750}.brand-note{display:block;margin-top:5px;font-size:11px;color:#bed1dc;font-weight:500;letter-spacing:1px;text-transform:uppercase}.header-right{display:flex;gap:18px;align-items:center}.version{font:12px var(--mono);padding:4px 9px;border:1px solid #8bb0c044;border-radius:5px;color:#d7e6ed}.search-wrap{position:relative;width:300px}.search-wrap input{width:100%;border:1px solid #87aabb50;border-radius:7px;background:#ffffff0c;color:white;padding:10px 38px 10px 13px;font:14px system-ui}.search-wrap input::placeholder{color:#b9cdd8}.search-key{position:absolute;right:11px;top:10px;border:1px solid #afcbd350;border-radius:3px;padding:0 5px;color:#bed1dc;font:12px/20px var(--mono)}.search-results{position:absolute;right:0;top:calc(100% + 12px);width:min(520px,calc(100vw - 32px));max-height:70vh;overflow:auto;background:white;color:var(--ink);border:1px solid var(--line);border-radius:10px;box-shadow:0 14px 40px #102c4030;padding:8px}.search-status{margin:4px 8px 8px;color:var(--muted);font-size:12px}.search-result{display:block;padding:10px 12px;border-radius:6px;text-decoration:none}.search-result:hover,.search-result:focus-visible{background:var(--teal-soft)}.search-result strong{display:block;font-size:14px;color:var(--ink);line-height:1.4}.search-result span{display:block;color:var(--muted);font-size:12px;line-height:1.55;margin-top:4px}
.layout{display:grid;grid-template-columns:var(--sidebar-width) minmax(0,1fr);align-items:start;max-width:1700px;margin:0 auto}.sidebar{position:sticky;top:var(--header-height);height:calc(100vh - var(--header-height));overflow-y:auto;background:#fbfcfd;border-right:1px solid var(--line);padding:24px 16px 30px}.sidebar>summary{display:none}.nav-group{margin:0 0 23px}.nav-heading{margin:0 10px 8px;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:1.15px;color:var(--muted)}.nav-group ul{list-style:none;margin:0;padding:0}.nav-group li{margin:2px 0}.nav-group a{display:block;padding:8px 11px;border-left:3px solid transparent;border-radius:0 5px 5px 0;color:#405665;text-decoration:none;font-size:13px;line-height:1.45}.nav-group a:hover{background:#ecf1f3}.nav-group a[aria-current=page]{background:var(--teal-soft);border-left-color:var(--teal);color:#08686c;font-weight:700}.sidebar-footer{margin:28px 12px 0;padding-top:15px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);line-height:1.65}
.content{min-width:0;padding:36px 40px 20px}.document-grid{display:grid;grid-template-columns:minmax(0,1fr) 184px;gap:32px;max-width:1180px;margin:auto}.article-column{min-width:0}.page-meta{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:22px;font-size:12px;color:var(--muted)}.breadcrumb{font-weight:650;letter-spacing:.15px}.breadcrumb a{text-decoration:none}.breadcrumb span{margin:0 8px;color:#8ea2ab}.source-link{white-space:nowrap;text-decoration:none;border:1px solid var(--line);padding:4px 10px;border-radius:5px;background:white;font-size:11px;font-weight:600}.document{background:var(--paper);border:1px solid var(--line);border-top:3px solid var(--teal);border-radius:8px;padding:34px 38px 40px;box-shadow:0 2px 5px #16394a03;min-width:0}.document>:first-child{margin-top:0}.document>h1{font-size:clamp(29px,2.55vw,39px);line-height:1.18;letter-spacing:-1.2px;font-weight:750;margin:0 0 25px;color:var(--navy);text-wrap:balance}.document>h1+p{font-size:17px;line-height:1.75;color:#425867}.document h2{font-size:24px;line-height:1.3;letter-spacing:-.55px;margin:42px 0 16px;padding-top:8px;color:var(--navy)}.document h3{font-size:18px;line-height:1.45;letter-spacing:-.15px;margin:29px 0 12px;color:#274c60}.document h4{font-size:16px;margin-top:26px}.document p{margin:0 0 18px}.document ul,.document ol{padding-left:1.4em;margin:0 0 22px}.document li{padding-left:3px;margin:6px 0}.document li p{margin-bottom:8px}.document strong{font-weight:700}.document blockquote{margin:24px 0;border-left:3px solid var(--teal);padding:13px 18px;background:#edf7f6;color:#31585c;border-radius:0 5px 5px 0}.document blockquote>:last-child{margin-bottom:0}.document hr{height:1px;border:0;background:var(--line);margin:34px 0}.document img,.document svg{max-width:100%;height:auto}.document code{font-family:var(--mono);font-size:.85em;color:#7c3548;background:#f3f5f6;padding:.16em .3em;border-radius:3px;overflow-wrap:anywhere}.document pre{margin:20px 0 24px;padding:18px 20px;max-width:100%;overflow:auto;border:1px solid #d9e3e8;border-radius:6px;background:#f5f8fa;line-height:1.55;font-size:13px;tab-size:4}.document pre code{font-size:1em;color:#263d4b;background:transparent;padding:0;border-radius:0;overflow-wrap:normal;white-space:pre}.document .sourceCode{max-width:100%;min-width:0}.document div.sourceCode{margin:0}.document code span.kw,.document code span.cf{color:#79509b;font-weight:600}.document code span.dt{color:#4765a4}.document code span.dv,.document code span.bn,.document code span.fl{color:#ad582f}.document code span.ch,.document code span.st,.document code span.ss{color:#21766a}.document code span.co,.document code span.do,.document code span.an{color:#637b83;font-style:italic}.document code span.fu{color:#2b648e}.document code span.al,.document code span.er{color:#a83240;font-weight:600}.document code span.op,.document code span.ot{color:#596571}.document code span.va{color:#284c65}.document code span.im,.document code span.bu{color:#76508b}.document pre code a{color:inherit;text-decoration:none}.table-scroll{overflow-x:auto;max-width:100%;margin:22px 0 27px;border:1px solid var(--line);border-radius:6px}.document table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.6;margin:0}.document thead{background:#eaf1f4;color:#244556}.document th{font-size:12px;font-weight:750;text-align:left;vertical-align:bottom}.document th,.document td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}.document tbody tr:last-child td{border-bottom:0}.document tbody tr:nth-child(even){background:#f8fafb}.document td code,.document th code{font-size:11.5px;white-space:normal}.document td{min-width:115px}.document .footnotes{font-size:13px;color:var(--muted)}
.page-toc{position:sticky;top:103px;max-height:calc(100vh - 130px);overflow:auto;padding-top:1px}.toc-title{font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:1px;margin:5px 0 14px;color:var(--muted)}.toc-list{list-style:none;padding:0;margin:0;border-left:1px solid var(--line)}.toc-list li{margin:0}.toc-list a{display:block;font-size:12px;line-height:1.5;padding:6px 0 6px 13px;color:#607583;text-decoration:none}.toc-list a:hover{color:var(--teal)}.toc-list .subsection a{padding-left:23px;font-size:11px;color:#72828e}.compact-toc{display:none;border:1px solid var(--line);padding:12px 16px;border-radius:6px;background:white;margin:0 0 20px}.compact-toc summary{font-size:13px;font-weight:650;cursor:pointer}.compact-toc .toc-list{margin-top:12px}.pagination{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:25px 0 30px}.pagination a{display:block;background:#fff;border:1px solid var(--line);border-radius:7px;padding:15px 18px;text-decoration:none;line-height:1.5}.pagination a:hover{border-color:#8abbbb;background:#fafdfe}.pagination .next{text-align:right}.pagination small{display:block;font-size:10px;font-weight:650;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:4px}.pagination strong{font-size:13px;font-weight:650}.page-footer{font-size:11px;line-height:1.6;color:var(--muted);padding:0 3px 25px}.page-footer a{color:var(--muted)}
@media(min-width:1500px){.content{padding-left:54px;padding-right:54px}.document{padding:40px 48px}}
@media(max-width:1200px){.document-grid{grid-template-columns:minmax(0,1fr);max-width:880px}.page-toc{display:none}.compact-toc{display:block}.content{padding:28px}.document{padding:30px 32px}}
@media(max-width:850px){:root{--sidebar-width:215px}.header-inner{padding:12px 18px}.version{display:none}.layout{display:block}.sidebar{position:static;height:auto;max-height:none;border-right:0;border-bottom:1px solid var(--line);padding:0;background:white}.sidebar>summary{display:block;padding:13px 20px;cursor:pointer;font-size:13px;font-weight:700;color:#385260;list-style:none}.sidebar>summary::after{content:"+";float:right;color:var(--teal);font-size:19px;line-height:20px}.sidebar[open]>summary::after{content:"−"}.sidebar nav{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 15px;padding:15px 16px 0}.sidebar-footer{display:none}.content{padding:24px 18px 10px}.document-grid{max-width:760px}.document{padding:28px 27px}.page-meta{margin-bottom:18px}.page-toc{display:none}}
@media(max-width:560px){.header-inner{flex-wrap:wrap;gap:12px;padding:14px 16px}.brand-note{font-size:10px}.brand-name{font-size:20px}.brand-mark{height:32px;width:32px}.header-right{width:100%}.search-wrap{width:100%}.search-wrap input{font-size:13px;padding:8px 36px 8px 12px}.search-key{top:7px}.search-results{left:0;right:auto;width:calc(100vw - 32px)}.content{padding:18px 12px 4px}.document{padding:23px 19px 28px}.document>h1{font-size:29px;letter-spacing:-.8px}.document>h1+p{font-size:16px}.document h2{font-size:22px;margin-top:32px}.document h3{font-size:18px}.document pre{padding:14px;font-size:12px}.document td,.document th{padding:10px 12px}.page-meta{font-size:11px;gap:8px}.source-link{font-size:10px;padding:4px 8px}.pagination{gap:8px}.pagination a{padding:12px}.pagination strong{font-size:12px}.pagination small{font-size:9px}.document p,.document li{overflow-wrap:break-word}}
@media(prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
@media print{html{scroll-padding-top:0}body{background:white;font-size:10pt;line-height:1.55}.site-header,.sidebar,.page-toc,.compact-toc,.pagination,.source-link,.skip-link{display:none!important}.layout,.document-grid{display:block}.content{padding:0}.page-meta{margin:0 0 12px}.document{padding:0;border:0;box-shadow:none}.document>h1{font-size:24pt}.document h2{font-size:17pt;break-after:avoid}.document h3{font-size:13pt;break-after:avoid}.document p{orphans:3;widows:3}.document pre{white-space:pre-wrap;font-size:8pt;background:#fafafa;padding:10px}.document pre code{white-space:pre-wrap;overflow-wrap:anywhere}.table-scroll{overflow:visible;break-inside:auto}.document table{font-size:8pt}.document td,.document th{padding:7px;min-width:0}.document tr{break-inside:avoid}.document a{color:#20566a}.page-footer{margin-top:18px}@page{margin:18mm}}
/* The search field adds a second header row on narrow screens. */
@media(max-width:560px){html{scroll-padding-top:136px}}
"""

SCRIPT = r"""
(() => {
  "use strict";
  const navigation = document.getElementById("xevo-navigation");
  const smallScreen = window.matchMedia("(max-width: 850px)");
  const fitNavigation = () => { navigation.open = !smallScreen.matches; };
  fitNavigation();
  smallScreen.addEventListener("change", fitNavigation);
  const entries = window.XEVO_SEARCH_INDEX;
  if (!Array.isArray(entries)) return;
  const wrap = document.getElementById("xevo-search");
  const input = document.getElementById("xevo-search-input");
  const results = document.getElementById("xevo-search-results");
  wrap.hidden = false;
  const close = () => { results.hidden = true; input.setAttribute("aria-expanded", "false"); };
  const search = () => {
    const query = input.value.trim().toLowerCase();
    if (query.length < 2) { close(); return; }
    const words = query.split(/\s+/);
    const matches = entries.map((entry) => {
      const title = entry.title.toLowerCase();
      const body = entry.text.toLowerCase();
      if (!words.every((word) => title.includes(word) || body.includes(word))) return null;
      const score = words.reduce((sum, word) => sum + (title.includes(word) ? 30 : 0) + (body.includes(word) ? 1 : 0), 0) + (title.includes(query) ? 60 : 0);
      return {entry, score};
    }).filter(Boolean).sort((a, b) => b.score - a.score || a.entry.order - b.entry.order);
    results.replaceChildren();
    const status = document.createElement("p");
    status.className = "search-status";
    status.textContent = matches.length ? `${matches.length} matching ${matches.length === 1 ? "page" : "pages"}${matches.length > 8 ? " · showing the first 8" : ""}` : "No matching pages. Try a different term.";
    results.append(status);
    for (const {entry} of matches.slice(0, 8)) {
      const link = document.createElement("a");
      link.className = "search-result";
      link.href = entry.url;
      const title = document.createElement("strong");
      title.textContent = entry.title;
      const snippet = document.createElement("span");
      const textLower = entry.text.toLowerCase();
      const positions = words.map((word) => textLower.indexOf(word)).filter((position) => position >= 0);
      const first = positions.length ? Math.min(...positions) : 0;
      const start = Math.max(0, first - 65);
      snippet.textContent = (start ? "…" : "") + entry.text.slice(start, start + 185) + (entry.text.length > start + 185 ? "…" : "");
      link.append(title, snippet);
      results.append(link);
    }
    results.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };
  input.addEventListener("input", search);
  input.addEventListener("focus", () => { if (input.value.trim().length >= 2) search(); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { close(); input.blur(); }
    if (event.key === "ArrowDown" && !results.hidden) {
      const first = results.querySelector("a");
      if (first) { event.preventDefault(); first.focus(); }
    }
  });
  results.addEventListener("keydown", (event) => { if (event.key === "Escape") { input.focus(); close(); } });
  document.addEventListener("click", (event) => { if (!wrap.contains(event.target)) close(); });
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (event.key === "/" && !(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement) && !target.isContentEditable) {
      event.preventDefault(); input.focus();
    }
  });
})();
"""


class BuildError(RuntimeError):
    """A source, converter, or portability contract prevents publication."""


@dataclass(frozen=True)
class Heading:
    level: int
    identifier: str
    text: str


@dataclass(frozen=True)
class Reference:
    target: str
    resource: bool


class DocumentIndex(HTMLParser):
    """Collect real Pandoc IDs and references rather than guessing Markdown IDs."""

    def __init__(self, content: str) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.duplicates: list[str] = []
        self.references: list[Reference] = []
        self.headings: list[Heading] = []
        self.words: list[str] = []
        self._heading: tuple[int, str] | None = None
        self._heading_text: list[str] = []
        self.feed(content)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        identifier = values.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicates.append(identifier)
            self.ids.add(identifier)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = (int(tag[1]), identifier or "")
            self._heading_text = []
        for attribute in ("href", "src", "poster", "data"):
            target = values.get(attribute)
            if target is not None:
                self.references.append(Reference(target, tag != "a"))
        if values.get("srcset"):
            for source in values["srcset"].split(","):
                self.references.append(Reference(source.strip().split()[0], True))
        if tag in {"p", "li", "td", "th", "pre", "h1", "h2", "h3", "h4"}:
            self.words.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._heading is not None and tag == f"h{self._heading[0]}":
            level, identifier = self._heading
            text = " ".join("".join(self._heading_text).split())
            self.headings.append(Heading(level, identifier, text))
            self._heading = None
        if tag in {"p", "li", "td", "th", "pre", "h1", "h2", "h3", "h4"}:
            self.words.append(" ")

    def handle_data(self, data: str) -> None:
        self.words.append(data)
        if self._heading is not None:
            self._heading_text.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self.words).split())


class LinkRewriter(HTMLParser):
    """Keep Pandoc content while changing local document destinations to HTML."""

    def __init__(self, content: str) -> None:
        super().__init__(convert_charrefs=False)
        self.output: list[str] = []
        self.feed(content)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.output.append(
                '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable table">'
            )
        rendered = []
        for key, value in attrs:
            if value is None:
                rendered.append(key)
                continue
            if key == "href":
                parsed = urlsplit(value)
                if (
                    not parsed.scheme
                    and not parsed.netloc
                    and parsed.path.endswith(".md")
                ):
                    value = urlunsplit(parsed._replace(path=parsed.path[:-3] + ".html"))
            rendered.append(f'{key}="{html.escape(value, quote=True)}"')
        self.output.append(f"<{tag}{' ' if rendered else ''}{' '.join(rendered)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self.output.append(f"</{tag}>")
        if tag == "table":
            self.output.append("</div>")

    def handle_data(self, data: str) -> None:
        self.output.append(data)

    def handle_entityref(self, name: str) -> None:
        self.output.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.output.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.output.append(f"<!--{data}-->")


@dataclass(frozen=True)
class Page:
    stem: str
    label: str
    group: str
    source: Path
    source_hash: str
    fragment: str
    index: DocumentIndex

    @property
    def title(self) -> str:
        return self.index.headings[0].text


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(source: Path, pandoc: str) -> tuple[str, str]:
    """Convert one captured UTF-8 source, reporting converter failures clearly."""
    data = source.read_bytes()
    process = subprocess.run(
        [
            pandoc,
            "--from=gfm",
            "--to=html5",
            "--wrap=none",
            "--highlight-style=pygments",
        ],
        input=data.decode("utf-8"),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise BuildError(f"Pandoc failed for {source.name}: {process.stderr.strip()}")
    return process.stdout, hashlib.sha256(data).hexdigest()


def navigation(current: str) -> str:
    sections = []
    for group, entries in GROUPS:
        links = []
        for stem, label in entries:
            selected = ' aria-current="page"' if stem == current else ""
            links.append(
                f'<li><a href="{stem}.html"{selected}>{html.escape(label)}</a></li>'
            )
        sections.append(
            f'<section class="nav-group"><h2 class="nav-heading">{html.escape(group)}</h2><ul>{"".join(links)}</ul></section>'
        )
    return "".join(sections)


def toc(page: Page) -> str:
    links = []
    for heading in page.index.headings:
        if heading.level not in {2, 3}:
            continue
        level = ' class="subsection"' if heading.level == 3 else ""
        anchor = quote(heading.identifier, safe="-_.~")
        links.append(
            f'<li{level}><a href="#{anchor}">{html.escape(heading.text)}</a></li>'
        )
    return f'<ul class="toc-list">{"".join(links)}</ul>'


def pagination(pages: Sequence[Page], position: int) -> str:
    links = []
    for offset, direction in ((-1, "Previous"), (1, "Next")):
        index = position + offset
        if not 0 <= index < len(pages):
            links.append('<span aria-hidden="true"></span>')
            continue
        page = pages[index]
        label = html.escape(page.label)
        links.append(
            f'<a class="{direction.lower()}" href="{page.stem}.html"><small>{direction}</small><strong>{label}</strong></a>'
        )
    return f'<nav class="pagination" aria-label="Previous and next articles">{"".join(links)}</nav>'


def render(page: Page, pages: Sequence[Page], position: int) -> str:
    title = html.escape(page.title)
    description = html.escape(page.index.text[:180], quote=True)
    contents = "".join(LinkRewriter(page.fragment).output)
    local_toc = toc(page)
    source_link = f"../markdown/{page.stem}.md"
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<title>{title} · xevolution documentation</title>
<link rel="stylesheet" href="assets/docs.css">
<script defer src="assets/search-index.js"></script>
<script defer src="assets/docs.js"></script>
</head>
<body>
<a class="skip-link" href="#xevo-main">Skip to content</a>
<header class="site-header"><div class="header-inner">
<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">x</span><span class="brand-name">xevolution<span class="brand-note">Evolutionary computation · Python</span></span></a>
<div class="header-right"><span class="version">v{VERSION}</span>
<div class="search-wrap" id="xevo-search" role="search" hidden>
<label class="sr-only" for="xevo-search-input">Search documentation</label>
<input id="xevo-search-input" type="search" placeholder="Search the documentation" autocomplete="off" aria-controls="xevo-search-results" aria-expanded="false">
<kbd class="search-key" aria-hidden="true">/</kbd>
<div id="xevo-search-results" class="search-results" role="region" aria-label="Search results" aria-live="polite" hidden></div>
</div></div></div></header>
<div class="layout">
<details class="sidebar" id="xevo-navigation" open><summary>Browse documentation</summary>
<nav aria-label="Documentation">{navigation(page.stem)}</nav>
<div class="sidebar-footer">Version {VERSION}<br>Offline reference documentation.<br>Markdown is the editable source.</div>
</details>
<main class="content" id="xevo-main"><div class="document-grid">
<div class="article-column">
<div class="page-meta"><div class="breadcrumb"><a href="index.html">Docs</a><span>/</span>{html.escape(page.group)}</div><a class="source-link" href="{source_link}">Markdown source</a></div>
<details class="compact-toc"><summary>On this page</summary>{local_toc}</details>
<article class="document">{contents}</article>
{pagination(pages, position)}
<footer class="page-footer">xevolution {VERSION} documentation · <a href="building_docs.html">Build and edit these docs</a><br>Framework behavior is documented for the supplied Python port. See <a href="verification.html">release verification</a> for the scope of executed checks.</footer>
</div>
<aside class="page-toc"><nav aria-label="On this page"><div class="toc-title">On this page</div>{local_toc}</nav></aside>
</div></main></div>
</body></html>
'''


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def validate(
    documents: dict[Path, DocumentIndex],
    all_documents: dict[Path, DocumentIndex],
    docs_root: Path,
    stage: Path,
) -> dict[str, int]:
    """Validate local file/fragment destinations and reject remote read assets."""
    failures: list[str] = []
    counts = {
        "documents": len(documents),
        "local_references": 0,
        "fragment_references": 0,
        "external_links": 0,
    }
    final_html = docs_root / "html"
    for source, document in documents.items():
        name = source.name
        if document.duplicates:
            failures.append(f"{name}: duplicate IDs {document.duplicates}")
        for reference in document.references:
            parsed = urlsplit(reference.target)
            if parsed.scheme or parsed.netloc:
                if reference.resource:
                    if parsed.scheme != "data":
                        failures.append(f"{name}: remote asset {reference.target!r}")
                elif parsed.scheme not in {"https", "http", "mailto"}:
                    failures.append(
                        f"{name}: unsupported link scheme {reference.target!r}"
                    )
                else:
                    counts["external_links"] += 1
                continue
            if parsed.path.startswith("/"):
                failures.append(f"{name}: absolute local path {reference.target!r}")
                continue
            candidate = (
                (source.parent / unquote(parsed.path)).resolve()
                if parsed.path
                else source
            )
            if candidate.is_relative_to(final_html):
                candidate = stage / candidate.relative_to(final_html)
            if not candidate.is_relative_to(docs_root):
                failures.append(
                    f"{name}: link leaves the documentation archive: {reference.target!r}"
                )
                continue
            counts["local_references"] += 1
            if not candidate.is_file():
                failures.append(f"{name}: missing local target {reference.target!r}")
                continue
            if parsed.fragment:
                counts["fragment_references"] += 1
                target = all_documents.get(candidate)
                anchor = unquote(parsed.fragment)
                if target is None or anchor not in target.ids:
                    failures.append(f"{name}: missing fragment {reference.target!r}")
    if failures:
        raise BuildError("Documentation validation failed:\n  " + "\n  ".join(failures))
    return counts


def build(docs_root: Path, pandoc: str, *, check: bool) -> dict[str, object]:
    """Build a complete validated tree, then publish it or check existing output."""
    source_root = docs_root / "markdown"
    output_root = docs_root / "html"
    expected = {stem for _, entries in GROUPS for stem, _ in entries}
    actual = {path.stem for path in source_root.glob("*.md")}
    if actual != expected:
        raise BuildError(
            f"Markdown navigation mismatch; missing={sorted(expected - actual)}, unregistered={sorted(actual - expected)}"
        )
    version = subprocess.run(
        [pandoc, "--version"], text=True, capture_output=True, check=True
    ).stdout.splitlines()[0]
    pages: list[Page] = []
    for group, entries in GROUPS:
        for stem, label in entries:
            source = source_root / f"{stem}.md"
            fragment, source_hash = convert(source, pandoc)
            index = DocumentIndex(fragment)
            if (
                not index.headings
                or index.headings[0].level != 1
                or sum(h.level == 1 for h in index.headings) != 1
            ):
                raise BuildError(f"{source.name}: provide exactly one leading H1 title")
            if any(not heading.identifier for heading in index.headings):
                raise BuildError(
                    f"{source.name}: every heading needs a generated or explicit ID"
                )
            pages.append(Page(stem, label, group, source, source_hash, fragment, index))
    stage = Path(tempfile.mkdtemp(prefix=".html-build-", dir=docs_root))
    try:
        assets = stage / "assets"
        assets.mkdir()
        (assets / "docs.css").write_text(STYLE.strip() + "\n", encoding="utf-8")
        (assets / "docs.js").write_text(SCRIPT.strip() + "\n", encoding="utf-8")
        entries = [
            {
                "title": page.title,
                "url": f"{page.stem}.html",
                "text": page.index.text,
                "order": position,
            }
            for position, page in enumerate(pages)
        ]
        search = (
            json.dumps(
                entries, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            .replace("<", "\\u003c")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
        (assets / "search-index.js").write_text(
            f"window.XEVO_SEARCH_INDEX = {search};\n", encoding="utf-8"
        )
        for source in sorted(source_root.rglob("*")):
            if source.is_symlink():
                raise BuildError(f"Symbolic-link resources are not portable: {source}")
            if not source.is_file() or source.suffix == ".md":
                continue
            destination = stage / source.relative_to(source_root)
            if destination.exists():
                raise BuildError(
                    f"Resource collides with generated output: {source.name}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        rendered: dict[Path, DocumentIndex] = {}
        for position, page in enumerate(pages):
            destination = stage / f"{page.stem}.html"
            document = render(page, pages, position)
            destination.write_text(document, encoding="utf-8")
            rendered[destination] = DocumentIndex(document)
        originals = {page.source: page.index for page in pages}
        all_documents = originals | rendered
        markdown_counts = validate(originals, all_documents, docs_root, stage)
        html_counts = validate(rendered, all_documents, docs_root, stage)
        for page in pages:
            if digest(page.source) != page.source_hash:
                raise BuildError(
                    f"{page.source.name} changed during the build; run again"
                )
        source_hashes = tree_hashes(source_root)
        output_hashes = tree_hashes(stage)
        report: dict[str, object] = {
            "schema_version": 1,
            "package_version_documented": VERSION,
            "generator": "docs/build_docs.py",
            "generator_sha256": digest(Path(__file__)),
            "converter": version,
            "python_version": sys.version.split()[0],
            "markdown_pages": len(pages),
            "html_pages": len(rendered),
            "html_files_including_assets": len(output_hashes),
            "validation": {
                "markdown": markdown_counts,
                "html": html_counts,
                "duplicate_ids": 0,
                "missing_targets": 0,
                "missing_fragments": 0,
                "remote_read_assets": 0,
            },
            "source_sha256": source_hashes,
            "html_sha256": output_hashes,
            "offline_reading": True,
            "search_requires_fetch": False,
            "core_tests_rerun": False,
        }
        if check:
            if not output_root.is_dir() or tree_hashes(output_root) != output_hashes:
                raise BuildError(
                    "Generated HTML is missing or out of date; run without --check"
                )
            return report
        if output_root.is_symlink():
            raise BuildError("Refusing to replace a symbolic-link HTML directory")
        backup = docs_root / (stage.name + "-previous")
        if output_root.exists():
            output_root.replace(backup)
        try:
            stage.replace(output_root)
        except OSError:
            if backup.exists():
                backup.replace(output_root)
            raise
        finally:
            if backup.exists() and output_root.exists():
                shutil.rmtree(backup)
        (docs_root / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local build; --check validates without replacing generated files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pandoc", default="pandoc", help="Pandoc executable (tested with 3.1.3)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that existing HTML exactly matches a fresh build",
    )
    args = parser.parse_args(argv)
    try:
        report = build(Path(__file__).resolve().parent, args.pandoc, check=args.check)
    except (BuildError, OSError, UnicodeError, subprocess.CalledProcessError) as error:
        print(f"docs: {error}", file=sys.stderr)
        return 1
    action = "Checked" if args.check else "Built"
    print(
        f"{action} {report['html_pages']} HTML pages from {report['markdown_pages']} Markdown sources; all local links, assets, fragments, and IDs validated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
