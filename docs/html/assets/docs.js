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
