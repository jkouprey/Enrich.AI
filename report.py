"""
HTML report for one Enrich.AI response.

Self-contained file: Enrich.AI header (logo, name, version if set), the question, the response,
the result tables and plots exactly as passed in (customised dialog figures included), and a
short methodology section built from the tool calls the agent actually made.
"""

import ast
import html
import re
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import markdown
import pandas as pd
import plotly.graph_objects as go

APP_VERSION = None  # e.g. "1.1.0" - shown in the report header when set

TOOL_LABELS = {
    "get_gene_info": "Gene information (MyGene.info)",
    "db_retrieve": "Database query (Enrichr libraries)",
    "run_enrichment_analysis": "Over-representation analysis (Enrichr)",
    "run_gsea": "Preranked GSEA (gseapy)",
    "search_literature": "Literature search (Europe PMC)",
    "get_paper_annotations": "Paper annotations (Europe PMC)",
}

CSS = """
:root { --bg:#f5f7fb; --ink:#0f172a; --muted:#64748b; --line:#e5e7eb; --card:#fff;
        --brand1:#6366f1; --brand2:#8b5cf6; --brand3:#06b6d4;
        --term:#3b82f6; --cluster:#8b5cf6; --net:#06b6d4; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 16px; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.65 Inter, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
a { color:var(--brand1); }

/* hero */
.hero { position:relative; overflow:hidden; color:#e2e8f0;
        background: radial-gradient(900px 300px at 10% -20%, rgba(99,102,241,.55), transparent 60%),
                    radial-gradient(700px 260px at 90% 0%, rgba(6,182,212,.35), transparent 60%),
                    linear-gradient(135deg,#0b1020 0%,#151a3a 55%,#1e1b4b 100%); }
.hero::after { content:""; position:absolute; inset:0; opacity:.12; pointer-events:none;
               background-image: linear-gradient(rgba(255,255,255,.4) 1px, transparent 1px),
                                 linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px);
               background-size: 36px 36px; mask-image: linear-gradient(to bottom, #000, transparent); }
.hero-in { position:relative; z-index:1; max-width:1320px; margin:0 auto; padding:34px 32px 30px;
           display:flex; align-items:center; gap:22px; }
.hero img { width:78px; height:78px; border-radius:18px; object-fit:cover;
            box-shadow:0 0 0 1px rgba(255,255,255,.15), 0 10px 30px rgba(99,102,241,.45); }
.wordmark { font-size:34px; font-weight:800; letter-spacing:.2px; line-height:1.1;
            background:linear-gradient(90deg,#a5b4fc,#c4b5fd 45%,#67e8f9); -webkit-background-clip:text;
            background-clip:text; color:transparent; }
.tagline { color:#cbd5e1; font-size:14px; margin-top:2px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
.chip { font-size:12px; padding:4px 10px; border-radius:999px; color:#e0e7ff;
        background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); }

/* layout + floating table of contents */
.layout { max-width:1320px; margin:0 auto; padding:28px 32px 10px; display:grid;
          grid-template-columns: 240px minmax(0,1fr); gap:32px; }
.toc { position:sticky; top:20px; align-self:start; max-height:calc(100vh - 40px); overflow:auto;
       background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 10px;
       box-shadow:0 1px 2px rgba(15,23,42,.04); }
.toc-title { font-size:11px; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
             color:var(--muted); padding:2px 10px 8px; }
.toc a { display:block; text-decoration:none; color:#334155; font-size:13.5px; padding:6px 10px;
         border-left:3px solid transparent; border-radius:0 8px 8px 0; transition:all .15s; }
.toc a.l2 { font-size:12.5px; padding:4px 10px 4px 24px; color:var(--muted); }
.toc a:hover { background:#f1f5f9; color:var(--ink); }
.toc a.active { border-left-color:var(--brand1); background:#eef2ff; color:#3730a3; font-weight:600; }

/* sections */
section { margin:0 0 34px; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:.14em; color:var(--brand1); text-transform:uppercase; }
h2 { font-size:22px; margin:2px 0 14px; letter-spacing:-.01em; }
h3 { font-size:16px; margin:20px 0 8px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px 24px;
        box-shadow:0 1px 2px rgba(15,23,42,.04), 0 6px 24px rgba(15,23,42,.04); }
.query { border-left:4px solid var(--brand1); font-size:16px; }
.response h2, .response h3 { margin-top:18px; }
.response table, .response th, .response td { border:1px solid var(--line); }
.meta { color:var(--muted); font-size:13px; }

/* at a glance */
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px,1fr)); gap:12px; margin-bottom:30px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; position:relative; overflow:hidden; }
.tile::before { content:""; position:absolute; inset:0 auto 0 0; width:4px;
                background:linear-gradient(var(--brand1),var(--brand3)); }
.tile .v { font-size:24px; font-weight:800; }
.tile .k { font-size:12px; color:var(--muted); }

/* tables */
.scroll { max-height:380px; overflow:auto; border:1px solid var(--line); border-radius:12px; background:#fff; }
table { border-collapse:collapse; width:100%; font-size:12px; line-height:1.45; }
th, td { padding:5px 10px; border-bottom:1px solid #eef0f4; text-align:left; vertical-align:top; }
th { background:#f8fafc; position:sticky; top:0; font-weight:600; color:#334155; z-index:1; }
tbody tr:nth-child(even) { background:#fbfcfe; }
tbody tr:hover { background:#eef2ff; }

/* panels + plots */
.panel { margin:22px 0 8px; }
.panel-h { display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:14px;
           padding:6px 14px; border-radius:999px; color:#fff; margin-bottom:10px; }
.panel-term .panel-h { background:var(--term); }  .panel-term .plot { border-top:3px solid var(--term); }
.panel-cluster .panel-h { background:var(--cluster); }  .panel-cluster .plot { border-top:3px solid var(--cluster); }
.panel-net .panel-h { background:var(--net); }  .panel-net .plot { border-top:3px solid var(--net); }
.plot { background:var(--card); border:1px solid var(--line); border-radius:14px; margin:12px 0 16px;
        box-shadow:0 1px 2px rgba(15,23,42,.04), 0 6px 24px rgba(15,23,42,.04); }
.plot-h { display:flex; align-items:center; gap:10px; padding:12px 16px; border-bottom:1px solid #eef0f4; }
.plot-h .t { font-weight:700; }
.badge { font-size:11px; padding:2px 8px; border-radius:999px; background:#fef3c7; color:#92400e; }
.plot-h .dl { margin:0 0 0 auto; }
.plot-b { padding:8px 10px; overflow-x:auto; }
.plot-b > div { max-width:860px; margin:0 auto; }
.plot details { margin:0 16px 14px; }
.plot summary { cursor:pointer; font-weight:600; font-size:13.5px; padding:6px 0; }

/* buttons */
.dl { display:flex; gap:8px; margin:8px 0; flex-wrap:wrap; }
.dl a, .dl button { font-family:inherit; font-size:12px; font-weight:600; line-height:1; padding:7px 12px; border-radius:8px; border:1px solid #c7d2fe;
                    background:#eef2ff; color:#3730a3; text-decoration:none; cursor:pointer; }
.dl a:hover, .dl button:hover { background:var(--brand1); border-color:var(--brand1); color:#fff; }

/* methods timeline */
.steps { list-style:none; margin:0; padding:0; }
.steps li { position:relative; padding:0 0 16px 44px; }
.steps li::before { content:""; position:absolute; left:15px; top:30px; bottom:0; width:2px; background:#e0e7ff; }
.steps li:last-child::before { display:none; }
.steps .num { position:absolute; left:0; top:0; width:32px; height:32px; border-radius:50%; display:grid; place-items:center;
              font-weight:700; font-size:13px; color:#fff; background:linear-gradient(135deg,var(--brand1),var(--brand2)); }
.steps .tool { font-weight:700; }
.args { margin-top:4px; display:flex; flex-wrap:wrap; gap:6px; }
code { background:#f1f5f9; padding:2px 6px; border-radius:6px; font-size:12.5px; }
.defaults { margin-top:4px; font-size:12.5px; color:var(--muted); }
.facts { margin:14px 0 0; padding-left:18px; }
.facts li { margin:5px 0; }

/* footer */
footer { max-width:1320px; margin:10px auto 0; padding:0 32px 40px; }
.cite { color:var(--muted); font-size:12px; line-height:1.6; border-top:1px solid var(--line); padding-top:14px; }
.cite b { color:#475569; }  .cite a { color:var(--muted); }
.totop { position:fixed; right:22px; bottom:22px; width:42px; height:42px; border-radius:50%; display:grid;
         place-items:center; text-decoration:none; color:#fff; font-size:18px;
         background:linear-gradient(135deg,var(--brand1),var(--brand2)); box-shadow:0 8px 20px rgba(99,102,241,.35); }

@media (max-width: 980px) {
  .layout { grid-template-columns:1fr; padding:20px 16px; }
  .toc { position:static; max-height:none; }
  .hero-in { padding:24px 16px; } footer { padding:0 16px 30px; }
}
@media print {
  .toc, .totop, .dl { display:none !important; }
  .layout { display:block; } .hero { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
"""

CITATION = ("Koumadorakis, D. E., Exarchos, T. P., Vlamos, P., & Vrahatis, A. G. (2026). Enrich.AI: An agentic AI "
            "platform for interactive gene set enrichment analysis and biological interpretation. <i>bioRxiv</i>. "
            '<a href="https://doi.org/10.64898/2026.09.29.755561">https://doi.org/10.64898/2026.09.29.755561</a>')

PANELS = [("term", "📊 Term Analysis"), ("cluster", "🧬 Clustering"), ("net", "🔬 Networks")]

SCROLLSPY = """
<script>
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  var byId = {}; links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
  var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); })
                     .filter(Boolean);
  function update() {
    var y = window.scrollY + 120, current = targets[0];
    targets.forEach(function (t) { if (t.getBoundingClientRect().top + window.scrollY <= y) current = t; });
    links.forEach(function (a) { a.classList.remove('active'); });
    if (current && byId[current.id]) byId[current.id].classList.add('active');
  }
  window.addEventListener('scroll', update, {passive: true}); update();
})();
</script>"""


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _md(text: str) -> str:
    return markdown.markdown(text or "", extensions=["tables", "fenced_code", "sane_lists"])


def _table(df: pd.DataFrame, name: str = "table") -> str:
    """Escaped HTML table (link column rendered as links) with a CSV download button."""
    csv = "data:text/csv;charset=utf-8," + urllib.parse.quote(df.to_csv(index=False))
    button = f'<div class="dl"><a href="{csv}" download="{_esc(name)}.csv">⬇ Download table (CSV)</a></div>'
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c == "Link" and isinstance(v, str) and v.startswith("http"):
                cells.append(f'<td><a href="{_esc(v)}" target="_blank">open</a></td>')
            else:
                cells.append(f"<td>{_esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    return button + f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def tool_calls_from_trace(trace: List[str]) -> List[Tuple[str, Dict]]:
    """Tool calls recorded in the envelope trace as 'Tool: name | Args: {...}'."""
    calls = []
    for line in trace or []:
        m = re.match(r"Tool: (\S+) \| (?:Args|Input): (.*)$", str(line), re.S)
        if not m:
            continue
        try:
            args = ast.literal_eval(m.group(2))
        except Exception:
            args = {"input": m.group(2)}
        calls.append((m.group(1), args if isinstance(args, dict) else {"input": args}))
    return calls


def _enrichment_table(df: pd.DataFrame) -> pd.DataFrame:
    is_gsea = "nes" in df.columns
    d = df.sort_values("adjusted_p_value")
    out = pd.DataFrame({"Term": d["term"], "Library": d["library"]})
    out["FDR" if is_gsea else "Adjusted p-value"] = d["adjusted_p_value"].map(lambda x: f"{x:.2e}")
    if is_gsea:
        out["NES"] = d["nes"].map(lambda x: f"{x:+.2f}")
    for col, label in (("z_score", "Z-score"), ("combined_score", "Combined score")):
        if col in d.columns:
            out[label] = d[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    out["Overlap" if not is_gsea else "Leading edge / set"] = d["overlap"]
    out["Leading-edge genes" if is_gsea else "Genes"] = d["genes"].map(
        lambda g: ", ".join(g) if isinstance(g, list) else str(g))
    return out


def _literature_table(papers: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in papers:
        authors = p.get("authors") or []
        if isinstance(authors, list):
            authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        url = p.get("url") or (f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}" if p.get("pmid")
                               else f"https://doi.org/{p['doi']}" if p.get("doi") else "")
        rows.append({"Title": re.sub(r"<[^>]+>", "", str(p.get("title", ""))), "Authors": authors,
                     "Journal": p.get("journal", ""), "Year": p.get("year", ""),
                     "Citations": p.get("citations", 0),
                     "Relevance": p.get("specificity") or p.get("relevance_rating", ""), "Link": url})
    return pd.DataFrame(rows)


def _gene_html(gene_info) -> str:
    genes = gene_info if isinstance(gene_info, list) else [gene_info]
    parts = []
    for g in genes:
        if not isinstance(g, dict):
            continue
        aliases = g.get("alias") or g.get("aliases") or []
        aliases = [aliases] if isinstance(aliases, str) else aliases
        parts.append(
            f'<div class="card"><h3>{_esc(g.get("symbol") or g.get("gene", ""))} '
            f'<span class="meta">{_esc(g.get("name", ""))}</span></h3>'
            + (f'<p class="meta">Also known as: {_esc(", ".join(aliases[:10]))}</p>' if aliases else "")
            + f'<p>{_esc(g.get("summary", ""))}</p>'
            + f'<p class="meta">KEGG pathways: {len(g.get("kegg_pathways", []) or [])} · '
              f'Reactome pathways: {len(g.get("reactome_pathways", []) or [])}</p></div>')
    return "".join(parts)


def _methods_html(envelope: Dict, tool_defaults: Dict[str, Dict], model: str) -> str:
    calls = tool_calls_from_trace(envelope.get("reasoning_trace") or envelope.get("trace"))
    if calls:
        steps = []
        for i, (tool, args) in enumerate(calls, 1):
            defaults = tool_defaults.get(tool, {})
            given = {k: v for k, v in args.items() if v not in (None, "")}
            if tool == "run_enrichment_analysis" and len(str(given.get("genes", ""))) > 120:
                given["genes"] = f"{len(str(given['genes']).split(','))} genes"
            shown = "".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in given.items()) or \
                "<span class='meta'>no arguments</span>"
            dflt = ", ".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in defaults.items()
                             if k not in given and v not in (None, ""))
            steps.append(f'<li><span class="num">{i}</span><div class="tool">{_esc(TOOL_LABELS.get(tool, tool))}</div>'
                         f'<div class="args">{shown}</div>'
                         + (f'<div class="defaults">Defaults used: {dflt}</div>' if dflt else "") + "</li>")
        steps_html = f"<h3>Tools run by the agent</h3><ol class='steps'>{''.join(steps)}</ol>"
    else:
        steps_html = "<p>No tools were called; the answer is from the language model alone.</p>"

    facts = []
    ora = envelope.get("full_enrichment_results") or {}
    if envelope.get("enrichment_df") is not None:
        libs = ora.get("libraries_tested") or sorted(set(pd.DataFrame(envelope["enrichment_df"])["library"]))
        facts.append("<li><b>Over-representation analysis:</b> Fisher's exact test via the Enrichr API, "
                     "Benjamini–Hochberg adjusted p-values"
                     + (f", {ora['gene_count']} input genes" if ora.get("gene_count") else "")
                     + f". Libraries: {_esc(', '.join(libs))}.</li>")
    gsea = envelope.get("full_gsea_results") or {}
    if envelope.get("gsea_df") is not None:
        prm = gsea.get("parameters", {})
        libs = gsea.get("libraries_tested") or sorted(set(pd.DataFrame(envelope["gsea_df"])["library"]))
        facts.append(
            f"<li><b>GSEA:</b> {_esc(gsea.get('method', 'preranked GSEA (gseapy)'))}; ranking metric "
            f"<code>{_esc(gsea.get('ranking_metric', ''))}</code>, {_esc(gsea.get('genes_ranked', ''))} genes ranked; "
            f"gene-set size {_esc(prm.get('min_size', ''))}–{_esc(prm.get('max_size', ''))}, "
            f"FDR ≤ {_esc(prm.get('fdr_threshold', ''))} (Benjamini–Hochberg), seed {_esc(prm.get('seed', ''))}. "
            f"Libraries: {_esc(', '.join(libs))}.</li>")
    if envelope.get("full_literature_results") or envelope.get("literature"):
        q = envelope.get("literature_query", "")
        facts.append("<li><b>Literature:</b> Europe PMC search" + (f" for <code>{_esc(q)}</code>" if q else "")
                     + "; each paper's relevance rated by the same language model.</li>")
    if model:
        facts.append(f"<li><b>Language model:</b> {_esc(model)}.</li>")
    if envelope.get("execution_time"):
        facts.append(f"<li><b>Run time:</b> {envelope['execution_time']:.1f} s.</li>")
    facts_html = f"<h3>Statistics and data</h3><ul class='facts'>{''.join(facts)}</ul>" if facts else ""
    return f'<div class="card">{steps_html}{facts_html}</div>'


def build_report_html(content: str, envelope: Dict, query: str = "",
                      figures: Optional[Dict[str, List[Tuple]]] = None,
                      tool_defaults: Optional[Dict[str, Dict]] = None, model: str = "",
                      created: str = "", logo_b64: str = "") -> str:
    """figures: {"ora"|"gsea": [(title, figure, customised_in_app, cluster_table_or_None, panel), ...]}
    panel is one of "term", "cluster", "net"."""
    envelope = envelope or {}
    figures = figures or {}
    plotly_js_done = False
    n_fig = 0
    toc, body = [], []

    def section(sid, title, html_, sub=()):
        """Numbered section + its table-of-contents entry (sub: [(id, label)] second-level entries)."""
        toc.append(f'<a href="#{sid}">{_esc(title)}</a>')
        toc.extend(f'<a class="l2" href="#{i}">{_esc(lbl)}</a>' for i, lbl in sub)
        body.append(f'<section id="{sid}"><div class="eyebrow">{len(body) + 1:02d}</div>'
                    f'<h2>{_esc(title)}</h2>{html_}</section>')

    def fig_html(sec, title, fig, customised, clusters):
        nonlocal plotly_js_done, n_fig
        n_fig += 1
        div_id, fname = f"fig{n_fig}", re.sub(r"\W+", "_", f"{sec}_{title}").lower()
        if not customised:  # default figures take the column width; customised keep the user's size
            fig.update_layout(width=None, autosize=True)
        inner = fig.to_html(full_html=False, include_plotlyjs=not plotly_js_done, div_id=div_id,
                            config={"displaylogo": False, "responsive": True})
        plotly_js_done = True
        badge = '<span class="badge">customised in the app</span>' if customised else ""
        buttons = ''.join(
            f'<button onclick="Plotly.downloadImage(\'{div_id}\', {{format: \'{fmt}\', scale: {sc}, '
            f'filename: \'{fname}\'}})">⬇ {fmt.upper()}</button>' for fmt, sc in (("png", 3), ("svg", 1)))
        table = ""
        if clusters is not None and not clusters.empty:
            table = (f"<details open><summary>Terms per cluster ({clusters['Cluster'].nunique()} clusters)</summary>"
                     f"{_table(clusters, fname + '_clusters')}</details>")
        return (f'<div class="plot"><div class="plot-h"><span class="t">{_esc(title)}</span>{badge}'
                f'<div class="dl">{buttons}</div></div><div class="plot-b">{inner}</div>{table}</div>')

    calls = tool_calls_from_trace(envelope.get("reasoning_trace") or envelope.get("trace"))
    papers = envelope.get("full_literature_results") or envelope.get("literature")
    genes = envelope.get("gene_info")
    genes = genes if isinstance(genes, list) else ([genes] if genes else [])
    dfs = {}
    for sec, key in (("ora", "enrichment_df"), ("gsea", "gsea_df")):
        df = envelope.get(key)
        df = pd.DataFrame(df) if isinstance(df, list) else df
        if df is not None and not df.empty:
            dfs[sec] = df

    # at a glance
    tiles = [(len(calls), "tools run")]
    if "ora" in dfs:
        tiles.append((len(dfs["ora"]), "ORA significant terms"))
    if "gsea" in dfs:
        g = dfs["gsea"]
        tiles.append((f"{int((g['nes'] > 0).sum())} ↑ / {int((g['nes'] < 0).sum())} ↓", "GSEA terms (NES)"))
    if papers:
        tiles.append((len(papers), "papers"))
    if genes:
        tiles.append((len(genes), "genes profiled"))
    tiles_html = '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>' for v, k in tiles) + "</div>"

    if query:
        section("question", "Question", f'<div class="card query">{_md(query)}</div>')
    section("response", "Response", f'<div class="card response">{_md(content)}</div>')
    if genes:
        section("genes", "Gene information", _gene_html(genes))
    if papers:
        section("literature", "Literature", f"<p class='meta'>{len(papers)} papers</p>"
                + _table(_literature_table(papers), "literature"))

    for sec, title in (("ora", "Over-representation analysis (ORA)"), ("gsea", "Gene set enrichment analysis (GSEA)")):
        if sec not in dfs:
            continue
        df = dfs[sec]
        parts = [f"<p class='meta'>{len(df)} significant terms across {df['library'].nunique()} libraries</p>",
                 _table(_enrichment_table(df), f"{sec}_results")]
        sub = []
        for panel, label in PANELS:
            figs = [f for f in figures.get(sec, []) if (f[4] if len(f) > 4 else "term") == panel]
            if not figs:
                continue
            pid = f"{sec}-{panel}"
            sub.append((pid, label.split(" ", 1)[1]))
            parts.append(f'<div class="panel panel-{panel}" id="{pid}"><div class="panel-h">{label}</div>'
                         + "".join(fig_html(sec, f[0], f[1], f[2], f[3]) for f in figs) + "</div>")
        section(sec, title, "".join(parts), sub)

    db = envelope.get("full_db_results") or envelope.get("db")
    if isinstance(db, dict) and db.get("results"):
        rdf = pd.DataFrame(db["results"])
        cols = [c for c in ["term_name", "term_id", "library", "num_genes", "overlap_count", "matched_query",
                            "description"] if c in rdf.columns]
        section("database", "Database query", f"<p class='meta'>Task: {_esc(db.get('task_performed', ''))}</p>"
                + _table(rdf[cols] if cols else rdf, "database_results"))

    section("methods", "Methodology", _methods_html(envelope, tool_defaults or {}, model))

    chips = [f"📅 Analysis {_esc(created or 'time not recorded')}", f"💾 Saved {datetime.now():%Y-%m-%d %H:%M}"]
    if model:
        chips.append(f"🤖 {_esc(model)}")
    if APP_VERSION:
        chips.append(f"🏷️ v{_esc(APP_VERSION)}")
    logo = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Enrich.AI logo">' if logo_b64 else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enrich.AI report</title><style>{CSS}</style></head>
<body id="top">
<header class="hero"><div class="hero-in">{logo}<div>
<div class="wordmark">Enrich.AI</div>
<div class="tagline">Agentic gene set enrichment analysis and biological interpretation</div>
<div class="chips">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
</div></div></header>
<div class="layout">
<nav class="toc"><div class="toc-title">Contents</div>{''.join(toc)}</nav>
<main>{tiles_html}{''.join(body)}</main>
</div>
<footer><div class="cite"><b>How to cite.</b> {CITATION}<br>
<span class="meta">Generated with Enrich.AI · <a href="https://enrichai.streamlit.app">enrichai.streamlit.app</a></span></div></footer>
<a class="totop" href="#top" title="Back to top">↑</a>
{SCROLLSPY}
</body></html>"""