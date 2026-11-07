"""
HTML report for one Enrich.AI response.

Self-contained file: Enrich.AI header (logo, name, version if set), the question, the response,
the result tables and plots exactly as passed in (customised dialog figures included), and a
short methodology section built from the tool calls the agent actually made.
"""

import ast
import html
import re
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
:root { --ink:#1e293b; --muted:#64748b; --line:#e2e8f0; --accent:#3b82f6; --accent2:#8b5cf6; }
* { box-sizing: border-box; }
body { margin:0; background:#f8fafc; color:var(--ink); font:15px/1.6 -apple-system, "Segoe UI", Roboto, Arial, sans-serif; }
.page { max-width: 1100px; margin: 0 auto; padding: 0 24px 48px; }
header { display:flex; align-items:center; gap:18px; padding:22px 24px; margin:0 -24px 28px;
         background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%); color:#f1f5f9; }
header img { width:64px; height:64px; border-radius:14px; object-fit:cover; }
header .name { font-size:28px; font-weight:800; letter-spacing:.3px;
               background:linear-gradient(135deg,#93c5fd,#c4b5fd,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent; }
header .sub { color:#cbd5e1; font-size:13px; }
h2 { font-size:20px; margin:34px 0 12px; padding-bottom:6px; border-bottom:2px solid var(--line); }
h3 { font-size:16px; margin:22px 0 8px; }
.card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:18px 22px; margin:12px 0; }
.query { border-left:4px solid var(--accent); }
.meta { color:var(--muted); font-size:13px; }
.scroll { max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:13px; background:#fff; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#f1f5f9; position:sticky; top:0; }
.plot { background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; margin:14px 0; overflow-x:auto; }
.plot .cap { font-weight:600; margin:2px 6px 6px; }
.plot .note { color:var(--muted); font-weight:400; font-size:12px; }
.methods li { margin:4px 0; }
.methods code, .response code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:12.5px; }
footer { margin-top:40px; color:var(--muted); font-size:12px; text-align:center; }
"""


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _md(text: str) -> str:
    return markdown.markdown(text or "", extensions=["tables", "fenced_code", "sane_lists"])


def _table(df: pd.DataFrame) -> str:
    """Escaped HTML table (link column rendered as links)."""
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
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


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
    items = []
    calls = tool_calls_from_trace(envelope.get("reasoning_trace") or envelope.get("trace"))
    if calls:
        steps = []
        for i, (tool, args) in enumerate(calls, 1):
            defaults = tool_defaults.get(tool, {})
            given = {k: v for k, v in args.items() if v not in (None, "")}
            if tool in ("run_enrichment_analysis",) and len(str(given.get("genes", ""))) > 120:
                given["genes"] = f"{len(str(given['genes']).split(','))} genes"
            shown = ", ".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in given.items()) or "no arguments"
            dflt = ", ".join(f"<code>{_esc(k)}={_esc(v)}</code>" for k, v in defaults.items()
                             if k not in given and v not in (None, ""))
            steps.append(f"<li><b>{i}. {_esc(TOOL_LABELS.get(tool, tool))}</b> — {shown}"
                         + (f'<br><span class="meta">defaults: {dflt}</span>' if dflt else "") + "</li>")
        items.append(f"<li>Tools run by the agent, in order:<ol style='list-style:none;padding-left:8px'>"
                     f"{''.join(steps)}</ol></li>")
    else:
        items.append("<li>No tools were called; the answer is from the language model alone.</li>")

    ora = envelope.get("full_enrichment_results") or {}
    if envelope.get("enrichment_df") is not None:
        libs = ora.get("libraries_tested") or sorted(set(envelope["enrichment_df"]["library"]))
        items.append(
            "<li><b>Over-representation analysis:</b> Fisher's exact test via the Enrichr API, "
            "Benjamini–Hochberg adjusted p-values"
            + (f", {ora['gene_count']} input genes" if ora.get("gene_count") else "")
            + f". Libraries: {_esc(', '.join(libs))}.</li>")
    gsea = envelope.get("full_gsea_results") or {}
    if envelope.get("gsea_df") is not None:
        prm = gsea.get("parameters", {})
        libs = gsea.get("libraries_tested") or sorted(set(envelope["gsea_df"]["library"]))
        items.append(
            f"<li><b>GSEA:</b> {_esc(gsea.get('method', 'preranked GSEA (gseapy)'))}; ranking metric "
            f"<code>{_esc(gsea.get('ranking_metric', ''))}</code>, {_esc(gsea.get('genes_ranked', ''))} genes ranked; "
            f"gene-set size {_esc(prm.get('min_size', ''))}–{_esc(prm.get('max_size', ''))}, "
            f"FDR ≤ {_esc(prm.get('fdr_threshold', ''))} (Benjamini–Hochberg), seed {_esc(prm.get('seed', ''))}. "
            f"Libraries: {_esc(', '.join(libs))}.</li>")
    if envelope.get("full_literature_results") or envelope.get("literature"):
        q = envelope.get("literature_query", "")
        items.append("<li><b>Literature:</b> Europe PMC search"
                     + (f" for <code>{_esc(q)}</code>" if q else "")
                     + "; each paper's relevance rated by the same language model.</li>")
    if model:
        items.append(f"<li><b>Language model:</b> {_esc(model)}.</li>")
    if envelope.get("execution_time"):
        items.append(f"<li><b>Run time:</b> {envelope['execution_time']:.1f} s.</li>")
    return f'<ul class="methods">{"".join(items)}</ul>'


def build_report_html(content: str, envelope: Dict, query: str = "",
                      figures: Optional[Dict[str, List[Tuple[str, go.Figure, bool]]]] = None,
                      tool_defaults: Optional[Dict[str, Dict]] = None, model: str = "",
                      created: str = "", logo_b64: str = "") -> str:
    """figures: {"ora"|"gsea": [(title, figure, customised_in_app), ...]}"""
    envelope = envelope or {}
    figures = figures or {}
    plotly_js_done = False

    def fig_html(title, fig, customised):
        nonlocal plotly_js_done
        inner = fig.to_html(full_html=False, include_plotlyjs=not plotly_js_done,
                            config={"displaylogo": False, "responsive": True})
        plotly_js_done = True
        note = ' <span class="note">(as customised in the app)</span>' if customised else ""
        return f'<div class="plot"><div class="cap">{_esc(title)}{note}</div>{inner}</div>'

    body = []
    if query:
        body.append(f'<h2>Question</h2><div class="card query">{_md(query)}</div>')
    body.append(f'<h2>Response</h2><div class="card response">{_md(content)}</div>')

    if envelope.get("gene_info"):
        body.append("<h2>Gene information</h2>" + _gene_html(envelope["gene_info"]))

    papers = envelope.get("full_literature_results") or envelope.get("literature")
    if papers:
        body.append(f"<h2>Literature search results</h2><p class='meta'>{len(papers)} papers</p>"
                    + _table(_literature_table(papers)))

    for key, title in (("enrichment_df", "Over-representation analysis (ORA)"), ("gsea_df", "GSEA")):
        df = envelope.get(key)
        if isinstance(df, list):
            df = pd.DataFrame(df)
        if df is None or df.empty:
            continue
        sec = "ora" if key == "enrichment_df" else "gsea"
        body.append(f"<h2>{title} results</h2><p class='meta'>{len(df)} significant terms</p>"
                    + _table(_enrichment_table(df)))
        for t, f, c in figures.get(sec, []):
            body.append(fig_html(t, f, c))

    db = envelope.get("full_db_results") or envelope.get("db")
    if isinstance(db, dict) and db.get("results"):
        rdf = pd.DataFrame(db["results"])
        cols = [c for c in ["term_name", "term_id", "library", "num_genes", "overlap_count", "matched_query",
                            "description"] if c in rdf.columns]
        body.append(f"<h2>Database query results</h2><p class='meta'>Task: {_esc(db.get('task_performed', ''))}</p>"
                    + _table(rdf[cols] if cols else rdf))

    body.append("<h2>Methodology</h2>" + _methods_html(envelope, tool_defaults or {}, model))

    created = created or "time not recorded"
    version = f" · version {APP_VERSION}" if APP_VERSION else ""
    logo = f'<img src="data:image/jpeg;base64,{logo_b64}" alt="Enrich.AI logo">' if logo_b64 else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Enrich.AI report</title><style>{CSS}</style></head>
<body><div class="page">
<header>{logo}<div><div class="name">Enrich.AI</div>
<div class="sub">Agentic gene set enrichment analysis and biological interpretation{version}</div>
<div class="sub">Analysis run {_esc(created)} · report saved {datetime.now():%Y-%m-%d %H:%M}</div></div></header>
{''.join(body)}
<footer>Generated with Enrich.AI · enrichai.streamlit.app</footer>
</div></body></html>"""
