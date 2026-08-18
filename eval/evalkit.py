"""
eval/evalkit.py — shared pieces for the 50-set Hallmark benchmark.

The QUERY, DECOMPOSE_PROMPT, JUDGE_PROMPT, build_evidence() and the scoring buckets
are copied VERBATIM from run_variance.py (the pilot script). Do not edit them here
without editing them there — the benchmark's comparability depends on it.

Phase 1 (phase1_run.py) uses: QUERY, tool plumbing, error classification.
Phase 2 (phase2_judge.py) uses: DECOMPOSE_PROMPT, JUDGE_PROMPT, build_evidence, tally.
"""
from __future__ import annotations
import html, json, re
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA, RESULTS, RUNS = BASE / "data", BASE / "results", BASE / "runs"
MSIGDB_FILE = DATA / "msigdb_sets.json"
FULL_RUN_DIR = RUNS / "full_run"
SUMMARY_FILE = RESULTS / "summary.json"

# --- knobs ---
# MAX_TERMS_PER_LIB was 15: a token-limit safeguard from the pilot. It meant the judge
# saw ~9% of the significant terms, so "absent from the evidence" was uninformative and
# leaked into UNSUPPORTED. None = no cap; the judge sees every significant term the tool
# actually retrieved, which is what makes absence a meaningful signal.
# MAX_ABSTRACT_CHARS was 400: abstracts were truncated mid-sentence, so a fact stated
# in the second half of an abstract looked absent to the judge. None = full abstracts.
MAX_TERMS_PER_LIB, ADJ_P_CUTOFF, MAX_ABSTRACT_CHARS = None, 0.05, None

# --- the pinned benchmark query (from run_variance.py) ---
# ONE deliberate deviation from the pilot: the gene set's name is passed as biological
# context, so the agent can anchor its literature queries to the actual topic instead of
# guessing from the gene list. Same phrasing as check_literature.py's existing pattern.
QUERY = ("Analyze this gene set (biological context: {context}) and give a biological interpretation. "
         "Use the GO_Biological_Process_2023 and KEGG_2021_Human libraries for enrichment, "
         "and you must call search_literature to retrieve at most the five most relevant papers for "
         "supporting evidence, and cite only papers you actually retrieved from the search results. "
         "Investigate thoroughly, but keep the final interpretation focused and brief: {genes}")

# --- decomposition prompt: note the p-value-recitation filter on the 4th bullet ---
DECOMPOSE_PROMPT = """Break the biological interpretation below into ATOMIC FACTUAL CLAIMS.
- Each claim: one self-contained, verifiable biological assertion. Resolve references to explicit gene/pathway names.
- INCLUDE: gene functions, molecular mechanisms, pathway roles, regulatory relationships, disease associations.
- EXCLUDE: generic framing, hedging, recommendations, restatements of the input gene list.
- EXCLUDE claims that merely recite enrichment statistics (e.g. "Term X enriched with p=1e-20"). Extract biology, not the statistical readout.
- Decompose the ENTIRE interpretation. There is NO limit on the number of claims - emit every
  distinct factual assertion the text makes. A sentence carrying two facts becomes two claims.
  Omitting claims biases the faithfulness score, so completeness matters more than brevity.
- One sentence per claim.
Return ONLY a JSON array of strings. No markdown.
TEXT:
---
{interpretation}
---
"""

# --- BACKGROUND-aware judge prompt (lenient variant, verbatim) ---
JUDGE_PROMPT = """You are a molecular biology expert. Given EVIDENCE (enriched terms with overlapping genes, and abstracts) and numbered CLAIMS, label each claim:
- "GROUNDED": explicitly supported by the evidence. For "GENE does FUNCTION", the gene must appear in a matching term's gene list.
- "BACKGROUND": not stated in the evidence, but well-established textbook-correct biology.
- "UNSUPPORTED": the claim is factually WRONG.

The EVIDENCE below contains every significant enriched term that was retrieved, so it is
reasonably complete - but it is still not a complete account of biology.

Rules for choosing between BACKGROUND and UNSUPPORTED - apply these strictly:
1. UNSUPPORTED requires you to state, in "why", what is factually INCORRECT about the claim:
   the wrong gene, the wrong direction of an effect, a relationship that does not exist, a
   contradicted fact. Name the error.
2. Absence from the evidence is NOT an error. If you cannot say what is factually wrong with
   a claim, and it is simply not stated in the evidence, the label is BACKGROUND.
3. Never justify an UNSUPPORTED label with wording like "X does not appear in the evidence",
   "no term matches", or "not listed". A justification of that form means the label is
   BACKGROUND, not UNSUPPORTED.
4. A claim about a gene's well-established function is BACKGROUND when the evidence does not
   happen to state it - it is only UNSUPPORTED if the stated function is actually wrong.

Do not mark GROUNDED merely for topical similarity; verify the specific gene/fact.
Return ONLY a JSON array, one object per claim in order: [{{"i":0,"label":"GROUNDED","why":"short"}}, ...]
EVIDENCE:
{evidence}
CLAIMS:
{claims}
"""


# --- parsers / helpers (verbatim from run_variance.py) ---
def parse_json_array(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text); text = re.sub(r'\s*```$', '', text)
    s = text.find('[')
    if s != -1: text = text[s:]
    try:
        e = text.rfind(']')
        return json.loads(text[:e + 1] if e != -1 else text)
    except Exception:
        pass
    objs, depth, start, instr, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if esc: esc = False; continue
        if ch == '\\': esc = True; continue
        if ch == '"': instr = not instr
        if instr: continue
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: objs.append(json.loads(text[start:i + 1]))
                except Exception: pass
                start = None
    return objs


def build_evidence(fer, papers):
    parts = ["=== ENRICHED TERMS (adjusted p<0.05), each lists overlapping genes ==="]
    enr = (fer or {}).get("enrichment_results") or {}
    for lib, terms in enr.items():
        if not isinstance(terms, list): continue
        kept = [t for t in terms if isinstance(t, dict) and isinstance(t.get("adjusted_p_value"), (int, float))
                and t["adjusted_p_value"] <= ADJ_P_CUTOFF]
        if MAX_TERMS_PER_LIB is not None:
            kept = kept[:MAX_TERMS_PER_LIB]
        if not kept: continue
        parts.append(f"\n[{lib}]")
        for t in kept:
            g = ", ".join(t.get("overlap_genes", []) or [])
            parts.append(f"- {t.get('term_name','?')} -> genes: [{g}]")
    lit = papers or []      # every retrieved paper, not just the first 5
    if lit:
        parts.append("\n=== ABSTRACTS ===")
        for i, p in enumerate(lit, 1):
            if isinstance(p, dict):
                abstract = p.get('abstract', '') or ''
                if MAX_ABSTRACT_CHARS is not None:
                    abstract = abstract[:MAX_ABSTRACT_CHARS]
                parts.append(f"[{i}] {p.get('title','')}\n{abstract}")
    return "\n".join(parts)


def gtext(resp):
    c = resp.content
    return c if isinstance(c, str) else "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def cleanj(o):
    try: json.dumps(o); return o
    except Exception: return json.loads(json.dumps(o, default=str))


def tally(claims, labels):
    """Scoring buckets, verbatim from run_variance.py score().

    Unscored claims are SKIPPED (not defaulted to UNSUPPORTED), so a truncated
    judge response cannot inflate the hallucination rate.
    """
    by_i = {int(o["i"]): o for o in labels if isinstance(o, dict) and "i" in o}
    counts = {"GROUNDED": 0, "BACKGROUND": 0, "UNSUPPORTED": 0}
    scored = []
    for i, c in enumerate(claims):
        if i not in by_i: continue
        lab = str(by_i[i].get("label", "UNSUPPORTED")).upper()
        if lab not in counts: lab = "UNSUPPORTED"
        counts[lab] += 1
        scored.append({"claim": c, "label": lab, "why": by_i[i].get("why", "")})
    tot = sum(counts.values()) or 1
    metrics = {"n_claims": len(claims), "n_scored": sum(counts.values()),
               "grounded_rate": round(counts["GROUNDED"] / tot, 3),
               "hallucination_rate": round(counts["UNSUPPORTED"] / tot, 3),
               **{k.lower(): v for k, v in counts.items()}}
    return metrics, scored


def load_gene_sets():
    if not MSIGDB_FILE.exists():
        raise SystemExit("msigdb_sets.json not found. Run fetch_msigdb.py first.")
    data = json.load(open(MSIGDB_FILE, encoding="utf-8"))
    return {name: info["genes"] for name, info in sorted(data.items())}


def load_contexts():
    """{set_name: display name} - the Hallmark set's own title, used as the
    biological context handed to the agent (e.g. 'Allograft Rejection')."""
    data = json.load(open(MSIGDB_FILE, encoding="utf-8"))
    return {name: (info.get("ground_truth") or name.replace("_", " "))
            for name, info in sorted(data.items())}


# ===============================================================================
# CITATION AUDIT - deterministic, no LLM call, no cost
# ===============================================================================

def _norm_title(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def enrichment_term_names(fer) -> set:
    """Every enriched term name the tool actually returned.

    Needed so the auditor does not mistake a quoted GO/KEGG term (which the model
    legitimately quotes from its own evidence) for a fabricated paper title.
    """
    names = set()
    for terms in ((fer or {}).get("enrichment_results") or {}).values():
        if not isinstance(terms, list): continue
        for t in terms:
            if isinstance(t, dict):
                n = _norm_title(t.get("term_name") or t.get("term") or "")
                if n: names.add(n)
    return names


def audit_citations(interpretation: str, papers: list, term_names: set | None = None) -> dict:
    """Does the interpretation cite a paper it did not actually retrieve?

    Three independent signals:
      * a PMID that is not in the retrieved set
      * a quoted paper-length title matching no retrieved title (quoted enrichment
        terms are excluded - those are evidence the tool really returned)
      * an author-year citation, e.g. "(Tomlins et al., 2008)", whose surname
        appears in no retrieved paper's author list
    """
    txt = interpretation or ""
    papers = [p for p in (papers or []) if isinstance(p, dict)]
    term_names = term_names or set()

    got_pmids = {str(p.get("pmid")).strip() for p in papers if p.get("pmid")}
    got_titles = [t for t in (_norm_title(p.get("title", "")) for p in papers) if t]
    got_authors = " ".join(
        " ".join(p.get("authors", [])) if isinstance(p.get("authors"), list) else str(p.get("authors", ""))
        for p in papers
    ).lower()

    cited_pmids = set(re.findall(r"PMID[:\s]*(\d{6,9})", txt))
    fabricated_pmids = sorted(cited_pmids - got_pmids)

    def _matches(nq, pool):
        return any(gt and (SequenceMatcher(None, nq, gt).ratio() > 0.60 or nq in gt or gt in nq)
                   for gt in pool)

    # A quoted string only counts as a claimed paper title when it sits in citation
    # context. Without this the auditor flags quoted GO terms, search queries the model
    # echoes, and ordinary quoted phrases - all false positives seen in smoke testing.
    _CITE_CONTEXT = re.compile(r"et al\.|PMID|doi|journal|paper|study|publication|\(19|\(20", re.I)
    fabricated_titles = []
    for m in re.finditer(r'["“]([^"”\n]{25,200})["”]', txt):
        q = m.group(1)
        if re.search(r"GO:\d|p\s*=|\*\*", q):         # markdown / stats artefact, not a title
            continue
        # A paper title starts with a capital. Quoted mid-sentence prose, quoted
        # abstract excerpts, and fragments the regex caught between two adjacent
        # quotes all start lowercase - those are not citations.
        if not re.match(r"^[A-Z0-9]", q.strip()):
            continue
        before = txt[max(0, m.start() - 60): m.start()]
        if re.search(r"search(?:ed|ing)?\s+(?:for|with)?\s*$|quer(?:y|ied|ies)\s*\S*\s*$",
                     before, re.I):                   # the model quoting its own search query
            continue
        window = txt[max(0, m.start() - 100): m.end() + 100]
        if not _CITE_CONTEXT.search(window):          # nobody is claiming this is a paper
            continue
        nq = _norm_title(q)
        if _matches(nq, term_names):                  # it's an enriched term, not a citation
            continue
        if not _matches(nq, got_titles):
            fabricated_titles.append(q)

    # "(Surname et al., 2008)" style references
    fabricated_author_years = []
    for surname, year in re.findall(r"\(([A-Z][A-Za-z\-']{2,})\s+et al\.,?\s*(\d{4})\)", txt):
        if surname.lower() not in got_authors:
            fabricated_author_years.append(f"{surname} et al., {year}")

    # bare "et al." with nothing retrieved at all
    etal_with_no_papers = len(re.findall(r"et al\.", txt)) if not papers else 0

    return {
        "n_papers_retrieved": len(papers),
        "cited_pmids": sorted(cited_pmids),
        "fabricated_pmids": fabricated_pmids,
        "fabricated_titles": fabricated_titles,
        "fabricated_author_years": fabricated_author_years,
        "et_al_with_no_papers": etal_with_no_papers,
        "has_fabricated_citation": bool(
            fabricated_pmids or fabricated_titles or fabricated_author_years or etal_with_no_papers),
    }


# ===============================================================================
# ERROR CLASSIFICATION - transient (retry) vs fatal (stop the whole run)
# ===============================================================================

# Won't clear on retry: quota exhausted, out of credit, bad/blocked key.
_FATAL_PATTERNS = (
    "resource_exhausted", "resource has been exhausted", "quota exceeded",
    "quota_exceeded", "insufficient_quota", "exceeded your current quota",
    "billing", "credit balance", "insufficient credit", "payment required",
    "per day", "daily limit", "tokens per day", "requests per day",
    "invalid api key", "api key not valid", "api_key_invalid",
    "unauthorized", "permission denied", "permission_denied",
    "authentication", "401", "403",
)
# Clears on its own: per-minute rate limits, transient server/network faults.
_TRANSIENT_PATTERNS = (
    "per minute", "rate limit", "rate_limit", "429", "too many requests",
    "timeout", "timed out", "deadline exceeded", "connection", "temporarily",
    "unavailable", "503", "502", "500", "internal error", "overloaded",
)


def classify_error(exc) -> str:
    """Return 'fatal' or 'transient'.

    Fatal wins over transient when both match, EXCEPT a 429 that only names a
    per-minute limit, which is genuinely transient. A 429 naming a daily/quota
    limit is fatal - it will not clear within the run.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    fatal_hit = any(p in msg for p in _FATAL_PATTERNS)
    transient_hit = any(p in msg for p in _TRANSIENT_PATTERNS)
    if fatal_hit and transient_hit:
        per_minute_only = ("per minute" in msg or "per-minute" in msg) and not any(
            p in msg for p in ("per day", "daily", "quota exceeded", "credit", "billing")
        )
        return "transient" if per_minute_only else "fatal"
    if fatal_hit:
        return "fatal"
    return "transient"
